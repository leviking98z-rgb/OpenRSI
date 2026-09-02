#!/usr/bin/env python3
"""Materialize an OpenMLE `recipe` task into the same on-disk layout that
`continuous_worker.py` expects from a `built_task`.

A recipe ships everything except the data: task_metadata.json, metric.py and
prepare.py. We fetch the upstream Kaggle dataset ourselves, run the recipe's own
prepare.py to derive the train/test/answer split, and lay the result out as

    <out>/<task_id>/
        RELEASE_METADATA.json
        info/task_metadata.json
        info/data_description.txt
        data/public/{train,test,sample_submission}.csv
        data/private/test_answer.csv
        utils/{metric.py,prepare.py,samples/...}
        utils/public -> ../data/public      (symlink, as the worker expects)
        .verified.json                      (marker, mirrors the built_task one)

prepare.py MUST run inside the canonical sandbox image, not on the host: the
whole point of the canonical-image rule is that data derived under a different
pandas/numpy is not comparable. See MEMORY openrsi-ma1-round3-canonical-sandbox.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import quote

REVISION = "f56e4b31252a9b81d95fea100098cd49b7290398"
REPOSITORY = "https://huggingface.co/datasets/FrontisAI/OpenMLE-Tasks/resolve"
IMAGE = "ccr.ccs.tencentyun.com/frontisai-openmle/openmle-sandbox-worker:1.0.0"
# Per-container memory ceiling; overridable for nodes with more headroom.
MEM_LIMIT = os.environ.get("MATERIALIZE_MEM_LIMIT", "6g")
# Above this many files, publish as a single tar instead of a directory tree
# (see publish() -- cephfs caps small-file creation at ~9.5 files/s).
PACK_ABOVE_FILES = int(os.environ.get("MATERIALIZE_PACK_ABOVE", "500"))
# Refuse to start a task with less than this much scratch headroom; a large
# archive plus its extracted tree needs several GB and ENOSPC mid-task looks
# exactly like a broken task in the results file.
MIN_FREE_GB = float(os.environ.get("MATERIALIZE_MIN_FREE_GB", "15"))

RECIPE_FILES = (
    "RELEASE_METADATA.json",
    "info/task_metadata.json",
    "info/data_description.txt",
    "description.txt",
    "utils/metric.py",
    "utils/prepare.py",
    "utils/samples/base_metric.py",
    "utils/samples/sample_metric.py",
    "utils/samples/sample_prepare.py",
    "utils/samples/sample_utils.py",
)
# Files the worker's own validation requires to exist after materialization.
REQUIRED = (
    "RELEASE_METADATA.json",
    "info/task_metadata.json",
    "data/public/train.csv",
    "data/public/test.csv",
    "data/public/sample_submission.csv",
    "data/private/test_answer.csv",
    "utils/metric.py",
)
# Recipe files that are absent for some tasks and must not abort materialization.
OPTIONAL_RECIPE_FILES = frozenset(
    ("info/data_description.txt", "description.txt")
) | frozenset(f for f in RECIPE_FILES if f.startswith("utils/samples/"))

# The `raw/` layout prepare.py expects varies per task: some address files by
# their upstream name, some expect a single `dataset.csv`, some walk directories.
# prepare.py asserts on what it needs, so we try each shape and let it decide.
# `flat` keeps upstream filenames and goes first — it is right most of the time.
RAW_LAYOUTS = ("flat", "nested_dir", "single_csv_as_dataset", "flat_plus_dataset_alias")


class Fail(RuntimeError):
    """Materialization failed for a reason we want recorded, not a crash."""


def log(*a):
    print(*a, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_get(artifact_path: str, relative: str, token: str, timeout=90) -> bytes | None:
    # '@' must stay literal: HF's CDN redirect rejects the %40 form and returns
    # the redirect page as a 200 body, which then fails as a corrupt archive.
    url = f"{REPOSITORY}/{REVISION}/{artifact_path}/{relative}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            if attempt == 3:
                raise Fail(f"hf {relative}: HTTP {e.code}")
        except Exception as e:
            if attempt == 3:
                raise Fail(f"hf {relative}: {type(e).__name__}")
        time.sleep(2 * (attempt + 1))
    return None


def kaggle_size(cand: str, user: str, key: str, timeout=60) -> int | None:
    """Total bytes of a Kaggle dataset zip, via a 1-byte range request."""
    auth = base64.b64encode(f"{user}:{key}".encode()).decode()
    url = f"https://www.kaggle.com/api/v1/datasets/download/{cand}"
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}",
                                               "Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            cr = r.headers.get("Content-Range")  # "bytes 0-0/<total>"
            if cr and "/" in cr:
                tail = cr.rsplit("/", 1)[1]
                if tail.isdigit():
                    return int(tail)
    except Exception:
        return None
    return None


def kaggle_download(ref: str, dest: Path, user: str, key: str, timeout=1800,
                    max_bytes: int | None = None) -> str:
    """Download a Kaggle dataset zip. `ref` may list fallback mirrors as "a/b | c/d".

    `max_bytes` guards against upstream datasets far larger than the recipe
    metadata claims (some entries understate size by 4 orders of magnitude — e.g.
    an LLM-weights dataset listed as tiny). Checked before transferring.

    The dataset endpoint has its own per-account quota, separate from the
    competition one but just as exhaustible: 24 shards across three nodes hit it
    within minutes. As with competitions, a 429 parks every shard on the shared
    backoff and the task is retried rather than consumed.
    """
    auth = base64.b64encode(f"{user}:{key}".encode()).decode()
    for attempt in range(COMP_429_RETRIES + 1):
        wait_out_shared_backoff("dataset")
        try:
            return _dataset_fetch(ref, dest, auth, user, key, timeout, max_bytes)
        except RateLimited as e:
            if attempt == COMP_429_RETRIES:
                raise Fail(f"kaggle download {ref}: rate limited (429) after "
                           f"{COMP_429_RETRIES} retries")
            delay = e.retry_after or min(COMP_429_BASE_SLEEP * (2 ** attempt), 1800)
            note_shared_backoff(delay, "dataset")
            log(f"  429 on dataset {ref}; backing off {delay}s (attempt {attempt+1})")
            time.sleep(delay)
    raise Fail(f"kaggle download {ref}: rate limited (429)")


def _dataset_fetch(ref: str, dest: Path, auth: str, user: str, key: str,
                   timeout: int, max_bytes: int | None) -> str:
    errors = []
    saw_429 = False
    for cand in [c.strip() for c in ref.split("|") if c.strip()]:
        if max_bytes:
            total = kaggle_size(cand, user, key)
            if total and total > max_bytes:
                errors.append(f"{cand}: {total/1e9:.1f}GB exceeds cap {max_bytes/1e9:.1f}GB")
                continue
        url = f"https://www.kaggle.com/api/v1/datasets/download/{cand}"
        tmp = dest.with_suffix(".part")
        cmd = ["curl", "-L", "--fail", "--silent", "--show-error", "--retry", "4",
               "--connect-timeout", "20", "--max-time", str(timeout),
               "-H", f"Authorization: Basic {auth}", "-o", str(tmp), url]
        if max_bytes:
            cmd.extend(["--max-filesize", str(max_bytes)])
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
            if not zipfile.is_zipfile(tmp):
                errors.append(f"{cand}: not a zip")
                tmp.unlink(missing_ok=True)
                continue
            tmp.replace(dest)
            return cand
        if "429" in (p.stderr or ""):
            saw_429 = True
        errors.append(f"{cand}: rc={p.returncode} {(p.stderr or '')[:80]}")
        tmp.unlink(missing_ok=True)
    # A 429 says nothing about whether the task is materializable, so surface it
    # as a retryable condition instead of burning the task on a quota blip.
    if saw_429:
        raise RateLimited()
    raise Fail("kaggle download failed: " + "; ".join(errors))


def kaggle_competition_download(slug: str, dest: Path, user: str, key: str,
                                timeout=3600, max_bytes: int | None = None) -> str:
    """Download a competition's data bundle.

    Competitions use a different endpoint from datasets and enforce rules
    acceptance at download time (403 with a "must accept this competition's
    rules" message). The account must already have accepted; see accept_rules.py.

    The competition download quota is per-account, so parallel shards share it
    and can exhaust it between them (8 shards burned it in minutes and then spun
    through hundreds of tasks recording 429s that are not real failures). On a
    429 the shard sleeps out the backoff and retries rather than consuming the
    task, and writes the deadline to a shared file so sibling shards on every
    node wait too instead of each rediscovering the limit.
    """
    auth = base64.b64encode(f"{user}:{key}".encode()).decode()
    url = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{slug}"

    for attempt in range(COMP_429_RETRIES + 1):
        wait_out_shared_backoff()
        try:
            return _competition_fetch(slug, dest, url, auth, timeout, max_bytes)
        except RateLimited as e:
            if attempt == COMP_429_RETRIES:
                raise Fail(f"competition {slug}: rate limited (429) after "
                           f"{COMP_429_RETRIES} retries")
            delay = e.retry_after or min(COMP_429_BASE_SLEEP * (2 ** attempt), 1800)
            note_shared_backoff(delay)
            log(f"  429 on {slug}; backing off {delay}s (attempt {attempt+1})")
            time.sleep(delay)
    raise Fail(f"competition {slug}: rate limited (429)")


class RateLimited(Exception):
    def __init__(self, retry_after=None):
        super().__init__("429")
        self.retry_after = retry_after


COMP_429_RETRIES = int(os.environ.get("MATERIALIZE_429_RETRIES", "4"))
COMP_429_BASE_SLEEP = int(os.environ.get("MATERIALIZE_429_SLEEP", "120"))
# Shared across every shard on every node, so one shard hitting the quota parks
# the rest instead of each burning its own request to learn the same thing.
# Keyed per endpoint: the dataset and competition quotas are independent, so a
# competition block must not stall dataset work (and vice versa).
BACKOFF_DIR = Path(os.environ.get(
    "MATERIALIZE_BACKOFF_DIR",
    "/root/shared/.clusters/.tmp/kaggle-expand/.backoff"))


def _backoff_file(kind: str) -> Path:
    BACKOFF_DIR.mkdir(parents=True, exist_ok=True)
    return BACKOFF_DIR / f"{kind}_until"


def note_shared_backoff(seconds: float, kind: str = "competition") -> None:
    try:
        _backoff_file(kind).write_text(str(time.time() + seconds))
    except OSError:
        pass


def wait_out_shared_backoff(kind: str = "competition") -> None:
    try:
        until = float(_backoff_file(kind).read_text().strip())
    except (OSError, ValueError):
        return
    remaining = until - time.time()
    if remaining > 0:
        log(f"  shared 429 backoff ({kind}): sleeping {int(remaining)}s")
        time.sleep(min(remaining, 1800))


def _competition_fetch(slug: str, dest: Path, url: str, auth: str,
                       timeout: int, max_bytes: int | None) -> str:
    if max_bytes:
        req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}",
                                                  "Range": "bytes=0-0"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                cr = r.headers.get("Content-Range")
                if cr and "/" in cr:
                    tail = cr.rsplit("/", 1)[1]
                    if tail.isdigit() and int(tail) > max_bytes:
                        raise Fail(f"competition {slug}: {int(tail)/1e9:.1f}GB exceeds cap "
                                   f"{max_bytes/1e9:.1f}GB")
        except urllib.error.HTTPError as e:
            if e.code == 403:
                raise Fail(f"competition {slug}: rules not accepted (403)")
            if e.code == 429:
                ra = e.headers.get("retry-after")
                raise RateLimited(int(ra) if ra and ra.isdigit() else None)
        except (Fail, RateLimited):
            raise
        except Exception:
            pass  # size check is advisory; the transfer below still has --max-filesize

    tmp = dest.with_suffix(".part")
    cmd = ["curl", "-L", "--fail", "--silent", "--show-error", "--retry", "3",
           "--connect-timeout", "20", "--max-time", str(timeout),
           "-w", "%{http_code}",
           "-H", f"Authorization: Basic {auth}", "-o", str(tmp), url]
    if max_bytes:
        cmd.extend(["--max-filesize", str(max_bytes)])
    p = subprocess.run(cmd, capture_output=True, text=True)
    if "429" in (p.stdout or "") or "429" in (p.stderr or ""):
        tmp.unlink(missing_ok=True)
        raise RateLimited()
    if p.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise Fail(f"competition {slug}: download rc={p.returncode} {(p.stderr or '')[:120]}")
    if not zipfile.is_zipfile(tmp):
        head = tmp.read_bytes()[:200].decode("utf-8", "replace")
        tmp.unlink(missing_ok=True)
        raise Fail(f"competition {slug}: not a zip ({head[:120]})")
    tmp.replace(dest)
    return slug


def materialize_hf_package(rec: dict, work: Path, token: str) -> dict:
    """Unpack an already-built task from HuggingFace.

    A built_task ships its data pre-split inside task.tar.zst, so unlike a recipe
    there is nothing to reconstruct: no Kaggle download and no prepare.py run.
    That makes this the cheapest and least failure-prone path, and it touches no
    Kaggle quota at all.

    The archive's sha256 is checked against package_manifest before unpacking, so
    a truncated download fails loudly instead of producing a half task.
    """
    ap = rec["artifact_path"]
    info = {}

    # metadata and scorer live outside the archive
    for rel in ("RELEASE_METADATA.json", "info/task_metadata.json",
                "info/data_description.txt", "description.txt",
                "utils/metric.py", "utils/prepare.py",
                "utils/samples/base_metric.py", "utils/samples/sample_metric.py",
                "utils/samples/sample_prepare.py", "utils/samples/sample_utils.py"):
        blob = hf_get(ap, rel, token)
        if blob is None:
            if rel in OPTIONAL_RECIPE_FILES:
                continue
            raise Fail(f"missing built_task file {rel}")
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_bytes(blob)

    expected = rec.get("archive_sha256") or rec.get("package_sha256")
    archive = work / "task.tar.zst"
    # Stream to disk while hashing. Reading the whole archive into memory first
    # exhausted RAM on multi-GB image tasks (six shards x >1 GB each) and stalled
    # every download; the sha256 is computed on the fly instead.
    url = f"{REPOSITORY}/{REVISION}/{ap}/task.tar.zst"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    digest = hashlib.sha256()
    total = 0
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=300) as r, archive.open("wb") as out:
            while True:
                chunk = r.read(8 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                out.write(chunk)
                total += len(chunk)
    except urllib.error.HTTPError as e:
        raise Fail(f"task.tar.zst HTTP {e.code}")
    except Exception as e:
        raise Fail(f"task.tar.zst download failed: {type(e).__name__}: {e}")
    if total == 0:
        raise Fail("task.tar.zst is empty")
    info["archive_bytes"] = total
    if expected:
        got = digest.hexdigest()
        if got != expected:
            raise Fail(f"archive sha256 mismatch: {got[:12]} != {expected[:12]}")
        info["sha256_verified"] = True

    tar_path = work / "task.tar"
    with tar_path.open("wb") as out:
        proc = subprocess.run(["unzstd", "-c", str(archive)], stdout=out,
                              stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise Fail(f"unzstd failed: {(proc.stderr or b'').decode()[:160]}")
    extracted = work / "_extracted"
    extracted.mkdir(exist_ok=True)
    safe_extract(tar_path, extracted)
    tar_path.unlink(missing_ok=True)
    archive.unlink(missing_ok=True)

    # Members are stored with a './' prefix and the payload may sit at the root or
    # one level down, so locate the directory that actually holds data/public.
    src = None
    for cand in [extracted, *(d for d in extracted.iterdir() if d.is_dir())]:
        if (cand / "data" / "public").is_dir():
            src = cand
            break
    if src is None:
        listing = [str(pp.relative_to(extracted))
                   for pp in list(extracted.rglob("*"))[:12]]
        raise Fail(f"no data/public in archive (saw: {listing})")
    for item in src.iterdir():
        target = work / item.name
        if target.exists():
            continue
        shutil.move(str(item), str(target))
    shutil.rmtree(extracted, ignore_errors=True)
    return info


def safe_extract(tar_path: Path, dest: Path) -> None:
    """Extract without letting member paths escape dest."""
    import tarfile
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            target = (dest / m.name).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise Fail(f"unsafe tar member: {m.name}")
        tf.extractall(dest)


def seed_descriptions(work: Path) -> None:
    """Put description.txt everywhere a recipe might expect to already find it.

    Recipes disagree on where it lives when prepare() runs: some copy it from the
    parent of public/, some assert public/description.txt, some look in utils/.
    In a real built_task it is simply already on disk, so seed every location they
    check instead of guessing per recipe. Re-run before each prepare attempt: a
    failed attempt may have cleared public/.
    """
    desc = work / "description.txt"
    if not desc.is_file() and (work / "info/data_description.txt").is_file():
        shutil.copyfile(work / "info/data_description.txt", desc)
    if not desc.is_file():
        return
    for rel in ("data/description.txt", "data/public/description.txt",
                "utils/description.txt", "info/description.txt"):
        target = work / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(desc, target)


def stage_raw(zip_path: Path, raw: Path, layout: str) -> None:
    """Unpack the Kaggle zip into raw/ using one candidate convention."""
    shutil.rmtree(raw, ignore_errors=True)
    raw.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        if not names:
            raise Fail("kaggle zip is empty")

        def extract_flat():
            for n in names:
                target = raw / Path(n).name
                with z.open(n) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

        if layout == "flat":
            # Keep upstream filenames; prepare.py usually addresses them directly.
            extract_flat()
        elif layout == "nested_dir":
            z.extractall(raw)
        elif layout == "single_csv_as_dataset":
            # Some recipes assert raw/dataset.csv regardless of the upstream name.
            csvs = [n for n in names if n.lower().endswith(".csv")]
            if len(csvs) != 1:
                raise Fail(f"layout single_csv: expected 1 csv, got {len(csvs)}")
            with z.open(csvs[0]) as src, (raw / "dataset.csv").open("wb") as dst:
                shutil.copyfileobj(src, dst)
        elif layout == "flat_plus_dataset_alias":
            # Upstream names AND a dataset.csv alias, for recipes that want either.
            extract_flat()
            csvs = [n for n in names if n.lower().endswith(".csv")]
            if len(csvs) != 1:
                raise Fail(f"layout flat+alias: expected 1 csv, got {len(csvs)}")
            alias = raw / "dataset.csv"
            if not alias.exists():
                shutil.copyfile(raw / Path(csvs[0]).name, alias)
        else:
            raise Fail(f"unknown layout {layout}")


RUNNER = r'''
import importlib.util, sys, json
from pathlib import Path
root = Path("/work")
spec = importlib.util.spec_from_file_location("prep", root / "utils/prepare.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.prepare(root / "raw", root / "data/public", root / "data/private")
print("PREPARE_OK")
'''

SCORER = r'''
import importlib.util, inspect, json, math
from pathlib import Path
import numpy as np
import pandas as pd
root = Path("/work")
spec = importlib.util.spec_from_file_location("m", root / "utils/metric.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
classes = [c for _, c in vars(m).items()
           if inspect.isclass(c) and hasattr(c, "evaluate") and c.__module__ == "m"]
if not classes:
    raise SystemExit("no metric class found")
M = classes[0]()
gt = pd.read_csv(root / "data/private/test_answer.csv")
sub = pd.read_csv(root / "data/public/sample_submission.csv")

# The gate: the recipe's own sample_submission must score to a finite number.
# This is what the sandbox scorer does for a model submission, so it exercises
# the whole reward path. We deliberately do NOT score ground-truth-as-submission
# or a perturbed submission: submission and answer schemas legitimately differ
# per task (label vs probability vs quantile columns), and a perturbation that
# stays schema-valid for every task is not something we can construct blindly.
score = float(M.evaluate(gt, sub))
if not math.isfinite(score):
    raise SystemExit(f"sample_submission scored non-finite: {score}")

print("SCORES " + json.dumps({
    "metric_class": classes[0].__name__,
    "sample_score": score,
    "n_test_rows": int(len(gt)),
    "sub_columns": [str(c) for c in sub.columns][:12],
    "higher_is_better": bool(getattr(M, "higher_is_better", True)),
}))
'''


def in_image(work: Path, script: str, tag: str, runtime: str, timeout=1800) -> str:
    """Run a python snippet against work/ inside the canonical image."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(work.parent)) as f:
        f.write(script)
        script_path = Path(f.name)
    try:
        cmd = [runtime, "run", "--rm", "--entrypoint", "bash", "--network=none",
               # Container root would otherwise leave __pycache__ that the host
               # user cannot delete (see MEMORY openrsi-ma1-round3-canonical-sandbox).
               "-e", "PYTHONDONTWRITEBYTECODE=1",
               "-u", f"{os.getuid()}:{os.getgid()}",
               # Cap container memory. Without this a wide CSV (e.g. a 1.3M-row
               # jobs dataset) grew pandas to 8.3 GB and the pod's memory cgroup
               # OOM-killed the whole python process, losing the shard silently.
               # A cap turns that into one failed task instead.
               "--memory", MEM_LIMIT, "--memory-swap", MEM_LIMIT,
               "-v", f"{work}:/work", "-v", f"{script_path}:/run.py:ro",
               IMAGE, "-lc", "python3 /run.py"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
        if p.returncode != 0:
            raise Fail(f"{tag} failed rc={p.returncode}: {out.strip()[-400:]}")
        return out
    except subprocess.TimeoutExpired:
        raise Fail(f"{tag} timed out after {timeout}s")
    finally:
        script_path.unlink(missing_ok=True)


def publish(work: Path, out_root: Path, tid: str, result: dict) -> None:
    """Move a finished task from node-local scratch onto the shared filesystem.

    cephfs creates small files at a hard ~9.5 files/s regardless of parallelism
    (measured: 1 worker and 16 workers score identically), while sequential
    writes run at 264 MB/s. Media tasks routinely hold tens of thousands of
    files, so a per-file move takes hours -- four shards spent an hour each
    stuck in fuse_lookup partway through one task. Above the threshold, stream
    the tree into a single archive instead and leave it packed: one big
    sequential write, and the consumer unpacks to its own local disk.
    """
    final = out_root / tid
    n_files = sum(1 for _ in work.rglob("*") if _.is_file())
    result["n_files"] = n_files

    if n_files <= PACK_ABOVE_FILES:
        staged = out_root / f".{tid.replace('/', '_')}.new"
        shutil.rmtree(staged, ignore_errors=True)
        shutil.move(str(work), str(staged))
        shutil.rmtree(final, ignore_errors=True)
        final.parent.mkdir(parents=True, exist_ok=True)
        staged.replace(final)
        result["publish"] = "tree"
        return

    # Packed form: <task_id>.tar (+ a sibling marker so resume and the
    # catalogue can see it without unpacking).
    import tarfile
    tmp_tar = work.parent / f"{tid.replace('/', '_')}.tar"
    tmp_tar.unlink(missing_ok=True)
    with tarfile.open(tmp_tar, "w") as tf:
        tf.add(str(work), arcname=tid, recursive=True)
    staged_tar = out_root / f".{tid.replace('/', '_')}.tar.new"
    shutil.rmtree(final, ignore_errors=True)
    shutil.copyfile(str(tmp_tar), str(staged_tar))
    staged_tar.replace(out_root / f"{tid}.tar")
    (out_root / f"{tid}.packed.json").write_text(
        json.dumps({"task_id": tid, "archive": f"{tid}.tar",
                    "n_files": n_files,
                    "tar_bytes": tmp_tar.stat().st_size,
                    "verified": json.loads((work / ".verified.json").read_text())},
                   indent=1), encoding="utf-8")
    tmp_tar.unlink(missing_ok=True)
    shutil.rmtree(work, ignore_errors=True)
    result["publish"] = "packed"
    result["tar_bytes"] = (out_root / f"{tid}.tar").stat().st_size


def materialize(rec: dict, out_root: Path, scratch: Path, token: str,
                kuser: str, kkey: str, runtime: str, keep_zip: bool,
                max_download_bytes: int | None = None) -> dict:
    tid = rec["task_id"]
    result = {"task_id": tid, "download_ref": rec["download_ref"],
              "artifact_path": rec["artifact_path"], "started": time.time()}
    work = scratch / tid.replace("/", "_")
    shutil.rmtree(work, ignore_errors=True)
    (work / "info").mkdir(parents=True)
    (work / "utils/samples").mkdir(parents=True)

    # built_task: data already split inside task.tar.zst, no Kaggle, no prepare.py
    if str(rec.get("acquisition")) == "hf_package" or rec.get("release_type") == "built_task":
        result.update(materialize_hf_package(rec, work, token))
        result["acquisition_kind"] = "hf_package"
        missing = [r for r in REQUIRED if not (work / r).is_file()]
        if missing:
            raise Fail(f"incomplete built_task: {missing}")
        out = in_image(work, SCORER, "score", runtime)
        line = next((l for l in out.splitlines() if l.startswith("SCORES ")), None)
        if not line:
            raise Fail(f"scorer produced no verdict: {out.strip()[-300:]}")
        scores = json.loads(line[len("SCORES "):])
        meta = json.loads((work / "info/task_metadata.json").read_text())
        if bool(scores["higher_is_better"]) != bool(meta["higher_is_better"]):
            raise Fail(f"higher_is_better mismatch: metric={scores['higher_is_better']} "
                       f"metadata={meta['higher_is_better']}")
        result.update(scores)
        link = work / "utils/public"
        if link.is_symlink() or link.exists():
            link.unlink() if link.is_symlink() else shutil.rmtree(link, ignore_errors=True)
        link.symlink_to("../data/public", target_is_directory=True)
        (work / ".verified.json").write_text(json.dumps({
            "task_id": tid, "artifact_path": rec["artifact_path"],
            "release_type": "built_task", "acquisition": "hf_package",
            "sha256_verified": result.get("sha256_verified", False),
            "prepared_in_image": IMAGE,
            "metric_probe": {k: scores[k] for k in ("metric_class", "sample_score",
                                                    "n_test_rows")},
            "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        }, indent=1), encoding="utf-8")
        sizes = {p.name: p.stat().st_size
                 for p in (work / "data/public").iterdir() if p.is_file()}
        result["public_sizes"] = sizes
        publish(work, out_root, tid, result)
        result["ok"] = True
        result["elapsed"] = round(time.time() - result["started"], 1)
        return result

    # 1. recipe files from HF
    for rel in RECIPE_FILES:
        blob = hf_get(rec["artifact_path"], rel, token)
        if blob is None:
            if rel in OPTIONAL_RECIPE_FILES:
                continue
            raise Fail(f"missing recipe file {rel}")
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_bytes(blob)

    seed_descriptions(work)


    # 2. upstream Kaggle data
    zip_path = scratch / f"{tid.replace('/', '_')}.zip"
    if not (keep_zip and zip_path.is_file() and zipfile.is_zipfile(zip_path)):
        if str(rec.get("source_type", "")) == "KAGGLE_COMPETITION":
            result["kaggle_used"] = kaggle_competition_download(
                rec["download_ref"], zip_path, kuser, kkey, max_bytes=max_download_bytes)
            result["acquisition_kind"] = "competition"
        else:
            result["kaggle_used"] = kaggle_download(rec["download_ref"], zip_path,
                                                    kuser, kkey,
                                                    max_bytes=max_download_bytes)
            result["acquisition_kind"] = "dataset"
    result["zip_bytes"] = zip_path.stat().st_size

    # 3. run the recipe's prepare.py in the canonical image, trying raw layouts
    (work / "data/public").mkdir(parents=True, exist_ok=True)
    (work / "data/private").mkdir(parents=True, exist_ok=True)
    errors = []
    for layout in RAW_LAYOUTS:
        try:
            stage_raw(zip_path, work / "raw", layout)
        except Fail as e:
            errors.append(f"{layout}: {e}")
            continue
        (work / "data/public").mkdir(parents=True, exist_ok=True)
        (work / "data/private").mkdir(parents=True, exist_ok=True)
        seed_descriptions(work)
        try:
            in_image(work, RUNNER, f"prepare[{layout}]", runtime)
            result["raw_layout"] = layout
            break
        except Fail as e:
            errors.append(f"{layout}: {e}")
    else:
        raise Fail("prepare.py failed for all raw layouts | " + " || ".join(errors)[:900])

    # 4. the worker's own required-file check
    missing = [r for r in REQUIRED if not (work / r).is_file()]
    if missing:
        raise Fail(f"incomplete after prepare: {missing}")

    # 5. prove the reward path works end to end
    out = in_image(work, SCORER, "score", runtime)
    line = next((l for l in out.splitlines() if l.startswith("SCORES ")), None)
    if not line:
        raise Fail(f"scorer produced no verdict: {out.strip()[-300:]}")
    scores = json.loads(line[len("SCORES "):])
    meta = json.loads((work / "info/task_metadata.json").read_text())
    if bool(scores["higher_is_better"]) != bool(meta["higher_is_better"]):
        raise Fail(f"higher_is_better mismatch: metric={scores['higher_is_better']} "
                   f"metadata={meta['higher_is_better']}")
    result.update(scores)

    # 6. lay out exactly like a built_task and publish atomically
    link = work / "utils/public"
    if link.is_symlink() or link.exists():
        link.unlink() if link.is_symlink() else shutil.rmtree(link, ignore_errors=True)
    link.symlink_to("../data/public", target_is_directory=True)
    shutil.rmtree(work / "raw", ignore_errors=True)

    sizes = {p.name: p.stat().st_size for p in (work / "data/public").iterdir() if p.is_file()}
    result["public_sizes"] = sizes
    (work / ".verified.json").write_text(json.dumps({
        "task_id": tid,
        "artifact_path": rec["artifact_path"],
        "release_type": "recipe",
        "acquisition": "kaggle_recipe",
        "kaggle_ref_used": result.get("kaggle_used"),
        "raw_layout": result["raw_layout"],
        "prepared_in_image": IMAGE,
        "metric_probe": {k: scores[k] for k in ("metric_class", "sample_score", "n_test_rows")},
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
    }, indent=1), encoding="utf-8")

    publish(work, out_root, tid, result)
    if not keep_zip:
        zip_path.unlink(missing_ok=True)
    result["ok"] = True
    result["elapsed"] = round(time.time() - result["started"], 1)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--out", required=True, help="materialized task root")
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--results", required=True, help="jsonl of per-task outcomes")
    ap.add_argument("--runtime", default="docker", choices=("docker", "podman"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n — process only tasks where index%n==i")
    ap.add_argument("--max-raw-gb", type=float, default=2.0,
                    help="skip tasks whose metadata claims more than this")
    ap.add_argument("--max-download-gb", type=float, default=3.0,
                    help="hard cap on the actual Kaggle zip; metadata understates "
                         "some datasets by orders of magnitude")
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--skip", default=None,
                    help="json list of task_ids to skip entirely. A failure record is "
                         "not terminal for resume (only ok:true is), so tasks we "
                         "deliberately defer would otherwise be retried on every "
                         "restart.")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    kuser = os.environ["KAGGLE_USERNAME"]
    kkey = os.environ["KAGGLE_KEY"]

    i, n = (int(x) for x in args.shard.split("/"))
    recs = [json.loads(l) for l in open(args.inventory)]
    recs = [r for r in recs if (r.get("raw_size_gb") or 0) <= args.max_raw_gb]
    recs = [r for k, r in enumerate(recs) if k % n == i]
    if args.limit:
        recs = recs[: args.limit]

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch); scratch.mkdir(parents=True, exist_ok=True)
    results = Path(args.results); results.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.skip:
        done |= set(json.load(open(args.skip)))
        log(f"skipping {len(done)} deliberately deferred task(s)")
    if results.is_file():
        for l in results.open():
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("ok"):
                done.add(r["task_id"])

    log(f"shard {args.shard}: {len(recs)} tasks ({len(done)} already done), runtime={args.runtime}")
    ok = fail = 0
    with results.open("a") as rf:
        for k, rec in enumerate(recs, 1):
            tid = rec["task_id"]
            if (tid in done
                    or (out_root / tid / ".verified.json").is_file()
                    or (out_root / f"{tid}.packed.json").is_file()):
                continue
            # A failed task can leave its half-extracted tree and zip behind, and
            # scratch on a node-local disk is far smaller than cephfs: /data filled
            # to 0 bytes free and the next 756 tasks all died with ENOSPC, recorded
            # as if the TASKS were broken. Sweep before each task and stop cleanly
            # if the disk is genuinely too full, rather than burning the inventory.
            free_gb = shutil.disk_usage(scratch).free / 1e9
            if free_gb < MIN_FREE_GB:
                for leftover in scratch.iterdir():
                    shutil.rmtree(leftover, ignore_errors=True) if leftover.is_dir() \
                        else leftover.unlink(missing_ok=True)
                free_gb = shutil.disk_usage(scratch).free / 1e9
                if free_gb < MIN_FREE_GB:
                    log(f"[{k}/{len(recs)}] STOP: only {free_gb:.1f}GB free on "
                        f"{scratch} after sweeping; refusing to continue")
                    break
            try:
                res = materialize(rec, out_root, scratch, token, kuser, kkey,
                                  args.runtime, args.keep_zip,
                                  max_download_bytes=int(args.max_download_gb * 1e9))
                ok += 1
                log(f"[{k}/{len(recs)}] OK {tid} "
                    f"sample={res['sample_score']:.4g} rows={res['n_test_rows']} "
                    f"{res['elapsed']}s")
            except Fail as e:
                fail += 1
                res = {"task_id": tid, "ok": False, "error": str(e)[:900]}
                log(f"[{k}/{len(recs)}] FAIL {tid}: {str(e)[:220]}")
            except Exception as e:  # unexpected — record and keep going
                fail += 1
                res = {"task_id": tid, "ok": False,
                       "error": f"unexpected {type(e).__name__}: {e}"[:900]}
                log(f"[{k}/{len(recs)}] ERROR {tid}: {type(e).__name__}: {e}")
            rf.write(json.dumps(res, ensure_ascii=False) + "\n")
            rf.flush()
            shutil.rmtree(Path(args.scratch) / tid.replace("/", "_"), ignore_errors=True)
    log(f"shard {args.shard} finished: ok={ok} fail={fail}")


if __name__ == "__main__":
    sys.exit(main())
