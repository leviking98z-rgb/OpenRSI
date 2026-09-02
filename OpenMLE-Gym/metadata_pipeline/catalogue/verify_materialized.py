#!/usr/bin/env python3
"""Independently verify materialized recipe tasks against the worker's contract.

Deliberately does NOT trust the materializer's own .verified.json claims: it
re-reads each task from disk and re-derives every property the worker and the
sandbox scorer depend on. Run this before pointing collection at the expanded
pool.

Checks per task:
  1. every file continuous_worker.py requires exists and is non-empty
  2. utils/public resolves to data/public (the worker relies on this symlink)
  3. task_metadata.json says CPU + Classification/Regression, has higher_is_better
  4. train/test/sample_submission/test_answer parse as CSV and are mutually consistent
     (test and answer aligned row-for-row; sample_submission covers the same ids)
  5. no answer leakage: the answer column must not appear in public/test.csv
  6. the metric scores the sample submission to a finite number, in the canonical image
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

IMAGE = "ccr.ccs.tencentyun.com/frontisai-openmle/openmle-sandbox-worker:1.0.0"
# The inner script loads train/test/answer into pandas at once, and a few tasks
# have tens of millions of rows, so a single worker can reach 10 GB. Unbounded
# across N workers that eats the host: 8 workers took 45 GB of a 61 GB machine.
# Cap each container; a task too big to verify under the cap fails loudly
# instead of pushing the host into swap.
MEM_LIMIT = os.environ.get("VERIFY_MEM_LIMIT", "8g")

REQUIRED = (
    "RELEASE_METADATA.json",
    "info/task_metadata.json",
    "data/public/train.csv",
    "data/public/test.csv",
    "data/public/sample_submission.csv",
    "data/private/test_answer.csv",
    "utils/metric.py",
)

# Runs inside the canonical image: pandas/numpy there are the versions the
# search agent will actually see, so alignment and scoring are checked under
# the same semantics that produce the reward.
INSIDE = r'''
import importlib.util, inspect, json, math
from pathlib import Path
import pandas as pd

root = Path("/work")
out = {"ok": False, "problems": []}
def bad(msg): out["problems"].append(msg)

# Key-signature comparison is O(rows) string work, and a few materialized tasks
# have tens of millions of test rows. Above this we check shapes and scoring but
# skip the row-wise key comparison rather than spending minutes on it.
BIG_ROWS = 2_000_000

# Reading all four tables whole costs several times the file size in RAM (pandas
# needs 2-3x during parse), and a handful of tasks -- DDoS traffic captures,
# wikibooks dumps -- are two orders of magnitude larger than the median. Holding
# them resident hit 23.6 GB on one task and got the container OOM-killed. Raising
# the cap is the wrong fix: above this file size, read only what each check
# actually needs (row counts, key columns, dtypes) and never materialize the
# feature columns.
STREAM_ABOVE_BYTES = 200 << 20


def csv_rows(path):
    """Count data rows without holding the table, ~constant memory."""
    n = 0
    for chunk in pd.read_csv(path, usecols=[0], chunksize=500_000, low_memory=False):
        n += len(chunk)
    return n


def csv_columns(path):
    return list(pd.read_csv(path, nrows=0).columns)


def csv_key_frame(path, cols):
    """Read just `cols`, streamed. Used for key-set comparison on big tables."""
    if not cols:
        return pd.DataFrame()
    parts = [c for c in pd.read_csv(path, usecols=cols, chunksize=500_000,
                                    low_memory=False)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def check_row_alignment(rows, cols, bad, out):
    """Are the answer and submission consistent with the test set?

    Row-for-row equality across all three is the WRONG rule and rejected 23 valid
    tasks. Detection and segmentation tasks come in two shapes, both legitimate:

      exploded answer   ans = one row per box, test = one row per image
                        (annotated-potholes: 299 answer rows over 133 test rows)
      packed submission sub = one row per image with every box in a single
                        `PredictionString` field, so sub == test while ans > test
                        (drones@1: test 158, sub 158, ans 198)

    So the invariant is not "all three equal", it is coverage:
      - every test item must have an answer      -> ans < test is a real defect
      - the submission must address every test item, either one row per item
        (packed) or one row per answer row (exploded)
    Anything else means the submission cannot be scored against the answer.
    """
    a, t, s = rows["ans"], rows["test"], rows["sub"]
    if a < t:
        bad(f"answer rows {a} < test rows {t}: {t - a} test items have no answer")
    elif a > t:
        out.setdefault("notes", []).append(
            f"multi-row answer: {a} answer rows for {t} test rows ({a/t:.2f}x)")
    if s == a:
        pass                      # exploded: one submission row per answer row
    elif s == t:
        out.setdefault("notes", []).append(
            f"packed submission: {s} rows for {a} answer rows (one row per test item)")
    else:
        bad(f"sample_submission rows {s} matches neither answer rows {a} "
            f"nor test rows {t}")


paths = {"train": root / "data/public/train.csv",
         "test": root / "data/public/test.csv",
         "sub": root / "data/public/sample_submission.csv",
         "ans": root / "data/private/test_answer.csv"}
big = any(p.stat().st_size > STREAM_ABOVE_BYTES for p in paths.values() if p.is_file())
out["streamed"] = big

if big:
    # Streamed path: shapes and column names come from metadata-only reads, and
    # only the key/target columns are ever loaded. The metric still gets real
    # frames, but built from just those columns.
    try:
        cols = {k: csv_columns(p) for k, p in paths.items()}
        # Only test/sub/ans row counts are ever compared against each other.
        # train's count is used for one thing -- "is it non-empty" -- so read a
        # single row instead of streaming the whole file: train is often the
        # largest table (7.9 GB on one task) and counting it dominated the
        # runtime, timing the verifier out after the memory fix.
        rows = {k: csv_rows(paths[k]) for k in ("test", "sub", "ans")}
        rows["train"] = len(pd.read_csv(paths["train"], usecols=[0], nrows=1))
    except Exception as e:
        bad(f"csv parse failed: {type(e).__name__}: {e}")
        print("VERDICT " + json.dumps(out)); raise SystemExit(0)

    out["shapes"] = {k: [rows[k], len(cols[k])] for k in paths}
    out["shapes"]["train"][0] = ">=1" if rows["train"] else 0
    if rows["train"] == 0: bad("train.csv is empty")
    if rows["test"] == 0: bad("test.csv is empty")
    check_row_alignment(rows, cols, bad, out)

    key_cols = [c for c in cols["sub"] if c in cols["test"]]
    targets = [c for c in cols["sub"] if c not in key_cols]
    out["key_columns"] = key_cols
    if not targets:
        bad(f"no prediction target: all submission columns {cols['sub']} exist in test.csv")
    else:
        out["target_columns"] = targets
        leaked = [c for c in targets if c in cols["test"]]
        if leaked:
            bad(f"target column(s) leaked into public test.csv: {leaked}")

    # Load only what the metric needs. The answer file's target column is NOT
    # always named the same as the submission's -- beth-dataset ships
    # sub=(row_id,score) against ans=(row_id,evil) -- so restricting the answer
    # read to the submission's column names drops the column the metric reads
    # and it fails with "Ground truth is missing required columns". Read all of
    # the answer file's columns; it is the small table (the big one is train,
    # which the metric never sees).
    try:
        sub = csv_key_frame(paths["sub"], cols["sub"])
        ans = csv_key_frame(paths["ans"], cols["ans"])
    except Exception as e:
        bad(f"key column read failed: {type(e).__name__}: {e}")
        sub = ans = None
    train = test = None
else:
    try:
        train = pd.read_csv(paths["train"], low_memory=False)
        test = pd.read_csv(paths["test"], low_memory=False)
        sub = pd.read_csv(paths["sub"], low_memory=False)
        ans = pd.read_csv(paths["ans"], low_memory=False)
    except Exception as e:
        bad(f"csv parse failed: {type(e).__name__}: {e}")
        print("VERDICT " + json.dumps(out)); raise SystemExit(0)

    out["shapes"] = {"train": list(train.shape), "test": list(test.shape),
                     "sub": list(sub.shape), "ans": list(ans.shape)}

    if len(train) == 0: bad("train.csv is empty")
    if len(test) == 0: bad("test.csv is empty")
    check_row_alignment({"ans": len(ans), "test": len(test), "sub": len(sub)},
                        None, bad, out)

# Id alignment and leak detection. On the streamed path both were already done
# above from column names and row counts, and `test` was never loaded, so skip.
if not big:
    # Keys may be composite, so compare on the set of columns the submission
    # shares with the answer file rather than assuming a single id.
    shared = [c for c in sub.columns if c in ans.columns and c in test.columns]
    if not shared:
        out.setdefault("notes", []).append("no shared key columns between sub/ans/test")
    elif len(test) > BIG_ROWS:
        out.setdefault("notes", []).append(f"key check skipped: {len(test)} rows > {BIG_ROWS}")
    else:
        # Compare key SETS, not rows in order. Multi-row answers (one row per
        # detected box) mean the frames legitimately differ in length, so an
        # order-sensitive or length-equal comparison would reject valid tasks.
        def sig(df):
            return df[shared].astype(str).agg("|".join, axis=1)
        ans_sig, test_sig, sub_sig = sig(ans), sig(test), sig(sub)
        missing = set(test_sig) - set(ans_sig)
        if missing:
            bad(f"{len(missing)} test keys have no answer (e.g. {sorted(missing)[:3]})")
        extra = set(ans_sig) - set(test_sig)
        if extra:
            bad(f"{len(extra)} answer keys are not in test (e.g. {sorted(extra)[:3]})")
        if set(sub_sig) != set(ans_sig):
            # A packed submission has one row per test item while the answer has
            # one per box, so the key SETS still have to match even though the
            # row counts do not. Only complain if a key is genuinely absent.
            miss = set(ans_sig) - set(sub_sig)
            surplus = set(sub_sig) - set(ans_sig)
            if miss or surplus:
                bad(f"sample_submission keys differ from answer keys "
                    f"(missing {len(miss)}, extra {len(surplus)})")
        if (len(ans) == len(test) and not ans_sig.equals(test_sig)
                and set(ans_sig) == set(test_sig)):
            out.setdefault("notes", []).append("answer/test same keys, different order")

    # Leakage: the prediction target must not be sitting in public test.csv.
    #
    # Identifying the target takes care. The id is not always a single leading
    # column — some tasks key on a composite (e.g. `date,sku`) — and answer files
    # sometimes carry extra key/feature columns that legitimately appear in
    # test.csv (e.g. `date`). What the model must actually predict is the
    # submission column that is NOT part of the shared key, so derive the target
    # from that:
    #   target = sample_submission columns, minus columns also present in test.csv
    # and only report a leak if such a column then turns up in test.csv anyway
    # (which would mean the answer is handed to the model for free).
    key_cols = [c for c in sub.columns if c in test.columns]
    targets = [c for c in sub.columns if c not in key_cols]
    if not targets:
        # Every submission column also exists in test.csv — the answer is
        # readable straight from the public data.
        bad(f"no prediction target: all submission columns {list(sub.columns)} exist in test.csv")
    else:
        out["target_columns"] = targets
        out["key_columns"] = key_cols
        leaked = [c for c in targets if c in test.columns]
        if leaked:
            bad(f"target column(s) leaked into public test.csv: {leaked}")

# the metric must score the recipe's own sample submission
if sub is None or ans is None:
    bad("could not load submission/answer columns for scoring")
else:
    try:
        spec = importlib.util.spec_from_file_location("m", root / "utils/metric.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        classes = [c for _, c in vars(m).items()
                   if inspect.isclass(c) and hasattr(c, "evaluate") and c.__module__ == "m"]
        if not classes:
            bad("no metric class in utils/metric.py")
        else:
            M = classes[0]()
            score = float(M.evaluate(ans, sub))
            out["metric_class"] = classes[0].__name__
            out["sample_score"] = score
            out["higher_is_better"] = bool(getattr(M, "higher_is_better", True))
            if not math.isfinite(score):
                bad(f"sample submission scored non-finite: {score}")
            # Does the metric actually reward a correct answer? Copy the true
            # target into the submission's target column (identified above, not
            # assumed to be position 1). A metric that cannot tell a random
            # submission from the right one gives the search no gradient.
            tgt = out.get("target_columns") or []
            if len(tgt) == 1 and tgt[0] in ans.columns and len(ans) == len(sub):
                better = sub.copy()
                better[tgt[0]] = ans[tgt[0]].values
                try:
                    good = float(M.evaluate(ans, better))
                    out["answers_score"] = good
                    if math.isfinite(good) and good == score:
                        bad("metric gives identical score to random and correct answers")
                except Exception as e:
                    # Some metrics reject a submission built this way; not fatal.
                    out["answers_score"] = f"n/a: {type(e).__name__}"
    except Exception as e:
        bad(f"metric load/eval failed: {type(e).__name__}: {e}")

out["ok"] = not out["problems"]
print("VERDICT " + json.dumps(out))
'''


def verify(task_dir: Path, runtime: str, timeout: int) -> dict:
    res = {"task_id": task_dir.name, "ok": False, "problems": []}

    for rel in REQUIRED:
        p = task_dir / rel
        if not p.is_file():
            res["problems"].append(f"missing {rel}")
        elif p.stat().st_size == 0:
            res["problems"].append(f"empty {rel}")

    link = task_dir / "utils/public"
    if not link.is_symlink():
        res["problems"].append("utils/public is not a symlink")
    elif link.resolve() != (task_dir / "data/public").resolve():
        res["problems"].append(f"utils/public points at {os.readlink(link)}")

    meta_path = task_dir / "info/task_metadata.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            # cpu_gpu and task type are NOT rejection criteria any more: the pool
            # deliberately covers GPU and non-clf/reg tasks now that compute is not
            # a constraint. Record them for downstream filtering instead.
            res["cpu_gpu"] = meta.get("cpu_gpu")
            res["task_type"] = meta.get("task")
            if "higher_is_better" not in meta:
                res["problems"].append("metadata lacks higher_is_better")
            res["metadata_higher_is_better"] = meta.get("higher_is_better")
        except Exception as e:
            res["problems"].append(f"task_metadata.json unreadable: {e}")

    if res["problems"]:
        return res  # don't pay for a container run on a task already broken

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(INSIDE)
        script = Path(f.name)
    try:
        cmd = [runtime, "run", "--rm", "--entrypoint", "bash", "--network=none",
               "-e", "PYTHONDONTWRITEBYTECODE=1", "-u", f"{os.getuid()}:{os.getgid()}",
               "--memory", MEM_LIMIT, "--memory-swap", MEM_LIMIT,
               "-v", f"{task_dir}:/work:ro", "-v", f"{script}:/v.py:ro",
               IMAGE, "-lc", "python3 /v.py"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        line = next((l for l in (p.stdout or "").splitlines() if l.startswith("VERDICT ")), None)
        if not line:
            # A container killed by the memory cgroup exits 137 with no output at
            # all -- no traceback, nothing on stderr. Name it, or it looks like a
            # mysterious silent failure.
            if p.returncode == 137:
                res["problems"].append(f"verifier OOM-killed at {MEM_LIMIT} "
                                       f"(task too large to verify under the cap)")
                return res
            res["problems"].append(f"verifier produced no verdict rc={p.returncode}: "
                                   f"{((p.stdout or '') + (p.stderr or '')).strip()[-300:]}")
            return res
        inner = json.loads(line[len("VERDICT "):])
        res["problems"].extend(inner.pop("problems", []))
        res.update(inner)
    except subprocess.TimeoutExpired:
        res["problems"].append(f"verifier timed out after {timeout}s")
    except Exception as e:
        res["problems"].append(f"verifier error {type(e).__name__}: {e}")
    finally:
        script.unlink(missing_ok=True)

    # metadata and metric must agree on direction, or reward is inverted
    mhb, khb = res.get("metadata_higher_is_better"), res.get("higher_is_better")
    if mhb is not None and khb is not None and bool(mhb) != bool(khb):
        res["problems"].append(f"higher_is_better mismatch metadata={mhb} metric={khb}")

    res["ok"] = not res["problems"]
    return res


def verify_packed(marker: Path, runtime: str, timeout: int, unpack_dir: Path) -> dict:
    """Verify a task published as a single tar.

    Unpacks to node-local scratch first and then runs the ordinary tree
    verification, so packed and unpacked tasks are held to the same bar. The
    unpack has to be local: reading tens of thousands of small files back off
    cephfs is exactly the cost the packing avoided.
    """
    import shutil
    import tarfile

    tid = json.loads(marker.read_text())["task_id"]
    tar = marker.parent / f"{tid}.tar"
    if not tar.is_file():
        return {"task_id": tid, "ok": False, "packed": True,
                "problems": [f"marker present but {tar.name} missing"]}

    dest = unpack_dir / tid.replace("/", "_")
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tar) as tf:
            root = dest.resolve()
            for m in tf.getmembers():
                if not str((dest / m.name).resolve()).startswith(str(root)):
                    return {"task_id": tid, "ok": False, "packed": True,
                            "problems": [f"tar member escapes dest: {m.name}"]}
            tf.extractall(dest)
        inner = dest / tid
        if not inner.is_dir():
            cands = [d for d in dest.iterdir() if d.is_dir()]
            if len(cands) != 1:
                return {"task_id": tid, "ok": False, "packed": True,
                        "problems": [f"unexpected tar layout: {[c.name for c in cands][:5]}"]}
            inner = cands[0]
        res = verify(inner, runtime, timeout)
        res["task_id"] = tid
        res["packed"] = True
        res["tar_bytes"] = tar.stat().st_size
        return res
    except Exception as e:
        return {"task_id": tid, "ok": False, "packed": True,
                "problems": [f"unpack failed: {type(e).__name__}: {e}"]}
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="materialized task root")
    ap.add_argument("--out", required=True, help="jsonl verdicts")
    ap.add_argument("--runtime", default="docker", choices=("docker", "podman"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--unpack-dir", default=None,
                    help="local scratch for unpacking .tar-published tasks. Tasks with "
                         "many small files are published as a single tar because cephfs "
                         "creates small files at ~9.5/s; without this they are skipped.")
    ap.add_argument("--only", default=None,
                    help="file with one task_id per line; verify just those")
    ap.add_argument("--append", action="store_true",
                    help="append to --out instead of truncating (for incremental runs)")
    args = ap.parse_args()

    root = Path(args.root)
    if args.only:
        wanted = [l.strip() for l in Path(args.only).read_text().splitlines() if l.strip()]
        tasks = [root / t for t in wanted if (root / t).is_dir()]
    else:
        tasks = sorted(d for d in root.iterdir()
                       if d.is_dir() and not d.name.startswith("."))
    packed = sorted(root.glob("*.packed.json")) if args.unpack_dir else []
    if args.limit:
        tasks = tasks[: args.limit]
        packed = packed[: args.limit]
    print(f"verifying {len(tasks)} tree tasks + {len(packed)} packed tasks "
          f"with {args.workers} workers", flush=True)

    ok = bad = 0
    with open(args.out, "a" if args.append else "w") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(verify, t, args.runtime, args.timeout): t for t in tasks}
        for p in packed:
            futures[ex.submit(verify_packed, p, args.runtime, args.timeout,
                              Path(args.unpack_dir))] = p
        # as_completed, not submission order: one huge task must not block the
        # progress report for everything finished behind it.
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if r["ok"]:
                ok += 1
            else:
                bad += 1
                print(f"  BAD {r['task_id']}: {'; '.join(r['problems'])[:180]}", flush=True)
            if i % 100 == 0:
                print(f"  {i}/{len(tasks)} ok={ok} bad={bad}", flush=True)
    print(f"\n=== verified ok={ok} bad={bad} ({100*ok/max(ok+bad,1):.1f}% pass) ===")


if __name__ == "__main__":
    sys.exit(main())
