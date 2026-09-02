#!/usr/bin/env python3
"""Build a complete metadata catalogue for every usable OpenMLE task.

The pool no longer excludes tasks by device or modality — resources will be
provided later — so the catalogue covers everything that can form a static,
automatically-scorable task. What it does NOT do is download data: each entry
records how to fetch it (HF artifact path for built_task, Kaggle ref for recipe)
plus the real byte size, so large tasks can stay unmaterialized until needed.

Per task it records:
  - identity + provenance (source list, release type, upstream refs, license)
  - scoring contract (task type, higher_is_better, theoretical/leaderboard bounds)
  - modality, both raw and normalized (the raw field is free-text: 103 variants)
  - cleaned task type (the raw field carries unclean LLM annotation residue)
  - acquisition plan (endpoint, whether rules are accepted, measured size)
  - materialization state (already in pool / ready / deferred-large)

`task` and `modality` in the official metadata are LLM-annotated and dirty:
values like 'Classification\\n\\n Let', '[TaskType]', 'One-Detection', 'XXXXXX'
appear verbatim. Both raw and cleaned forms are kept so nothing is silently lost.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

TMP = Path("/root/shared/.clusters/.tmp")
META_DIR = TMP / "openrsi-official-ma1-20260830"
HERE = TMP / "kaggle-expand"
COLLECT = TMP / "openrsi-round3-collect-20260831"

SOURCE_LIST = {"paper_tasks.txt": "curated_anchors",
               "smith_tasks.txt": "kaggle_datasets",
               "ours_tasks.txt": "kaggle_competitions"}

# Canonical task types. The raw field is LLM-written and frequently trails into
# reasoning text, so match a known type at the start and drop the rest.
KNOWN_TASKS = ("classification", "regression", "segmentation", "object-detection",
               "generation", "clustering", "ranking", "retrieval", "prediction")


def clean_task(raw) -> tuple[str, bool]:
    """Return (cleaned, was_dirty). Empty string when nothing recognizable."""
    s = str(raw or "").strip()
    if not s:
        return "", False
    # cut at the first newline / quote / bracket / markdown marker
    head = re.split(r"[\n\r\"*\[]", s)[0].strip().rstrip(".,;:")
    parts = [p.strip().lower() for p in head.split(",") if p.strip()]
    keep = [p for p in parts if p in KNOWN_TASKS]
    cleaned = ",".join(keep)
    return cleaned, cleaned != s.lower()


# The modality field is LLM-written and sometimes contains the model's entire
# reasoning transcript instead of a label, hundreds of words long and full of
# commas. Splitting on commas therefore cannot be trusted; only accept tokens
# that are actual modality names.
KNOWN_MODALITIES = ("tabular", "time-series", "text", "image", "audio", "video",
                    "graph", "multimodal")


def clean_modality(raw) -> tuple[str, bool]:
    """Return (normalized, was_dirty).

    ' Tabular', 'Text,Tabular' and 'Text, Tabular' all mean the same thing, so
    normalize to a sorted comma list of recognized names. Anything unrecognized
    is dropped rather than carried through — a runaway annotation would otherwise
    inject a whole paragraph into the catalogue.
    """
    s = str(raw or "").strip()
    low = s.lower()
    found = sorted({m for m in KNOWN_MODALITIES if m in low})
    # 'time-series' contains no substring clash, but 'text' appears inside words
    # like 'context'; require a token-ish boundary for the short names.
    import re as _re
    found = sorted({m for m in KNOWN_MODALITIES
                    if _re.search(rf"(?<![a-z]){_re.escape(m)}(?![a-z])", low)})
    cleaned = ",".join(found)
    # dirty when the raw value was not already exactly the normalized form
    return cleaned, cleaned != ",".join(sorted(
        p.strip().lower() for p in s.replace(";", ",").split(",") if p.strip()))


def load_jsonl(path: Path, key: str | None = None):
    out = {} if key else []
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if key:
            out.setdefault(r[key], r)
        else:
            out.append(r)
    return out


def num(v):
    try:
        f = float(v)
        return f if f == f else None  # drop NaN
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "task_catalogue.jsonl"))
    ap.add_argument("--report", default=str(HERE / "catalogue_report.json"))
    ap.add_argument("--defer-gb", type=float, default=10.0,
                    help="tasks larger than this are catalogued but not materialized")
    args = ap.parse_args()

    index = load_jsonl(META_DIR / "task_index.jsonl", key="task_id")
    manifest = load_jsonl(META_DIR / "package_manifest.jsonl", key="task_id")

    meta: dict[str, dict] = {}
    for f in ("recipe_meta.jsonl", "recipe_meta_retry.jsonl", "missing_built_meta.jsonl"):
        for r in load_jsonl(HERE / f):
            if not r.get("error"):
                meta.setdefault(r["task_id"], r)
    for f in ("h20_inventory.jsonl", "l20_inventory.jsonl"):
        for r in load_jsonl(TMP / "openrsi-continuous-20260831" / f):
            meta.setdefault(r["task_id"], r)

    recl = load_jsonl(HERE / "comp_class_v2.jsonl", key="task_id")

    # Measured sizes from competitions/data/list. The official raw_size_gb is
    # missing for most competition tasks and under-reports by orders of magnitude
    # where present, so a measured value always wins.
    measured: dict[str, dict] = {}
    for r in load_jsonl(HERE / "comp_sizes.jsonl"):
        if r.get("code") == 200 and r.get("slug"):
            measured[r["slug"]] = r

    # competition slugs whose rules we hold
    accepted = set()
    for f in ("accept_rules_log.jsonl", "accept_retry_log.jsonl", "accept_r2_log.jsonl"):
        for r in load_jsonl(HERE / f):
            if r.get("outcome") in ("accepted", "no_rules_needed"):
                accepted.add(r["slug"])

    ceiling = set(json.load(open(HERE / "ceiling_all_resources.json")))
    in_pool = {r["task_id"] for r in load_jsonl(HERE / "expanded_inventory.jsonl")}
    ready = {r["task_id"] for r in load_jsonl(HERE / "comp_ready_inventory.jsonl")}

    eval_tasks = set(json.load(open(COLLECT / "eval_tasks.json")))
    fam = lambda t: t.rsplit("@", 1)[0]
    eval_fams = {fam(t) for t in eval_tasks}

    rows, report = [], Counter()
    for tid in sorted(ceiling):
        idx = index.get(tid, {})
        m = meta.get(tid, {})
        rc = recl.get(tid)

        task_clean, task_dirty = clean_task(m.get("task"))
        mod_clean, mod_dirty = clean_modality(m.get("modality"))
        release = idx.get("release_type", "")
        source = idx.get("source_type", "")
        ref = idx.get("download_ref", "")

        # acquisition plan: how would we actually fetch this?
        if release == "built_task":
            acq = {"kind": "hf_package", "endpoint": idx.get("artifact_path", ""),
                   "needs_rules": False, "ready": True}
            mf = manifest.get(tid, {})
            size_bytes = mf.get("compressed_bytes") or idx.get("package_bytes")
        else:
            size_bytes = None
            if source == "KAGGLE_DATASET":
                acq = {"kind": "kaggle_dataset",
                       "endpoint": f"api/v1/datasets/download/{ref}",
                       "needs_rules": False, "ready": bool(ref)}
            elif source == "KAGGLE_COMPETITION":
                acq = {"kind": "kaggle_competition",
                       "endpoint": f"api/v1/competitions/data/download-all/{ref}",
                       "needs_rules": True, "ready": ref in accepted}
            else:
                acq = {"kind": source.lower() or "unknown",
                       "endpoint": ref, "needs_rules": False, "ready": False}

        raw_gb = num(m.get("raw_size_gb"))
        final_gb = num(m.get("final_size_gb"))
        mz = measured.get(ref) if ref else None
        if mz and mz.get("bytes"):
            size_gb = mz["bytes"] / 1e9
            size_source = "measured"
        elif raw_gb is not None:
            size_gb = raw_gb
            size_source = "declared"
        elif size_bytes:
            size_gb = size_bytes / 1e9
            size_source = "package_manifest"
        else:
            size_gb = None
            size_source = None

        # Materialization state. Sizes come from metadata that is known to
        # under-report by orders of magnitude, so 'deferred_large' is advisory —
        # the downloader still enforces its own cap.
        if tid in in_pool:
            state = "materialized"
        elif tid in ready:
            state = "ready"
        elif size_gb is not None and size_gb > args.defer_gb:
            state = "deferred_large"
        elif not acq["ready"]:
            state = "blocked_needs_rules" if acq["needs_rules"] else "blocked"
        else:
            state = "pending"

        rows.append({
            "task_id": tid,
            "source_list": SOURCE_LIST.get(idx.get("task_key", "").split(":")[0], "?"),
            "release_type": release,
            "source_type": source,
            "artifact_path": idx.get("artifact_path", ""),
            "download_ref": ref,
            "source_urls": idx.get("source_urls", []),
            "license": idx.get("license_name_or_permission") or m.get("license_name_or_permission"),
            # scoring contract
            "task_raw": m.get("task"),
            "task": task_clean,
            "task_annotation_dirty": task_dirty,
            "higher_is_better": m.get("higher_is_better"),
            "theoretical_min": num(m.get("theoretical_min")),
            "theoretical_max": num(m.get("theoretical_max")),
            "leaderboard_min": num(m.get("leaderboard_min")),
            "leaderboard_max": num(m.get("leaderboard_max")),
            # compute + modality
            "cpu_gpu_official": m.get("cpu_gpu"),
            "cpu_gpu_reclassified": (rc or {}).get("verdict"),
            "modality_raw": (str(m.get("modality"))[:200] if m.get("modality") else None),
            "modality": mod_clean,
            "modality_annotation_dirty": mod_dirty,
            # size + acquisition
            "size_gb": round(size_gb, 4) if size_gb is not None else None,
            "size_source": size_source,
            "n_files": (mz or {}).get("n_files"),
            "declared_size_gb": raw_gb,
            "final_size_gb": final_gb,
            "acquisition": acq,
            "state": state,
            "eval_leak": tid in eval_tasks or fam(tid) in eval_fams,
        })
        report[state] += 1

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "total": len(rows),
        "by_state": dict(report),
        "by_source_list": dict(Counter(r["source_list"] for r in rows)),
        "by_acquisition": dict(Counter(r["acquisition"]["kind"] for r in rows)),
        "by_task": dict(Counter(r["task"] or "(unrecognized)" for r in rows).most_common(12)),
        "by_modality": dict(Counter(r["modality"] or "(none)" for r in rows).most_common(12)),
        "size_declared_vs_measured_note": ("declared sizes are mostly accurate "
            "(median ratio 0.9x over 11 comparable); a few are wildly off, so a "
            "measured value always wins and the downloader still enforces a cap"),
        "dirty_task_annotations": sum(1 for r in rows if r["task_annotation_dirty"]),
        "dirty_modality_annotations": sum(1 for r in rows if r["modality_annotation_dirty"]),
        "missing_higher_is_better": sum(1 for r in rows if r["higher_is_better"] is None),
        "size_known": sum(1 for r in rows if r["size_gb"] is not None),
        "size_measured": sum(1 for r in rows if r["size_source"] == "measured"),
        "size_declared": sum(1 for r in rows if r["size_source"] == "declared"),
        "size_unknown": sum(1 for r in rows if r["size_gb"] is None),
        "size_total_gb": round(sum(r["size_gb"] or 0 for r in rows), 1),
        "deferred_large_gb": round(sum(r["size_gb"] or 0 for r in rows
                                       if r["state"] == "deferred_large"), 1),
        "eval_leak_count": sum(1 for r in rows if r["eval_leak"]),
        "defer_threshold_gb": args.defer_gb,
    }
    Path(args.report).write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                 encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
