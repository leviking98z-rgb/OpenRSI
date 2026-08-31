#!/usr/bin/env python3
"""Merge frozen quick-SFT recipes on the L20 node.

The input paths below intentionally identify the immutable 20260831T1047
snapshot. Source priority is fixed-d16, L20 continuous, then H20 continuous.
Exact duplicate messages retain the first occurrence.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

BASE = Path(
    "/data2/openrsi/experiments/g0_quick_sft_20260831/"
    "snapshot-20260831T1047"
)
SOURCES = [
    ("fixed-d16", BASE / "fixed/fixed"),
    ("l20-continuous", BASE / "l20/recipe"),
    ("h20-continuous", BASE / "h20/recipe"),
]
OUT = BASE / "combined"
MAX_TOKENS = 32768


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def message_hash(messages: list[dict]) -> str:
    blob = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(blob).hexdigest()


def percentile(values: list[int], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(ordered[low])
    return (
        ordered[low] * (high - position)
        + ordered[high] * (position - low)
    )


def distribution(values: list[int]) -> dict:
    return {
        "count": len(values),
        "sum": sum(values),
        "mean": mean(values) if values else None,
        "min": min(values) if values else None,
        "p25": percentile(values, 0.25),
        "median": median(values) if values else None,
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict] = []
    output_manifest: list[dict] = []
    task_rows: list[dict] = []
    duplicate_rows: list[dict] = []
    source_stats: dict[str, dict] = {}
    source_snapshots: list[dict] = []
    seen_hashes: set[str] = set()
    seen_ids: set[str] = set()

    for source_name, root in SOURCES:
        rows = read_jsonl(root / "train.jsonl")
        manifest = read_jsonl(root / "manifest.jsonl")
        tasks = read_jsonl(root / "tasks.jsonl")
        source_summary = json.loads((root / "summary.json").read_text())
        assert len(rows) == len(manifest)
        manifest_by_id = {row["id"]: row for row in manifest}
        assert len(manifest_by_id) == len(manifest)
        accepted = 0

        for row in rows:
            record_id = row["id"]
            item = manifest_by_id[record_id]
            messages = row["messages"]
            assert [message["role"] for message in messages] == [
                "system",
                "user",
                "assistant",
            ]
            assert all(message["content"].strip() for message in messages)
            digest = message_hash(messages)
            assert digest == item["message_sha256"]
            assert item["training_tokens"] <= MAX_TOKENS
            assert record_id not in seen_ids
            seen_ids.add(record_id)
            if digest in seen_hashes:
                duplicate_rows.append(
                    {
                        "source": source_name,
                        "id": record_id,
                        "message_sha256": digest,
                    }
                )
                continue
            seen_hashes.add(digest)
            row = dict(row, source=source_name)
            item = dict(item, aggregation_source=source_name)
            output_rows.append(row)
            output_manifest.append(item)
            accepted += 1

        for task in tasks:
            task_rows.append(dict(task, source=source_name))

        source_stats[source_name] = {
            "input_rows": len(rows),
            "accepted_rows": accepted,
            "exact_duplicate_drops": len(rows) - accepted,
            "task_manifest_rows": len(tasks),
            "summary_tasks_with_sft": source_summary.get("tasks"),
            "snapshot_completed_tasks": source_summary.get(
                "snapshot_completed_tasks"
            ),
            "operator_calls": source_summary.get("operator_calls"),
            "train_jsonl_sha256_before_merge": sha256_file(
                root / "train.jsonl"
            ),
            "manifest_jsonl_sha256_before_merge": sha256_file(
                root / "manifest.jsonl"
            ),
        }
        source_snapshots.append(
            {
                "source": source_name,
                "path": str(root),
                "summary": source_summary,
                "merge": source_stats[source_name],
            }
        )

    assert len(output_rows) == len(output_manifest)
    assert len({row["id"] for row in output_rows}) == len(output_rows)
    assert len(
        {row["message_sha256"] for row in output_manifest}
    ) == len(output_manifest)

    write_jsonl(OUT / "train.jsonl", output_rows)
    write_jsonl(OUT / "manifest.jsonl", output_manifest)
    write_jsonl(OUT / "tasks.jsonl", task_rows)
    write_jsonl(OUT / "duplicates_dropped.jsonl", duplicate_rows)
    (OUT / "source_snapshots.json").write_text(
        json.dumps(
            source_snapshots,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )

    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.Table.from_pylist(output_rows),
        OUT / "train.parquet",
        compression="zstd",
    )

    operators = collections.Counter(
        row["operator"] for row in output_manifest
    )
    selections = collections.Counter(
        row["selection"] for row in output_manifest
    )
    tasks_by_source: dict[str, set[str]] = collections.defaultdict(set)
    for row in output_manifest:
        tasks_by_source[row["aggregation_source"]].add(row["task_name"])

    summary = {
        "schema_version": 1,
        "recipe": (
            "strict transitions + best verified endpoint rewritten as Draft; "
            "no anchor/replay"
        ),
        "snapshot": "20260831T1047",
        "source_priority": [source for source, _ in SOURCES],
        "rows": len(output_rows),
        "input_rows": sum(
            item["input_rows"] for item in source_stats.values()
        ),
        "exact_duplicate_drops": len(duplicate_rows),
        "source_rows": {
            source: item["accepted_rows"]
            for source, item in source_stats.items()
        },
        "source_stats": source_stats,
        "operators": dict(sorted(operators.items())),
        "selection_types": dict(sorted(selections.items())),
        "tasks_with_sft_by_source": {
            source: len(tasks)
            for source, tasks in tasks_by_source.items()
        },
        "task_source_pairs_with_sft": sum(
            len(tasks) for tasks in tasks_by_source.values()
        ),
        "unique_task_names_with_sft": len(
            {row["task_name"] for row in output_manifest}
        ),
        "task_manifest_rows": len(task_rows),
        "training_tokens": distribution(
            [int(row["training_tokens"]) for row in output_manifest]
        ),
        "assistant_chars": distribution(
            [int(row["assistant_chars"]) for row in output_manifest]
        ),
        "full_chars": distribution(
            [int(row["full_chars"]) for row in output_manifest]
        ),
        "max_tokens_gate": MAX_TOKENS,
        "validation": {
            "ids_unique": True,
            "message_hashes_unique": True,
            "roles_exactly_system_user_assistant": True,
            "all_contents_nonempty": True,
            "all_training_tokens_within_gate": True,
            "manifest_matches_train": True,
        },
    }
    summary["artifacts"] = {
        name: str(OUT / filename)
        for name, filename in {
            "train_jsonl": "train.jsonl",
            "train_parquet": "train.parquet",
            "manifest_jsonl": "manifest.jsonl",
            "tasks_jsonl": "tasks.jsonl",
        }.items()
    }
    summary["artifacts"].update(
        {
            f"{name}_sha256": sha256_file(OUT / filename)
            for name, filename in {
                "train_jsonl": "train.jsonl",
                "train_parquet": "train.parquet",
                "manifest_jsonl": "manifest.jsonl",
                "tasks_jsonl": "tasks.jsonl",
            }.items()
        }
    )
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
