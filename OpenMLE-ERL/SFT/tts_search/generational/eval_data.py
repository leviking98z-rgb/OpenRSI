"""Build the three frozen MA1 evaluation parquet files from installed tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tts_search.generational.common import sha256_file, write_json, write_records

SYSTEM_PROMPT = (
    "You are an ML coding agent running in an offline sandbox. "
    "Return exactly one complete executable Python program in one ```python``` "
    "block and no other prose. Use only files under the DATA_DIR environment "
    "variable. Never use network access, package installation, subprocesses, "
    "absolute host paths, private labels, hidden answers, or reference solutions. "
    "The working directory is writable. You must write ./submission.csv in the "
    "exact format described by the task. Be deterministic and CPU-efficient."
)
USER_PROMPT = (
    "Use the public train/test/sample_submission files available through DATA_DIR. "
    "Train a sensible model, produce submission.csv, and verify its rows, columns, "
    "IDs, and missing values before exiting."
)


def load_split(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    splits = payload.get("splits") if isinstance(payload, dict) else None
    if not isinstance(splits, dict):
        raise ValueError(f"missing splits object: {path}")
    result: dict[str, list[str]] = {}
    all_tasks: list[str] = []
    for name in ("search_train", "promotion", "final_test"):
        values = splits.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"split {name!r} is missing or empty")
        tasks = [
            str(item["task_name"] if isinstance(item, dict) else item)
            for item in values
        ]
        result[name] = tasks
        all_tasks.extend(tasks)
    if len(all_tasks) != len(set(all_tasks)):
        raise ValueError("task split contains duplicates")
    return result


def _task_row(task_root: Path, task_name: str, split_name: str) -> dict[str, Any]:
    task_dir = task_root / task_name
    public_dir = task_dir / "data" / "public"
    required = [
        task_dir / "info" / "task_metadata.json",
        public_dir / "train.csv",
        public_dir / "test.csv",
        public_dir / "sample_submission.csv",
        task_dir / "data" / "private" / "test_answer.csv",
        task_dir / "utils" / "metric.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete task {task_name}: {missing}")

    metadata = json.loads(required[0].read_text(encoding="utf-8"))
    task_description_path = public_dir / "description.txt"
    if not task_description_path.is_file():
        task_description_path = task_dir / "description.txt"
    data_description_path = task_dir / "info" / "data_description.txt"
    task_description = task_description_path.read_text(encoding="utf-8").strip()
    data_description = (
        data_description_path.read_text(encoding="utf-8").strip()
        if data_description_path.is_file()
        else ""
    )
    record_metadata = {
        "uuid": f"ma1::{split_name}::{task_name}",
        "task_id": f"ma1::{split_name}::{task_name}",
        "task_name": task_name,
        "task": str(metadata.get("task") or ""),
        "source": str(metadata.get("source") or ""),
        "modality": metadata.get("modality"),
        "cpu_gpu": str(metadata.get("cpu_gpu") or "CPU"),
        "data_dir": f"/mnt/pubdatasets2/MLTasks/Selected_Dojo/{task_name}",
        "higher_is_better": bool(metadata["higher_is_better"]),
        "theoretical_min": metadata.get("theoretical_min"),
        "theoretical_max": metadata.get("theoretical_max"),
        "leaderboard_min": metadata.get("leaderboard_min"),
        "leaderboard_max": metadata.get("leaderboard_max"),
        "task_description": task_description,
        "data_description": data_description,
        "experiment_split": split_name,
    }
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "metadata": record_metadata,
    }


def build_eval_data(
    split_path: Path, task_root: Path, output_dir: Path
) -> dict[str, Any]:
    splits = load_split(split_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "split_manifest": str(split_path.resolve()),
        "split_manifest_sha256": sha256_file(split_path),
        "task_root": str(task_root.resolve()),
        "splits": {},
    }
    for split_name, task_names in splits.items():
        rows = [_task_row(task_root, task_name, split_name) for task_name in task_names]
        parquet = output_dir / f"{split_name}.parquet"
        jsonl = output_dir / f"{split_name}.jsonl"
        write_records(parquet, rows)
        write_records(jsonl, rows)
        result["splits"][split_name] = {
            "rows": len(rows),
            "parquet": str(parquet),
            "parquet_sha256": sha256_file(parquet),
            "jsonl": str(jsonl),
            "jsonl_sha256": sha256_file(jsonl),
        }
    write_json(output_dir / "summary.json", result)
    return result
