"""Export fixed-budget Evo executions for paired promotion evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tts_search.generational.common import finite_float, write_json, write_jsonl


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _epoch_dirs(root: Path) -> list[Path]:
    if root.name.startswith("program_ep_"):
        return [root]
    return sorted(
        (path for path in root.glob("program_ep_*") if path.is_dir()),
        key=lambda path: int(path.name.removeprefix("program_ep_")),
    )


def export_evolutionary_eval(
    rollout_root: Path, output_path: Path, expected_budget: int = 4
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for epoch_dir in _epoch_dirs(rollout_root):
        for task_dir in sorted(path for path in epoch_dir.iterdir() if path.is_dir()):
            stat_path = task_dir / "stat.json"
            config_path = task_dir / "aira_evo" / "dojo_config.json"
            if not stat_path.is_file():
                continue
            task_stat = _object(stat_path)
            config = _object(config_path)
            task = config.get("task") or {}
            task_name = str(task_stat.get("task_name") or task_dir.name)
            seed = int(config.get("seed", config.get("sample_index", 0)))
            key = (task_name, seed)
            if key in seen:
                raise ValueError(f"duplicate task/seed: {key}")
            seen.add(key)
            steps = sorted(
                (dict(step) for step in task_stat.get("steps") or []),
                key=lambda step: int(step["step"]),
            )
            indices = [int(step["step"]) for step in steps]
            if indices != list(range(len(steps))):
                raise ValueError(
                    f"non-contiguous execution indices for {key}: {indices}"
                )
            if len(steps) != expected_budget:
                raise ValueError(
                    f"{key} has {len(steps)} executions, expected {expected_budget}"
                )
            for step in steps:
                score = finite_float(step.get("score"))
                valid = not bool(step.get("is_buggy")) and score is not None
                records.append(
                    {
                        "task_name": task_name,
                        "seed": seed,
                        "execution_index": int(step["step"]),
                        "operator": str(
                            step.get("operator") or step.get("mode") or "unknown"
                        ).lower(),
                        "score": score,
                        "reward": finite_float(step.get("reward")),
                        "valid": valid,
                        "higher_is_better": bool(task["higher_is_better"]),
                        "theoretical_min": finite_float(task.get("theoretical_min")),
                        "theoretical_max": finite_float(task.get("theoretical_max")),
                    }
                )
    if not records:
        raise ValueError(f"no rollout records under {rollout_root}")
    write_jsonl(output_path, records)
    summary = {
        "records": len(records),
        "task_seed_pairs": len(seen),
        "tasks": len({task for task, _ in seen}),
        "valid_records": sum(row["valid"] for row in records),
        "expected_budget": expected_budget,
        "output": str(output_path.resolve()),
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    return summary
