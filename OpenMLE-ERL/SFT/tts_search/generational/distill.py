"""Distill only verified Improve and Debug edges from an Evo rollout."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tts_search.generational.common import (
    finite_float,
    messages,
    sha256_file,
    write_json,
    write_sft_pair,
)
from tts_search.generational.eval_data import load_split


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _valid(step: dict[str, Any]) -> bool:
    return (
        not bool(step.get("is_buggy")) and finite_float(step.get("score")) is not None
    )


def _assistant(step_dir: Path) -> str:
    def read(name: str) -> str:
        path = step_dir / name
        return (
            path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        )

    reasoning = read("reasoning.md").strip()
    code = read("valid_code.py").rstrip()
    response = read("response.md").strip()
    if reasoning and code:
        return f"<think>\n{reasoning}\n</think>\n\n```python\n{code}\n```"
    if code:
        return f"```python\n{code}\n```"
    return response


def _epoch_dirs(rollout_root: Path) -> list[Path]:
    if rollout_root.name.startswith("program_ep_"):
        return [rollout_root]
    return sorted(
        (path for path in rollout_root.glob("program_ep_*") if path.is_dir()),
        key=lambda path: int(path.name.removeprefix("program_ep_")),
    )


def distill_transitions(
    rollout_root: Path,
    split_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    allowed = set(load_split(split_path)["search_train"])
    selected: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    seen_messages: set[str] = set()
    drops: Counter[str] = Counter()

    for epoch_dir in _epoch_dirs(rollout_root):
        for task_dir in sorted(path for path in epoch_dir.iterdir() if path.is_dir()):
            stat_path = task_dir / "stat.json"
            config_path = task_dir / "aira_evo" / "dojo_config.json"
            if not stat_path.is_file():
                continue
            task_stat = _load_object(stat_path)
            task_name = str(task_stat.get("task_name") or task_dir.name)
            if task_name not in allowed:
                drops["outside_search_train"] += 1
                continue
            if not config_path.is_file():
                raise FileNotFoundError(config_path)
            task_cfg = _load_object(config_path).get("task") or {}
            higher_is_better = task_cfg.get("higher_is_better")
            if not isinstance(higher_is_better, bool):
                raise ValueError(f"missing higher_is_better: {config_path}")
            step_limit = int(task_stat.get("step_limit") or 0)
            raw_steps = [
                dict(step)
                for step in task_stat.get("steps") or []
                if isinstance(step, dict) and "step" in step
            ]
            if step_limit > 0:
                drops["outside_execution_budget"] += sum(
                    int(step["step"]) >= step_limit for step in raw_steps
                )
                raw_steps = [
                    step for step in raw_steps if int(step["step"]) < step_limit
                ]
            steps = {
                int(step["step"]): dict(step)
                for step in raw_steps
            }

            for child_index, child in sorted(steps.items()):
                operator = str(child.get("operator") or child.get("mode") or "").lower()
                if operator not in {"improve", "debug"}:
                    continue
                parent_indices = [
                    int(value) for value in child.get("parent_steps") or []
                ]
                if len(parent_indices) != 1 or parent_indices[0] not in steps:
                    drops["missing_single_parent"] += 1
                    continue
                parent_index = parent_indices[0]
                parent = steps[parent_index]
                parent_score = finite_float(parent.get("score"))
                child_score = finite_float(child.get("score"))
                child_valid = _valid(child)
                parent_valid = _valid(parent)
                if operator == "improve":
                    better = (
                        child_score is not None
                        and parent_score is not None
                        and (
                            child_score > parent_score
                            if higher_is_better
                            else child_score < parent_score
                        )
                    )
                    keep = parent_valid and child_valid and better
                    reason = "strict_improve"
                else:
                    keep = (not parent_valid) and child_valid
                    reason = "invalid_to_valid"
                if not keep:
                    drops[f"rejected_{operator}"] += 1
                    continue

                step_dir = task_dir / f"step_{child_index}"
                system = (
                    (step_dir / "system_prompt.md")
                    .read_text(encoding="utf-8", errors="replace")
                    .strip()
                )
                user = (
                    (step_dir / "user_prompt.md")
                    .read_text(encoding="utf-8", errors="replace")
                    .strip()
                )
                assistant = _assistant(step_dir)
                if not user or not assistant:
                    drops["missing_prompt_or_target"] += 1
                    continue
                conversation = messages(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ]
                )
                digest = hashlib.sha256(
                    json.dumps(
                        conversation, sort_keys=True, ensure_ascii=False
                    ).encode()
                ).hexdigest()
                if digest in seen_messages:
                    drops["duplicate_messages"] += 1
                    continue
                seen_messages.add(digest)
                row_id = f"{epoch_dir.name}::{task_name}::step_{child_index}"
                selected.append(
                    {"id": row_id, "task_name": task_name, "messages": conversation}
                )
                manifest.append(
                    {
                        "id": row_id,
                        "task_name": task_name,
                        "seed": int(_load_object(config_path).get("seed", 0)),
                        "operator": operator,
                        "selection": reason,
                        "parent_step": parent_index,
                        "child_step": child_index,
                        "parent_valid": parent_valid,
                        "child_valid": child_valid,
                        "parent_score": parent_score,
                        "child_score": child_score,
                        "higher_is_better": higher_is_better,
                        "signed_delta": (
                            None
                            if operator == "debug"
                            else (child_score - parent_score)
                            * (1.0 if higher_is_better else -1.0)
                        ),
                        "step_dir": str(step_dir.resolve()),
                        "step_stat_sha256": sha256_file(step_dir / "stat.json"),
                    }
                )

    if not selected:
        raise ValueError("no verified Improve/Debug transitions were found")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = write_sft_pair(output_dir, "evo_transitions", selected, manifest)
    summary = {
        "rows": len(selected),
        "tasks": len({row["task_name"] for row in selected}),
        "operators": dict(Counter(row["operator"] for row in manifest)),
        "drops": dict(drops),
        "rollout_root": str(rollout_root.resolve()),
        "split_sha256": sha256_file(split_path),
        "artifacts": artifacts,
    }
    write_json(output_dir / "summary.json", summary)
    return summary
