from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tts_search.generational.common import read_jsonl, write_jsonl
from tts_search.generational.distill import distill_transitions
from tts_search.generational.eval_data import build_eval_data, load_split
from tts_search.generational.evo_eval import export_evolutionary_eval
from tts_search.generational.mixture import build_matched_mixtures
from tts_search.generational.promotion import evaluate_promotion


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _split(path: Path, search_train: list[str]) -> None:
    _json(
        path,
        {
            "splits": {
                "search_train": search_train,
                "promotion": ["held-out-a"],
                "final_test": ["held-out-b"],
            }
        },
    )


def _step(task_dir: Path, index: int, operator: str, score: float | None) -> None:
    step_dir = task_dir / f"step_{index}"
    step_dir.mkdir(parents=True)
    _json(
        step_dir / "stat.json",
        {
            "operator": operator,
            "score": score,
            "is_buggy": score is None,
            "node_id": f"node-{index}",
        },
    )
    (step_dir / "system_prompt.md").write_text("system", encoding="utf-8")
    (step_dir / "user_prompt.md").write_text(f"prompt-{index}", encoding="utf-8")
    (step_dir / "reasoning.md").write_text(f"reason-{index}", encoding="utf-8")
    (step_dir / "response.md").write_text("fallback", encoding="utf-8")
    (step_dir / "valid_code.py").write_text(f"print({index})\n", encoding="utf-8")


def test_build_eval_data_from_frozen_split(tmp_path: Path) -> None:
    pq = __import__("pytest").importorskip("pyarrow.parquet")
    split_path = tmp_path / "split.json"
    _split(split_path, ["train-task"])
    for task_name in ("train-task", "held-out-a", "held-out-b"):
        task = tmp_path / "tasks" / task_name
        _json(
            task / "info/task_metadata.json",
            {
                "task_name": task_name,
                "task": "Classification",
                "cpu_gpu": "CPU",
                "higher_is_better": True,
                "theoretical_min": 0.0,
                "theoretical_max": 1.0,
            },
        )
        (task / "info/data_description.txt").write_text("two csv files")
        public = task / "data/public"
        private = task / "data/private"
        public.mkdir(parents=True)
        private.mkdir(parents=True)
        for name in ("train.csv", "test.csv", "sample_submission.csv"):
            (public / name).write_text("id,y\n1,0\n")
        (public / "description.txt").write_text("predict y")
        (private / "test_answer.csv").write_text("id,y\n1,0\n")
        metric = task / "utils/metric.py"
        metric.parent.mkdir(parents=True)
        metric.write_text("class Metric: pass\n")

    summary = build_eval_data(split_path, tmp_path / "tasks", tmp_path / "eval")
    loaded = load_split(split_path)
    assert [len(loaded[name]) for name in loaded] == [1, 1, 1]
    assert summary["splits"]["search_train"]["rows"] == 1
    row = pq.read_table(tmp_path / "eval/search_train.parquet").to_pylist()[0]
    assert (
        row["metadata"]["data_dir"]
        == "/mnt/pubdatasets2/MLTasks/Selected_Dojo/train-task"
    )
    system = row["prompt"][0]["content"]
    assert "Never use network access" in system
    assert "private labels" in system


def test_distill_keeps_only_strict_improve_and_invalid_to_valid(tmp_path: Path) -> None:
    split_path = tmp_path / "split.json"
    _split(split_path, ["task"])
    task = tmp_path / "rollout/program_ep_0/task"
    _json(
        task / "aira_evo/dojo_config.json",
        {"seed": 11, "task": {"higher_is_better": True}},
    )
    steps = [
        {"step": 0, "operator": "draft", "score": 0.2, "is_buggy": False},
        {
            "step": 1,
            "operator": "improve",
            "score": 0.4,
            "is_buggy": False,
            "parent_steps": [0],
        },
        {
            "step": 2,
            "operator": "improve",
            "score": 0.3,
            "is_buggy": False,
            "parent_steps": [1],
        },
        {"step": 3, "operator": "draft", "score": None, "is_buggy": True},
        {
            "step": 4,
            "operator": "debug",
            "score": 0.1,
            "is_buggy": False,
            "parent_steps": [3],
        },
    ]
    _json(task / "stat.json", {"task_name": "task", "steps": steps})
    for step in steps:
        _step(task, step["step"], step["operator"], step["score"])

    summary = distill_transitions(
        tmp_path / "rollout", split_path, tmp_path / "distilled"
    )
    rows = read_jsonl(tmp_path / "distilled/evo_transitions.manifest.jsonl")
    assert summary["operators"] == {"debug": 1, "improve": 1}
    assert [(row["child_step"], row["selection"]) for row in rows] == [
        (1, "strict_improve"),
        (4, "invalid_to_valid"),
    ]
    assert read_jsonl(tmp_path / "distilled/evo_transitions.jsonl")[0]["messages"][-1][
        "content"
    ].endswith("```python\nprint(1)\n```")


def test_distill_drops_steps_beyond_declared_execution_budget(
    tmp_path: Path,
) -> None:
    split_path = tmp_path / "split.json"
    _split(split_path, ["task"])
    task = tmp_path / "rollout/program_ep_0/task"
    _json(
        task / "aira_evo/dojo_config.json",
        {"seed": 11, "task": {"higher_is_better": True}},
    )
    steps = [
        {"step": 0, "operator": "draft", "score": 0.2, "is_buggy": False},
        {
            "step": 1,
            "operator": "improve",
            "score": 0.4,
            "is_buggy": False,
            "parent_steps": [0],
        },
        {
            "step": 2,
            "operator": "improve",
            "score": 0.5,
            "is_buggy": False,
            "parent_steps": [1],
        },
    ]
    _json(
        task / "stat.json",
        {"task_name": "task", "step_limit": 2, "steps": steps},
    )
    for step in steps:
        _step(task, step["step"], step["operator"], step["score"])

    summary = distill_transitions(
        tmp_path / "rollout", split_path, tmp_path / "distilled"
    )
    rows = read_jsonl(tmp_path / "distilled/evo_transitions.manifest.jsonl")
    assert [row["child_step"] for row in rows] == [1]
    assert summary["drops"]["outside_execution_budget"] == 1


def _messages(label: str, length: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": label},
        {"role": "assistant", "content": "x" * length},
    ]


def test_mixture_has_shared_replay_and_token_matched_control(tmp_path: Path) -> None:
    split_path = tmp_path / "split.json"
    _split(split_path, ["train-a", "train-b"])
    anchor_path = tmp_path / "anchor.jsonl"
    evo_path = tmp_path / "evo.jsonl"
    anchor_rows = [
        {
            "id": f"a-{index}",
            "task_name": f"anchor-{index}",
            "messages": _messages(str(index), length),
        }
        for index, length in enumerate([10, 20, 30, 40, 50, 60, 70, 80])
    ]
    evo_rows = [
        {"id": "e-0", "task_name": "train-a", "messages": _messages("e0", 20)},
        {"id": "e-1", "task_name": "train-b", "messages": _messages("e1", 70)},
    ]
    write_jsonl(anchor_path, anchor_rows)
    write_jsonl(evo_path, evo_rows)

    summary = build_matched_mixtures(
        anchor_path=anchor_path,
        evo_path=evo_path,
        split_path=split_path,
        tokenizer_model=Path("/required-but-injected"),
        output_dir=tmp_path / "mix",
        total_rows=6,
        max_token_mismatch_fraction=0.01,
        count_tokens=lambda value: len(value[-1]["content"]),
    )
    candidate = read_jsonl(tmp_path / "mix/candidate_train.manifest.jsonl")
    control = read_jsonl(tmp_path / "mix/control_train.manifest.jsonl")
    assert summary["candidate_rows"] == summary["control_rows"] == 6
    assert sum(row["source"] == "evo" for row in candidate) == 2
    assert sum(row["source"] == "evo" for row in control) == 0
    candidate_anchor_ids = {row["id"] for row in candidate if row["source"] == "anchor"}
    control_ids = {row["id"] for row in control}
    assert candidate_anchor_ids < control_ids
    assert summary["token_mismatch_fraction"] == 0.0


def _eval_rows(values: dict[str, float], seed: int) -> list[dict[str, Any]]:
    return [
        {
            "task_name": task,
            "seed": seed,
            "execution_index": index,
            "reward": max(0.0, score - 0.03 * index),
            "valid": True,
            "higher_is_better": True,
        }
        for task, score in values.items()
        for index in range(4)
    ]


def test_export_and_task_level_promotion_gate(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout/program_ep_0/task-a"
    _json(
        rollout / "aira_evo/dojo_config.json",
        {
            "seed": 7,
            "task": {
                "higher_is_better": True,
                "theoretical_min": 0.0,
                "theoretical_max": 1.0,
            },
        },
    )
    steps = [
        {
            "step": index,
            "operator": "draft" if index == 0 else "improve",
            "score": 0.1 + index / 10,
            "reward": 0.1 + index / 10,
            "is_buggy": False,
        }
        for index in range(4)
    ]
    _json(rollout / "stat.json", {"task_name": "task-a", "steps": steps})
    exported = tmp_path / "export.jsonl"
    assert export_evolutionary_eval(tmp_path / "rollout", exported)["records"] == 4
    assert read_jsonl(exported)[0]["execution_index"] == 0

    parent = tmp_path / "parent.jsonl"
    control = tmp_path / "control.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_jsonl(parent, _eval_rows({"task-a": 0.4, "task-b": 0.3}, 1))
    write_jsonl(control, _eval_rows({"task-a": 0.45, "task-b": 0.35}, 1))
    write_jsonl(candidate, _eval_rows({"task-a": 0.7, "task-b": 0.6}, 1))
    result = evaluate_promotion(
        parent_paths=[parent],
        candidate_paths=[candidate],
        control_paths=[control],
        output_dir=tmp_path / "promotion",
        bootstrap_samples=100,
    )
    assert result["decision"] == "accept"
    assert result["comparisons"]["candidate_vs_parent"]["best"]["wins"] == 2


def test_promotion_uses_raw_scores_when_reward_is_constant_zero(
    tmp_path: Path,
) -> None:
    def rows(score: float, label: str) -> list[dict[str, Any]]:
        return [
            {
                "task_name": "lower-is-better",
                "seed": seed,
                "execution_index": index,
                "score": score + 0.1 * index,
                "reward": 0.0,
                "valid": True,
                "higher_is_better": False,
                "theoretical_min": 0.0,
                "theoretical_max": None,
                "label": label,
            }
            for seed in (1, 2)
            for index in range(4)
        ]

    parent = tmp_path / "parent.jsonl"
    control = tmp_path / "control.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_jsonl(parent, rows(3.0, "parent"))
    write_jsonl(control, rows(2.0, "control"))
    write_jsonl(candidate, rows(1.0, "candidate"))

    result = evaluate_promotion(
        parent_paths=[parent],
        candidate_paths=[candidate],
        control_paths=[control],
        output_dir=tmp_path / "promotion",
        bootstrap_samples=100,
    )

    assert result["normalization_by_task"]["lower-is-better"]["mode"] == (
        "pooled_observed_range"
    )
    assert result["comparisons"]["candidate_vs_parent"]["best"]["mean_delta"] > 0
    assert result["comparisons"]["candidate_vs_control"]["best"]["mean_delta"] > 0
    assert result["decision"] == "accept"
