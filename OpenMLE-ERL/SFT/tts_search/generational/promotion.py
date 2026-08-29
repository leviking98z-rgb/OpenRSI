"""Paired, task-level promotion gate for G1 versus G0 and replay control."""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from tts_search.generational.common import finite_float, read_jsonl, write_json


def _load(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def _normal(row: dict[str, Any]) -> float:
    if not bool(row.get("valid")):
        return 0.0
    reward = finite_float(row.get("reward"))
    if reward is not None:
        if 0.0 <= reward <= 1.0:
            return reward
        raise ValueError(f"reward outside [0, 1]: {reward}")
    score = finite_float(row.get("score"))
    lower = finite_float(row.get("theoretical_min"))
    upper = finite_float(row.get("theoretical_max"))
    if score is None:
        return 0.0
    if lower is not None and upper is not None and upper > lower:
        value = (score - lower) / (upper - lower)
        if not bool(row.get("higher_is_better", True)):
            value = 1.0 - value
        return min(1.0, max(0.0, value))
    raise ValueError("valid row needs a unit reward or finite theoretical bounds")


def _task_metrics(
    rows: list[dict[str, Any]], budget: int
) -> dict[str, dict[str, float]]:
    by_pair: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[(str(row["task_name"]), int(row["seed"]))].append(row)
    per_task: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (task, _seed), group in by_pair.items():
        group.sort(key=lambda row: int(row["execution_index"]))
        indices = [int(row["execution_index"]) for row in group]
        if len(group) != budget or indices != list(range(budget)):
            raise ValueError(f"{task} does not contain exactly Evo@{budget}")
        scores = [_normal(row) for row in group]
        running: list[float] = []
        for score in scores:
            running.append(max(running[-1], score) if running else score)
        per_task[task].append(
            {
                "direct": scores[0],
                "best": max(scores),
                "auc": statistics.fmean(running),
                "valid_rate": sum(bool(row.get("valid")) for row in group) / budget,
            }
        )
    return {
        task: {
            metric: statistics.fmean(seed_row[metric] for seed_row in seed_rows)
            for metric in ("direct", "best", "auc", "valid_rate")
        }
        for task, seed_rows in per_task.items()
    }


def _bootstrap(values: list[float], seed: int, samples: int) -> list[float]:
    if len(values) < 2:
        return values * 2
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)
    )
    return [means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]]


def _compare(
    candidate: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
    metric: str,
    seed: int,
    samples: int,
    tie_tolerance: float,
) -> dict[str, Any]:
    if set(candidate) != set(baseline):
        raise ValueError("evaluation task sets do not match")
    pairs = []
    for task in sorted(candidate):
        delta = candidate[task][metric] - baseline[task][metric]
        outcome = (
            "win"
            if delta > tie_tolerance
            else "loss"
            if delta < -tie_tolerance
            else "tie"
        )
        pairs.append({"task_name": task, "delta": delta, "outcome": outcome})
    deltas = [row["delta"] for row in pairs]
    return {
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "wins": sum(row["outcome"] == "win" for row in pairs),
        "ties": sum(row["outcome"] == "tie" for row in pairs),
        "losses": sum(row["outcome"] == "loss" for row in pairs),
        "bootstrap_95_ci": _bootstrap(deltas, seed, samples),
        "pairs": pairs,
    }


def evaluate_promotion(
    *,
    parent_paths: list[Path],
    candidate_paths: list[Path],
    control_paths: list[Path],
    output_dir: Path,
    budget: int = 4,
    tie_tolerance: float = 0.0,
    min_valid_rate_delta: float = -0.05,
    bootstrap_seed: int = 20260829,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    if budget <= 0 or bootstrap_samples <= 0 or tie_tolerance < 0:
        raise ValueError("invalid promotion arguments")
    groups = {
        "parent": _task_metrics(_load(parent_paths), budget),
        "candidate": _task_metrics(_load(candidate_paths), budget),
        "control": _task_metrics(_load(control_paths), budget),
    }
    comparisons: dict[str, dict[str, Any]] = {}
    for offset, baseline in enumerate(("parent", "control")):
        comparisons[f"candidate_vs_{baseline}"] = {
            metric: _compare(
                groups["candidate"],
                groups[baseline],
                metric,
                bootstrap_seed + offset,
                bootstrap_samples,
                tie_tolerance,
            )
            for metric in ("direct", "best", "auc", "valid_rate")
        }
    parent_best = comparisons["candidate_vs_parent"]["best"]
    control_best = comparisons["candidate_vs_control"]["best"]
    valid = comparisons["candidate_vs_parent"]["valid_rate"]
    checks = {
        "best_mean_gt_parent": parent_best["mean_delta"] > 0,
        "best_wins_gt_losses_parent": parent_best["wins"] > parent_best["losses"],
        "best_mean_gt_control": control_best["mean_delta"] > 0,
        "best_wins_gt_losses_control": control_best["wins"] > control_best["losses"],
        "valid_rate_not_regressed": valid["mean_delta"] >= min_valid_rate_delta,
    }
    result = {
        "decision": "accept" if all(checks.values()) else "reject",
        "budget": budget,
        "metrics_by_task": groups,
        "comparisons": comparisons,
        "gate_checks": checks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "promotion.json", result)
    return result
