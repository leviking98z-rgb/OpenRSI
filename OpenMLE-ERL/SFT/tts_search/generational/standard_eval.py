"""Small standardized reports for operator and end-to-end MA1 evaluation."""

from __future__ import annotations

import statistics
import tempfile
from pathlib import Path
from typing import Any

from tts_search.generational.common import (
    finite_float,
    read_rows,
    write_json,
    write_jsonl,
)
from tts_search.generational.promotion import evaluate_promotion

SUPPORTED_OPERATORS = ("debug", "improve")


def _load(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_rows(path))
    return rows


def _manifest_task_name(row: dict[str, Any]) -> str:
    value = row.get("task_name")
    if value is None and isinstance(row.get("metadata"), dict):
        value = row["metadata"].get("task_name")
    task_name = str(value or "").strip()
    if not task_name:
        raise ValueError("task manifest row is missing task_name")
    return task_name


def select_task_names(task_manifest: Path, num_tasks: int) -> list[str]:
    """Select the first N unique tasks from a frozen JSONL/Parquet manifest."""

    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    task_names: list[str] = []
    seen: set[str] = set()
    for row in read_rows(task_manifest):
        task_name = _manifest_task_name(row)
        if task_name in seen:
            raise ValueError(f"duplicate task in manifest: {task_name}")
        seen.add(task_name)
        task_names.append(task_name)
    if len(task_names) < num_tasks:
        raise ValueError(
            f"task manifest contains {len(task_names)} tasks, requested {num_tasks}"
        )
    return task_names[:num_tasks]


def _select_e2e_rows(
    rows: list[dict[str, Any]], task_names: list[str], label: str
) -> list[dict[str, Any]]:
    selected = set(task_names)
    filtered = [
        row for row in rows if str(row.get("task_name") or "") in selected
    ]
    present = {str(row.get("task_name") or "") for row in filtered}
    missing = [task_name for task_name in task_names if task_name not in present]
    if missing:
        raise ValueError(f"{label} results are missing tasks: {missing}")
    return filtered


def evaluate_e2e(
    *,
    parent_paths: list[Path],
    candidate_paths: list[Path],
    task_manifest: Path,
    num_tasks: int,
    budget: int,
    output_dir: Path,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Report two already-exported fixed-budget Evo runs on a frozen task prefix."""

    task_names = select_task_names(task_manifest, num_tasks)
    parent_rows = _select_e2e_rows(_load(parent_paths), task_names, "parent")
    candidate_rows = _select_e2e_rows(
        _load(candidate_paths), task_names, "candidate"
    )
    with tempfile.TemporaryDirectory(prefix="ma1-standard-eval-") as temp_dir:
        temp_root = Path(temp_dir)
        parent_path = temp_root / "parent.jsonl"
        candidate_path = temp_root / "candidate.jsonl"
        write_jsonl(parent_path, parent_rows)
        write_jsonl(candidate_path, candidate_rows)
        analysis = evaluate_promotion(
            parent_paths=[parent_path],
            candidate_paths=[candidate_path],
            output_dir=temp_root / "promotion",
            budget=budget,
            min_valid_rate_delta=0.0,
            bootstrap_samples=bootstrap_samples,
        )

    metrics_by_task = analysis["metrics_by_task"]
    overall_scores = {
        model: {
            metric: statistics.fmean(
                task_metrics[metric] for task_metrics in model_metrics.values()
            )
            for metric in ("direct", "best", "auc", "valid_rate")
        }
        for model, model_metrics in metrics_by_task.items()
    }
    parent_best = overall_scores["parent"]["best"]
    candidate_best = overall_scores["candidate"]["best"]
    result = {
        "mode": "e2e",
        "num_tasks": num_tasks,
        "selected_tasks": task_names,
        "budget": budget,
        "primary_metric": {
            "name": "mean_normalized_best",
            "parent": parent_best,
            "candidate": candidate_best,
            "delta": candidate_best - parent_best,
        },
        "overall_scores": overall_scores,
        "normalization_by_task": analysis["normalization_by_task"],
        "metrics_by_task": metrics_by_task,
        "comparisons": analysis["comparisons"],
    }
    write_json(output_dir / "standard_eval.json", result)
    return result


def _operators(values: list[str]) -> list[str]:
    operators: list[str] = []
    for value in values:
        operator = value.strip().lower()
        if operator not in SUPPORTED_OPERATORS:
            raise ValueError(
                f"unsupported operator {value!r}; choose from {SUPPORTED_OPERATORS}"
            )
        if operator not in operators:
            operators.append(operator)
    if not operators:
        raise ValueError("at least one operator is required")
    return operators


def _select_operator_cases(
    case_path: Path,
    num_tasks: int,
    operators: list[str],
    cases_per_operator: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    if num_tasks <= 0 or cases_per_operator <= 0:
        raise ValueError("num_tasks and cases_per_operator must be positive")
    allowed = set(operators)
    rows = [
        row
        for row in read_rows(case_path)
        if str(row.get("operator") or "").lower() in allowed
    ]
    task_names: list[str] = []
    seen_tasks: set[str] = set()
    seen_cases: set[str] = set()
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        task_name = str(row.get("task_name") or "").strip()
        if not case_id or not task_name:
            raise ValueError("operator case is missing case_id or task_name")
        if case_id in seen_cases:
            raise ValueError(f"duplicate operator case: {case_id}")
        seen_cases.add(case_id)
        if task_name not in seen_tasks:
            seen_tasks.add(task_name)
            task_names.append(task_name)
    if len(task_names) < num_tasks:
        raise ValueError(
            f"operator cases contain {len(task_names)} tasks, requested {num_tasks}"
        )
    task_names = task_names[:num_tasks]
    selected_rows: list[dict[str, Any]] = []
    for task_name in task_names:
        for operator in operators:
            matching = [
                row
                for row in rows
                if str(row["task_name"]) == task_name
                and str(row["operator"]).lower() == operator
            ]
            if len(matching) < cases_per_operator:
                raise ValueError(
                    f"{task_name} has {len(matching)} {operator} cases, "
                    f"requested {cases_per_operator}"
                )
            selected_rows.extend(matching[:cases_per_operator])
    return task_names, selected_rows


def _result_index(
    paths: list[Path], selected_case_ids: set[str], label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in _load(paths):
        case_id = str(row.get("case_id") or "").strip()
        if not case_id or case_id not in selected_case_ids:
            continue
        if case_id in indexed:
            raise ValueError(f"duplicate {label} result for case {case_id}")
        indexed[case_id] = row
    missing = sorted(selected_case_ids - set(indexed))
    if missing:
        raise ValueError(f"{label} results are missing cases: {missing}")
    return indexed


def _valid(row: dict[str, Any], label: str) -> bool:
    value = row.get("valid")
    if not isinstance(value, bool):
        raise ValueError(f"{label} valid must be a boolean")
    return value


def _operator_success(
    case: dict[str, Any], result: dict[str, Any], label: str
) -> bool:
    operator = str(case["operator"]).lower()
    valid = _valid(result, label)
    if operator == "debug":
        return valid

    parent_score = finite_float(case.get("parent_score"))
    if parent_score is None:
        raise ValueError(f"{label} improve case is missing finite parent_score")
    child_score = finite_float(result.get("score"))
    if not valid or child_score is None:
        return False
    higher_is_better = case.get("higher_is_better", True)
    if not isinstance(higher_is_better, bool):
        raise ValueError(f"{label} higher_is_better must be a boolean")
    return (
        child_score > parent_score
        if higher_is_better
        else child_score < parent_score
    )


def _check_result_identity(
    case: dict[str, Any], result: dict[str, Any], label: str
) -> None:
    for field in ("task_name", "operator"):
        if field not in result:
            continue
        expected = str(case[field]).lower() if field == "operator" else str(case[field])
        actual = (
            str(result[field]).lower() if field == "operator" else str(result[field])
        )
        if actual != expected:
            raise ValueError(
                f"{label} result {case['case_id']} has {field}={actual!r}, "
                f"expected {expected!r}"
            )


def evaluate_operator(
    *,
    case_path: Path,
    parent_paths: list[Path],
    candidate_paths: list[Path],
    num_tasks: int,
    operators: list[str],
    output_dir: Path,
    cases_per_operator: int = 2,
) -> dict[str, Any]:
    """Report fixed-context Debug/Improve success for two models."""

    selected_operators = _operators(operators)
    task_names, cases = _select_operator_cases(
        case_path, num_tasks, selected_operators, cases_per_operator
    )
    case_ids = {str(case["case_id"]) for case in cases}
    parent = _result_index(parent_paths, case_ids, "parent")
    candidate = _result_index(candidate_paths, case_ids, "candidate")

    comparisons: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        parent_result = parent[case_id]
        candidate_result = candidate[case_id]
        _check_result_identity(case, parent_result, "parent")
        _check_result_identity(case, candidate_result, "candidate")
        parent_success = _operator_success(
            case, parent_result, f"parent result {case_id}"
        )
        candidate_success = _operator_success(
            case, candidate_result, f"candidate result {case_id}"
        )
        comparisons.append(
            {
                "case_id": case_id,
                "task_name": str(case["task_name"]),
                "operator": str(case["operator"]).lower(),
                "parent_program_valid": case.get("parent_valid"),
                "parent_program_score": finite_float(case.get("parent_score")),
                "higher_is_better": case.get("higher_is_better", True),
                "parent_model": {
                    "valid": _valid(parent_result, f"parent result {case_id}"),
                    "score": finite_float(parent_result.get("score")),
                    "success": parent_success,
                },
                "candidate_model": {
                    "valid": _valid(
                        candidate_result, f"candidate result {case_id}"
                    ),
                    "score": finite_float(candidate_result.get("score")),
                    "success": candidate_success,
                },
            }
        )

    metrics: dict[str, dict[str, Any]] = {}
    for operator in selected_operators:
        rows = [row for row in comparisons if row["operator"] == operator]
        parent_successes = sum(row["parent_model"]["success"] for row in rows)
        candidate_successes = sum(
            row["candidate_model"]["success"] for row in rows
        )
        metrics[operator] = {
            "cases": len(rows),
            "parent_successes": parent_successes,
            "candidate_successes": candidate_successes,
            "parent_success_rate": parent_successes / len(rows),
            "candidate_success_rate": candidate_successes / len(rows),
            "success_delta": candidate_successes - parent_successes,
        }

    parent_total = sum(row["parent_model"]["success"] for row in comparisons)
    candidate_total = sum(
        row["candidate_model"]["success"] for row in comparisons
    )
    total_cases = len(comparisons)
    result = {
        "mode": "operator",
        "num_tasks": num_tasks,
        "selected_tasks": task_names,
        "operators": selected_operators,
        "cases_per_operator_per_task": cases_per_operator,
        "metrics_by_operator": metrics,
        "overall": {
            "cases": total_cases,
            "parent_successes": parent_total,
            "candidate_successes": candidate_total,
            "parent_success_rate": parent_total / total_cases,
            "candidate_success_rate": candidate_total / total_cases,
            "success_rate_delta": (
                candidate_total / total_cases - parent_total / total_cases
            ),
        },
        "cases": comparisons,
    }
    write_json(output_dir / "standard_eval.json", result)
    return result
