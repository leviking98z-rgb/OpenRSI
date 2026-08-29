#!/usr/bin/env python3
"""Trusted scorer for an installed OpenMLE task package."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(module: Any) -> Any:
    classes = []
    for _, value in inspect.getmembers(module, inspect.isclass):
        if value.__module__ != module.__name__:
            continue
        if callable(getattr(value, "evaluate", None)) and callable(
            getattr(value, "validate_submission", None)
        ):
            classes.append(value)
    if len(classes) != 1:
        raise RuntimeError(
            f"expected one metric class, found {[c.__name__ for c in classes]}"
        )
    return classes[0]()


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: score_submission.py TASK_ROOT SUBMISSION_CSV")
    task_root = Path(sys.argv[1]).resolve()
    submission_path = Path(sys.argv[2]).resolve()
    metric_path = task_root / "utils" / "metric.py"
    answer_path = task_root / "data" / "private" / "test_answer.csv"
    for path in (metric_path, answer_path, submission_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location("openmle_task_metric", metric_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {metric_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metric = _metric(module)
    ground_truth = pd.read_csv(answer_path)
    submission = pd.read_csv(submission_path)
    validation = metric.validate_submission(submission, ground_truth)
    score = float(metric.evaluate(ground_truth, submission))
    if not math.isfinite(score):
        raise ValueError(f"non-finite score: {score}")
    result = {
        "status": "success",
        "metric_class": type(metric).__name__,
        "validation": validation,
        "score": score,
        "rows": len(submission),
        "submission_sha256": _sha256(submission_path),
        "metric_sha256": _sha256(metric_path),
        "answer_sha256": _sha256(answer_path),
    }
    print("OPENRSI_SCORE_JSON=" + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
