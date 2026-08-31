#!/usr/bin/env python3
"""Run the standardized operator or end-to-end promotion gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SFT_ROOT = Path(__file__).resolve().parents[2]
if str(SFT_ROOT) not in sys.path:
    sys.path.insert(0, str(SFT_ROOT))

from tts_search.generational.standard_eval import (  # noqa: E402
    evaluate_e2e,
    evaluate_operator,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    operator = subparsers.add_parser("operator")
    operator.add_argument("--cases", type=Path, required=True)
    operator.add_argument("--parent", type=Path, nargs="+", required=True)
    operator.add_argument("--candidate", type=Path, nargs="+", required=True)
    operator.add_argument("--num-tasks", type=int, required=True)
    operator.add_argument(
        "--operators", nargs="+", default=["debug", "improve"]
    )
    operator.add_argument("--output-dir", type=Path, required=True)

    e2e = subparsers.add_parser("e2e")
    e2e.add_argument("--parent", type=Path, nargs="+", required=True)
    e2e.add_argument("--candidate", type=Path, nargs="+", required=True)
    e2e.add_argument("--task-manifest", type=Path, required=True)
    e2e.add_argument("--num-tasks", type=int, required=True)
    e2e.add_argument("--budget", type=int, required=True)
    e2e.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.mode == "operator":
        result = evaluate_operator(
            case_path=args.cases,
            parent_paths=args.parent,
            candidate_paths=args.candidate,
            num_tasks=args.num_tasks,
            operators=args.operators,
            output_dir=args.output_dir,
        )
    else:
        result = evaluate_e2e(
            parent_paths=args.parent,
            candidate_paths=args.candidate,
            task_manifest=args.task_manifest,
            num_tasks=args.num_tasks,
            budget=args.budget,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
