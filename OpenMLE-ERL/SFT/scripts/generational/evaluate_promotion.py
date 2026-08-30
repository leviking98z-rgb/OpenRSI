#!/usr/bin/env python3
"""Apply the candidate-versus-parent promotion gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SFT_ROOT = Path(__file__).resolve().parents[2]
if str(SFT_ROOT) not in sys.path:
    sys.path.insert(0, str(SFT_ROOT))

from tts_search.generational.promotion import evaluate_promotion  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--tie-tolerance", type=float, default=0.0)
    parser.add_argument("--min-valid-rate-delta", type=float, default=-0.05)
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    result = evaluate_promotion(
        parent_paths=args.parent,
        candidate_paths=args.candidate,
        output_dir=args.output_dir,
        budget=args.budget,
        tie_tolerance=args.tie_tolerance,
        min_valid_rate_delta=args.min_valid_rate_delta,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
