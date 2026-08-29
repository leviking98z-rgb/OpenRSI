#!/usr/bin/env python3
"""Export one fixed-budget Evo run to JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SFT_ROOT = Path(__file__).resolve().parents[2]
if str(SFT_ROOT) not in sys.path:
    sys.path.insert(0, str(SFT_ROOT))

from tts_search.generational.evo_eval import export_evolutionary_eval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-budget", type=int, default=4)
    args = parser.parse_args()
    result = export_evolutionary_eval(
        args.rollout_root, args.output, args.expected_budget
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
