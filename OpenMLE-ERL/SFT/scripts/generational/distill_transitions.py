#!/usr/bin/env python3
"""Extract strict Improve/Debug transitions from an Evo run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SFT_ROOT = Path(__file__).resolve().parents[2]
if str(SFT_ROOT) not in sys.path:
    sys.path.insert(0, str(SFT_ROOT))

from tts_search.generational.distill import distill_transitions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = distill_transitions(args.rollout_root, args.split, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
