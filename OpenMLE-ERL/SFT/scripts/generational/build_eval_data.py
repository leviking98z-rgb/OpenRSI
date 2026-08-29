#!/usr/bin/env python3
"""Build frozen search-train, promotion, and final-test inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SFT_ROOT = Path(__file__).resolve().parents[2]
if str(SFT_ROOT) not in sys.path:
    sys.path.insert(0, str(SFT_ROOT))

from tts_search.generational.eval_data import build_eval_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_eval_data(args.split, args.task_root, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
