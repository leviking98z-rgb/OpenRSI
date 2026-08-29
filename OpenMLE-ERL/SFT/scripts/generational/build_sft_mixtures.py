#!/usr/bin/env python3
"""Build candidate/control SFT data with matched rows and exact tokens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SFT_ROOT = Path(__file__).resolve().parents[2]
if str(SFT_ROOT) not in sys.path:
    sys.path.insert(0, str(SFT_ROOT))

from tts_search.generational.mixture import build_matched_mixtures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--evo", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-rows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-token-mismatch-fraction", type=float, default=0.01)
    args = parser.parse_args()
    result = build_matched_mixtures(
        anchor_path=args.anchor,
        evo_path=args.evo,
        split_path=args.split,
        tokenizer_model=args.tokenizer_model,
        output_dir=args.output_dir,
        total_rows=args.total_rows,
        seed=args.seed,
        max_tokens=args.max_tokens,
        max_token_mismatch_fraction=args.max_token_mismatch_fraction,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
