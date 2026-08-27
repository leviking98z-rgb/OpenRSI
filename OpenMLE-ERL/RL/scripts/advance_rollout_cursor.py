#!/usr/bin/env python3
"""Advance a slime global-dataset cursor after a completed, unsnapshotted rollout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def advance_cursor(
    state: dict,
    *,
    dataset_size: int,
    rollout_batch_size: int,
    samples_per_prompt: int,
) -> dict:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if not 0 < rollout_batch_size <= dataset_size:
        raise ValueError("rollout_batch_size must be in [1, dataset_size]")
    if samples_per_prompt <= 0:
        raise ValueError("samples_per_prompt must be positive")

    result = dict(state)
    sample_offset = int(result.get("sample_offset", 0))
    epoch_id = int(result.get("epoch_id", 0))
    sample_group_index = int(result.get("sample_group_index", 0))
    sample_index = int(result.get("sample_index", 0))
    if not 0 <= sample_offset <= dataset_size:
        raise ValueError(f"sample_offset {sample_offset} is outside [0, {dataset_size}]")

    if sample_offset + rollout_batch_size <= dataset_size:
        sample_offset += rollout_batch_size
    else:
        remaining = rollout_batch_size - (dataset_size - sample_offset)
        epoch_id += 1
        sample_offset = remaining

    result.update(
        sample_offset=sample_offset,
        epoch_id=epoch_id,
        sample_group_index=sample_group_index + rollout_batch_size,
        sample_index=sample_index + rollout_batch_size * samples_per_prompt,
    )
    result.setdefault("metadata", {})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-size", required=True, type=int)
    parser.add_argument("--rollout-batch-size", required=True, type=int)
    parser.add_argument("--samples-per-prompt", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing cursor: {args.output}")

    state = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"expected a dict cursor, got {type(state).__name__}")
    advanced = advance_cursor(
        state,
        dataset_size=args.dataset_size,
        rollout_batch_size=args.rollout_batch_size,
        samples_per_prompt=args.samples_per_prompt,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    try:
        torch.save(advanced, temporary)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    print(json.dumps(advanced, sort_keys=True))


if __name__ == "__main__":
    main()
