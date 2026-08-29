"""Build equal-row, exact-token-audited candidate and replay controls."""

from __future__ import annotations

import hashlib
import json
import random
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from tts_search.generational.common import (
    messages,
    read_rows,
    write_json,
    write_sft_pair,
)
from tts_search.generational.eval_data import load_split


def _digest(value: list[dict[str, str]]) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _normalize(
    rows: list[dict[str, Any]], source: str, reserved: set[str]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    kept: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        task_name = str(raw.get("task_name") or "")
        if task_name in reserved:
            drops["reserved_task"] += 1
            continue
        conversation = messages(raw.get("messages"))
        digest = _digest(conversation)
        if digest in seen:
            drops["duplicate"] += 1
            continue
        seen.add(digest)
        kept.append(
            {
                "id": f"{source}::{raw.get('id', index)}",
                "task_name": task_name,
                "messages": conversation,
                "source": source,
                "source_id": str(raw.get("id", index)),
                "message_sha256": digest,
            }
        )
    return kept, drops


def _token_counter(tokenizer_model: Path) -> Callable[[list[dict[str, str]]], int]:
    from tts_search.data_produce.token_filter import (
        count_chat_template_tokens,
        load_tokenizer,
    )

    tokenizer = load_tokenizer(tokenizer_model, local_files_only=True)
    return lambda value: count_chat_template_tokens(value, tokenizer)


def _match_tokens(
    targets: list[dict[str, Any]], pool: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    ranked = sorted((int(row["training_tokens"]), str(row["id"]), row) for row in pool)
    lengths = [item[0] for item in ranked]
    result: list[dict[str, Any]] = []
    for target in sorted(
        targets, key=lambda row: (-int(row["training_tokens"]), str(row["id"]))
    ):
        wanted = int(target["training_tokens"])
        position = bisect_left(lengths, wanted)
        candidates = {max(0, position - 1), min(len(ranked) - 1, position)}
        selected = min(
            candidates,
            key=lambda index: (
                abs(ranked[index][0] - wanted),
                ranked[index][1],
            ),
        )
        result.append(ranked.pop(selected)[2])
        lengths.pop(selected)
    return result


def build_matched_mixtures(
    *,
    anchor_path: Path,
    evo_path: Path,
    split_path: Path,
    tokenizer_model: Path,
    output_dir: Path,
    total_rows: int = 128,
    seed: int = 20260829,
    max_tokens: int = 32768,
    max_token_mismatch_fraction: float = 0.01,
    count_tokens: Callable[[list[dict[str, str]]], int] | None = None,
) -> dict[str, Any]:
    if total_rows <= 0:
        raise ValueError("total_rows must be positive")
    if not tokenizer_model and count_tokens is None:
        raise ValueError("an exact tokenizer model is required")
    splits = load_split(split_path)
    held_out = set(splits["promotion"]) | set(splits["final_test"])
    all_experiment_tasks = set().union(*splits.values())
    anchor, anchor_drops = _normalize(
        read_rows(anchor_path), "anchor", all_experiment_tasks
    )
    evo, evo_drops = _normalize(read_rows(evo_path), "evo", held_out)
    outside_search_train = [
        row for row in evo if row["task_name"] not in set(splits["search_train"])
    ]
    if outside_search_train:
        raise ValueError("Evo input contains rows outside search_train")
    anchor_keys = {row["message_sha256"] for row in anchor}
    evo_without_overlap = [
        row for row in evo if row["message_sha256"] not in anchor_keys
    ]
    evo_drops["anchor_overlap"] += len(evo) - len(evo_without_overlap)
    evo = evo_without_overlap
    if not evo:
        raise ValueError("no unique Evo rows remain")

    counter = count_tokens or _token_counter(tokenizer_model)
    for row in anchor + evo:
        row["training_tokens"] = int(counter(row["messages"]))
    before_anchor, before_evo = len(anchor), len(evo)
    anchor = [row for row in anchor if row["training_tokens"] <= max_tokens]
    evo = [row for row in evo if row["training_tokens"] <= max_tokens]
    anchor_drops["over_token_limit"] += before_anchor - len(anchor)
    evo_drops["over_token_limit"] += before_evo - len(evo)
    if not evo:
        raise ValueError("all Evo rows exceed the token limit")
    if len(evo) >= total_rows:
        raise ValueError(
            f"need room for replay anchors: {len(evo)} Evo rows >= {total_rows}"
        )

    if len(anchor) < total_rows:
        raise ValueError(f"need {total_rows} anchor rows, found {len(anchor)}")
    matched = _match_tokens(evo, anchor)
    matched_ids = {row["id"] for row in matched}
    remaining = [row for row in anchor if row["id"] not in matched_ids]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    shared = remaining[: total_rows - len(evo)]

    pairs = [(row, row) for row in shared] + list(zip(evo, matched, strict=True))
    rng.shuffle(pairs)
    candidate = [left for left, _ in pairs]
    control = [right for _, right in pairs]
    candidate_tokens = sum(row["training_tokens"] for row in candidate)
    control_tokens = sum(row["training_tokens"] for row in control)
    mismatch = abs(candidate_tokens - control_tokens) / max(
        candidate_tokens, control_tokens, 1
    )
    if mismatch > max_token_mismatch_fraction:
        raise ValueError(
            f"candidate/control token mismatch {mismatch:.6f} exceeds "
            f"{max_token_mismatch_fraction:.6f}"
        )

    def manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {key: value for key, value in row.items() if key != "messages"}
            for row in rows
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_artifacts = write_sft_pair(
        output_dir, "candidate_train", candidate, manifest(candidate)
    )
    control_artifacts = write_sft_pair(
        output_dir, "control_train", control, manifest(control)
    )
    summary = {
        "seed": seed,
        "requested_total_rows": total_rows,
        "evo_rows": len(evo),
        "shared_anchor_rows": len(shared),
        "matched_control_rows": len(matched),
        "candidate_rows": len(candidate),
        "control_rows": len(control),
        "candidate_tokens": candidate_tokens,
        "control_tokens": control_tokens,
        "token_mismatch_fraction": mismatch,
        "tokenizer_model": str(tokenizer_model),
        "max_tokens": max_tokens,
        "anchor_drops": dict(anchor_drops),
        "evo_drops": dict(evo_drops),
        "candidate_artifacts": candidate_artifacts,
        "control_artifacts": control_artifacts,
    }
    write_json(output_dir / "mixture_summary.json", summary)
    return summary
