"""Build one candidate SFT mixture from verified Evo rows and replay."""

from __future__ import annotations

import hashlib
import json
import random
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


def build_candidate_mixture(
    *,
    anchor_path: Path,
    evo_path: Path,
    split_path: Path,
    tokenizer_model: Path,
    output_dir: Path,
    total_rows: int = 128,
    seed: int = 20260829,
    max_tokens: int = 32768,
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

    replay_rows = total_rows - len(evo)
    if len(anchor) < replay_rows:
        raise ValueError(f"need {replay_rows} anchor rows, found {len(anchor)}")
    rng = random.Random(seed)
    rng.shuffle(anchor)
    replay = anchor[:replay_rows]
    candidate = replay + evo
    rng.shuffle(candidate)
    candidate_tokens = sum(row["training_tokens"] for row in candidate)

    def manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {key: value for key, value in row.items() if key != "messages"}
            for row in rows
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_artifacts = write_sft_pair(
        output_dir, "candidate_train", candidate, manifest(candidate)
    )
    summary = {
        "seed": seed,
        "requested_total_rows": total_rows,
        "evo_rows": len(evo),
        "replay_anchor_rows": len(replay),
        "candidate_rows": len(candidate),
        "candidate_tokens": candidate_tokens,
        "tokenizer_model": str(tokenizer_model),
        "max_tokens": max_tokens,
        "anchor_drops": dict(anchor_drops),
        "evo_drops": dict(evo_drops),
        "candidate_artifacts": candidate_artifacts,
    }
    write_json(output_dir / "mixture_summary.json", summary)
    return summary
