"""Small I/O helpers shared by the MA1 SFT-only experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def messages(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError("messages must be a list")
    result = []
    for item in value:
        item = dict(item)
        role = str(item.get("role") or "").strip()
        if not role:
            raise ValueError("message role is empty")
        result.append({"role": role, "content": str(item.get("content") or "")})
    return result


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    raise ValueError(f"expected .jsonl or .parquet: {path}")


def write_records(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        write_jsonl(path, rows)
        return
    if path.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(rows), path)
        return
    raise ValueError(f"expected .jsonl or .parquet: {path}")


def write_sft_pair(
    output_dir: Path,
    stem: str,
    rows: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> dict[str, str]:
    training_rows = [
        {
            "id": str(row["id"]),
            "task_name": str(row.get("task_name") or ""),
            "messages": messages(row["messages"]),
        }
        for row in rows
    ]
    jsonl = output_dir / f"{stem}.jsonl"
    parquet = output_dir / f"{stem}.parquet"
    manifest_path = output_dir / f"{stem}.manifest.jsonl"
    write_records(jsonl, training_rows)
    write_records(parquet, training_rows)
    write_jsonl(manifest_path, manifest)
    return {
        "jsonl": str(jsonl),
        "jsonl_sha256": sha256_file(jsonl),
        "parquet": str(parquet),
        "parquet_sha256": sha256_file(parquet),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
