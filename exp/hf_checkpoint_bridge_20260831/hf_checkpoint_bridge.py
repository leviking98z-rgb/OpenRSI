#!/usr/bin/env python3
"""Resumable private-Hugging-Face checkpoint transfer for OpenRSI.

Use one private Hugging Face repository per HF-format checkpoint. Uploads use
`upload_large_folder`, whose local metadata makes an interrupted upload
resumable by rerunning the same command. Downloads use `snapshot_download`.
Both directions verify a SHA-256 manifest unless explicitly disabled.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


MANIFEST_NAME = "OPENRSI_HF_BRIDGE_MANIFEST.json"
DEFAULT_IGNORE = [
    ".cache/**",
    "**/.cache/**",
    "**/*.lock",
    "**/*.tmp",
    "**/__pycache__/**",
]


def read_token(token_file: str | None) -> str:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if token:
        return token.strip()

    candidates: list[Path] = []
    if token_file:
        candidates.append(Path(token_file).expanduser())
    if os.environ.get("HF_TOKEN_FILE"):
        candidates.append(Path(os.environ["HF_TOKEN_FILE"]).expanduser())
    candidates.extend(
        [
            Path.home() / ".cache/openrsi/.secrets/hf_token",
            Path("/data2/openrsi/.secrets/hf_token"),
        ]
    )
    for path in candidates:
        if path.is_file():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
    raise SystemExit(
        "No Hugging Face token found. Set HF_TOKEN/HF_TOKEN_FILE or pass "
        "--token-file. Do not put the token in this repository."
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(root: Path) -> list[Path]:
    ignored_parts = {".cache", "__pycache__"}
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        relative = path.relative_to(root)
        if any(part in ignored_parts for part in relative.parts):
            continue
        if path.suffix in {".lock", ".tmp"}:
            continue
        files.append(path)
    return sorted(files)


def make_manifest(
    root: Path, repo_id: str, revision: str, hash_workers: int
) -> dict[str, Any]:
    paths = source_files(root)
    workers = max(1, min(hash_workers, len(paths) or 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        hashes = list(pool.map(sha256_file, paths))
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": digest,
        }
        for path, digest in zip(paths, hashes, strict=True)
    ]
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": repo_id,
        "revision": revision,
        "root_name": root.name,
        "file_count": len(entries),
        "total_bytes": sum(entry["size"] for entry in entries),
        "files": entries,
    }


def verify_manifest(root: Path, manifest: dict[str, Any], workers: int) -> None:
    entries = manifest.get("files", [])

    def verify(entry: dict[str, Any]) -> str | None:
        relative = entry["path"]
        path = root / relative
        if not path.is_file():
            return f"missing: {relative}"
        actual_size = path.stat().st_size
        if actual_size != entry["size"]:
            return (
                f"size mismatch: {relative}: "
                f"{actual_size} != {entry['size']}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != entry["sha256"]:
            return f"sha256 mismatch: {relative}"
        return None

    pool_size = max(1, min(workers, len(entries) or 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as pool:
        errors = [error for error in pool.map(verify, entries) if error]
    if errors:
        print("\n".join(errors[:20]), file=sys.stderr)
        raise SystemExit(f"Checkpoint verification failed ({len(errors)} errors)")
    print(
        json.dumps(
            {
                "verified": True,
                "file_count": len(entries),
                "total_bytes": manifest.get("total_bytes"),
                "root": str(root),
            },
            sort_keys=True,
        )
    )


def auth_command(args: argparse.Namespace) -> None:
    api = HfApi(token=read_token(args.token_file))
    identity = api.whoami()
    result: dict[str, Any] = {
        "authenticated": True,
        "user": identity.get("name") or identity.get("fullname"),
    }
    if args.repo_id:
        info = api.repo_info(repo_id=args.repo_id, repo_type="model")
        result.update({"repo_id": args.repo_id, "private": info.private})
    print(json.dumps(result, sort_keys=True))


def upload_command(args: argparse.Namespace) -> None:
    source = Path(args.source).expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"Source directory does not exist: {source}")
    token = read_token(args.token_file)
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=True,
        exist_ok=True,
    )

    print("Building SHA-256 manifest...", flush=True)
    manifest = make_manifest(
        source,
        repo_id=args.repo_id,
        revision=args.revision,
        hash_workers=args.hash_workers,
    )
    print(
        json.dumps(
            {
                "file_count": manifest["file_count"],
                "total_bytes": manifest["total_bytes"],
                "repo_id": args.repo_id,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    # With hf_xet installed, this streams and chunks large files. Rerunning the
    # same command skips committed files and reuses already uploaded chunks.
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=source,
        repo_type="model",
        revision=args.revision,
        commit_message="Upload OpenRSI HF checkpoint",
        ignore_patterns=DEFAULT_IGNORE,
    )
    commit = api.upload_file(
        path_or_fileobj=json.dumps(manifest, indent=2, sort_keys=True).encode(),
        path_in_repo=MANIFEST_NAME,
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        commit_message="Add OpenRSI checkpoint integrity manifest",
    )
    print(
        json.dumps(
            {
                "uploaded": True,
                "repo_id": args.repo_id,
                "revision": args.revision,
                "commit_revision": commit.oid,
                "file_count": manifest["file_count"],
                "total_bytes": manifest["total_bytes"],
            },
            sort_keys=True,
        )
    )


def download_command(args: argparse.Namespace) -> None:
    destination = Path(args.destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    token = read_token(args.token_file)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=destination,
        token=token,
        max_workers=args.download_workers,
    )
    if args.skip_verify:
        print(
            json.dumps(
                {
                    "downloaded": True,
                    "verified": False,
                    "repo_id": args.repo_id,
                    "destination": str(destination),
                },
                sort_keys=True,
            )
        )
        return
    manifest_path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=MANIFEST_NAME,
            repo_type="model",
            revision=args.revision,
            local_dir=destination,
            token=token,
        )
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_manifest(destination, manifest, workers=args.hash_workers)


def verify_command(args: argparse.Namespace) -> None:
    root = Path(args.directory).expanduser().resolve()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_manifest(root, manifest, workers=args.hash_workers)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument(
        "--token-file",
        help="Private token file; HF_TOKEN and HF_TOKEN_FILE are also supported",
    )
    commands = root.add_subparsers(dest="command", required=True)

    auth = commands.add_parser("auth", help="Validate token and optional repo access")
    auth.add_argument("--repo-id")
    auth.set_defaults(func=auth_command)

    upload = commands.add_parser(
        "upload", help="Create a private repo and resumably upload one checkpoint"
    )
    upload.add_argument("source")
    upload.add_argument("--repo-id", required=True)
    upload.add_argument("--revision", default="main")
    upload.add_argument("--hash-workers", type=int, default=2)
    upload.set_defaults(func=upload_command)

    download = commands.add_parser(
        "download", help="Download a checkpoint snapshot and verify its manifest"
    )
    download.add_argument("destination")
    download.add_argument("--repo-id", required=True)
    download.add_argument("--revision", default="main")
    download.add_argument("--download-workers", type=int, default=8)
    download.add_argument("--hash-workers", type=int, default=2)
    download.add_argument("--skip-verify", action="store_true")
    download.set_defaults(func=download_command)

    verify = commands.add_parser("verify", help="Re-verify an existing download")
    verify.add_argument("directory")
    verify.add_argument("--hash-workers", type=int, default=2)
    verify.set_defaults(func=verify_command)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
