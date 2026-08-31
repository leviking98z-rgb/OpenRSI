#!/usr/bin/env python3
"""Continuously collect one D16 AIRA-Evo trace for each uniquely claimed task."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from collections import deque
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from typing import Any
from urllib.parse import quote

import yaml


REVISION = "f56e4b31252a9b81d95fea100098cd49b7290398"
REPOSITORY = "https://huggingface.co/datasets/FrontisAI/OpenMLE-Tasks/resolve"
PARENT_CHECKPOINT = "FrontisAI/Frontis-MA1-35B"
PARENT_REVISION = "79a29e43e7f96b96b06eaf24dcc885ad0318aa01"
SYSTEM_PROMPT = (
    "You are an ML coding agent running in an offline sandbox. "
    "Return exactly one complete executable Python program in one ```python``` "
    "block and no other prose. Use only files under the DATA_DIR environment "
    "variable. Never use network access, package installation, subprocesses, "
    "absolute host paths, private labels, hidden answers, or reference solutions. "
    "The working directory is writable. You must write ./submission.csv in the "
    "exact format described by the task. Be deterministic and CPU-efficient."
)
USER_PROMPT = (
    "Use the public train/test/sample_submission files available through DATA_DIR. "
    "Train a sensible model, produce submission.csv, and verify its rows, columns, "
    "IDs, and missing values before exiting."
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(task_id: str) -> str:
    return task_id.replace("/", "__").replace("\\", "__")


def safe_extract(tar_path: Path, output_dir: Path) -> None:
    with tarfile.open(tar_path, "r:") as archive:
        members = archive.getmembers()
        for member in members:
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts:
                raise RuntimeError(f"unsafe archive path: {member.name}")
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise RuntimeError(f"unsupported archive member: {member.name}")
        for member in members:
            pure = PurePosixPath(member.name)
            relative = Path(*[part for part in pure.parts if part not in ("", ".")])
            if not relative.parts:
                continue
            destination = output_dir / relative
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"cannot read archive member: {member.name}")
                with source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                os.chmod(destination, 0o644)
            else:
                raise RuntimeError(f"unsupported archive member type: {member.name}")


class Materializer:
    def __init__(self, metadata_root: Path, local_root: Path) -> None:
        self.local_root = local_root
        self.archives = local_root / "archives"
        self.tasks = local_root / "tasks"
        self.archives.mkdir(parents=True, exist_ok=True)
        self.tasks.mkdir(parents=True, exist_ok=True)
        self.package_manifest = {
            row["task_id"]: row
            for row in load_jsonl(metadata_root / "package_manifest.jsonl")
        }
        self.task_index = {
            row["task_id"]: row
            for row in load_jsonl(metadata_root / "task_index.jsonl")
        }
        self.checksums_by_prefix: dict[str, list[tuple[str, str]]] = {}
        with (metadata_root / "checksums.sha256").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                digest, relative = line.rstrip("\n").split("  ", 1)
                parts = relative.split("/")
                if len(parts) < 3:
                    continue
                prefix = "/".join(parts[:2]) + "/"
                self.checksums_by_prefix.setdefault(prefix, []).append(
                    (relative[len(prefix) :], digest)
                )

    def prepare(self, task_id: str) -> tuple[Path, dict[str, Any]]:
        record = self.package_manifest[task_id]
        archive_dir = self.archives / safe_name(task_id)
        archive_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        for part in record["parts"]:
            destination = archive_dir / Path(str(part["path"])).name
            expected_sha = str(part["sha256"])
            expected_bytes = int(part["bytes"])
            valid = (
                destination.is_file()
                and destination.stat().st_size == expected_bytes
                and sha256(destination) == expected_sha
            )
            if not valid:
                temporary = destination.with_suffix(destination.suffix + ".part")
                if temporary.is_file() and temporary.stat().st_size > expected_bytes:
                    temporary.unlink()
                artifact = quote(str(record["artifact_path"]), safe="/@")
                part_path = quote(str(part["path"]), safe="/@")
                url = f"{REPOSITORY}/{REVISION}/{artifact}/{part_path}"
                command = [
                    "curl",
                    "-L",
                    "--fail",
                    "--retry",
                    "8",
                    "--connect-timeout",
                    "20",
                    "--max-time",
                    "7200",
                    "--silent",
                    "--show-error",
                ]
                if temporary.is_file() and temporary.stat().st_size:
                    command.extend(["-C", "-"])
                command.extend(["-o", str(temporary), url])
                subprocess.run(command, check=True)
                if (
                    temporary.stat().st_size != expected_bytes
                    or sha256(temporary) != expected_sha
                ):
                    raise RuntimeError(f"archive verification failed: {task_id}")
                temporary.replace(destination)
            downloaded.append(destination)

        output = self.tasks / task_id
        marker = output / ".verified.json"
        if marker.is_file():
            previous = json.loads(marker.read_text(encoding="utf-8"))
            if previous.get("package_sha256") == record.get("archive_sha256"):
                public_link = output / "utils" / "public"
                if not public_link.exists():
                    public_link.symlink_to("../data/public", target_is_directory=True)
                metadata = json.loads(
                    (output / "info/task_metadata.json").read_text(encoding="utf-8")
                )
                return output, metadata

        with tempfile.TemporaryDirectory(prefix="extract-", dir=self.local_root) as raw:
            temporary_dir = Path(raw)
            tar_path = temporary_dir / "task.tar"
            with tar_path.open("wb") as target:
                process = subprocess.Popen(
                    ["unzstd", "-c", *map(str, downloaded)], stdout=target
                )
                if process.wait():
                    raise RuntimeError(f"unzstd failed: {task_id}")
            extracted = temporary_dir / "extracted"
            extracted.mkdir()
            safe_extract(tar_path, extracted)
            required = [
                "RELEASE_METADATA.json",
                "info/task_metadata.json",
                "data/public/train.csv",
                "data/public/test.csv",
                "data/public/sample_submission.csv",
                "data/private/test_answer.csv",
                "utils/metric.py",
            ]
            missing = [item for item in required if not (extracted / item).is_file()]
            if missing:
                raise RuntimeError(f"incomplete task {task_id}: {missing}")
            release = json.loads(
                (extracted / "RELEASE_METADATA.json").read_text(encoding="utf-8")
            )
            metadata = json.loads(
                (extracted / "info/task_metadata.json").read_text(encoding="utf-8")
            )
            if release.get("task_id") != task_id:
                raise RuntimeError(f"release id mismatch: {task_id}")
            if str(metadata.get("cpu_gpu", "")).strip().upper() != "CPU":
                raise RuntimeError(f"not a CPU task: {task_id}")
            if str(metadata.get("task", "")).strip().lower() not in {
                "classification",
                "regression",
            }:
                raise RuntimeError(f"unsupported task type: {task_id}")
            prefix = str(record["artifact_path"]).rstrip("/") + "/"
            verified_files = 0
            for relative, expected_sha in self.checksums_by_prefix.get(prefix, []):
                path = extracted / relative
                if not path.is_file():
                    if relative == "package.json":
                        continue
                    raise RuntimeError(
                        f"checksummed file missing: {task_id}/{relative}"
                    )
                if sha256(path) != expected_sha:
                    raise RuntimeError(
                        f"internal checksum mismatch: {task_id}/{relative}"
                    )
                verified_files += 1
            if verified_files < 2:
                raise RuntimeError(f"too few verified files for {task_id}")
            public_link = extracted / "utils" / "public"
            if public_link.exists() or public_link.is_symlink():
                if public_link.is_dir() and not public_link.is_symlink():
                    shutil.rmtree(public_link)
                else:
                    public_link.unlink()
            public_link.symlink_to("../data/public", target_is_directory=True)
            replacement = self.local_root / f".{safe_name(task_id)}.new"
            shutil.rmtree(replacement, ignore_errors=True)
            extracted.replace(replacement)
            shutil.rmtree(output, ignore_errors=True)
            replacement.replace(output)

        marker_payload = {
            "task_id": task_id,
            "artifact_path": record["artifact_path"],
            "package_sha256": record.get("archive_sha256"),
            "compressed_bytes": record.get("compressed_bytes"),
            "internal_checksums_verified": verified_files,
            "package_index": self.task_index[task_id],
            "verified_at": utc_now(),
        }
        atomic_json(marker, marker_payload)
        return output, metadata


def read_text_if_present(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def write_task_config(
    *,
    task_dir: Path,
    config_dir: Path,
    task_id: str,
    metadata: dict[str, Any],
    task_data_root: Path,
    sandbox_url: str,
) -> Path:
    public = task_dir / "data/public"
    task_description_path = public / "description.txt"
    if not task_description_path.is_file():
        task_description_path = task_dir / "description.txt"
    payload = {
        "uuid": f"ma1::search_train::{task_id}",
        "task_name": task_id,
        "task": str(metadata.get("task") or ""),
        "source": str(metadata.get("source") or "MLE-Smith"),
        "modality": metadata.get("modality"),
        "data_dir": f"/mnt/pubdatasets2/MLTasks/Selected_Dojo/{task_id}",
        "higher_is_better": bool(metadata["higher_is_better"]),
        "theoretical_max": metadata.get("theoretical_max"),
        "theoretical_min": metadata.get("theoretical_min"),
        "leaderboard_max": metadata.get("leaderboard_max"),
        "leaderboard_min": metadata.get("leaderboard_min"),
        "leaderboard_dir": "",
        "submit_dir": "MLTasks/MLE-Bench-Lite-Modified",
        "submit_data_dir_root": str(task_data_root),
        "sandbox": {
            "resource": "cpu",
            "base_url": sandbox_url,
            "job_timeout": 1200,
            "wait_timeout": 1800,
            "poll_interval": 5,
            "use_score2reward": True,
        },
        "task_description": read_text_if_present(task_description_path),
        "data_description": read_text_if_present(
            task_dir / "info/data_description.txt"
        ),
        "public_system_prompt": SYSTEM_PROMPT,
        "public_user_prompt": USER_PROMPT,
    }
    output = config_dir / task_id
    output.mkdir(parents=True, exist_ok=True)
    config = output / "config.yaml"
    config.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    atomic_json(output / "task_metadata.json", payload)
    return output


def write_runner_config(
    *,
    path: Path,
    output_root: Path,
    config_root: Path,
    task_id: str,
    llm_url: str,
    model_id: str,
) -> None:
    payload = {
        "experiment_name": "g0_unique_continuous_d16",
        "output_dir": str(output_root),
        "seed": 20260831,
        "time_budget": 14400,
        "model_plus_sandbox_time_budget": 18000,
        "n_samples_per_task": 1,
        "accepted_target": 16,
        "rejection_policy": "accept_scored",
        "rejection_target": 16,
        "rejection_score_threshold": None,
        "rejection_reward_threshold": None,
        "rejection_reference_scores_path": None,
        "rejection_accepted_medals": [],
        "rejection_apply_baseline_filters": False,
        "rejection_baseline_token_limit": 32768,
        "rejection_baseline_tokenizer_model": None,
        "rejection_baseline_relative_gap_limit": 0.12,
        "rejection_mixed_leaderboard_target": 16,
        "rejection_mixed_no_leaderboard_target": 16,
        "candidates_per_step": 5,
        "task_concurrency": 8,
        "llm_concurrency": 8,
        "sandbox_concurrency": 8,
        "strict_resume": False,
        "progress_history_interval_seconds": 600.0,
        "task_root": str(config_root),
        "task_list": [task_id],
        "leaderboard_dir": "",
        "llm": {
            "api": "litellm",
            "model_id": model_id,
            "base_url": llm_url,
            "generation_kwargs": {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 12288,
                "extra_body": {
                    "top_k": 20,
                    "min_p": 0.0,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            },
            "use_azure_client": False,
            "provider": "selfhosted",
        },
        "solver": {
            "step_limit": 16,
            "num_islands": 1,
            "max_island_size": 500,
            "crossover_prob": 0.5,
            "migration_prob": 0.0,
            "initial_temp": 1.0,
            "final_temp": 1.0,
            "num_generations_till_migration": 999,
            "num_generations_till_crossover": 2,
            "num_generations": 100,
            "individuals_per_generation": 5,
            "max_debug_depth": 10,
            "max_debug_time": 1200,
            "data_preview": True,
            "use_test_score": False,
            "use_complexity": False,
            "execution_timeout": 1200,
            "export_search_results": True,
            "available_packages": [
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "torch",
                "torchvision",
            ],
        },
        "interpreter": {"use_symlinks": True},
        "logger": {"use_console": True, "use_json": True, "use_wandb": False},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def run_command(command: list[str], *, env: dict[str, str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab", buffering=0) as handle:
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        return process.wait()


class Dispatcher:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.inventory = load_jsonl(args.inventory)
        self.state = args.state_root
        self.claims = self.state / "claims"
        self.done = self.state / "done"
        self.failed = self.state / "failed"
        self.heartbeats = self.state / "heartbeats"
        for path in (self.claims, self.done, self.failed, self.heartbeats):
            path.mkdir(parents=True, exist_ok=True)
        self.local_root = args.local_root
        self.output_root = self.local_root / "rollouts"
        self.config_root = self.local_root / "task-configs"
        self.runner_configs = self.local_root / "runner-configs"
        self.logs = self.local_root / "logs"
        self.archive_root = args.archive_root
        for path in (
            self.output_root,
            self.config_root,
            self.runner_configs,
            self.logs,
            self.archive_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.materializer = Materializer(args.metadata_root, args.materialize_root)
        self.lock = threading.Lock()
        self.claim_lock = threading.Lock()
        self.worker_id = (
            args.worker_id
            or f"{socket.gethostname()}-{os.getpid()}-{int(time.time())}"
        )
        self.started_at = utc_now()
        self.completed_local = 0
        self.failed_local = 0
        inventory_by_id = {
            str(record["task_id"]): record for record in self.inventory
        }
        resumable: list[dict[str, Any]] = []
        for claim_path in sorted(self.claims.glob("*/claim.json")):
            try:
                claim = json.loads(claim_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            task_id = str(claim.get("task_id") or "")
            name = safe_name(task_id)
            if (
                claim.get("worker_id") == self.worker_id
                and task_id in inventory_by_id
                and not (self.done / f"{name}.json").exists()
                and not (self.failed / f"{name}.json").exists()
            ):
                resumable.append(inventory_by_id[task_id])
        self.resume_records = deque(resumable)

    def claim_next(self, slot: int) -> dict[str, Any] | None:
        with self.claim_lock:
            if self.resume_records:
                return self.resume_records.popleft()
            for record in self.inventory:
                task_id = str(record["task_id"])
                name = safe_name(task_id)
                if (
                    (self.done / f"{name}.json").exists()
                    or (self.failed / f"{name}.json").exists()
                ):
                    continue
                claim_dir = self.claims / name
                try:
                    claim_dir.mkdir()
                except FileExistsError:
                    continue
                claim = {
                    "schema_version": 1,
                    "task_id": task_id,
                    "worker_id": self.worker_id,
                    "slot": slot,
                    "claimed_at": utc_now(),
                    "pool": self.args.pool,
                    "inventory_record": record,
                }
                atomic_json(claim_dir / "claim.json", claim)
                return record
        return None

    def heartbeat(self) -> None:
        while True:
            payload = {
                "schema_version": 1,
                "worker_id": self.worker_id,
                "pool": self.args.pool,
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "started_at": self.started_at,
                "updated_at": utc_now(),
                "completed_local": self.completed_local,
                "failed_local": self.failed_local,
            }
            atomic_json(self.heartbeats / f"{safe_name(self.worker_id)}.json", payload)
            time.sleep(30)

    def archive_task(
        self,
        *,
        task_id: str,
        output_dir: Path,
        task_config_dir: Path,
        runner_config: Path,
        manifest: dict[str, Any],
    ) -> tuple[Path, str]:
        archive = self.archive_root / f"{safe_name(task_id)}.tar.zst"
        temporary = archive.with_suffix(archive.suffix + f".partial.{os.getpid()}")
        manifest_path = output_dir / "continuous_attempt.json"
        atomic_json(manifest_path, manifest)
        command = (
            f"tar -cf - "
            f"-C {shlex_quote(str(self.local_root))} "
            f"{shlex_quote(str(output_dir.relative_to(self.local_root)))} "
            f"{shlex_quote(str(task_config_dir.relative_to(self.local_root)))} "
            f"{shlex_quote(str(runner_config.relative_to(self.local_root)))} "
            f"| zstd -T0 -3 -q -o {shlex_quote(str(temporary))}"
        )
        subprocess.run(["bash", "-lc", command], check=True)
        os.replace(temporary, archive)
        digest = sha256(archive)
        (archive.with_suffix(archive.suffix + ".sha256")).write_text(
            f"{digest}  {archive.name}\n", encoding="utf-8"
        )
        return archive, digest

    def run_one(self, slot: int, record: dict[str, Any]) -> None:
        task_id = str(record["task_id"])
        name = safe_name(task_id)
        claim = json.loads((self.claims / name / "claim.json").read_text())
        started_at = utc_now()
        output_dir = self.output_root / "program_ep_0" / task_id
        log_path = self.logs / f"{name}.log"
        runner_config = self.runner_configs / f"{name}.yaml"
        try:
            if (self.done / f"{name}.json").exists() or (
                self.failed / f"{name}.json"
            ).exists():
                return
            task_dir, metadata = self.materializer.prepare(task_id)
            task_config_dir = write_task_config(
                task_dir=task_dir,
                config_dir=self.config_root,
                task_id=task_id,
                metadata=metadata,
                task_data_root=self.args.materialize_root / "tasks",
                sandbox_url=self.args.sandbox_url,
            )
            write_runner_config(
                path=runner_config,
                output_root=self.output_root,
                config_root=self.config_root,
                task_id=task_id,
                llm_url=self.args.llm_url,
                model_id=self.args.model_id,
            )
            env = os.environ.copy()
            env.update(self.args.shared_concurrency_env)
            env.update(
                {
                    "OPENMLE_TASK_DATA_ROOT": str(
                        self.args.materialize_root / "tasks"
                    ),
                    "OPENMLE_SEED": "20260831",
                    "OPENMLE_LLM_CONCURRENCY": str(self.args.concurrency),
                    "OPENMLE_TASK_CONCURRENCY": str(self.args.concurrency),
                    "OPENMLE_SANDBOX_CONCURRENCY": str(self.args.concurrency),
                    "OPENMLE_CANDIDATES_PER_STEP": "5",
                    "OPENMLE_MAX_TOKENS": "12288",
                    "OPENMLE_LLM_TIMEOUT": "900",
                    "OPENMLE_LLM_RETRIES": "1",
                    "OPENMLE_TASK_TIME_BUDGET": "14400",
                    "OPENMLE_TOTAL_TIME_BUDGET": "18000",
                    "OPENMLE_SANDBOX_JOB_TIMEOUT": "1200",
                    "OPENMLE_SANDBOX_WAIT_TIMEOUT": "1800",
                    "SGLANG_BASE_URL": self.args.llm_url,
                    "OPENMLE_MODEL_ID": self.args.model_id,
                    "PRIMARY_KEY": self.args.sandbox_key,
                    "SANDBOX_API_KEY": self.args.sandbox_key,
                    "SANDBOX_CPU_API_KEY": self.args.sandbox_key,
                    "SANDBOX_GPU_API_KEY": self.args.sandbox_key,
                    "AIRA_LITELLM_STREAM": "1",
                    "AIRA_LITELLM_TIMEOUT": "900",
                    "AIRA_LITELLM_NUM_RETRIES": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            python_path = [
                str(self.args.source),
                str(self.args.aira_root / "src"),
                str(self.args.aira_root / "examples/mle_bench"),
            ]
            if env.get("PYTHONPATH"):
                python_path.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = os.pathsep.join(python_path)
            command = [
                str(self.args.python),
                str(
                    self.args.aira_root
                    / "examples/mle_bench/single_task_runner.py"
                ),
                "--task-dir",
                str(task_config_dir),
                "--output-dir",
                str(output_dir),
                "--runner-config",
                str(runner_config),
                "--sample-index",
                "0",
            ]
            return_code = run_command(command, env=env, log=log_path)
            stat_path = output_dir / "stat.json"
            stat = {}
            if stat_path.is_file():
                stat = json.loads(stat_path.read_text(encoding="utf-8"))
            success = return_code == 0 and stat_path.is_file()
            manifest = {
                **claim,
                "started_at": started_at,
                "finished_at": utc_now(),
                "return_code": return_code,
                "success": success,
                "parent_checkpoint": PARENT_CHECKPOINT,
                "parent_revision": PARENT_REVISION,
                "sample_index": 0,
                "max_operator_executions": 16,
                "operators": ["draft", "debug", "improve", "crossover"],
                "individuals_per_generation": 5,
                "num_generations": 100,
                "crossover_prob": 0.5,
                "num_generations_till_crossover": 2,
                "max_debug_depth": 10,
                "max_output_tokens": 12288,
                "stat": stat,
            }
            archive, archive_sha = self.archive_task(
                task_id=task_id,
                output_dir=output_dir,
                task_config_dir=task_config_dir,
                runner_config=runner_config,
                manifest=manifest,
            )
            manifest["archive"] = str(archive)
            manifest["archive_sha256"] = archive_sha
            if success:
                atomic_json(self.done / f"{name}.json", manifest)
                with self.lock:
                    self.completed_local += 1
            else:
                atomic_json(self.failed / f"{name}.json", manifest)
                with self.lock:
                    self.failed_local += 1
        except BaseException as error:
            failure = {
                **claim,
                "started_at": started_at,
                "finished_at": utc_now(),
                "success": False,
                "error": repr(error),
                "parent_checkpoint": PARENT_CHECKPOINT,
                "parent_revision": PARENT_REVISION,
                "sample_index": 0,
            }
            atomic_json(self.failed / f"{name}.json", failure)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\nCONTINUOUS_WORKER_ERROR {utc_now()} {error!r}\n")
            with self.lock:
                self.failed_local += 1

    def slot_loop(self, slot: int) -> None:
        while True:
            record = self.claim_next(slot)
            if record is None:
                time.sleep(self.args.empty_sleep)
                continue
            self.run_one(slot, record)

    def run(self) -> None:
        atomic_json(
            self.local_root / "worker_manifest.json",
            {
                "schema_version": 1,
                "worker_id": self.worker_id,
                "pool": self.args.pool,
                "started_at": self.started_at,
                "inventory": str(self.args.inventory),
                "inventory_sha256": sha256(self.args.inventory),
                "concurrency": self.args.concurrency,
                "parent_checkpoint": PARENT_CHECKPOINT,
                "parent_revision": PARENT_REVISION,
            },
        )
        threading.Thread(target=self.heartbeat, daemon=True).start()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.args.concurrency
        ) as executor:
            futures = [
                executor.submit(self.slot_loop, slot)
                for slot in range(self.args.concurrency)
            ]
            for future in futures:
                future.result()


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True, choices=("h20", "l20"))
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--materialize-root", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--aira-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--llm-url", required=True)
    parser.add_argument("--model-id", default="Frontis-MA1-35B")
    parser.add_argument("--sandbox-url", required=True)
    parser.add_argument("--sandbox-key-file", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--worker-id")
    parser.add_argument("--empty-sleep", type=float, default=60.0)
    args = parser.parse_args()
    args.sandbox_key = args.sandbox_key_file.read_text(encoding="utf-8").strip()
    if not args.sandbox_key:
        parser.error("sandbox key file is empty")
    for path in (args.inventory, args.metadata_root, args.source, args.aira_root):
        if not path.exists():
            parser.error(f"path does not exist: {path}")
    return args


def main() -> None:
    args = parse_args()
    args.local_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.local_root / "worker.lock"
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another continuous worker owns {lock_path}")
    lock_handle.write(f"pid={os.getpid()}\nstarted_at={utc_now()}\n")
    lock_handle.flush()
    sys.path.insert(0, str(args.source))
    from tts_search.airaevo_concurrency import SharedConcurrencyServer

    server = SharedConcurrencyServer(
        llm_concurrency=args.concurrency,
        sandbox_concurrency=args.concurrency,
    )
    args.shared_concurrency_env = server.env
    try:
        Dispatcher(args).run()
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
