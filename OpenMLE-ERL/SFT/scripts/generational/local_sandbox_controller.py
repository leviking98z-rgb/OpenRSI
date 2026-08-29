#!/usr/bin/env python3
"""Minimal OpenMLE sandbox API with an untrusted runner and trusted scorer."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import signal
import subprocess
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

TASKS_ROOT = Path(os.environ["OPENRSI_TASKS_ROOT"]).resolve()
JOBS_ROOT = Path(os.environ["OPENRSI_SANDBOX_JOBS_ROOT"]).resolve()
SCORER = Path(
    os.environ.get("OPENRSI_SCORER", Path(__file__).with_name("score_submission.py"))
).resolve()
IMAGE = os.environ.get("OPENRSI_SANDBOX_IMAGE", "openrsi-sandbox:cpu-v1")
API_KEY = os.environ["OPENRSI_SANDBOX_API_KEY"]
WORKERS = int(os.environ.get("OPENRSI_SANDBOX_WORKERS", "4"))
MAX_TIMEOUT = int(os.environ.get("OPENRSI_SANDBOX_MAX_JOB_TIMEOUT", "1200"))
VIRTUAL_PREFIX = os.environ.get(
    "OPENRSI_VIRTUAL_TASK_PREFIX", "/mnt/pubdatasets2/MLTasks/Selected_Dojo/"
)
JOBS: dict[str, dict[str, Any]] = {}
LOCK = threading.Lock()
RUNNING: dict[str, str] = {}
POOL = ThreadPoolExecutor(max_workers=WORKERS)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _save(job: dict[str, Any]) -> None:
    path = Path(job["artifact_dir"]) / "job.json"
    path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _get(job_id: str) -> dict[str, Any] | None:
    with LOCK:
        job = JOBS.get(job_id)
        return json.loads(json.dumps(job)) if job else None


def _update(job_id: str, **values: Any) -> None:
    with LOCK:
        JOBS[job_id].update(values)
        _save(JOBS[job_id])


def _task(data_dir: str) -> Path:
    if data_dir.startswith(VIRTUAL_PREFIX):
        name = data_dir[len(VIRTUAL_PREFIX) :].strip("/")
    else:
        path = Path(data_dir).resolve()
        try:
            name = str(path.relative_to(TASKS_ROOT))
        except ValueError as exc:
            raise ValueError(f"unsupported data_dir: {data_dir}") from exc
    if not name or "/" in name or name in {".", ".."}:
        raise ValueError(f"invalid task name: {name!r}")
    task = (TASKS_ROOT / name).resolve()
    if task.parent != TASKS_ROOT:
        raise ValueError("task path escaped root")
    required = [
        task / "data/public/train.csv",
        task / "data/public/test.csv",
        task / "data/public/sample_submission.csv",
        task / "data/private/test_answer.csv",
        task / "utils/metric.py",
    ]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(f"incomplete task: {missing}")
    return task


def _run(
    command: list[str], timeout: int, container: str
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command, text=True, capture_output=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command, 124, exc.stdout or "", exc.stderr or "timeout"
        )
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container], capture_output=True, check=False
        )


def _docker_base(name: str, cpus: str, memory: str) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--runtime=runc",
        "--network",
        "none",
        "--cpus",
        cpus,
        "--memory",
        memory,
        "--memory-swap",
        memory,
        "--pids-limit",
        "256",
        "--read-only",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=512m",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "CUDA_VISIBLE_DEVICES=",
        "-e",
        "NVIDIA_VISIBLE_DEVICES=void",
    ]


def _execute(job_id: str) -> None:
    started = time.monotonic()
    job = _get(job_id)
    if not job:
        return
    work = Path(job["artifact_dir"]) / "work"
    work.mkdir(parents=True)
    (work / "main.py").write_text(job["request"]["code"], encoding="utf-8")
    task = _task(str(job["request"]["data_dir"]))
    timeout = min(
        MAX_TIMEOUT, max(1, int(job["request"].get("timeout") or MAX_TIMEOUT))
    )
    container = "ma1-sandbox-" + job_id[-20:]
    _update(job_id, status="running", started_at=_now())
    with LOCK:
        RUNNING[job_id] = container
    try:
        command = _docker_base(container, "4", "8g") + [
            "-e",
            "DATA_DIR=/data/public",
            "-e",
            "SANDBOX_DATA_DIR=/data/public",
            "-e",
            "OMP_NUM_THREADS=4",
            "-v",
            f"{task / 'data/public'}:/data/public:ro",
            "-v",
            f"{work}:/workspace:rw",
            "-w",
            "/workspace",
            "--entrypoint",
            "python",
            IMAGE,
            "-B",
            "/workspace/main.py",
        ]
        result = _run(command, timeout, container)
        run_log = (
            "--- SANDBOX STDOUT START ---\n"
            + result.stdout
            + "\n--- SANDBOX STDOUT END ---\n--- SANDBOX STDERR START ---\n"
            + result.stderr
            + "\n--- SANDBOX STDERR END ---"
        )
        if result.returncode != 0:
            payload = {
                "result": "timeout"
                if result.returncode == 124
                else "code_execution_error",
                "score": None,
                "run_log": run_log,
            }
        elif not (work / "submission.csv").is_file():
            payload = {
                "result": "submission_missing",
                "score": None,
                "run_log": run_log,
            }
        else:
            scorer_name = container + "-score"
            score_cmd = _docker_base(scorer_name, "1", "2g") + [
                "-v",
                f"{task}:/task:ro",
                "-v",
                f"{work / 'submission.csv'}:/submission.csv:ro",
                "-v",
                f"{SCORER}:/score_submission.py:ro",
                "--entrypoint",
                "python",
                IMAGE,
                "-B",
                "/score_submission.py",
                "/task",
                "/submission.csv",
            ]
            scored = _run(score_cmd, 120, scorer_name)
            lines = [
                line
                for line in scored.stdout.splitlines()
                if line.startswith("OPENRSI_SCORE_JSON=")
            ]
            if scored.returncode or not lines:
                payload = {
                    "result": "scoring_failed",
                    "score": None,
                    "run_log": run_log + "\n" + scored.stderr,
                }
            else:
                detail = json.loads(lines[-1].split("=", 1)[1])
                payload = {
                    "result": "success",
                    "score": float(detail["score"]),
                    "run_log": run_log,
                    "score_details": detail,
                }
        finished = _now()
        _update(
            job_id,
            status="completed",
            completed_at=finished,
            finished_at=finished,
            running_time=time.monotonic() - started,
            result=payload,
        )
    except Exception as exc:
        finished = _now()
        _update(
            job_id,
            status="failed",
            completed_at=finished,
            finished_at=finished,
            result={
                "result": "sandbox_unconfirmed",
                "score": None,
                "run_log": traceback.format_exc(),
                "error": repr(exc),
            },
        )
    finally:
        with LOCK:
            RUNNING.pop(job_id, None)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{_now()} {fmt % args}", flush=True)

    def _json(self, status: int, value: Any) -> None:
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        if self.headers.get("X-API-Key") == API_KEY:
            return True
        self._json(HTTPStatus.UNAUTHORIZED, {"detail": "invalid API key"})
        return False

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/health"}:
            self._json(200, {"status": "ok", "workers": WORKERS, "network": "none"})
            return
        if not self._authorized():
            return
        if path == "/api/v1/workers/status":
            with LOCK:
                busy = len(RUNNING)
            self._json(
                200, {"gpu": {"total": WORKERS, "idle": WORKERS - busy, "busy": busy}}
            )
            return
        match = re.fullmatch(r"/api/v1/jobs/([^/]+)(/logs)?", path)
        job = _get(match.group(1)) if match else None
        if not job:
            self._json(404, {"detail": "job not found"})
        elif match and match.group(2):
            self._json(
                200,
                {
                    "job_id": job["job_id"],
                    "status": job["status"],
                    "run_log": (job.get("result") or {}).get("run_log", ""),
                },
            )
        else:
            self._json(200, job)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/v1/jobs":
            self._json(404, {"detail": "not found"})
            return
        if not self._authorized():
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= 2 * 1024 * 1024:
                raise ValueError("invalid request size")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request.get("code"), str) or not request["code"].strip():
                raise ValueError("code must be non-empty")
            task = _task(str(request.get("data_dir") or ""))
            job_id = "job-" + uuid.uuid4().hex
            artifact_dir = JOBS_ROOT / dt.date.today().isoformat() / task.name / job_id
            artifact_dir.mkdir(parents=True)
            job = {
                "job_id": job_id,
                "status": "queued",
                "created_at": _now(),
                "started_at": None,
                "completed_at": None,
                "request": request,
                "result": None,
                "artifact_dir": str(artifact_dir),
            }
            with LOCK:
                JOBS[job_id] = job
                _save(job)
            POOL.submit(_execute, job_id)
            self._json(200, {"job_id": job_id, "status": "queued"})
        except Exception as exc:
            self._json(400, {"detail": str(exc), "type": type(exc).__name__})

    def do_DELETE(self) -> None:
        if not self._authorized():
            return
        match = re.fullmatch(r"/api/v1/jobs/([^/]+)", urlparse(self.path).path)
        job = _get(match.group(1)) if match else None
        if not job:
            self._json(404, {"detail": "job not found"})
            return
        with LOCK:
            container = RUNNING.get(job["job_id"])
        if container:
            subprocess.run(
                ["docker", "rm", "-f", container], capture_output=True, check=False
            )
        _update(job["job_id"], status="cancelled", completed_at=_now())
        self._json(200, _get(job["job_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6581)
    args = parser.parse_args()
    for path in (TASKS_ROOT, SCORER):
        if not path.exists():
            raise FileNotFoundError(path)
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["docker", "image", "inspect", IMAGE], check=True, capture_output=True
    )
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        json.dumps({"event": "started", "host": args.host, "port": args.port}),
        flush=True,
    )
    try:
        server.serve_forever(0.25)
    finally:
        server.server_close()
        POOL.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
