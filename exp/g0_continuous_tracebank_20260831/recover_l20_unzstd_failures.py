#!/usr/bin/env python3
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
import shutil

root = Path("/data2/openrsi/experiments/g0_unique_continuous_20260831")
state = root / "l20-state"
failed_root = state / "failed"
stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
stash = state / f"infra-failures-unzstd-{stamp}"
stash.mkdir(parents=True, exist_ok=False)
retried = []
skipped = []
for failed in sorted(failed_root.glob("*.json")):
    try:
        record = json.loads(failed.read_text(encoding="utf-8"))
    except Exception as error:
        skipped.append([failed.name, f"bad_json:{error!r}"])
        continue
    if record.get("error") != "FileNotFoundError(2, 'No such file or directory')":
        skipped.append([failed.name, "different_error"])
        continue
    task = str(record.get("task_id") or failed.stem)
    output = root / "work" / "rollouts" / "program_ep_0" / task
    if output.exists() and any(output.rglob("search_events.jsonl")):
        skipped.append([task, "search_started"])
        continue
    shutil.copy2(failed, stash / failed.name)
    failed.unlink()
    claim = state / "claims" / failed.stem
    if claim.exists():
        shutil.rmtree(claim)
    retried.append(task)
manifest = {
    "schema_version": 1,
    "reason": "worker container lacked unzstd; materialization failed before search",
    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "released_for_same-seed_retry": retried,
    "skipped": skipped,
}
(stash / "retry_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({"released": len(retried), "skipped": len(skipped), "stash": str(stash)}))
