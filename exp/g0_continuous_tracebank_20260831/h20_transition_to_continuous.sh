#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 SHARD" >&2
  exit 2
fi

SHARD=$(printf '%02d' "$((10#$1))")
D16=/root/.cache/openrsi/experiments/g0_d16_64_20260831
ROOT=/root/.cache/openrsi/experiments/g0_unique_continuous_20260831
STAGING=/root/shared/.clusters/.tmp/openrsi-continuous-20260831
STATE=$STAGING/h20-state
META=/root/shared/.clusters/.tmp/openrsi-official-ma1-20260830
SRC=/root/.cache/openrsi/src/OpenRSI-ma1-eval-20f0eb0-SFT
ARCHIVE_ROOT=/root/sync/openrsi/g0_unique_continuous_20260831/raw
LOG=$ROOT/logs/transition-shard-${SHARD}.log
WORKER_LOG=$ROOT/logs/worker-shard-${SHARD}.log
WORKER_PID=$ROOT/worker-shard-${SHARD}.pid

mkdir -p "$ROOT/logs" "$STATE" "$ARCHIVE_ROOT"
exec >>"$LOG" 2>&1
echo "$(date -Is) transition_watcher_start shard=$SHARD"

# Preserve the original 64-task run and its archive before taking new claims.
while true; do
  static_pid=""
  [[ -s "$D16/pids/shard-${SHARD}.rollout.pid" ]] \
    && static_pid=$(cat "$D16/pids/shard-${SHARD}.rollout.pid")
  if [[ -n "$static_pid" ]] && kill -0 "$static_pid" 2>/dev/null; then
    sleep 30
    continue
  fi
  if [[ -s "/root/sync/openrsi/g0_d16_64_20260831/raw/g0-d16-shard-${SHARD}.completion.json" ]]; then
    break
  fi
  sleep 30
done

if [[ -s "$WORKER_PID" ]] && kill -0 "$(cat "$WORKER_PID")" 2>/dev/null; then
  echo "$(date -Is) worker_already_running pid=$(cat "$WORKER_PID")"
  exit 0
fi

curl -fsS --max-time 10 http://127.0.0.1:8000/health >/dev/null
curl -fsS --max-time 10 http://127.0.0.1:6582/health >/dev/null

cd "$SRC"
nohup env \
  no_proxy="${no_proxy:-localhost,127.0.0.1},localhost,127.0.0.1" \
  NO_PROXY="${NO_PROXY:-localhost,127.0.0.1},localhost,127.0.0.1" \
  /root/.cache/openrsi/envs/rollout/bin/python -u \
  "$STAGING/continuous_worker.py" \
  --pool h20 \
  --inventory "$STAGING/h20_inventory.jsonl" \
  --state-root "$STATE" \
  --metadata-root "$META" \
  --materialize-root "$D16/tasks/shard-${SHARD}" \
  --local-root "$ROOT" \
  --archive-root "$ARCHIVE_ROOT" \
  --source "$SRC" \
  --aira-root "$SRC/third_party/aira-evo" \
  --python /root/.cache/openrsi/envs/rollout/bin/python \
  --llm-url http://127.0.0.1:8000/v1 \
  --model-id Frontis-MA1-35B \
  --sandbox-url http://127.0.0.1:6582 \
  --sandbox-key-file "$D16/secrets/sandbox-api-key" \
  --concurrency 8 \
  --worker-id "h20-shard-${SHARD}" \
  >"$WORKER_LOG" 2>&1 </dev/null &
echo $! >"$WORKER_PID"
echo "$(date -Is) continuous_worker_started shard=$SHARD pid=$(cat "$WORKER_PID")"
