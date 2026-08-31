#!/usr/bin/env bash
set -euo pipefail
ROOT=/data2/openrsi/experiments/g0_unique_continuous_20260831
BOOT=$ROOT/bootstrap
STATE=$ROOT/l20-state
LOCAL=$ROOT/work
ARCHIVE=$ROOT/archives
LOG=$ROOT/logs/worker-l20-main.log
PID=$ROOT/run/worker-l20-main.pid
CONTAINER=openrsi-g0-continuous-l20
IMAGE=slimerl/slime:nightly-dev-20260706a
SRC=/data2/openrsi/src/OpenRSI-ma1-tracebank-ea4ad81/OpenMLE-ERL/SFT
AIRA=/data2/openrsi/experiments/ma1_recursive_sft_20260829/runtime/aira-evo-2ced6d0
PY=/data2/openrsi/experiments/ma1_recursive_sft_20260829/envs/rollout-min/bin/python
MATERIAL=/data2/openrsi/experiments/official_ma1_headroom_20260830/fresh-task-staging
KEY=/data2/openrsi/experiments/official_ma1_headroom_20260830/run/sandbox-6583.key
mkdir -p "$STATE" "$LOCAL" "$ARCHIVE" "$ROOT/logs" "$ROOT/run"
if docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER"; then
  echo "container_already_running=$CONTAINER"
  exit 0
fi
if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER"; then
  docker rm "$CONTAINER" >/dev/null
fi
curl -fsS --max-time 10 http://127.0.0.1:30010/health >/dev/null
curl -fsS --max-time 10 http://127.0.0.1:6583/health >/dev/null
# Keep host proxy values if present; localhost services must bypass them.
docker run -d \
  --name "$CONTAINER" \
  --network host \
  --restart unless-stopped \
  -v /data2:/data2 \
  -v "$BOOT/zstd_python:/usr/local/bin/zstd:ro" \
  -v "$BOOT/zstd_python:/usr/local/bin/unzstd:ro" \
  -e no_proxy="${no_proxy:-localhost,127.0.0.1},localhost,127.0.0.1" \
  -e NO_PROXY="${NO_PROXY:-localhost,127.0.0.1},localhost,127.0.0.1" \
  "$IMAGE" \
  /bin/bash -lc "cd '$SRC' && exec '$PY' -u '$BOOT/continuous_worker.py' \
    --pool l20 \
    --inventory '$BOOT/l20_inventory.jsonl' \
    --state-root '$STATE' \
    --metadata-root '$BOOT' \
    --materialize-root '$MATERIAL' \
    --local-root '$LOCAL' \
    --archive-root '$ARCHIVE' \
    --source '$SRC' \
    --aira-root '$AIRA' \
    --python '$PY' \
    --llm-url http://127.0.0.1:30010/v1 \
    --model-id Frontis-MA1-35B \
    --sandbox-url http://127.0.0.1:6583 \
    --sandbox-key-file '$KEY' \
    --concurrency 8 \
    --worker-id l20-main \
    >>'$LOG' 2>&1"
docker inspect -f '{{.State.Pid}}' "$CONTAINER" > "$PID"
echo "started container=$CONTAINER host_pid=$(cat "$PID")"
