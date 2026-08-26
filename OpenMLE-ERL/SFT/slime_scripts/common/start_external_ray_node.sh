#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ( "$1" != "head" && "$1" != "worker" ) ]]; then
  echo "Usage: $0 head|worker" >&2
  exit 2
fi

role="$1"
MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to the Ray head IP.}"
NODE_IP="${NODE_IP:?Set NODE_IP to the local routable IP.}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-8}"
RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
RAY_CLIENT_SERVER_PORT="${RAY_CLIENT_SERVER_PORT:-10001}"
DASHBOARD_AGENT_LISTEN_PORT="${DASHBOARD_AGENT_LISTEN_PORT:-52365}"
MIN_WORKER_PORT="${MIN_WORKER_PORT:-10002}"
MAX_WORKER_PORT="${MAX_WORKER_PORT:-19999}"
RUNTIME_DIR="${RUNTIME_DIR:-/tmp/openrsi-sft-ray}"
RAY_TMPDIR="${RAY_TMPDIR:-${RUNTIME_DIR}/ray}"

if [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
  echo "[ERROR] MASTER_ADDR must be routable from every Ray node." >&2
  exit 2
fi
if [[ "${role}" == "head" && "${NODE_IP}" != "${MASTER_ADDR}" ]]; then
  echo "[ERROR] The head NODE_IP must equal MASTER_ADDR." >&2
  exit 2
fi

mkdir -p "${RAY_TMPDIR}"
export RAY_TMPDIR

python3 - "${RAY_PORT}" "${DASHBOARD_PORT}" "${RAY_CLIENT_SERVER_PORT}" \
  "${DASHBOARD_AGENT_LISTEN_PORT}" "${MIN_WORKER_PORT}" "${MAX_WORKER_PORT}" <<'PY'
import sys
from ray._private.parameter import RayParams

values = [int(value) for value in sys.argv[1:]]
params = RayParams(
    gcs_server_port=values[0],
    dashboard_port=values[1],
    ray_client_server_port=values[2],
    dashboard_agent_listen_port=values[3],
    min_worker_port=values[4],
    max_worker_port=values[5],
)
params.update_pre_selected_port()
PY

common_args=(
  --node-ip-address "${NODE_IP}"
  --num-gpus "${RAY_NUM_GPUS}"
  --disable-usage-stats
  --dashboard-agent-listen-port "${DASHBOARD_AGENT_LISTEN_PORT}"
  --min-worker-port "${MIN_WORKER_PORT}"
  --max-worker-port "${MAX_WORKER_PORT}"
  --temp-dir "${RAY_TMPDIR}"
  --block
)

if [[ "${role}" == "head" ]]; then
  exec ray start --head \
    --port "${RAY_PORT}" \
    --dashboard-host=127.0.0.1 \
    --dashboard-port "${DASHBOARD_PORT}" \
    --ray-client-server-port "${RAY_CLIENT_SERVER_PORT}" \
    "${common_args[@]}"
else
  exec ray start \
    --address "${MASTER_ADDR}:${RAY_PORT}" \
    "${common_args[@]}"
fi
