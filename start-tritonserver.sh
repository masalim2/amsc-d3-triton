#!/bin/bash

# Launch one Triton server inside the remote job allocation and integrate it with
# the remote-triton-controller registration model.
#
# Lifecycle:
# 1. start Triton locally via podman-hpc
# 2. wait for Triton HTTP readiness on localhost
# 3. register the backend with the controller API
# 4. send periodic heartbeats while Triton stays alive
# 5. keep writing the legacy endpoint JSON file for compatibility/debugging
#
# Expected environment variables:
# - CONTROLLER_API_TOKEN
#   required; bearer token used for POST /register and POST /heartbeat
# - REMOTE_TRITON_LAUNCH_TOKEN
#   optional; correlation token injected by the controller so registration can
#   repair a crashed leader's unresolved launch reservation
# - REMOTE_TRITON_CONTROLLER_URL
#   optional; defaults to NERSC endpoint URL
# - TRITON_HEARTBEAT_INTERVAL_SECONDS
#   optional; heartbeat interval, defaults to 10 seconds
# - TRITON_CONTROLLER_CONNECT_TIMEOUT_SECONDS
#   optional; curl connect timeout for controller requests, defaults to 5 seconds
# - TRITON_CONTROLLER_MAX_TIME_SECONDS
#   optional; total curl timeout for controller requests, defaults to 15 seconds
# - SLURM_JOBID / SLURMD_NODENAME / HOSTNAME
#   provided by the remote job environment and used to derive backend identity
#   and log/output locations

set -euo pipefail

OUTPUTFILE="${SLURM_JOBID:-triton}.json"
OUTPUT_DIR="."
MODEL_NAME=""

while getopts "o:d:m:" opt; do
  case $opt in
    o)
      OUTPUTFILE=$OPTARG
      ;;
    d)
      OUTPUT_DIR=$OPTARG
      ;;
    m)
      MODEL_NAME=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

INSTANCE_ID="${SLURM_JOBID:-${HOSTNAME:-triton}}"
LAUNCH_TOKEN="${REMOTE_TRITON_LAUNCH_TOKEN:-}"
CONTROLLER_URL="${REMOTE_TRITON_CONTROLLER_URL:-https://amsc-d3.ml4phys.com}"
CONTROLLER_TOKEN="${CONTROLLER_API_TOKEN:-}"
HEARTBEAT_INTERVAL_SECONDS="${TRITON_HEARTBEAT_INTERVAL_SECONDS:-10}"
CONTROLLER_CONNECT_TIMEOUT_SECONDS="${TRITON_CONTROLLER_CONNECT_TIMEOUT_SECONDS:-5}"
CONTROLLER_MAX_TIME_SECONDS="${TRITON_CONTROLLER_MAX_TIME_SECONDS:-15}"
TRITON_SHUTDOWN_GRACE_SECONDS="${TRITON_SHUTDOWN_GRACE_SECONDS:-20}"

if [ -z "$CONTROLLER_TOKEN" ]; then
  echo "CONTROLLER_API_TOKEN must be set" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
if ! FILE_UPDATE_DIR_BASE="$(realpath "$OUTPUT_DIR")"; then
  echo "Failed to resolve output directory: $OUTPUT_DIR" >&2
  exit 1
fi
FILE_UPDATE_DIR="${FILE_UPDATE_DIR_BASE}/nodes"
mkdir -p "$FILE_UPDATE_DIR"
OUTPUTFILE="$FILE_UPDATE_DIR/$(basename "$OUTPUTFILE")"

WORK_DIR="${OUTPUT_DIR}/${INSTANCE_ID}"
TRITON_MODELS="/global/cfs/cdirs/m2845/amsc-d3/triton-models/cuda_models"
TRITON_IMAGE="docker.io/docexoty/tritonserver:light"
TRITON_JOBS_DIR="${WORK_DIR}/jobs"
TRITON_LOG_VERBOSE=true
TRITON_SERVER_NAME="${SLURMD_NODENAME:-$(hostname)}"
TRITON_HTTP_PORT=8000
TRITON_GRPC_PORT=8001
TRITON_METRICS_PORT=8002
TRITON_READY_URL="http://127.0.0.1:${TRITON_HTTP_PORT}/v2/health/ready"
TRITON_READY_TIMEOUT_SECONDS=300
TRITON_READY_POLL_INTERVAL_SECONDS=1
READY_DEADLINE=$(( $(date +%s) + TRITON_READY_TIMEOUT_SECONDS ))
TRITON_GRPC_READY_TIMEOUT_SECONDS="${TRITON_GRPC_READY_TIMEOUT_SECONDS:-30}"

mkdir -p "$TRITON_JOBS_DIR"

TRITON_LOG_VERBOSE_FLAGS=""
TRITON_MODEL_FLAGS="--model-repository=/models"
if [ -n "$MODEL_NAME" ]; then
  TRITON_MODEL_FLAGS="${TRITON_MODEL_FLAGS} --model-control-mode=explicit --load-model=${MODEL_NAME}"
  echo "Using explicit model load for: ${MODEL_NAME}"
else
  echo "No model provided; loading all models (implicit control mode)"
fi

if [ "$TRITON_LOG_VERBOSE" = true ]; then
  TRITON_LOG_VERBOSE_FLAGS="--log-verbose=3 --log-info=1 --log-warning=1 --log-error=1"
fi

# Post one JSON payload to the controller using the shared bearer token.
controller_post() {
  local endpoint="$1"
  local payload="$2"
  curl --silent --show-error --fail \
    --connect-timeout "${CONTROLLER_CONNECT_TIMEOUT_SECONDS}" \
    --max-time "${CONTROLLER_MAX_TIME_SECONDS}" \
    -X POST \
    -H "Authorization: Bearer ${CONTROLLER_TOKEN}" \
    -H "Content-Type: application/json" \
    --data "$payload" \
    "${CONTROLLER_URL}${endpoint}"
}

# Return success when a TCP listener is accepting connections.
tcp_port_open() {
  local host="$1"
  local port="$2"
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.0)
try:
    sock.connect((host, port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
    return $?
  fi

  if command -v nc >/dev/null 2>&1; then
    nc -z -w 1 "$host" "$port" >/dev/null 2>&1
    return $?
  fi

  echo "[slurm] neither python3 nor nc is available for TCP readiness checks" >&2
  return 1
}

# Stop the background heartbeat loop when the main Triton process exits.
stop_heartbeat_loop() {
  if [ -n "${HEARTBEAT_PID:-}" ]; then
    kill "$HEARTBEAT_PID" 2>/dev/null || true
    wait "$HEARTBEAT_PID" 2>/dev/null || true
    HEARTBEAT_PID=""
  fi
}

graceful_shutdown() {
  if [ "${TRITON_SHUTDOWN_IN_PROGRESS:-0}" = "1" ]; then
    return
  fi
  TRITON_SHUTDOWN_IN_PROGRESS=1

  echo "[slurm] received shutdown signal; stopping heartbeats and terminating Triton"
  stop_heartbeat_loop

  if [ -z "${TRITON_PID:-}" ] || ! kill -0 "$TRITON_PID" 2>/dev/null; then
    exit 0
  fi

  kill -TERM "$TRITON_PID" 2>/dev/null || true

  SHUTDOWN_DEADLINE=$(( $(date +%s) + TRITON_SHUTDOWN_GRACE_SECONDS ))
  while kill -0 "$TRITON_PID" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$SHUTDOWN_DEADLINE" ]; then
      echo "[slurm] Triton did not exit within ${TRITON_SHUTDOWN_GRACE_SECONDS}s; sending SIGKILL" >&2
      kill -KILL "$TRITON_PID" 2>/dev/null || true
      break
    fi
    sleep 1
  done

  wait "$TRITON_PID" 2>/dev/null || true
  exit 0
}

cleanup() {
  stop_heartbeat_loop
}
trap cleanup EXIT
trap graceful_shutdown TERM INT

echo "[slurm] starting ${TRITON_SERVER_NAME}"
podman-hpc run -it --rm --gpu --shm-size=20GB \
  -p ${TRITON_METRICS_PORT}:${TRITON_METRICS_PORT} \
  -p ${TRITON_GRPC_PORT}:${TRITON_GRPC_PORT} \
  -p ${TRITON_HTTP_PORT}:${TRITON_HTTP_PORT} \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --volume="$TRITON_MODELS:/models" \
  -w "$WORK_DIR" \
  -v "$WORK_DIR:$WORK_DIR" \
  "$TRITON_IMAGE" \
  tritonserver \
    $TRITON_MODEL_FLAGS \
    --allow-metrics=true \
    $TRITON_LOG_VERBOSE_FLAGS > >(tee "$TRITON_JOBS_DIR/$TRITON_SERVER_NAME.log") 2>&1 &
TRITON_PID=$!

# The controller only wants backends that can already answer Triton readiness.
echo "[slurm] waiting for Triton readiness at ${TRITON_READY_URL}"
while true; do
  if curl --silent --fail "$TRITON_READY_URL" >/dev/null 2>&1; then
    break
  fi

  echo "[slurm] Triton not ready yet; waiting..."

  if ! kill -0 "$TRITON_PID" 2>/dev/null; then
    wait "$TRITON_PID"
    exit $?
  fi

  if [ "$(date +%s)" -ge "$READY_DEADLINE" ]; then
    echo "[slurm] Triton failed to become ready within ${TRITON_READY_TIMEOUT_SECONDS}s"
    kill "$TRITON_PID" 2>/dev/null || true
    wait "$TRITON_PID" || true
    exit 1
  fi

  sleep "$TRITON_READY_POLL_INTERVAL_SECONDS"
done

HOSTNAME_IP=$(getent hosts "$(hostname)" | awk '{print $1; exit}')
if [ -z "$HOSTNAME_IP" ]; then
  echo "[slurm] failed to resolve host IP" >&2
  kill "$TRITON_PID" 2>/dev/null || true
  wait "$TRITON_PID" || true
  exit 1
fi

GRPC_READY_DEADLINE=$(( $(date +%s) + TRITON_GRPC_READY_TIMEOUT_SECONDS ))
echo "[slurm] waiting for Triton gRPC readiness at ${HOSTNAME_IP}:${TRITON_GRPC_PORT}"
while true; do
  if tcp_port_open "$HOSTNAME_IP" "$TRITON_GRPC_PORT"; then
    break
  fi

  echo "[slurm] Triton gRPC port not ready yet; waiting..."

  if ! kill -0 "$TRITON_PID" 2>/dev/null; then
    wait "$TRITON_PID"
    exit $?
  fi

  if [ "$(date +%s)" -ge "$GRPC_READY_DEADLINE" ]; then
    echo "[slurm] Triton gRPC port failed to open within ${TRITON_GRPC_READY_TIMEOUT_SECONDS}s"
    kill "$TRITON_PID" 2>/dev/null || true
    wait "$TRITON_PID" || true
    exit 1
  fi

  sleep "$TRITON_READY_POLL_INTERVAL_SECONDS"
done

# Registration is the first durable signal that this backend is ready for use.
LAUNCH_TOKEN_JSON_FRAGMENT=""
if [ -n "$LAUNCH_TOKEN" ]; then
  LAUNCH_TOKEN_JSON_FRAGMENT=",\"launch_token\":\"${LAUNCH_TOKEN}\""
fi

REGISTER_PAYLOAD=$(cat <<JSON
{"instance_id":"${INSTANCE_ID}","ip":"${HOSTNAME_IP}","http_port":${TRITON_HTTP_PORT},"grpc_port":${TRITON_GRPC_PORT},"metrics_port":${TRITON_METRICS_PORT}${LAUNCH_TOKEN_JSON_FRAGMENT}}
JSON
)

echo "[slurm] registering Triton backend with controller at ${CONTROLLER_URL}/register"
if ! controller_post "/api/register" "$REGISTER_PAYLOAD" >/dev/null; then
  echo "[slurm] failed to register backend with controller" >&2
  kill "$TRITON_PID" 2>/dev/null || true
  wait "$TRITON_PID" || true
  exit 1
fi

# Heartbeats keep the backend eligible for discovery after initial registration.
HEARTBEAT_PAYLOAD=$(cat <<JSON
{"instance_id":"${INSTANCE_ID}","ip":"${HOSTNAME_IP}","http_port":${TRITON_HTTP_PORT},"grpc_port":${TRITON_GRPC_PORT},"metrics_port":${TRITON_METRICS_PORT}}
JSON
)

# Retry heartbeats forever while Triton is still running; the controller will
# age the backend out if heartbeats stop arriving within its timeout window.
heartbeat_loop() {
  while kill -0 "$TRITON_PID" 2>/dev/null; do
    if ! controller_post "/api/heartbeat" "$HEARTBEAT_PAYLOAD" >/dev/null; then
      echo "[slurm] heartbeat failed; will retry in ${HEARTBEAT_INTERVAL_SECONDS}s" >&2
    fi
    sleep "$HEARTBEAT_INTERVAL_SECONDS"
  done
}

heartbeat_loop &
HEARTBEAT_PID=$!

# Preserve the endpoint file for debugging
TMP_OUTPUTFILE="${OUTPUTFILE}.tmp"
{
  echo "{"
  echo "  \"ip\": \"$HOSTNAME_IP\"," 
  echo "  \"http_port\": ${TRITON_HTTP_PORT},"
  echo "  \"grpc_port\": ${TRITON_GRPC_PORT},"
  echo "  \"metrics_port\": ${TRITON_METRICS_PORT}"
  echo "}"
} > "$TMP_OUTPUTFILE"
mv "$TMP_OUTPUTFILE" "$OUTPUTFILE"
echo "[slurm] Triton is ready; registered with controller and wrote endpoint file: $OUTPUTFILE"

wait "$TRITON_PID"
