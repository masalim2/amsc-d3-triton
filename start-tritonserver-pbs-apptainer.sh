#!/bin/bash
#PBS -N tritonserver
#PBS -A AmSC_Demos
#PBS -q by-gpu
#PBS -l select=1
#PBS -l singularity_fakeroot=True
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -V

# Launch one Triton server on a Sophia PBS allocation using Apptainer.
#
# Typical use from a Sophia login node:
#   qsub -V start-tritonserver-pbs-apptainer.sh
#
# Typical use from an interactive PBS allocation:
#   ./start-tritonserver-pbs-apptainer.sh
#
# Required environment:
# - CONTROLLER_API_TOKEN
#   bearer token used for POST /register and POST /heartbeat
#
# Useful overrides:
# - TRITON_IMAGE
#   Apptainer/Singularity image containing tritonserver. Defaults to
#   /eagle/AmSC_Demos/amsc-d3/apptainer/tritonserver_light.sif
# - TRITON_MODELS
#   model repository to mount at /models. Defaults to ./cuda_models
# - REMOTE_TRITON_CONTROLLER_URL
#   controller API base URL. Defaults to https://amsc-d3.ml4phys.com

set -euo pipefail

OUTPUTFILE="${OUTPUTFILE:-${PBS_JOBID:-triton}.json}"
OUTPUT_DIR="${OUTPUT_DIR:-.}"
MODEL_NAME="${MODEL_NAME:-}"

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

if [ -n "${PBS_O_WORKDIR:-}" ]; then
  cd "$PBS_O_WORKDIR"
fi
SCRIPT_DIR="$(pwd)"

INSTANCE_ID="${PBS_JOBID:-${HOSTNAME:-triton}}"
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

if command -v module >/dev/null 2>&1; then
  module use /soft/spack/base/0.7.1/install/modulefiles/Core/ >/dev/null 2>&1 || true
  module load apptainer >/dev/null 2>&1 || true
elif command -v ml >/dev/null 2>&1; then
  ml use /soft/modulefiles >/dev/null 2>&1 || true
  ml spack-pe-base >/dev/null 2>&1 || true
  ml apptainer >/dev/null 2>&1 || true
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "apptainer is not available; run from a Sophia compute node after loading the Apptainer module" >&2
  exit 1
fi

APPTAINER_DIR="${APPTAINER_DIR:-/eagle/AmSC_Demos/amsc-d3/apptainer}"
export BASE_SCRATCH_DIR="${BASE_SCRATCH_DIR:-$APPTAINER_DIR}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$APPTAINER_DIR/apptainer-cachedir}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$APPTAINER_DIR/apptainer-tmpdir}"
export HTTP_PROXY="${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.alcf.anl.gov:3128}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-$NO_PROXY}"

mkdir -p "$OUTPUT_DIR"
if ! FILE_UPDATE_DIR_BASE="$(realpath "$OUTPUT_DIR")"; then
  echo "Failed to resolve output directory: $OUTPUT_DIR" >&2
  exit 1
fi
FILE_UPDATE_DIR="${FILE_UPDATE_DIR_BASE}/nodes"
mkdir -p "$FILE_UPDATE_DIR"
OUTPUTFILE="$FILE_UPDATE_DIR/$(basename "$OUTPUTFILE")"

WORK_DIR="${FILE_UPDATE_DIR_BASE}/${INSTANCE_ID}"
TRITON_MODELS="${TRITON_MODELS:-${SCRIPT_DIR}/cuda_models}"
TRITON_IMAGE="${TRITON_IMAGE:-${APPTAINER_DIR}/tritonserver_light.sif}"
TRITON_JOBS_DIR="${WORK_DIR}/jobs"
TRITON_LOG_VERBOSE="${TRITON_LOG_VERBOSE:-true}"
TRITON_SERVER_NAME="${HOSTNAME:-$(hostname)}"
TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-8000}"
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-8001}"
TRITON_METRICS_PORT="${TRITON_METRICS_PORT:-8002}"
TRITON_READY_URL="http://127.0.0.1:${TRITON_HTTP_PORT}/v2/health/ready"
TRITON_READY_TIMEOUT_SECONDS="${TRITON_READY_TIMEOUT_SECONDS:-300}"
TRITON_READY_POLL_INTERVAL_SECONDS="${TRITON_READY_POLL_INTERVAL_SECONDS:-1}"
READY_DEADLINE=$(( $(date +%s) + TRITON_READY_TIMEOUT_SECONDS ))
TRITON_GRPC_READY_TIMEOUT_SECONDS="${TRITON_GRPC_READY_TIMEOUT_SECONDS:-30}"

if [ ! -d "$TRITON_MODELS" ]; then
  echo "Triton model repository does not exist: $TRITON_MODELS" >&2
  exit 1
fi

if [ ! -f "$TRITON_IMAGE" ]; then
  echo "Triton Apptainer image does not exist: $TRITON_IMAGE" >&2
  echo "Set TRITON_IMAGE to a .sif image containing tritonserver." >&2
  exit 1
fi

mkdir -p "$TRITON_JOBS_DIR"

TRITON_LOG_VERBOSE_FLAGS=""
TRITON_MODEL_FLAGS="--model-repository=/models"
if [ -n "$MODEL_NAME" ]; then
  TRITON_MODEL_FLAGS="${TRITON_MODEL_FLAGS} --model-control-mode=explicit --load-model=${MODEL_NAME}"
  echo "[pbs] using explicit model load for: ${MODEL_NAME}"
else
  echo "[pbs] no model provided; loading all models (implicit control mode)"
fi

if [ "$TRITON_LOG_VERBOSE" = true ]; then
  TRITON_LOG_VERBOSE_FLAGS="--log-verbose=3 --log-info=1 --log-warning=1 --log-error=1"
fi

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

  echo "[pbs] neither python3 nor nc is available for TCP readiness checks" >&2
  return 1
}

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

  echo "[pbs] received shutdown signal; stopping heartbeats and terminating Triton"
  stop_heartbeat_loop

  if [ -z "${TRITON_PID:-}" ] || ! kill -0 "$TRITON_PID" 2>/dev/null; then
    exit 0
  fi

  kill -TERM "$TRITON_PID" 2>/dev/null || true

  SHUTDOWN_DEADLINE=$(( $(date +%s) + TRITON_SHUTDOWN_GRACE_SECONDS ))
  while kill -0 "$TRITON_PID" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$SHUTDOWN_DEADLINE" ]; then
      echo "[pbs] Triton did not exit within ${TRITON_SHUTDOWN_GRACE_SECONDS}s; sending SIGKILL" >&2
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

echo "[pbs] starting ${TRITON_SERVER_NAME}"
echo "[pbs] model repository: ${TRITON_MODELS}"
echo "[pbs] apptainer image: ${TRITON_IMAGE}"
apptainer exec --nv \
  --bind="${TRITON_MODELS}:/models" \
  --bind="${WORK_DIR}:${WORK_DIR}" \
  --pwd "$WORK_DIR" \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$TRITON_IMAGE" \
  tritonserver \
    $TRITON_MODEL_FLAGS \
    --allow-metrics=true \
    --http-port="${TRITON_HTTP_PORT}" \
    --grpc-port="${TRITON_GRPC_PORT}" \
    --metrics-port="${TRITON_METRICS_PORT}" \
    $TRITON_LOG_VERBOSE_FLAGS > >(tee "$TRITON_JOBS_DIR/$TRITON_SERVER_NAME.log") 2>&1 &
TRITON_PID=$!

echo "[pbs] waiting for Triton readiness at ${TRITON_READY_URL}"
while true; do
  if curl --silent --fail "$TRITON_READY_URL" >/dev/null 2>&1; then
    break
  fi

  echo "[pbs] Triton not ready yet; waiting..."

  if ! kill -0 "$TRITON_PID" 2>/dev/null; then
    wait "$TRITON_PID"
    exit $?
  fi

  if [ "$(date +%s)" -ge "$READY_DEADLINE" ]; then
    echo "[pbs] Triton failed to become ready within ${TRITON_READY_TIMEOUT_SECONDS}s"
    kill "$TRITON_PID" 2>/dev/null || true
    wait "$TRITON_PID" || true
    exit 1
  fi

  sleep "$TRITON_READY_POLL_INTERVAL_SECONDS"
done

HOSTNAME_IP=$(getent hosts "$(hostname)" | awk '{print $1; exit}')
if [ -z "$HOSTNAME_IP" ]; then
  echo "[pbs] failed to resolve host IP" >&2
  kill "$TRITON_PID" 2>/dev/null || true
  wait "$TRITON_PID" || true
  exit 1
fi

GRPC_READY_DEADLINE=$(( $(date +%s) + TRITON_GRPC_READY_TIMEOUT_SECONDS ))
echo "[pbs] waiting for Triton gRPC readiness at ${HOSTNAME_IP}:${TRITON_GRPC_PORT}"
while true; do
  if tcp_port_open "$HOSTNAME_IP" "$TRITON_GRPC_PORT"; then
    break
  fi

  echo "[pbs] Triton gRPC port not ready yet; waiting..."

  if ! kill -0 "$TRITON_PID" 2>/dev/null; then
    wait "$TRITON_PID"
    exit $?
  fi

  if [ "$(date +%s)" -ge "$GRPC_READY_DEADLINE" ]; then
    echo "[pbs] Triton gRPC port failed to open within ${TRITON_GRPC_READY_TIMEOUT_SECONDS}s"
    kill "$TRITON_PID" 2>/dev/null || true
    wait "$TRITON_PID" || true
    exit 1
  fi

  sleep "$TRITON_READY_POLL_INTERVAL_SECONDS"
done

LAUNCH_TOKEN_JSON_FRAGMENT=""
if [ -n "$LAUNCH_TOKEN" ]; then
  LAUNCH_TOKEN_JSON_FRAGMENT=",\"launch_token\":\"${LAUNCH_TOKEN}\""
fi

REGISTER_PAYLOAD=$(cat <<JSON
{"instance_id":"${INSTANCE_ID}","ip":"${HOSTNAME_IP}","http_port":${TRITON_HTTP_PORT},"grpc_port":${TRITON_GRPC_PORT},"metrics_port":${TRITON_METRICS_PORT}${LAUNCH_TOKEN_JSON_FRAGMENT}}
JSON
)

echo "[pbs] registering Triton backend with controller at ${CONTROLLER_URL}/register"
if ! controller_post "/api/register" "$REGISTER_PAYLOAD" >/dev/null; then
  echo "[pbs] failed to register backend with controller" >&2
  kill "$TRITON_PID" 2>/dev/null || true
  wait "$TRITON_PID" || true
  exit 1
fi

HEARTBEAT_PAYLOAD=$(cat <<JSON
{"instance_id":"${INSTANCE_ID}","ip":"${HOSTNAME_IP}","http_port":${TRITON_HTTP_PORT},"grpc_port":${TRITON_GRPC_PORT},"metrics_port":${TRITON_METRICS_PORT}}
JSON
)

heartbeat_loop() {
  while kill -0 "$TRITON_PID" 2>/dev/null; do
    if ! controller_post "/api/heartbeat" "$HEARTBEAT_PAYLOAD" >/dev/null; then
      echo "[pbs] heartbeat failed; will retry in ${HEARTBEAT_INTERVAL_SECONDS}s" >&2
    fi
    sleep "$HEARTBEAT_INTERVAL_SECONDS"
  done
}

heartbeat_loop &
HEARTBEAT_PID=$!

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
echo "[pbs] Triton is ready; registered with controller and wrote endpoint file: $OUTPUTFILE"

wait "$TRITON_PID"
