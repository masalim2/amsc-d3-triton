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

# Dual-mode Sophia launcher for a standalone Triton server.
#
# From a login node, this script submits itself to PBS, waits for the compute
# node to write an endpoint JSON file, and prints the HTTP/gRPC/metrics URLs.
# Inside the PBS job, it starts Triton with Apptainer and keeps it running until
# PBS walltime expires, qdel is used, or Triton exits.

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-.}"
TRITON_MODELS="${TRITON_MODELS:-}"
TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-8000}"
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-8001}"
TRITON_METRICS_PORT="${TRITON_METRICS_PORT:-8002}"
TRITON_SUBMIT_WAIT_SECONDS="${TRITON_SUBMIT_WAIT_SECONDS:-600}"
TRITON_READY_TIMEOUT_SECONDS="${TRITON_READY_TIMEOUT_SECONDS:-300}"
TRITON_READY_POLL_INTERVAL_SECONDS="${TRITON_READY_POLL_INTERVAL_SECONDS:-1}"
TRITON_WALLTIME="${TRITON_WALLTIME:-01:00:00}"
TRITON_QUEUE="${TRITON_QUEUE:-by-gpu}"
TRITON_SELECT="${TRITON_SELECT:-1}"
TRITON_ACCOUNT="${TRITON_ACCOUNT:-AmSC_Demos}"
TRITON_PATCH_GPU_IDS="${TRITON_PATCH_GPU_IDS:-false}"

usage() {
  cat <<USAGE
Usage: $0 [options]

Submit a PBS job that runs Triton under Apptainer and prints the endpoint.

Options:
  -m MODEL        Load only MODEL with Triton explicit model control mode.
  -r REPO         Triton model repository. Default inside job: ./cuda_models.
  -o DIR          Output/status directory. Default: current directory.
  -t WALLTIME     PBS walltime. Default: ${TRITON_WALLTIME}.
  -q QUEUE        PBS queue. Default: ${TRITON_QUEUE}.
  -s SELECT       PBS select value. Default: ${TRITON_SELECT}.
  -w SECONDS      Wait this long for endpoint after qsub. Default: ${TRITON_SUBMIT_WAIT_SECONDS}.
  -h              Show this help.

Environment:
  TRITON_IMAGE defaults to /eagle/AmSC_Demos/amsc-d3/apptainer/tritonserver_light.sif
  TRITON_PATCH_GPU_IDS=true copies the selected model to the output directory
  and rewrites simple 'gpus: [ ... ]' config entries to 'gpus: [ 0 ]' for
  one-GPU by-gpu jobs.
USAGE
}

while getopts "m:r:o:t:q:s:w:h" opt; do
  case "$opt" in
    m) MODEL_NAME="$OPTARG" ;;
    r) TRITON_MODELS="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    t) TRITON_WALLTIME="$OPTARG" ;;
    q) TRITON_QUEUE="$OPTARG" ;;
    s) TRITON_SELECT="$OPTARG" ;;
    w) TRITON_SUBMIT_WAIT_SECONDS="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) usage >&2; exit 1 ;;
  esac
done

json_get_string() {
  local key="$1"
  local file="$2"
  python3 - "$key" "$file" <<'PY'
import json
import sys

with open(sys.argv[2], encoding="utf-8") as handle:
    data = json.load(handle)
value = data.get(sys.argv[1], "")
print(value)
PY
}

submit_job() {
  if ! command -v qsub >/dev/null 2>&1; then
    echo "qsub is not available; run this from a Sophia login node" >&2
    exit 1
  fi

  mkdir -p "$OUTPUT_DIR"
  local output_abs
  output_abs="$(realpath "$OUTPUT_DIR")"
  mkdir -p "$output_abs/nodes"

  local script_path
  script_path="$(realpath "$0")"
  local var_list
  var_list="MODEL_NAME=${MODEL_NAME},OUTPUT_DIR=${output_abs},TRITON_HTTP_PORT=${TRITON_HTTP_PORT},TRITON_GRPC_PORT=${TRITON_GRPC_PORT},TRITON_METRICS_PORT=${TRITON_METRICS_PORT},TRITON_PATCH_GPU_IDS=${TRITON_PATCH_GPU_IDS}"

  if [ -n "$TRITON_MODELS" ]; then
    var_list="${var_list},TRITON_MODELS=$(realpath "$TRITON_MODELS")"
  fi
  if [ -n "${TRITON_IMAGE:-}" ]; then
    var_list="${var_list},TRITON_IMAGE=${TRITON_IMAGE}"
  fi

  local job_id
  job_id="$(qsub -V \
    -A "$TRITON_ACCOUNT" \
    -q "$TRITON_QUEUE" \
    -l "select=${TRITON_SELECT}" \
    -l "walltime=${TRITON_WALLTIME}" \
    -l singularity_fakeroot=True \
    -l filesystems=home:eagle \
    -v "$var_list" \
    "$script_path")"

  local endpoint_file="$output_abs/nodes/${job_id}.json"
  echo "Submitted PBS job: ${job_id}"
  echo "Waiting for Triton endpoint: ${endpoint_file}"

  local deadline=$(( $(date +%s) + TRITON_SUBMIT_WAIT_SECONDS ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if [ -s "$endpoint_file" ]; then
      local hostname ip http_port grpc_port metrics_port http_url grpc_url metrics_url
      hostname="$(json_get_string hostname "$endpoint_file")"
      ip="$(json_get_string ip "$endpoint_file")"
      http_port="$(json_get_string http_port "$endpoint_file")"
      grpc_port="$(json_get_string grpc_port "$endpoint_file")"
      metrics_port="$(json_get_string metrics_port "$endpoint_file")"
      http_url="$(json_get_string http_url "$endpoint_file")"
      grpc_url="$(json_get_string grpc_url "$endpoint_file")"
      metrics_url="$(json_get_string metrics_url "$endpoint_file")"

      echo "Triton is ready."
      echo "  PBS job:     ${job_id}"
      echo "  Hostname:    ${hostname}"
      echo "  IP:          ${ip}"
      echo "  HTTP:        ${http_url}  (port ${http_port})"
      echo "  gRPC:        ${grpc_url}  (port ${grpc_port})"
      echo "  Metrics:     ${metrics_url}  (port ${metrics_port})"
      echo "  Endpoint:    ${endpoint_file}"
      echo "The server will keep running until PBS walltime, qdel, or Triton exit."
      return 0
    fi

    if command -v qstat >/dev/null 2>&1 && ! qstat "$job_id" >/dev/null 2>&1; then
      echo "PBS job finished before writing an endpoint file: ${job_id}" >&2
      local job_num="${job_id%%.*}"
      if [ -f "${output_abs}/tritonserver.o${job_num}" ]; then
        echo "Last 80 lines of ${output_abs}/tritonserver.o${job_num}:" >&2
        tail -n 80 "${output_abs}/tritonserver.o${job_num}" >&2
      fi
      exit 1
    fi

    sleep 2
  done

  echo "Timed out waiting for endpoint file after ${TRITON_SUBMIT_WAIT_SECONDS}s" >&2
  echo "PBS job is still submitted as: ${job_id}" >&2
  exit 1
}

load_apptainer() {
  if command -v module >/dev/null 2>&1; then
    module use /soft/spack/base/0.7.1/install/modulefiles/Core/ >/dev/null 2>&1 || true
    module load apptainer >/dev/null 2>&1 || true
  elif command -v ml >/dev/null 2>&1; then
    ml use /soft/modulefiles >/dev/null 2>&1 || true
    ml spack-pe-base >/dev/null 2>&1 || true
    ml apptainer >/dev/null 2>&1 || true
  fi

  if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer is not available on this node" >&2
    exit 1
  fi
}

tcp_port_open() {
  local host="$1"
  local port="$2"
  python3 - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.0)
try:
    sock.connect((sys.argv[1], int(sys.argv[2])))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

shutdown_triton() {
  if [ -n "${TRITON_PID:-}" ] && kill -0 "$TRITON_PID" 2>/dev/null; then
    echo "[pbs] stopping Triton"
    kill -TERM "$TRITON_PID" 2>/dev/null || true
    wait "$TRITON_PID" 2>/dev/null || true
  fi
}

run_job() {
  if [ -n "${PBS_O_WORKDIR:-}" ]; then
    cd "$PBS_O_WORKDIR"
  fi
  local script_dir
  script_dir="$(pwd)"

  load_apptainer

  local apptainer_dir="${APPTAINER_DIR:-/eagle/AmSC_Demos/amsc-d3/apptainer}"
  local triton_image="${TRITON_IMAGE:-${apptainer_dir}/tritonserver_light.sif}"
  local triton_models="${TRITON_MODELS:-${script_dir}/cuda_models}"

  export BASE_SCRATCH_DIR="${BASE_SCRATCH_DIR:-$apptainer_dir}"
  export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$apptainer_dir/apptainer-cachedir}"
  export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$apptainer_dir/apptainer-tmpdir}"
  export HTTP_PROXY="${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}"
  export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.alcf.anl.gov:3128}"
  export http_proxy="${http_proxy:-$HTTP_PROXY}"
  export https_proxy="${https_proxy:-$HTTPS_PROXY}"
  export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
  export no_proxy="${no_proxy:-$NO_PROXY}"

  if [ ! -f "$triton_image" ]; then
    echo "Triton Apptainer image does not exist: $triton_image" >&2
    exit 1
  fi
  if [ ! -d "$triton_models" ]; then
    echo "Triton model repository does not exist: $triton_models" >&2
    exit 1
  fi

  local output_abs
  mkdir -p "$OUTPUT_DIR"
  output_abs="$(realpath "$OUTPUT_DIR")"
  local endpoint_dir="${output_abs}/nodes"
  local work_dir="${output_abs}/${PBS_JOBID}"
  local jobs_dir="${work_dir}/jobs"
  local endpoint_file="${endpoint_dir}/${PBS_JOBID}.json"
  mkdir -p "$endpoint_dir" "$jobs_dir"

  if [ "$TRITON_PATCH_GPU_IDS" = "true" ]; then
    if [ -z "$MODEL_NAME" ]; then
      echo "TRITON_PATCH_GPU_IDS=true requires -m MODEL / MODEL_NAME" >&2
      exit 1
    fi
    local patched_repo="${work_dir}/patched-model-repo"
    mkdir -p "$patched_repo"
    cp -a "${triton_models}/${MODEL_NAME}" "$patched_repo/"
    perl -0pi -e 's/gpus:\s*\[\s*[0-9,\s]+\s*\]/gpus: [ 0 ]/g' "${patched_repo}/${MODEL_NAME}/config.pbtxt"
    triton_models="$patched_repo"
    echo "[pbs] using patched one-GPU model copy: ${triton_models}"
  fi

  local model_flags="--model-repository=/models"
  if [ -n "$MODEL_NAME" ]; then
    model_flags="${model_flags} --model-control-mode=explicit --load-model=${MODEL_NAME}"
    echo "[pbs] loading model: ${MODEL_NAME}"
  else
    echo "[pbs] loading all models in repository"
  fi

  local ready_url="http://127.0.0.1:${TRITON_HTTP_PORT}/v2/health/ready"
  local deadline=$(( $(date +%s) + TRITON_READY_TIMEOUT_SECONDS ))
  local log_file="${jobs_dir}/$(hostname).log"

  trap shutdown_triton TERM INT EXIT

  echo "[pbs] job id: ${PBS_JOBID}"
  echo "[pbs] node: $(hostname)"
  echo "[pbs] image: ${triton_image}"
  echo "[pbs] model repository: ${triton_models}"

  apptainer exec --nv \
    --bind="${triton_models}:/models" \
    --bind="${work_dir}:${work_dir}" \
    --pwd "$work_dir" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$triton_image" \
    tritonserver \
      $model_flags \
      --allow-metrics=true \
      --http-port="${TRITON_HTTP_PORT}" \
      --grpc-port="${TRITON_GRPC_PORT}" \
      --metrics-port="${TRITON_METRICS_PORT}" \
      --log-info=1 --log-warning=1 --log-error=1 > >(tee "$log_file") 2>&1 &
  TRITON_PID=$!

  echo "[pbs] waiting for Triton HTTP readiness at ${ready_url}"
  while true; do
    if curl --silent --fail --noproxy '*' "$ready_url" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$TRITON_PID" 2>/dev/null; then
      wait "$TRITON_PID"
      exit $?
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "[pbs] Triton did not become ready within ${TRITON_READY_TIMEOUT_SECONDS}s" >&2
      exit 1
    fi
    sleep "$TRITON_READY_POLL_INTERVAL_SECONDS"
  done

  local host_name host_ip
  host_name="$(hostname -f 2>/dev/null || hostname)"
  host_ip="$(getent hosts "$(hostname)" | awk '{print $1; exit}')"
  if [ -z "$host_ip" ]; then
    echo "[pbs] failed to resolve host IP" >&2
    exit 1
  fi

  local grpc_deadline=$(( $(date +%s) + 30 ))
  while ! tcp_port_open "$host_ip" "$TRITON_GRPC_PORT"; do
    if [ "$(date +%s)" -ge "$grpc_deadline" ]; then
      echo "[pbs] Triton gRPC port did not open within 30s" >&2
      exit 1
    fi
    sleep 1
  done

  local tmp_endpoint="${endpoint_file}.tmp"
  {
    echo "{"
    echo "  \"job_id\": \"${PBS_JOBID}\","
    echo "  \"hostname\": \"${host_name}\","
    echo "  \"ip\": \"${host_ip}\","
    echo "  \"http_port\": ${TRITON_HTTP_PORT},"
    echo "  \"grpc_port\": ${TRITON_GRPC_PORT},"
    echo "  \"metrics_port\": ${TRITON_METRICS_PORT},"
    echo "  \"http_url\": \"http://${host_name}:${TRITON_HTTP_PORT}\","
    echo "  \"grpc_url\": \"${host_name}:${TRITON_GRPC_PORT}\","
    echo "  \"metrics_url\": \"http://${host_name}:${TRITON_METRICS_PORT}/metrics\""
    echo "}"
  } > "$tmp_endpoint"
  mv "$tmp_endpoint" "$endpoint_file"

  echo "[pbs] Triton is ready"
  echo "[pbs] HTTP URL: http://${host_name}:${TRITON_HTTP_PORT}"
  echo "[pbs] gRPC URL: ${host_name}:${TRITON_GRPC_PORT}"
  echo "[pbs] metrics URL: http://${host_name}:${TRITON_METRICS_PORT}/metrics"
  echo "[pbs] endpoint file: ${endpoint_file}"
  echo "[pbs] keeping Triton alive until PBS walltime, qdel, or Triton exit"

  wait "$TRITON_PID"
}

if [ -z "${PBS_JOBID:-}" ]; then
  submit_job
else
  run_job
fi
