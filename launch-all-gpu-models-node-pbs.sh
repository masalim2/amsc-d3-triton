#!/bin/bash
#PBS -N triton-all-gpu-models
#PBS -A AmSC_Demos
#PBS -q by-node
#PBS -l select=1
#PBS -l singularity_fakeroot=True
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -V

# Dual-mode Sophia launcher for serving all GPU-backed Triton models on one
# full node. From a login node, this script submits itself to PBS, waits for an
# endpoint JSON file, and prints one HTTP/gRPC/metrics URL set per model.
# Inside the PBS job it starts seven Triton servers concurrently:
#   - six models with one visible GPU each
#   - nugraph2 with two visible GPUs

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-.}"
TRITON_WALLTIME="${TRITON_WALLTIME:-01:00:00}"
TRITON_ACCOUNT="${TRITON_ACCOUNT:-AmSC_Demos}"
TRITON_QUEUE="${TRITON_QUEUE:-by-node}"
TRITON_SELECT="${TRITON_SELECT:-1}"
TRITON_SUBMIT_WAIT_SECONDS="${TRITON_SUBMIT_WAIT_SECONDS:-900}"
TRITON_READY_TIMEOUT_SECONDS="${TRITON_READY_TIMEOUT_SECONDS:-600}"
TRITON_READY_POLL_INTERVAL_SECONDS="${TRITON_READY_POLL_INTERVAL_SECONDS:-1}"
BASE_PORT="${BASE_PORT:-8500}"
PORT_STRIDE="${PORT_STRIDE:-10}"

usage() {
  cat <<USAGE
Usage: $0 [options]

Submit one full-node PBS job and serve all seven GPU-backed Triton models.

Options:
  -o DIR        Output/status directory. Default: current directory.
  -t WALLTIME   PBS walltime. Default: ${TRITON_WALLTIME}.
  -w SECONDS    Wait this long for endpoints after qsub. Default: ${TRITON_SUBMIT_WAIT_SECONDS}.
  -p PORT       Base HTTP port. Default: ${BASE_PORT}. gRPC and metrics use +1/+2.
  -h            Show this help.

Environment:
  TRITON_IMAGE defaults to /eagle/AmSC_Demos/amsc-d3/apptainer/tritonserver_light.sif
  TRITON_ACCOUNT defaults to AmSC_Demos
  TRITON_QUEUE defaults to by-node
  TRITON_SELECT defaults to 1

GPU layout inside the PBS job:
  GPU 0      snbamsc_2dcnn_u
  GPU 1      snbamsc_2dcnn_v
  GPU 2      snbamsc_2dcnn_z
  GPU 3      DoubleMetricLearning
  GPU 4      higgsInteractionNet
  GPU 5      particlenet_AK4_PT
  GPUs 6,7   nugraph2
USAGE
}

while getopts "o:t:w:p:h" opt; do
  case "$opt" in
    o) OUTPUT_DIR="$OPTARG" ;;
    t) TRITON_WALLTIME="$OPTARG" ;;
    w) TRITON_SUBMIT_WAIT_SECONDS="$OPTARG" ;;
    p) BASE_PORT="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) usage >&2; exit 1 ;;
  esac
done

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
  var_list="OUTPUT_DIR=${output_abs},BASE_PORT=${BASE_PORT},PORT_STRIDE=${PORT_STRIDE}"
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
  echo "Waiting for all Triton endpoints: ${endpoint_file}"

  local deadline=$(( $(date +%s) + TRITON_SUBMIT_WAIT_SECONDS ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if [ -s "$endpoint_file" ]; then
      echo "All Triton servers are ready."
      python3 - "$endpoint_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)

print(f"  PBS job:  {data['job_id']}")
print(f"  Hostname: {data['hostname']}")
print(f"  Endpoint: {sys.argv[1]}")
for model in data["models"]:
    print(f"  {model['model']}:")
    print(f"    GPUs:    {model['host_gpus']}")
    print(f"    HTTP:    {model['http_url']}")
    print(f"    gRPC:    {model['grpc_url']}")
    print(f"    Metrics: {model['metrics_url']}")
PY
      echo "The servers will keep running until PBS walltime, qdel, or Triton exit."
      return 0
    fi

    if command -v qstat >/dev/null 2>&1 && ! qstat "$job_id" >/dev/null 2>&1; then
      echo "PBS job finished before writing endpoint file: ${job_id}" >&2
      local job_num="${job_id%%.*}"
      if [ -f "${output_abs}/triton-all-gpu-models.o${job_num}" ]; then
        echo "Last 120 lines of ${output_abs}/triton-all-gpu-models.o${job_num}:" >&2
        tail -n 120 "${output_abs}/triton-all-gpu-models.o${job_num}" >&2
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

patch_gpu_ids() {
  local config="$1"
  local gpu_config="$2"
  perl -0pi -e "s/gpus:\\s*\\[\\s*[0-9,\\s]+\\s*\\]/gpus: [ ${gpu_config} ]/g" "$config"
}

shutdown_all() {
  local pid
  for pid in "${TRITON_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "[pbs] stopping Triton pid ${pid}"
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${TRITON_PIDS[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}

start_model() {
  local index="$1"
  local model="$2"
  local repo="$3"
  local host_gpus="$4"
  local container_gpus="$5"
  local http_port=$((BASE_PORT + index * PORT_STRIDE))
  local grpc_port=$((http_port + 1))
  local metrics_port=$((http_port + 2))
  local model_work_dir="${WORK_DIR}/${model}"
  local patched_repo="${model_work_dir}/model-repo"
  local log_file="${LOG_DIR}/${model}.log"

  mkdir -p "$patched_repo" "$model_work_dir"
  cp -a "${repo}/${model}" "$patched_repo/"
  patch_gpu_ids "${patched_repo}/${model}/config.pbtxt" "$container_gpus"

  echo "[pbs] starting ${model} on host GPU(s) ${host_gpus}; container config GPU(s) ${container_gpus}"
  echo "[pbs] ${model} ports: http=${http_port} grpc=${grpc_port} metrics=${metrics_port}"

  CUDA_VISIBLE_DEVICES="$host_gpus" apptainer exec --nv \
    --bind="${patched_repo}:/models" \
    --bind="${model_work_dir}:${model_work_dir}" \
    --pwd "$model_work_dir" \
    --env "CUDA_VISIBLE_DEVICES=${host_gpus}" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$TRITON_IMAGE_PATH" \
    tritonserver \
      --model-repository=/models \
      --model-control-mode=explicit \
      --load-model="${model}" \
      --allow-metrics=true \
      --http-port="${http_port}" \
      --grpc-port="${grpc_port}" \
      --metrics-port="${metrics_port}" \
      --log-info=1 --log-warning=1 --log-error=1 > >(tee "$log_file") 2>&1 &

  local pid=$!
  TRITON_PIDS+=("$pid")
  MODEL_ROWS+=("${model}|${host_gpus}|${http_port}|${grpc_port}|${metrics_port}|${pid}|${log_file}")
}

wait_for_model() {
  local row="$1"
  local model host_gpus http_port grpc_port metrics_port pid log_file
  IFS='|' read -r model host_gpus http_port grpc_port metrics_port pid log_file <<< "$row"

  local ready_url="http://127.0.0.1:${http_port}/v2/health/ready"
  local deadline=$(( $(date +%s) + TRITON_READY_TIMEOUT_SECONDS ))

  echo "[pbs] waiting for ${model}: ${ready_url}"
  while true; do
    if curl --silent --fail --noproxy '*' "$ready_url" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[pbs] ${model} exited before readiness; log: ${log_file}" >&2
      wait "$pid"
      return 1
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "[pbs] ${model} did not become ready within ${TRITON_READY_TIMEOUT_SECONDS}s; log: ${log_file}" >&2
      return 1
    fi
    sleep "$TRITON_READY_POLL_INTERVAL_SECONDS"
  done

  while ! tcp_port_open "127.0.0.1" "$grpc_port"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[pbs] ${model} exited before gRPC port opened; log: ${log_file}" >&2
      wait "$pid"
      return 1
    fi
    sleep 1
  done

  echo "[pbs] ${model} is ready"
}

write_endpoint_file() {
  local endpoint_file="$1"
  local host_name="$2"
  local host_ip="$3"
  python3 - "$endpoint_file" "$PBS_JOBID" "$host_name" "$host_ip" "${MODEL_ROWS[@]}" <<'PY'
import json
import sys

endpoint_file, job_id, hostname, ip, *rows = sys.argv[1:]
models = []
for row in rows:
    model, host_gpus, http_port, grpc_port, metrics_port, pid, log_file = row.split("|", 6)
    models.append({
        "model": model,
        "host_gpus": host_gpus,
        "pid": int(pid),
        "http_port": int(http_port),
        "grpc_port": int(grpc_port),
        "metrics_port": int(metrics_port),
        "http_url": f"http://{hostname}:{http_port}",
        "grpc_url": f"{hostname}:{grpc_port}",
        "metrics_url": f"http://{hostname}:{metrics_port}/metrics",
        "log_file": log_file,
    })

payload = {
    "job_id": job_id,
    "hostname": hostname,
    "ip": ip,
    "models": models,
}

with open(f"{endpoint_file}.tmp", "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY
  mv "${endpoint_file}.tmp" "$endpoint_file"
}

run_job() {
  if [ -n "${PBS_O_WORKDIR:-}" ]; then
    cd "$PBS_O_WORKDIR"
  fi

  load_apptainer

  local script_dir
  script_dir="$(pwd)"
  local apptainer_dir="${APPTAINER_DIR:-/eagle/AmSC_Demos/amsc-d3/apptainer}"
  TRITON_IMAGE_PATH="${TRITON_IMAGE:-${apptainer_dir}/tritonserver_light.sif}"

  export BASE_SCRATCH_DIR="${BASE_SCRATCH_DIR:-$apptainer_dir}"
  export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$apptainer_dir/apptainer-cachedir}"
  export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$apptainer_dir/apptainer-tmpdir}"
  export HTTP_PROXY="${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}"
  export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.alcf.anl.gov:3128}"
  export http_proxy="${http_proxy:-$HTTP_PROXY}"
  export https_proxy="${https_proxy:-$HTTPS_PROXY}"
  export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
  export no_proxy="${no_proxy:-$NO_PROXY}"

  if [ ! -f "$TRITON_IMAGE_PATH" ]; then
    echo "Triton Apptainer image does not exist: $TRITON_IMAGE_PATH" >&2
    exit 1
  fi

  local root_repo="${script_dir}"
  local cuda_repo="${script_dir}/cuda_models"
  local output_abs
  mkdir -p "$OUTPUT_DIR"
  output_abs="$(realpath "$OUTPUT_DIR")"
  local endpoint_dir="${output_abs}/nodes"
  WORK_DIR="${output_abs}/${PBS_JOBID}"
  LOG_DIR="${WORK_DIR}/logs"
  local endpoint_file="${endpoint_dir}/${PBS_JOBID}.json"
  mkdir -p "$endpoint_dir" "$WORK_DIR" "$LOG_DIR"

  TRITON_PIDS=()
  MODEL_ROWS=()
  trap shutdown_all TERM INT EXIT

  echo "[pbs] job id: ${PBS_JOBID}"
  echo "[pbs] node: $(hostname)"
  echo "[pbs] image: ${TRITON_IMAGE_PATH}"
  echo "[pbs] output: ${output_abs}"
  echo "[pbs] base port: ${BASE_PORT}"

  start_model 0 "snbamsc_2dcnn_u" "$root_repo" "0" "0"
  start_model 1 "snbamsc_2dcnn_v" "$root_repo" "1" "0"
  start_model 2 "snbamsc_2dcnn_z" "$root_repo" "2" "0"
  start_model 3 "DoubleMetricLearning" "$cuda_repo" "3" "0"
  start_model 4 "higgsInteractionNet" "$cuda_repo" "4" "0"
  start_model 5 "particlenet_AK4_PT" "$cuda_repo" "5" "0"
  start_model 6 "nugraph2" "$root_repo" "6,7" "0, 1"

  local row
  for row in "${MODEL_ROWS[@]}"; do
    wait_for_model "$row"
  done

  local host_name host_ip
  host_name="$(hostname -f 2>/dev/null || hostname)"
  host_ip="$(getent hosts "$(hostname)" | awk '{print $1; exit}')"
  if [ -z "$host_ip" ]; then
    echo "[pbs] failed to resolve host IP" >&2
    exit 1
  fi

  write_endpoint_file "$endpoint_file" "$host_name" "$host_ip"

  echo "[pbs] all Triton servers are ready"
  echo "[pbs] endpoint file: ${endpoint_file}"
  python3 - "$endpoint_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)
for model in data["models"]:
    print(f"[pbs] {model['model']}: GPUs {model['host_gpus']} HTTP {model['http_url']} gRPC {model['grpc_url']}")
PY
  echo "[pbs] keeping servers alive until PBS walltime, qdel, or process exit"

  local pid
  while true; do
    for pid in "${TRITON_PIDS[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "[pbs] Triton pid ${pid} exited; stopping remaining servers" >&2
        exit 1
      fi
    done
    sleep 10
  done
}

if [ -z "${PBS_JOBID:-}" ]; then
  submit_job
else
  run_job
fi
