#!/bin/bash

# Standalone compute-node launcher for serving all GPU-backed Triton models
# concurrently on one full Sophia node.
#
# This script does not submit a PBS job. Run it after obtaining a full-node
# allocation, for example through qsub -I on the by-node queue.

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-./all-gpu-models-node-$(date +%Y%m%d-%H%M%S)}"
BASE_PORT="${BASE_PORT:-8500}"
PORT_STRIDE="${PORT_STRIDE:-10}"
TRITON_READY_TIMEOUT_SECONDS="${TRITON_READY_TIMEOUT_SECONDS:-600}"
TRITON_READY_POLL_INTERVAL_SECONDS="${TRITON_READY_POLL_INTERVAL_SECONDS:-1}"
TRITON_IMAGE="${TRITON_IMAGE:-/eagle/AmSC_Demos/amsc-d3/apptainer/tritonserver_light.sif}"

usage() {
  cat <<USAGE
Usage: $0 [options]

Serve all seven GPU-backed Triton models concurrently on the current node.
This script assumes it is already running on a full GPU node.

Options:
  -o DIR      Output/status directory. Default: ${OUTPUT_DIR}
  -p PORT     Base HTTP port. Default: ${BASE_PORT}. gRPC and metrics use +1/+2.
  -i IMAGE    Triton Apptainer image. Default: ${TRITON_IMAGE}
  -h          Show this help.

GPU layout:
  GPU 0      snbamsc_2dcnn_u
  GPU 1      snbamsc_2dcnn_v
  GPU 2      snbamsc_2dcnn_z
  GPU 3      DoubleMetricLearning
  GPU 4      higgsInteractionNet
  GPU 5      particlenet_AK4_PT
  GPUs 6,7   nugraph2
USAGE
}

while getopts "o:p:i:h" opt; do
  case "$opt" in
    o) OUTPUT_DIR="$OPTARG" ;;
    p) BASE_PORT="$OPTARG" ;;
    i) TRITON_IMAGE="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) usage >&2; exit 1 ;;
  esac
done

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
    echo "apptainer is not available; run this on a Sophia compute node with Apptainer loaded" >&2
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
      echo "[serve] stopping Triton pid ${pid}"
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

  echo "[serve] starting ${model} on host GPU(s) ${host_gpus}; HTTP ${http_port}"

  CUDA_VISIBLE_DEVICES="$host_gpus" apptainer exec --nv \
    --bind="${patched_repo}:/models" \
    --bind="${model_work_dir}:${model_work_dir}" \
    --pwd "$model_work_dir" \
    --env "CUDA_VISIBLE_DEVICES=${host_gpus}" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$TRITON_IMAGE" \
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

  echo "[serve] waiting for ${model}: ${ready_url}"
  while true; do
    if curl --silent --fail --noproxy '*' "$ready_url" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[serve] ${model} exited before readiness; log: ${log_file}" >&2
      wait "$pid"
      return 1
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "[serve] ${model} did not become ready within ${TRITON_READY_TIMEOUT_SECONDS}s; log: ${log_file}" >&2
      return 1
    fi
    sleep "$TRITON_READY_POLL_INTERVAL_SECONDS"
  done

  while ! tcp_port_open "127.0.0.1" "$grpc_port"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[serve] ${model} exited before gRPC port opened; log: ${log_file}" >&2
      wait "$pid"
      return 1
    fi
    sleep 1
  done

  echo "[serve] ${model} is ready"
}

write_endpoint_file() {
  local endpoint_file="$1"
  local host_name="$2"
  local host_ip="$3"
  python3 - "$endpoint_file" "$host_name" "$host_ip" "${MODEL_ROWS[@]}" <<'PY'
import json
import sys

endpoint_file, hostname, ip, *rows = sys.argv[1:]
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

main() {
  load_apptainer

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local apptainer_dir="${APPTAINER_DIR:-/eagle/AmSC_Demos/amsc-d3/apptainer}"

  export BASE_SCRATCH_DIR="${BASE_SCRATCH_DIR:-$apptainer_dir}"
  export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$apptainer_dir/apptainer-cachedir}"
  export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$apptainer_dir/apptainer-tmpdir}"
  export HTTP_PROXY="${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}"
  export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.alcf.anl.gov:3128}"
  export http_proxy="${http_proxy:-$HTTP_PROXY}"
  export https_proxy="${https_proxy:-$HTTPS_PROXY}"
  export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
  export no_proxy="${no_proxy:-$NO_PROXY}"

  if [ ! -f "$TRITON_IMAGE" ]; then
    echo "Triton Apptainer image does not exist: $TRITON_IMAGE" >&2
    exit 1
  fi

  local root_repo="${script_dir}"
  local cuda_repo="${script_dir}/cuda_models"
  local output_abs
  mkdir -p "$OUTPUT_DIR"
  output_abs="$(realpath "$OUTPUT_DIR")"
  WORK_DIR="${output_abs}/work"
  LOG_DIR="${WORK_DIR}/logs"
  local endpoint_file="${output_abs}/endpoints.json"
  mkdir -p "$WORK_DIR" "$LOG_DIR"

  TRITON_PIDS=()
  MODEL_ROWS=()
  trap shutdown_all TERM INT EXIT

  echo "[serve] node: $(hostname)"
  echo "[serve] image: ${TRITON_IMAGE}"
  echo "[serve] output: ${output_abs}"
  echo "[serve] base port: ${BASE_PORT}"

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
    echo "[serve] failed to resolve host IP" >&2
    exit 1
  fi

  write_endpoint_file "$endpoint_file" "$host_name" "$host_ip"

  echo "[serve] all Triton servers are ready"
  echo "[serve] endpoint file: ${endpoint_file}"
  python3 - "$endpoint_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)
for model in data["models"]:
    print(f"[serve] {model['model']}: GPUs {model['host_gpus']} HTTP {model['http_url']} gRPC {model['grpc_url']}")
PY
  echo "[serve] keeping servers alive until interrupted or one Triton process exits"

  local pid
  while true; do
    for pid in "${TRITON_PIDS[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "[serve] Triton pid ${pid} exited; stopping remaining servers" >&2
        exit 1
      fi
    done
    sleep 10
  done
}

main "$@"
