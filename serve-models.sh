#!/bin/bash
set -euox pipefail


BASE_HTTP_PORT=8500
PORT_STRIDE=10

if [ -z "${TRITON_IMAGE:-}" ]; then
  echo "TRITON_IMAGE must be set" >&2
  exit 1
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "apptainer is not available" >&2
  exit 1
fi

if [ ! -f "$TRITON_IMAGE" ]; then
  echo "Triton image does not exist: $TRITON_IMAGE" >&2
  exit 1
fi

ROOT_REPO="/eagle/AmSC_Demos/amsc-d3/triton-models"
CUDA_REPO="${ROOT_REPO}/cuda_models"
WORK_DIR="`pwd`/triton-work"
LOG_DIR="${WORK_DIR}/logs"
mkdir -p "$LOG_DIR"

TRITON_PIDS=()

shutdown_all() {
  for pid in "${TRITON_PIDS[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${TRITON_PIDS[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap shutdown_all TERM INT EXIT

start_model() {
  local index="$1" model="$2" repo="$3" host_gpus="$4" container_gpus="$5"
  local http_port=$((BASE_HTTP_PORT + index * PORT_STRIDE))
  local grpc_port=$((http_port + 1))
  local metrics_port=$((http_port + 2))
  local model_dir="${WORK_DIR}/${model}"
  local patched_repo="${model_dir}/model-repo"

  mkdir -p "$patched_repo" "$model_dir"
  cp -a "${repo}/${model}" "$patched_repo/"
  perl -0pi -e "s/gpus:\\s*\\[\\s*[0-9,\\s]+\\s*\\]/gpus: [ ${container_gpus} ]/g" \
    "${patched_repo}/${model}/config.pbtxt"

  echo "starting ${model} on GPU(s) ${host_gpus} (http=${http_port} grpc=${grpc_port})"

  CUDA_VISIBLE_DEVICES="$host_gpus" apptainer exec --nv \
    --bind="${patched_repo}:/models" \
    --bind="${model_dir}:${model_dir}" \
    --pwd "$model_dir" \
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
      --log-info=1 --log-warning=1 --log-error=1 \
    > "${LOG_DIR}/${model}.log" 2>&1 &

  TRITON_PIDS+=("$!")
}


start_model 0 snbamsc_2dcnn_u      "$ROOT_REPO" 0   "0"
start_model 1 snbamsc_2dcnn_v      "$ROOT_REPO" 1   "0"
start_model 2 snbamsc_2dcnn_z      "$ROOT_REPO" 2   "0"
start_model 3 DoubleMetricLearning "$CUDA_REPO" 3   "0"
start_model 4 higgsInteractionNet  "$CUDA_REPO" 4   "0"
start_model 5 particlenet_AK4_PT   "$CUDA_REPO" 5   "0"
start_model 6 nugraph2             "$ROOT_REPO" 6,7 "0, 1"

echo "all models launched; logs in ${LOG_DIR}/"
wait
