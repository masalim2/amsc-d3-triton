#!/bin/bash

# Run perf_analyzer against the endpoint JSON produced by
# launch-all-gpu-models-node-pbs.sh.

set -uo pipefail

ENDPOINT_JSON="${ENDPOINT_JSON:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./perf-analyzer-all-gpu-models-$(date +%Y%m%d-%H%M%S)}"
SDK_IMAGE="${SDK_IMAGE:-/eagle/AmSC_Demos/amsc-d3/apptainer/tritonserver_26.05-py3-sdk.sif}"
NUGRAPH2_INPUT="${NUGRAPH2_INPUT:-/eagle/AmSC_Demos/amsc-d3/triton-models/perf_analyzer_inputs/nugraph2_input.json}"
CONCURRENCY_RANGE="${CONCURRENCY_RANGE:-1}"
MEASUREMENT_INTERVAL="${MEASUREMENT_INTERVAL:-5000}"

usage() {
  cat <<USAGE
Usage: $0 -e ENDPOINT_JSON [options]

Run perf_analyzer against all model endpoints written by
launch-all-gpu-models-node-pbs.sh.

Options:
  -e FILE     Combined endpoint JSON from launch-all-gpu-models-node-pbs.sh.
  -o DIR      Output directory. Default: ${OUTPUT_DIR}
  -s IMAGE    Triton SDK Apptainer image. Default: ${SDK_IMAGE}
  -c RANGE    perf_analyzer --concurrency-range. Default: ${CONCURRENCY_RANGE}
  -m MSEC     perf_analyzer --measurement-interval. Default: ${MEASUREMENT_INTERVAL}
  -n FILE     nugraph2 JSON input file. Default: ${NUGRAPH2_INPUT}
  -h          Show this help.

Outputs:
  <output>/logs/<model>.log
  <output>/results.tsv
USAGE
}

while getopts "e:o:s:c:m:n:h" opt; do
  case "$opt" in
    e) ENDPOINT_JSON="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    s) SDK_IMAGE="$OPTARG" ;;
    c) CONCURRENCY_RANGE="$OPTARG" ;;
    m) MEASUREMENT_INTERVAL="$OPTARG" ;;
    n) NUGRAPH2_INPUT="$OPTARG" ;;
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
    echo "apptainer is not available" >&2
    exit 1
  fi
}

model_shapes() {
  local model="$1"
  case "$model" in
    snbamsc_2dcnn_u)
      printf '%s\n' "--shape" "zero_padding2d_input:1149,128,1"
      ;;
    snbamsc_2dcnn_v)
      printf '%s\n' "--shape" "zero_padding2d_input:1148,128,1"
      ;;
    snbamsc_2dcnn_z)
      printf '%s\n' "--shape" "zero_padding2d_1_input:480,128,1"
      ;;
    DoubleMetricLearning)
      printf '%s\n' "--shape" "FEATURES:1,44"
      ;;
    higgsInteractionNet)
      printf '%s\n' "--shape" "input_cpf:30,60" "--shape" "input_sv:14,5"
      ;;
    particlenet_AK4_PT)
      printf '%s\n' \
        "--shape" "pf_points__0:2,100" \
        "--shape" "pf_features__1:20,100" \
        "--shape" "pf_mask__2:1,100" \
        "--shape" "sv_points__3:2,10" \
        "--shape" "sv_features__4:11,10" \
        "--shape" "sv_mask__5:1,10"
      ;;
    nugraph2)
      printf '%s\n' \
        "--shape" "hit_table_hit_id:90" \
        "--shape" "hit_table_local_plane:90" \
        "--shape" "hit_table_local_time:90" \
        "--shape" "hit_table_local_wire:90" \
        "--shape" "hit_table_integral:90" \
        "--shape" "hit_table_rms:90" \
        "--shape" "spacepoint_table_spacepoint_id:30" \
        "--shape" "spacepoint_table_hit_id_u:30" \
        "--shape" "spacepoint_table_hit_id_v:30" \
        "--shape" "spacepoint_table_hit_id_y:30"
      ;;
    *)
      return 1
      ;;
  esac
}

endpoint_rows() {
  python3 - "$ENDPOINT_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)

for model in data.get("models", []):
    name = model["model"]
    http_url = model["http_url"]
    host_port = http_url.removeprefix("http://").removeprefix("https://")
    print(f"{name}\t{host_port}\t{http_url}")
PY
}

parse_log_summary() {
  local log_file="$1"
  python3 - "$log_file" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
throughput = ""
latency = ""
server_latency = ""

match = re.search(r"Throughput:\s+([0-9.]+)\s+infer/sec", text)
if match:
    throughput = match.group(1)
match = re.search(r"Avg latency:\s+([0-9.]+)\s+usec", text)
if match:
    latency = match.group(1)
match = re.search(r"Avg request latency:\s+([0-9.]+)\s+usec", text)
if match:
    server_latency = match.group(1)

print(f"{throughput}\t{latency}\t{server_latency}")
PY
}

run_model() {
  local model="$1"
  local host_port="$2"
  local http_url="$3"
  local log_file="${LOG_DIR}/${model}.log"
  local input_data="random"
  local bind_args=(--bind /eagle:/eagle)
  local shape_args=()

  if ! mapfile -t shape_args < <(model_shapes "$model"); then
    echo "[perf] unknown model, skipping: ${model}" | tee "$log_file"
    printf '%s\t%s\tSKIP\t\t\tunknown model\n' "$model" "$http_url" >> "$RESULTS_FILE"
    return 0
  fi

  if [ "$model" = "nugraph2" ]; then
    input_data="$NUGRAPH2_INPUT"
  fi

  echo "[perf] ${model}: ${http_url}"
  apptainer exec "${bind_args[@]}" "$SDK_IMAGE" \
    perf_analyzer \
      -m "$model" \
      -i http \
      -u "$host_port" \
      --input-data "$input_data" \
      --concurrency-range "$CONCURRENCY_RANGE" \
      --measurement-interval "$MEASUREMENT_INTERVAL" \
      "${shape_args[@]}" > "$log_file" 2>&1

  local status="PASS"
  if [ "$?" -ne 0 ]; then
    status="FAIL"
  fi

  local summary
  summary="$(parse_log_summary "$log_file")"
  local throughput latency server_latency
  IFS=$'\t' read -r throughput latency server_latency <<< "$summary"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$model" "$http_url" "$status" "$throughput" "$latency" "$server_latency" "$log_file" >> "$RESULTS_FILE"

  if [ "$status" = "PASS" ]; then
    echo "[perf] ${model}: PASS throughput=${throughput} infer/sec avg_latency=${latency} usec"
  else
    echo "[perf] ${model}: FAIL log=${log_file}"
  fi
}

main() {
  if [ -z "$ENDPOINT_JSON" ]; then
    usage >&2
    exit 1
  fi
  if [ ! -f "$ENDPOINT_JSON" ]; then
    echo "Endpoint JSON does not exist: $ENDPOINT_JSON" >&2
    exit 1
  fi
  if [ ! -f "$SDK_IMAGE" ]; then
    echo "SDK Apptainer image does not exist: $SDK_IMAGE" >&2
    exit 1
  fi
  if [ ! -f "$NUGRAPH2_INPUT" ]; then
    echo "nugraph2 input JSON does not exist: $NUGRAPH2_INPUT" >&2
    exit 1
  fi

  load_apptainer

  mkdir -p "$OUTPUT_DIR"
  OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
  LOG_DIR="${OUTPUT_DIR}/logs"
  RESULTS_FILE="${OUTPUT_DIR}/results.tsv"
  mkdir -p "$LOG_DIR"
  printf 'model\thttp_url\tstatus\tthroughput_infer_per_sec\tavg_client_latency_usec\tavg_server_latency_usec\tlog_file\n' > "$RESULTS_FILE"

  echo "[perf] endpoint JSON: ${ENDPOINT_JSON}"
  echo "[perf] output: ${OUTPUT_DIR}"
  echo "[perf] SDK image: ${SDK_IMAGE}"
  echo "[perf] concurrency range: ${CONCURRENCY_RANGE}"
  echo "[perf] measurement interval: ${MEASUREMENT_INTERVAL} ms"

  local model host_port http_url
  while IFS=$'\t' read -r model host_port http_url; do
    [ -n "${model:-}" ] || continue
    run_model "$model" "$host_port" "$http_url"
  done < <(endpoint_rows)

  echo "[perf] results: ${RESULTS_FILE}"
  column -t -s $'\t' "$RESULTS_FILE" 2>/dev/null || cat "$RESULTS_FILE"
}

main "$@"
