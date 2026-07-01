#!/bin/bash

# Submit a parallel PBS smoke-test sweep for the GPU-backed Triton models.
# Each model gets its own one-GPU PBS job and unique HTTP/gRPC/metrics ports.
# By default, jobs are cancelled automatically after Triton reaches readiness.
# Set LEAVE_RUNNING_UNTIL_WALLTIME=true to keep ready servers running until
# PBS walltime, qdel, or Triton exit.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${LAUNCHER:-${ROOT_DIR}/launch-tritonserver-pbs.sh}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/gpu-model-sweep-$(date +%Y%m%d-%H%M%S)}"
WALLTIME="${WALLTIME:-00:10:00}"
WAIT_SECONDS="${WAIT_SECONDS:-600}"
BASE_PORT="${BASE_PORT:-8100}"
PORT_STRIDE="${PORT_STRIDE:-10}"
LEAVE_RUNNING_UNTIL_WALLTIME="${LEAVE_RUNNING_UNTIL_WALLTIME:-false}"

usage() {
  cat <<USAGE
Usage: $0 [options]

Submit a parallel PBS sweep for the GPU-backed Triton models.

Options:
  --leave-running-until-walltime
      Keep ready Triton servers running until PBS walltime, qdel, or Triton
      exit. By default, ready jobs are cancelled after the sweep records PASS.
  -h, --help
      Show this help.

Environment overrides:
  OUTPUT_DIR, WALLTIME, WAIT_SECONDS, BASE_PORT, PORT_STRIDE,
  LEAVE_RUNNING_UNTIL_WALLTIME
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --leave-running-until-walltime)
      LEAVE_RUNNING_UNTIL_WALLTIME=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

mkdir -p "$OUTPUT_DIR"
JOBS_FILE="${OUTPUT_DIR}/jobs.tsv"
RESULTS_FILE="${OUTPUT_DIR}/results.tsv"
: > "$JOBS_FILE"
: > "$RESULTS_FILE"

submit_model() {
  local index="$1"
  local model="$2"
  local repo="$3"
  local patch_gpu_ids="$4"
  local http_port=$((BASE_PORT + index * PORT_STRIDE))
  local grpc_port=$((http_port + 1))
  local metrics_port=$((http_port + 2))
  local job_id

  job_id="$(qsub -V \
    -A AmSC_Demos \
    -q by-gpu \
    -l select=1 \
    -l "walltime=${WALLTIME}" \
    -l singularity_fakeroot=True \
    -l filesystems=home:eagle \
    -v "MODEL_NAME=${model},OUTPUT_DIR=${OUTPUT_DIR},TRITON_MODELS=${repo},TRITON_PATCH_GPU_IDS=${patch_gpu_ids},TRITON_HTTP_PORT=${http_port},TRITON_GRPC_PORT=${grpc_port},TRITON_METRICS_PORT=${metrics_port}" \
    "$LAUNCHER")"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$job_id" "$model" "$repo" "$http_port" "$grpc_port" "$metrics_port" | tee -a "$JOBS_FILE"
}

classify_jobs() {
  local deadline=$(( $(date +%s) + WAIT_SECONDS ))

  while [ "$(date +%s)" -lt "$deadline" ]; do
    while IFS=$'\t' read -r job_id model repo http_port grpc_port metrics_port; do
      [ -n "${job_id:-}" ] || continue
      if grep -q "^${job_id}[[:space:]]" "$RESULTS_FILE" 2>/dev/null; then
        continue
      fi

      local endpoint="${OUTPUT_DIR}/nodes/${job_id}.json"
      local short_job="${job_id%%.*}"
      local log_file="${ROOT_DIR}/tritonserver.o${short_job}"

      if [ -s "$endpoint" ]; then
        local url
        url="$(python3 - "$endpoint" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
        print(json.load(handle)["http_url"])
PY
)"
        printf '%s\t%s\tPASS\t%s\n' "$job_id" "$model" "$url" | tee -a "$RESULTS_FILE"
        if [ "$LEAVE_RUNNING_UNTIL_WALLTIME" != "true" ]; then
          qdel "$job_id" >/dev/null 2>&1 || true
        fi
        continue
      fi

      if ! qstat "$job_id" >/dev/null 2>&1; then
        local reason="finished without endpoint"
        if [ -f "$log_file" ]; then
          local log_reason
          log_reason="$(grep -E "error: creating server|failed to load|UNAVAILABLE|No such file|does not exist|Triton did not become ready|Assertion|Exception|Traceback" "$log_file" | tail -1 || true)"
          if [ -n "$log_reason" ]; then
            reason="$log_reason"
          fi
        fi
        printf '%s\t%s\tFAIL\t%s\n' "$job_id" "$model" "$reason" | tee -a "$RESULTS_FILE"
      fi
    done < "$JOBS_FILE"

    local complete
    local total
    complete="$(wc -l < "$RESULTS_FILE")"
    total="$(wc -l < "$JOBS_FILE")"
    if [ "$complete" -ge "$total" ]; then
      break
    fi
    sleep 10
  done

  while IFS=$'\t' read -r job_id model repo http_port grpc_port metrics_port; do
    [ -n "${job_id:-}" ] || continue
    if grep -q "^${job_id}[[:space:]]" "$RESULTS_FILE" 2>/dev/null; then
      continue
    fi
    printf '%s\t%s\tFAIL\tsweep timed out after %ss without endpoint\n' \
      "$job_id" "$model" "$WAIT_SECONDS" | tee -a "$RESULTS_FILE"
    qdel "$job_id" >/dev/null 2>&1 || true
  done < "$JOBS_FILE"
}

print_summary() {
  echo
  echo "Sweep output: ${OUTPUT_DIR}"
  echo "Jobs: ${JOBS_FILE}"
  echo "Results: ${RESULTS_FILE}"
  echo

  python3 - "$RESULTS_FILE" <<'PY'
import sys
from pathlib import Path

rows = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    rows.append(line.split("\t", 3))

print(f"total {len(rows)}")
print(f"pass {sum(1 for row in rows if row[2] == 'PASS')}")
print(f"fail {sum(1 for row in rows if row[2] == 'FAIL')}")

print("\nPASS:")
for _, model, status, detail in rows:
    if status == "PASS":
        print(f"  {model}: {detail}")

print("\nFAIL:")
for _, model, status, detail in rows:
    if status == "FAIL":
        print(f"  {model}: {detail}")
PY
}

echo "Submitting GPU-backed model sweep to PBS"
echo "Output directory: ${OUTPUT_DIR}"
echo "Leave ready servers running until walltime: ${LEAVE_RUNNING_UNTIL_WALLTIME}"

submit_model 0 "snbamsc_2dcnn_u" "${ROOT_DIR}" "true"
submit_model 1 "snbamsc_2dcnn_v" "${ROOT_DIR}" "true"
submit_model 2 "snbamsc_2dcnn_z" "${ROOT_DIR}" "true"
submit_model 3 "DoubleMetricLearning" "${ROOT_DIR}/cuda_models" "true"
submit_model 4 "higgsInteractionNet" "${ROOT_DIR}/cuda_models" "true"
submit_model 5 "particlenet_AK4_PT" "${ROOT_DIR}/cuda_models" "true"
submit_model 6 "nugraph2" "${ROOT_DIR}" "true"

classify_jobs
print_summary
