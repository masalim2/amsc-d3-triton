# Triton Models on Sophia

This directory contains Triton model repositories and PBS/Apptainer launchers
for serving the models on ALCF Sophia compute nodes.

## Model Layout

The model repository root is:

```bash
/eagle/AmSC_Demos/amsc-d3/triton-models
```

The GPU-backed models used by the Sophia sweep are:

| Model | Repository path |
| --- | --- |
| `nugraph2` | `/eagle/AmSC_Demos/amsc-d3/triton-models/nugraph2` |
| `snbamsc_2dcnn_u` | `/eagle/AmSC_Demos/amsc-d3/triton-models/snbamsc_2dcnn_u` |
| `snbamsc_2dcnn_v` | `/eagle/AmSC_Demos/amsc-d3/triton-models/snbamsc_2dcnn_v` |
| `snbamsc_2dcnn_z` | `/eagle/AmSC_Demos/amsc-d3/triton-models/snbamsc_2dcnn_z` |
| `DoubleMetricLearning` | `/eagle/AmSC_Demos/amsc-d3/triton-models/cuda_models/DoubleMetricLearning` |
| `higgsInteractionNet` | `/eagle/AmSC_Demos/amsc-d3/triton-models/cuda_models/higgsInteractionNet` |
| `particlenet_AK4_PT` | `/eagle/AmSC_Demos/amsc-d3/triton-models/cuda_models/particlenet_AK4_PT` |

The Triton Apptainer image is:

```bash
/eagle/AmSC_Demos/amsc-d3/apptainer/tritonserver_light.sif
```

Models are served on compute nodes by running:

```bash
apptainer exec --nv ... tritonserver --model-repository=/models ...
```

The `--nv` option exposes the allocated NVIDIA GPU inside the Apptainer
container. The launcher scripts bind the selected model repository to `/models`
inside the container and start Triton on HTTP `8000`, gRPC `8001`, and metrics
`8002` by default.

Some checked-in model configs refer to GPU id `3`. A one-GPU `by-gpu`
allocation exposes that GPU as id `0` inside the container, so the launchers can
use `TRITON_PATCH_GPU_IDS=true` to make a temporary per-job model copy and
rewrite simple `gpus: [ ... ]` entries to `gpus: [ 0 ]`.

## Manual Serving on an Interactive Compute Node

Run these steps from a Sophia login node.

1. Change to the model directory.

```bash
cd /eagle/AmSC_Demos/amsc-d3/triton-models
```

2. Request an interactive compute node with one GPU.

```bash
qsub -I -A AmSC_Demos -l select=1 -q by-gpu \
  -l singularity_fakeroot=True \
  -l walltime=01:00:00 \
  -l filesystems=home:eagle
```

3. After PBS places you on a compute node, load Apptainer.

```bash
module use /soft/spack/base/0.7.1/install/modulefiles/Core/ 2>/dev/null || true
module load apptainer
```

If your shell uses the shorter ALCF module helper, this also works:

```bash
ml use /soft/modulefiles
ml spack-pe-base
ml apptainer
```

4. Set Apptainer cache/tmp locations and proxy environment.

```bash
export APPTAINER_DIR=/eagle/AmSC_Demos/amsc-d3/apptainer
export BASE_SCRATCH_DIR=$APPTAINER_DIR
export APPTAINER_CACHEDIR=$APPTAINER_DIR/apptainer-cachedir
export APPTAINER_TMPDIR=$APPTAINER_DIR/apptainer-tmpdir
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=$HTTP_PROXY
export https_proxy=$HTTPS_PROXY
export NO_PROXY=127.0.0.1,localhost
export no_proxy=$NO_PROXY
```

5. Start Triton manually.

For one-GPU allocations, make a temporary model repository and patch the
configured GPU ids to `0`. This leaves the checked-in model files unchanged.

For `nugraph2`, use the top-level repository:

```bash
rm -rf /tmp/triton-nugraph2-repo /tmp/triton-nugraph2-work
mkdir -p /tmp/triton-nugraph2-repo /tmp/triton-nugraph2-work
cp -a /eagle/AmSC_Demos/amsc-d3/triton-models/nugraph2 /tmp/triton-nugraph2-repo/
perl -0pi -e 's/gpus:\s*\[\s*[0-9,\s]+\s*\]/gpus: [ 0 ]/g' \
  /tmp/triton-nugraph2-repo/nugraph2/config.pbtxt

apptainer exec --nv \
  --bind /tmp/triton-nugraph2-repo:/models \
  --bind /tmp/triton-nugraph2-work:/tmp/triton-nugraph2-work \
  --pwd /tmp/triton-nugraph2-work \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /eagle/AmSC_Demos/amsc-d3/apptainer/tritonserver_light.sif \
  tritonserver \
    --model-repository=/models \
    --model-control-mode=explicit \
    --load-model=nugraph2 \
    --allow-metrics=true \
    --http-port=8000 \
    --grpc-port=8001 \
    --metrics-port=8002
```

For a model under `cuda_models`, bind that repository instead:

```bash
rm -rf /tmp/triton-dml-repo /tmp/triton-dml-work
mkdir -p /tmp/triton-dml-repo /tmp/triton-dml-work
cp -a /eagle/AmSC_Demos/amsc-d3/triton-models/cuda_models/DoubleMetricLearning \
  /tmp/triton-dml-repo/
perl -0pi -e 's/gpus:\s*\[\s*[0-9,\s]+\s*\]/gpus: [ 0 ]/g' \
  /tmp/triton-dml-repo/DoubleMetricLearning/config.pbtxt

apptainer exec --nv \
  --bind /tmp/triton-dml-repo:/models \
  --bind /tmp/triton-dml-work:/tmp/triton-dml-work \
  --pwd /tmp/triton-dml-work \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /eagle/AmSC_Demos/amsc-d3/apptainer/tritonserver_light.sif \
  tritonserver \
    --model-repository=/models \
    --model-control-mode=explicit \
    --load-model=DoubleMetricLearning \
    --allow-metrics=true \
    --http-port=8000 \
    --grpc-port=8001 \
    --metrics-port=8002
```

6. Check readiness from another shell on the same compute node.

```bash
curl --noproxy '*' http://127.0.0.1:8000/v2/health/ready
```

If the command exits successfully, Triton is ready. The service endpoints are:

```text
HTTP:    http://<compute-node-hostname>:8000
gRPC:    <compute-node-hostname>:8001
Metrics: http://<compute-node-hostname>:8002/metrics
```

7. Stop Triton with `Ctrl-C`, or let PBS terminate it when walltime expires.

## Launching One Model from the Login Node

`launch-tritonserver-pbs.sh` submits one PBS job, waits until the compute node
writes an endpoint JSON file, prints the hostname and ports, and leaves Triton
running until walltime, `qdel`, or Triton exit.

The most important options are:

```text
-m MODEL      model name to load
-r REPO       Triton model repository containing that model
-t WALLTIME   PBS walltime, for example 01:00:00
-o DIR        output/status directory
-w SECONDS    how long the submit-side command waits for the endpoint file
```

Use `TRITON_PATCH_GPU_IDS=true` for the one-GPU `by-gpu` jobs. The launcher
then copies the selected model into the job output directory and rewrites simple
`gpus: [ ... ]` config entries to `gpus: [ 0 ]` in that temporary copy.

Example for `nugraph2`:

```bash
TRITON_PATCH_GPU_IDS=true ./launch-tritonserver-pbs.sh \
  -m nugraph2 \
  -r /eagle/AmSC_Demos/amsc-d3/triton-models \
  -t 01:00:00
```

The six GPU-backed models use two repository roots:

| Model | Use this `-r` repository |
| --- | --- |
| `snbamsc_2dcnn_u` | `/eagle/AmSC_Demos/amsc-d3/triton-models` |
| `snbamsc_2dcnn_v` | `/eagle/AmSC_Demos/amsc-d3/triton-models` |
| `snbamsc_2dcnn_z` | `/eagle/AmSC_Demos/amsc-d3/triton-models` |
| `DoubleMetricLearning` | `/eagle/AmSC_Demos/amsc-d3/triton-models/cuda_models` |
| `higgsInteractionNet` | `/eagle/AmSC_Demos/amsc-d3/triton-models/cuda_models` |
| `particlenet_AK4_PT` | `/eagle/AmSC_Demos/amsc-d3/triton-models/cuda_models` |

Actual example for starting `higgsInteractionNet`:

```bash
cd /eagle/AmSC_Demos/amsc-d3/triton-models

TRITON_PATCH_GPU_IDS=true ./launch-tritonserver-pbs.sh \
  -m higgsInteractionNet \
  -r /eagle/AmSC_Demos/amsc-d3/triton-models/cuda_models \
  -o /eagle/AmSC_Demos/amsc-d3/triton-models/higgsInteractionNet-run \
  -t 01:00:00
```

Equivalent commands for the other five GPU-backed models:

```bash
TRITON_PATCH_GPU_IDS=true ./launch-tritonserver-pbs.sh \
  -m snbamsc_2dcnn_u \
  -r /eagle/AmSC_Demos/amsc-d3/triton-models \
  -t 01:00:00

TRITON_PATCH_GPU_IDS=true ./launch-tritonserver-pbs.sh \
  -m snbamsc_2dcnn_v \
  -r /eagle/AmSC_Demos/amsc-d3/triton-models \
  -t 01:00:00

TRITON_PATCH_GPU_IDS=true ./launch-tritonserver-pbs.sh \
  -m snbamsc_2dcnn_z \
  -r /eagle/AmSC_Demos/amsc-d3/triton-models \
  -t 01:00:00

TRITON_PATCH_GPU_IDS=true ./launch-tritonserver-pbs.sh \
  -m DoubleMetricLearning \
  -r /eagle/AmSC_Demos/amsc-d3/triton-models/cuda_models \
  -t 01:00:00

TRITON_PATCH_GPU_IDS=true ./launch-tritonserver-pbs.sh \
  -m particlenet_AK4_PT \
  -r /eagle/AmSC_Demos/amsc-d3/triton-models/cuda_models \
  -t 01:00:00
```

Placeholder form for another model:

```bash
TRITON_PATCH_GPU_IDS=true ./launch-tritonserver-pbs.sh \
  -m <model-name> \
  -r <triton-model-repository-containing-model> \
  -o <output-directory> \
  -t <HH:MM:SS>
```

The launcher prints output similar to:

```text
Submitted PBS job: 159768.sophia-pbs-01.lab.alcf.anl.gov
Triton is ready.
  Hostname:    sophia-gpu-05.lab.alcf.anl.gov
  IP:          10.140.49.232
  HTTP:        http://sophia-gpu-05.lab.alcf.anl.gov:8000
  gRPC:        sophia-gpu-05.lab.alcf.anl.gov:8001
  Metrics:     http://sophia-gpu-05.lab.alcf.anl.gov:8002/metrics
```

Cancel a running job with:

```bash
qdel <job-id>
```

## Launching All GPU Models on One Node

`launch-all-gpu-models-node-pbs.sh` submits one full-node PBS job to the
`by-node` queue and starts seven Triton servers concurrently on that node. The
six non-`nugraph2` models get one GPU each, and `nugraph2` gets two GPUs.

```bash
./launch-all-gpu-models-node-pbs.sh \
  -o /eagle/AmSC_Demos/amsc-d3/triton-models/all-gpu-models-node-run \
  -t 01:00:00 \
  -p 8500
```

The default GPU layout is:

| Model | Host GPU(s) | HTTP port |
| --- | --- | ---: |
| `snbamsc_2dcnn_u` | `0` | `8500` |
| `snbamsc_2dcnn_v` | `1` | `8510` |
| `snbamsc_2dcnn_z` | `2` | `8520` |
| `DoubleMetricLearning` | `3` | `8530` |
| `higgsInteractionNet` | `4` | `8540` |
| `particlenet_AK4_PT` | `5` | `8550` |
| `nugraph2` | `6,7` | `8560` |

For each model, gRPC uses HTTP port `+1` and metrics uses HTTP port `+2`.
The script writes a combined endpoint JSON file under `<output>/nodes/` and
prints every model URL when all servers are ready.

Run `perf_analyzer` against all seven endpoints from that combined endpoint
file with:

```bash
./perf-analyze-all-gpu-models-node.sh \
  -e /eagle/AmSC_Demos/amsc-d3/triton-models/all-gpu-models-node-run/nodes/<pbs-job-id>.json \
  -o /eagle/AmSC_Demos/amsc-d3/triton-models/all-gpu-models-node-perf \
  -c 1 \
  -m 5000
```

The analyzer writes one log per model under `<output>/logs/` and a summary TSV
at `<output>/results.tsv`. It uses random input data for all models except
`nugraph2`, where it uses
`/eagle/AmSC_Demos/amsc-d3/triton-models/perf_analyzer_inputs/nugraph2_input.json`.

Full example from a successful run:

```bash
cd /eagle/AmSC_Demos/amsc-d3/triton-models

./launch-all-gpu-models-node-pbs.sh \
  -o /eagle/AmSC_Demos/amsc-d3/triton-models/all-gpu-models-node-run-20260623-150211 \
  -t 01:00:00 \
  -p 8500 \
  -w 1200
```

The launcher printed:

```text
Submitted PBS job: 159893.sophia-pbs-01.lab.alcf.anl.gov
All Triton servers are ready.
  Hostname: sophia-gpu-05.lab.alcf.anl.gov
  snbamsc_2dcnn_u:      http://sophia-gpu-05.lab.alcf.anl.gov:8500
  snbamsc_2dcnn_v:      http://sophia-gpu-05.lab.alcf.anl.gov:8510
  snbamsc_2dcnn_z:      http://sophia-gpu-05.lab.alcf.anl.gov:8520
  DoubleMetricLearning: http://sophia-gpu-05.lab.alcf.anl.gov:8530
  higgsInteractionNet:  http://sophia-gpu-05.lab.alcf.anl.gov:8540
  particlenet_AK4_PT:   http://sophia-gpu-05.lab.alcf.anl.gov:8550
  nugraph2:             http://sophia-gpu-05.lab.alcf.anl.gov:8560
```

Then run `perf_analyzer` against that job's endpoint file:

```bash
./perf-analyze-all-gpu-models-node.sh \
  -e /eagle/AmSC_Demos/amsc-d3/triton-models/all-gpu-models-node-run-20260623-150211/nodes/159893.sophia-pbs-01.lab.alcf.anl.gov.json \
  -o /eagle/AmSC_Demos/amsc-d3/triton-models/all-gpu-models-node-perf-20260623-150211 \
  -c 1 \
  -m 5000
```

That run wrote:

```text
/eagle/AmSC_Demos/amsc-d3/triton-models/all-gpu-models-node-perf-20260623-150211/results.tsv
```

and produced:

| Model | HTTP port | Throughput | Avg client latency |
| --- | ---: | ---: | ---: |
| `snbamsc_2dcnn_u` | `8500` | 404.496 infer/sec | 2113 usec |
| `snbamsc_2dcnn_v` | `8510` | 425.754 infer/sec | 2044 usec |
| `snbamsc_2dcnn_z` | `8520` | 631.02 infer/sec | 1429 usec |
| `DoubleMetricLearning` | `8530` | 2649.87 infer/sec | 369 usec |
| `higgsInteractionNet` | `8540` | 1255.17 infer/sec | 779 usec |
| `particlenet_AK4_PT` | `8550` | 426.255 infer/sec | 2324 usec |
| `nugraph2` | `8560` | 14.1583 infer/sec | 70377 usec |

Keep the serving job running for interactive use, or cancel it manually:

```bash
qdel 159893.sophia-pbs-01.lab.alcf.anl.gov
```

## Serving BTagging on a Full Node

`BTagging_network_8085e6c5717c` is a CPU-backed ONNX Runtime model. Its
`config.pbtxt` uses `KIND_CPU`, so serve it on a full node through the `by-node`
queue rather than the one-GPU `by-gpu` queue.

Start the server from a login node:

```bash
cd /eagle/AmSC_Demos/amsc-d3/triton-models

./launch-tritonserver-pbs.sh \
  -m BTagging_network_8085e6c5717c \
  -r /eagle/AmSC_Demos/amsc-d3/triton-models/cpu_models \
  -o /eagle/AmSC_Demos/amsc-d3/triton-models/btagging-full-node-serve \
  -q by-node \
  -s 1 \
  -t 01:00:00 \
  -w 900
```

A successful run printed:

```text
Submitted PBS job: 159927.sophia-pbs-01.lab.alcf.anl.gov
Triton is ready.
  Hostname: sophia-gpu-04.lab.alcf.anl.gov
  HTTP:     http://sophia-gpu-04.lab.alcf.anl.gov:8000
  gRPC:     sophia-gpu-04.lab.alcf.anl.gov:8001
  Metrics:  http://sophia-gpu-04.lab.alcf.anl.gov:8002/metrics
```

Download the real-world request JSON:

```bash
mkdir -p /eagle/AmSC_Demos/amsc-d3/triton-models/benchmark_inputs

curl -L --fail \
  -o /eagle/AmSC_Demos/amsc-d3/triton-models/benchmark_inputs/daod_BTagging_network_8085e6c5717c_5000evts.json \
  https://portal.nersc.gov/cfs/m3443/xju/BenchmarkData/daod_BTagging_network_8085e6c5717c_5000evts.json
```

Run `perf_analyzer` with that JSON file:

```bash
SDK=/eagle/AmSC_Demos/amsc-d3/apptainer/tritonserver_26.05-py3-sdk.sif
INPUT=/eagle/AmSC_Demos/amsc-d3/triton-models/benchmark_inputs/daod_BTagging_network_8085e6c5717c_5000evts.json
OUT=/eagle/AmSC_Demos/amsc-d3/triton-models/btagging-full-node-serve/perf_analyzer

mkdir -p "$OUT"
module use /soft/spack/base/0.7.1/install/modulefiles/Core/ 2>/dev/null || true
module load apptainer

apptainer exec --bind /eagle:/eagle "$SDK" perf_analyzer \
  -m BTagging_network_8085e6c5717c \
  -i http \
  -u sophia-gpu-04.lab.alcf.anl.gov:8000 \
  --input-data "$INPUT" \
  --concurrency-range 1 \
  --measurement-interval 5000 \
  > "$OUT/BTagging_network_8085e6c5717c_5000evts.log" 2>&1
```

The tested JSON file contained `4999` request steps. The successful
concurrency-`1` result was:

| Metric | Value |
| --- | ---: |
| Throughput | 79.2008 infer/sec |
| Avg client latency | 12570 usec |
| p50 latency | 9341 usec |
| p90 latency | 24920 usec |
| p95 latency | 30892 usec |
| p99 latency | 52430 usec |
| Avg server request latency | 11954 usec |

Cancel the serving job when finished:

```bash
qdel <pbs-job-id>
```

If you already have a full-node allocation, use the standalone compute-node
launcher instead. It does not call `qsub`; it starts the same seven Triton
servers on the current node and writes `<output>/endpoints.json`.

```bash
qsub -I -A AmSC_Demos -l select=1 -q by-node \
  -l singularity_fakeroot=True \
  -l walltime=01:00:00 \
  -l filesystems=home:eagle

cd /eagle/AmSC_Demos/amsc-d3/triton-models

./serve-all-gpu-models-node.sh \
  -o /eagle/AmSC_Demos/amsc-d3/triton-models/all-gpu-models-node-run \
  -p 8500
```

## GPU Model Sweep

`sweep-gpu-models-pbs.sh` submits a parallel smoke-test sweep for only the
GPU-backed models:

```text
snbamsc_2dcnn_u
snbamsc_2dcnn_v
snbamsc_2dcnn_z
DoubleMetricLearning
higgsInteractionNet
particlenet_AK4_PT
nugraph2
```

For each model, the sweep:

1. Submits a one-GPU PBS job to the `by-gpu` queue.
2. Uses unique HTTP/gRPC/metrics ports so jobs can share a physical node.
3. Sets `TRITON_PATCH_GPU_IDS=true` for one-GPU Sophia allocations.
4. Waits for an endpoint JSON file under the sweep output directory.
5. Marks the model `PASS` when Triton reaches readiness.
6. Cancels the ready test job immediately to release the GPU by default.
7. Writes `jobs.tsv` and `results.tsv`.

Run the default sweep:

```bash
./sweep-gpu-models-pbs.sh
```

Useful overrides:

```bash
WALLTIME=00:15:00 WAIT_SECONDS=900 BASE_PORT=8200 ./sweep-gpu-models-pbs.sh
```

To leave passing Triton servers running until PBS walltime, use
`--leave-running-until-walltime`:

```bash
WALLTIME=01:00:00 \
  WAIT_SECONDS=900 \
  BASE_PORT=8200 \
  ./sweep-gpu-models-pbs.sh --leave-running-until-walltime
```

The same behavior can be selected with an environment variable:

```bash
LEAVE_RUNNING_UNTIL_WALLTIME=true ./sweep-gpu-models-pbs.sh
```

In this mode, `PASS` rows in `results.tsv` are live endpoints until walltime,
manual `qdel`, or Triton exit. Failed or timed-out jobs are still cancelled.
Use the job ids in `jobs.tsv` to stop running servers manually:

```bash
qdel <pbs-job-id>
```

The script prints the output directory at startup and summary at completion:

```text
Sweep output: /eagle/AmSC_Demos/amsc-d3/triton-models/gpu-model-sweep-YYYYMMDD-HHMMSS
Jobs: .../jobs.tsv
Results: .../results.tsv
```

`results.tsv` contains one row per model:

```text
<pbs-job-id>    <model-name>    PASS|FAIL    <endpoint-or-error>
```

If a model reaches readiness, the endpoint field is the HTTP URL. If a model
does not reach readiness before `WAIT_SECONDS`, the script records a failure and
attempts to cancel the job.

## Running Perf Analyzer

Use NVIDIA Triton's SDK image for `perf_analyzer`:

```bash
SDK=/eagle/AmSC_Demos/amsc-d3/apptainer/tritonserver_26.05-py3-sdk.sif
```

The examples below use HTTP, concurrency `1`, batch size `1`, and a 5 second
measurement window. Most models can use random input data. `nugraph2` needs a
structured JSON event because its inputs are correlated tables. Replace the host
and port with the endpoint reported by `launch-tritonserver-pbs.sh` or
`sweep-gpu-models-pbs.sh`.

Example for `snbamsc_2dcnn_u`:

```bash
module use /soft/spack/base/0.7.1/install/modulefiles/Core/ 2>/dev/null || true
module load apptainer

apptainer exec "$SDK" perf_analyzer \
  -m snbamsc_2dcnn_u \
  -i http \
  -u <hostname>:<http-port> \
  --input-data random \
  --shape zero_padding2d_input:1149,128,1 \
  --concurrency-range 1 \
  --measurement-interval 5000
```

Model-specific shape arguments:

```text
snbamsc_2dcnn_u:
  --shape zero_padding2d_input:1149,128,1

snbamsc_2dcnn_v:
  --shape zero_padding2d_input:1148,128,1

snbamsc_2dcnn_z:
  --shape zero_padding2d_1_input:480,128,1

DoubleMetricLearning:
  --shape FEATURES:1,44

higgsInteractionNet:
  --shape input_cpf:30,60 --shape input_sv:14,5

particlenet_AK4_PT:
  --shape pf_points__0:2,100
  --shape pf_features__1:20,100
  --shape pf_mask__2:1,100
  --shape sv_points__3:2,10
  --shape sv_features__4:11,10
  --shape sv_mask__5:1,10

nugraph2:
  --shape hit_table_hit_id:90
  --shape hit_table_local_plane:90
  --shape hit_table_local_time:90
  --shape hit_table_local_wire:90
  --shape hit_table_integral:90
  --shape hit_table_rms:90
  --shape spacepoint_table_spacepoint_id:30
  --shape spacepoint_table_hit_id_u:30
  --shape spacepoint_table_hit_id_v:30
  --shape spacepoint_table_hit_id_y:30
```

For `nugraph2`, use the checked-in JSON payload with valid hit and spacepoint
table relationships:

```bash
NUGRAPH2_INPUT=/eagle/AmSC_Demos/amsc-d3/triton-models/perf_analyzer_inputs/nugraph2_input.json

apptainer exec --bind /eagle:/eagle "$SDK" perf_analyzer \
  -m nugraph2 \
  -i http \
  -u <hostname>:<http-port> \
  --input-data "$NUGRAPH2_INPUT" \
  --shape hit_table_hit_id:90 \
  --shape hit_table_local_plane:90 \
  --shape hit_table_local_time:90 \
  --shape hit_table_local_wire:90 \
  --shape hit_table_integral:90 \
  --shape hit_table_rms:90 \
  --shape spacepoint_table_spacepoint_id:30 \
  --shape spacepoint_table_hit_id_u:30 \
  --shape spacepoint_table_hit_id_v:30 \
  --shape spacepoint_table_hit_id_y:30 \
  --concurrency-range 1 \
  --measurement-interval 5000
```

The checked-in file was generated with:

```bash
mkdir -p /eagle/AmSC_Demos/amsc-d3/triton-models/perf_analyzer_inputs

python3 - <<'PY'
import json
from pathlib import Path

out = Path("/eagle/AmSC_Demos/amsc-d3/triton-models/perf_analyzer_inputs/nugraph2_input.json")
n_hits = 90
n_sp = 30

payload = {
    "data": [
        {
            "hit_table_hit_id": list(range(n_hits)),
            "hit_table_local_plane": [i // 30 for i in range(n_hits)],
            "hit_table_local_time": [
                float((i % 30) * 2 + (i // 30) * 0.1) for i in range(n_hits)
            ],
            "hit_table_local_wire": [
                int((i % 30) * 3 + (i // 30)) for i in range(n_hits)
            ],
            "hit_table_integral": [float(100.0 + (i % 11)) for i in range(n_hits)],
            "hit_table_rms": [float(2.0 + (i % 5) * 0.1) for i in range(n_hits)],
            "spacepoint_table_spacepoint_id": list(range(n_sp)),
            "spacepoint_table_hit_id_u": list(range(0, 30)),
            "spacepoint_table_hit_id_v": list(range(30, 60)),
            "spacepoint_table_hit_id_y": list(range(60, 90)),
        }
    ]
}

out.write_text(json.dumps(payload), encoding="utf-8")
print(out)
PY
```

Measured results from a keep-alive sweep on June 23, 2026, using
`tritonserver_26.05-py3-sdk.sif`, concurrency `1`, and batch size `1`.
Random inputs were used except for `nugraph2`, which used the structured JSON
payload above:

| Model | Throughput | Avg client latency |
| --- | ---: | ---: |
| `snbamsc_2dcnn_u` | 431.884 infer/sec | 2030 usec |
| `snbamsc_2dcnn_v` | 425.053 infer/sec | 2045 usec |
| `snbamsc_2dcnn_z` | 644.279 infer/sec | 1413 usec |
| `DoubleMetricLearning` | 2858.04 infer/sec | 342 usec |
| `higgsInteractionNet` | 1180.36 infer/sec | 827 usec |
| `particlenet_AK4_PT` | 433.956 infer/sec | 2281 usec |
| `nugraph2` | 14.2143 infer/sec | 70504 usec |
