## Environment Setup
- Run `uv pip install -e.` to install the Triton model client and dependencies.  This should be installed in the same venv as the Globus Compute Endpoint serving the Triton models.
- `python register_globus_compute_func.py` -- registers the GC client function that invokes the local triton server.
- `user_config_template.yaml.j2` -- Globus Compute Endpoint configuration for Sophia.  Starts the models in the background on the designated ports. This is copied into your actual GC endpoint.

## Triton Client
- `triton_hep_client.py` -- implements the `triton_rpc` function that wraps each gRPC inference request.
This is invoked by the Globus Compute worker running on the compute node: the Triton models are contacted directly at `localhost:{grpc_port}`.

## Local gRPC Sidecar

A local proxy gRPC server that runs on the end user's machine, alongside the Triton client application.
This presents a true gRPC server to local applications and handles the upstream translation of unary Triton inference requests into ALCF inference gateway service calls.

```bash
export INFERENCE_TOKEN=$(uvx alcf-ai auth get-access-token)
python triton_sidecar.py --listen 127.0.0.1:8001
```

Then, point your gRPC client directly at the sidecar:

```cpp
InferenceServerGrpcClient::Create(&client, "localhost:8001")
```