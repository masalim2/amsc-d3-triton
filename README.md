## Compute Environment Setup

On the HPC compute cluster: 

- Run `uv pip install -e.` to install the Triton model client and dependencies.  This should be installed in the same venv as the Globus Compute Endpoint serving the Triton models.
- `python register_globus_compute_func.py` -- registers the GC client function that invokes the local triton server.
- `user_config_template.yaml.j2` -- Globus Compute Endpoint configuration for Sophia.  Starts the models in the background on the designated ports. This is copied into your actual GC endpoint.

Globus Compute invokes the `triton_rpc` function in `triton_hep_client.py`.  This takes the raw protobuf payload that was relayed from the inference gateway and sends it to the Triton server on a Sophia compute node. 

## Local gRPC Sidecar

On the user side, wherever you are running the gRPC client application that connects to the Triton inference server: 

- Run `uv pip install -e.` to install the  dependencies.  (Globus Compute Endpoint is not necessary)

Run `triton_sidecar.py`: a local proxy gRPC server that forwards gRPC requests from to the upstream inference gateway.  

```bash
export INFERENCE_TOKEN=$(uvx alcf-ai auth get-access-token)
python triton_sidecar.py --listen 127.0.0.1:8001
```

Then, point your gRPC client directly at the sidecar:

```cpp
InferenceServerGrpcClient::Create(&client, "localhost:8001")
```
