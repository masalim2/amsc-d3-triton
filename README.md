## Compute Environment Setup

On the HPC compute cluster:

- Run `uv pip install -e.` to install the `triton-relay` entrypoint and its dependencies.

Each Triton model is a FIRST deployment (see the `sophia/triton` LaunchTemplate in gateway-api-fixtures): one `tritonserver` plus one `triton-relay`.  The relay listens on the replica's Unix domain socket, forwards the unary gRPC calls arriving through the gateway's generic tasks route to the local Triton server, and reports the replica healthy once Triton has the model ready.  Triton itself listens only on a gRPC Unix socket in a private directory, with HTTP, metrics and the model load/unload API turned off, so the relay is its only way in, and the relay pins every request to its replica's model.

## Local gRPC Sidecar

On the user side, wherever you are running the gRPC client application that connects to the Triton inference server: 

- Run `uv pip install -e ".[sidecar]"` to install the dependencies.  (The `sidecar` extra adds `alcf-ai`, which requires Python 3.12+.)

Run `triton_sidecar.py`: a local proxy gRPC server that forwards gRPC requests to the upstream inference gateway.  A Triton model `nugraph2` is the catalog model `amsc-d3/nugraph2`.  

```bash
uvx alcf-ai auth login   # once every 30 days
python triton_sidecar.py --listen 127.0.0.1:8001
```

The sidecar authenticates with the `AutoGlobusAuth` flow from the `alcf_ai` SDK: it mints an access token from the Globus refresh token on your machine and refreshes it as needed.  Access tokens expire after 2 hours, so there is no `INFERENCE_TOKEN` to export -- a sidecar can stay up for the 30-day life of the refresh token.

Then, point your gRPC client directly at the sidecar:

```cpp
InferenceServerGrpcClient::Create(&client, "localhost:8001")
```
