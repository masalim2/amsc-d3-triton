"""HTTP relay between the FIRST gateway and one local Triton gRPC server.

Triton speaks only gRPC (here on a private Unix socket), and the gateway speaks
JSON over HTTP, so each Triton model runs behind one of these.  The sidecar's unary RPCs
arrive as serialized protobuf, base64-encoded in the generic tasks route's JSON:

  POST /v1/triton-rpc   {"rpc": "ModelInfer", "request_bytes": "<base64>"}
      -> 200 {"grpc_code": "OK", "response_bytes": "<base64>"}
      -> 200 {"grpc_code": "<grpc.StatusCode name>", "details": "..."}
      -> 503 Triton is unreachable, so the gateway may try another backend

As in gRPC itself, a Triton-side failure is a successful transport carrying an
error status.  The gateway would otherwise rewrite the status on the way through.

Every request is pinned to this replica's model: the gateway routed it here by
catalog name, so the model named inside the protobuf is overwritten, not trusted.

GET /health is healthy once Triton reports the model ready, which is what keeps
the gateway from routing requests here while the model loads.

Usage:
    triton-relay --uds /tmp/replica.sock --grpc-target unix:/tmp/x/triton.sock:8001 \
        --model nugraph2
"""

from __future__ import annotations

import argparse
import base64
import binascii
from typing import Annotated

import grpc
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from tritonclient.grpc import service_pb2 as pb
from tritonclient.grpc import service_pb2_grpc

REQUEST_TYPES = {
    "ServerLive": pb.ServerLiveRequest,
    "ServerReady": pb.ServerReadyRequest,
    "ServerMetadata": pb.ServerMetadataRequest,
    "ModelReady": pb.ModelReadyRequest,
    "ModelMetadata": pb.ModelMetadataRequest,
    "ModelConfig": pb.ModelConfigRequest,
    "ModelStatistics": pb.ModelStatisticsRequest,
    "ModelInfer": pb.ModelInferRequest,
}

# Model-scoped requests name their model in one of these fields.
MODEL_FIELDS = ("model_name", "name")

HEALTH_TIMEOUT_SEC = 5.0


class RelayRequest(BaseModel):
    rpc: str
    request_bytes: str


def get_stub(request: Request) -> service_pb2_grpc.GRPCInferenceServiceStub:
    return request.app.state.stub


Stub = Annotated[service_pb2_grpc.GRPCInferenceServiceStub, Depends(get_stub)]


def create_app(grpc_target: str, model: str) -> FastAPI:
    app = FastAPI(title="Triton relay")
    channel = grpc.insecure_channel(
        grpc_target,
        options=[
            ("grpc.enable_http_proxy", 0),
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )
    app.state.stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    @app.get("/health")
    def health(stub: Stub) -> JSONResponse:
        try:
            probe = pb.ModelReadyRequest(name=model)
            ready = stub.ModelReady(probe, timeout=HEALTH_TIMEOUT_SEC).ready
        except grpc.RpcError:
            ready = False
        return JSONResponse(status_code=200 if ready else 503, content={"ready": ready})

    @app.post("/v1/triton-rpc")
    def triton_rpc(body: RelayRequest, stub: Stub) -> dict[str, str]:
        request_type = REQUEST_TYPES.get(body.rpc)
        if request_type is None:
            raise HTTPException(404, f"rpc {body.rpc!r} is not relayed")
        try:
            request = request_type.FromString(
                base64.b64decode(body.request_bytes, validate=True)
            )
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(400, f"bad request_bytes: {exc}")
        for field in MODEL_FIELDS:
            if field in request.DESCRIPTOR.fields_by_name:
                setattr(request, field, model)

        try:
            response = getattr(stub, body.rpc)(request)
        except grpc.RpcError as exc:
            if exc.code() is grpc.StatusCode.UNAVAILABLE:
                raise HTTPException(503, exc.details())
            return {"grpc_code": exc.code().name, "details": exc.details() or ""}
        encoded = base64.b64encode(response.SerializeToString()).decode()
        return {"grpc_code": "OK", "response_bytes": encoded}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--uds", required=True, help="Unix domain socket to listen on")
    parser.add_argument(
        "--grpc-target", required=True,
        help="Triton's gRPC address, e.g. unix:/path/triton.sock:8001",
    )
    parser.add_argument("--model", required=True, help="Triton model served by this replica")
    args = parser.parse_args()
    uvicorn.run(create_app(args.grpc_target, args.model), uds=args.uds)


if __name__ == "__main__":
    main()
