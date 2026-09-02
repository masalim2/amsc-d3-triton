"""Globus Compute worker for the Triton gRPC relay, with readiness gating.

Register ``triton_rpc`` with Globus Compute.  Everything the function needs is
imported inside it (Globus Compute serializes the function body, not the
module), except the module-level state below which persists for the life of a
worker process and is re-created after a respawn.

Contract:
    triton_rpc(rpc: str, request_bytes: bytes, ready_timeout: float = 300.0) -> bytes
    Raises RuntimeError("TRITON_GRPC_ERROR <StatusCode.name>: <details>") on a
    Triton-side failure so the gateway can forward the status to the sidecar.
"""

from __future__ import annotations

import threading
import time

import grpc
from tritonclient.grpc import service_pb2 as pb
from tritonclient.grpc import service_pb2_grpc

MODEL_PORTS: dict[str, int] = {
    "snbamsc_2dcnn_u": 8501,
    "snbamsc_2dcnn_v": 8511,
    "snbamsc_2dcnn_z": 8521,
    "DoubleMetricLearning": 8531,
    "higgsInteractionNet": 8541,
    "particlenet_AK4_PT": 8551,
    "nugraph2": 8561,
}
DEFAULT_PORT = 8501  # target for server-level RPCs that name no model

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

# RPCs that need the model loaded.  Probes (ServerLive/ServerReady/ModelReady)
# are deliberately NOT gated: they must answer truthfully and immediately so a
# client can poll them.
GATED_RPCS = {"ModelInfer", "ModelMetadata", "ModelConfig", "ModelStatistics"}

# Errors that mean "not yet" rather than "no".  Add NOT_FOUND here only if you
# run Triton in explicit/poll model-control mode, where the gRPC port can be up
# before a model is registered; in the default mode Triton loads the repository
# before it starts listening, so NOT_FOUND after the port is up is a real error.
TRANSIENT = {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED}

# Process-lifetime state: one channel per port, and the set of models already
# observed ready so steady-state calls skip the probe entirely.
_lock = threading.Lock()
_channels: dict[int, grpc.Channel] = {}
_ready: set[tuple[int, str, str]] = set()


def _channel(port: int) -> grpc.Channel:
    with _lock:
        ch = _channels.get(port)
        if ch is None:
            ch = _channels[port] = grpc.insecure_channel(f"localhost:{port}")
        return ch


def _triton_error(code: grpc.StatusCode, details: str) -> RuntimeError:
    return RuntimeError(f"TRITON_GRPC_ERROR {code.name}: {details}")


def _wait_model_ready(port: int, name: str, version: str, timeout: float) -> None:
    key = (port, name, version)
    if key in _ready:
        return

    deadline = time.monotonic() + timeout
    channel = _channel(port)

    # Phase 1: the Triton process may not be listening yet.
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
    except grpc.FutureTimeoutError:
        raise _triton_error(
            grpc.StatusCode.UNAVAILABLE,
            f"nothing listening on localhost:{port} after {timeout:.0f}s",
        )

    # Phase 2: the process is up but the model may still be loading.
    stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)
    probe = pb.ModelReadyRequest(name=name, version=version)
    last = "never probed"
    while True:
        try:
            if stub.ModelReady(probe, timeout=5.0).ready:
                with _lock:
                    _ready.add(key)
                return
            last = "ready=false"
        except grpc.RpcError as exc:
            if exc.code() not in TRANSIENT:
                raise _triton_error(exc.code(), exc.details())
            last = exc.code().name
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _triton_error(
                grpc.StatusCode.UNAVAILABLE,
                f"model {name!r} on localhost:{port} not ready after "
                f"{timeout:.0f}s (last probe: {last})",
            )
        time.sleep(min(1.0, remaining))


def triton_rpc(data) -> bytes:
    """Relay one unary Triton RPC to the local server that owns the model."""
    try:
        assert isinstance(data, dict)
        rpc = data["rpc"]
        request_bytes = data["request_bytes"]
        ready_timeout = data.get("ready_timeout", 300.0)
    except (KeyError, AssertionError):
        raise _triton_error(grpc.StatusCode.INVALID_ARGUMENT, f"Bad GlobusCompute data payload: must receive dict with 'rpc' and 'request_bytes' keys.")

    try:
        req_cls = REQUEST_TYPES[rpc]
    except KeyError:
        raise _triton_error(grpc.StatusCode.UNIMPLEMENTED, f"rpc {rpc!r} is not relayed")
    req = req_cls.FromString(request_bytes)

    # ModelInferRequest uses model_name/model_version; the rest use name/version.
    name = getattr(req, "model_name", None) or getattr(req, "name", None) or ""
    version = getattr(req, "model_version", None) or getattr(req, "version", None) or ""
    port = MODEL_PORTS.get(name, DEFAULT_PORT)

    if rpc in GATED_RPCS:
        if name and name not in MODEL_PORTS:
            raise _triton_error(grpc.StatusCode.NOT_FOUND, f"unknown model {name!r}")
        _wait_model_ready(port, name, version, ready_timeout)

    stub = service_pb2_grpc.GRPCInferenceServiceStub(_channel(port))
    try:
        return getattr(stub, rpc)(req).SerializeToString()
    except grpc.RpcError as exc:
        if exc.code() is grpc.StatusCode.UNAVAILABLE:
            # Triton went away under us; make the next call re-probe.
            with _lock:
                _ready.discard((port, name, version))
        raise _triton_error(exc.code(), exc.details())
