#!/usr/bin/env python3
"""Local gRPC sidecar for the Globus-Compute-fronted Triton deployment.

Presents Triton's ``GRPCInferenceService`` on localhost so an unmodified
Triton gRPC client (C++ or Python) can point at it as if it were the server.
Each unary RPC is serialized to its protobuf bytes, POSTed to the REST
gateway, executed on the remote endpoint, and the ``*Response`` bytes are
returned to the caller.  ``ModelStreamInfer`` is supported by unrolling the
stream into sequential unary relays.

Wire contract with the gateway (see triton_rpc_views.py):

  POST {gateway}/rpc/{rpc}            body: serialized {rpc}Request
      Content-Type: application/x-protobuf
      Authorization: Bearer $INFERENCE_TOKEN
      -> 200 JSON {"task_id": "..."}   (any extra fields are ignored)

  GET  {gateway}/rpc/result/{task_id}
      -> 202                            task still pending (Retry-After honored)
      -> 200 application/x-protobuf     serialized {rpc}Response
      -> 4xx/5xx JSON {"grpc_code": "<grpc.StatusCode name>", "details": "..."}

Usage:
    export INFERENCE_TOKEN=...
    python triton_sidecar.py --listen 127.0.0.1:8001

Then in the user's code:  InferenceServerGrpcClient::Create(&client, "localhost:8001")
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from concurrent import futures
from typing import Callable

import grpc
import requests
from requests.adapters import HTTPAdapter
from tritonclient.grpc import service_pb2, service_pb2_grpc
from urllib3.util.retry import Retry

log = logging.getLogger("triton_sidecar")

PROTOBUF_CONTENT_TYPES = {"application/x-protobuf", "application/octet-stream"}

# RPC name -> response class.  Only read-only, unary RPCs are relayed.
# Everything else (repository load/unload, shared-memory, trace/log settings)
# falls through to the base servicer, which aborts with UNIMPLEMENTED.
UNARY_RPCS: dict[str, type] = {
    "ServerLive": service_pb2.ServerLiveResponse,
    "ServerReady": service_pb2.ServerReadyResponse,
    "ServerMetadata": service_pb2.ServerMetadataResponse,
    "ModelReady": service_pb2.ModelReadyResponse,
    "ModelMetadata": service_pb2.ModelMetadataResponse,
    "ModelConfig": service_pb2.ModelConfigResponse,
    "ModelStatistics": service_pb2.ModelStatisticsResponse,
    "ModelInfer": service_pb2.ModelInferResponse,
}

# Responses that don't change while a model is loaded: cache them so the
# client's metadata handshake doesn't cost a Globus Compute round trip each time.
CACHED_RPCS = {"ServerMetadata", "ModelMetadata", "ModelConfig"}

HTTP_TO_GRPC: dict[int, grpc.StatusCode] = {
    400: grpc.StatusCode.INVALID_ARGUMENT,
    401: grpc.StatusCode.UNAUTHENTICATED,
    403: grpc.StatusCode.PERMISSION_DENIED,
    404: grpc.StatusCode.NOT_FOUND,
    408: grpc.StatusCode.DEADLINE_EXCEEDED,
    413: grpc.StatusCode.RESOURCE_EXHAUSTED,
    415: grpc.StatusCode.INVALID_ARGUMENT,
    429: grpc.StatusCode.RESOURCE_EXHAUSTED,
    500: grpc.StatusCode.INTERNAL,
    502: grpc.StatusCode.UNAVAILABLE,
    503: grpc.StatusCode.UNAVAILABLE,
    504: grpc.StatusCode.DEADLINE_EXCEEDED,
}


class RelayError(Exception):
    """A failure that should surface to the gRPC caller as a status code."""

    def __init__(self, code: grpc.StatusCode, details: str):
        super().__init__(f"{code.name}: {details}")
        self.code = code
        self.details = details


# --------------------------------------------------------------------------- #
# Gateway client
# --------------------------------------------------------------------------- #


class Gateway:
    """Submit-then-poll client for the REST gateway."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        http_timeout: float = 30.0,
        poll_interval: float = 0.2,
        max_poll_interval: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.http_timeout = http_timeout
        self.poll_interval = poll_interval
        self.max_poll_interval = max_poll_interval

        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"
        self.session.headers["Accept"] = "application/x-protobuf, application/json"
        # Retry only idempotent GETs on transient upstream errors; never retry
        # the POST, since that would double-submit the task.
        retry = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=(502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.mount("http://", HTTPAdapter(max_retries=retry))

    # -- public --

    def call(
        self,
        rpc: str,
        body: bytes,
        *,
        budget: float | None,
        cancelled: Callable[[], bool],
    ) -> bytes:
        """Relay one unary RPC.  ``budget`` is seconds allowed end-to-end."""
        deadline = None if budget is None else time.monotonic() + budget
        task_id = self._submit(rpc, body)
        return self._wait(task_id, deadline, cancelled)

    # -- internals --

    def _submit(self, rpc: str, body: bytes) -> str:
        try:
            resp = self.session.post(
                f"{self.base_url}/rpc/{rpc}",
                data=body,
                headers={"Content-Type": "application/x-protobuf"},
                timeout=self.http_timeout,
            )
        except requests.RequestException as exc:
            raise RelayError(grpc.StatusCode.UNAVAILABLE, f"gateway unreachable: {exc}")

        if resp.status_code != 200:
            raise self._error_from_response(resp)
        try:
            task_id = resp.json()["task_id"]
        except (ValueError, KeyError, TypeError):
            raise RelayError(
                grpc.StatusCode.INTERNAL,
                f"gateway submit returned unexpected body: {resp.text[:200]!r}",
            )
        return str(task_id)

    def _wait(
        self, task_id: str, deadline: float | None, cancelled: Callable[[], bool]
    ) -> bytes:
        interval = self.poll_interval
        url = f"{self.base_url}/rpc/result/{task_id}"
        while True:
            if cancelled():
                raise RelayError(grpc.StatusCode.CANCELLED, "client cancelled")
            if deadline is not None and time.monotonic() >= deadline:
                raise RelayError(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    f"task {task_id} did not complete within the client deadline",
                )
            try:
                resp = self.session.get(url, timeout=self.http_timeout)
            except requests.RequestException as exc:
                raise RelayError(grpc.StatusCode.UNAVAILABLE, f"gateway unreachable: {exc}")

            if resp.status_code == 200:
                ctype = resp.headers.get("Content-Type", "").split(";")[0].strip()
                if ctype not in PROTOBUF_CONTENT_TYPES:
                    raise RelayError(
                        grpc.StatusCode.INTERNAL,
                        f"gateway result had Content-Type {ctype!r}, expected protobuf",
                    )
                return resp.content
            if resp.status_code == 202:
                sleep_for = interval
                if deadline is not None:
                    sleep_for = min(sleep_for, max(0.0, deadline - time.monotonic()))
                time.sleep(sleep_for)
                interval = min(interval * 1.5, self.max_poll_interval)
                continue
            raise self._error_from_response(resp)

    @staticmethod
    def _error_from_response(resp: requests.Response) -> RelayError:
        # Preferred: gateway forwards the Triton status as {"grpc_code", "details"}.
        try:
            payload = resp.json()
            code = grpc.StatusCode[payload["grpc_code"]]
            return RelayError(code, str(payload.get("details", "")))
        except Exception:
            pass
        code = HTTP_TO_GRPC.get(resp.status_code, grpc.StatusCode.UNKNOWN)
        return RelayError(code, f"gateway HTTP {resp.status_code}: {resp.text[:300]}")


# --------------------------------------------------------------------------- #
# gRPC servicer
# --------------------------------------------------------------------------- #


class _TTLCache:
    def __init__(self, ttl: float):
        self.ttl = ttl
        self._data: dict[tuple, tuple[float, bytes]] = {}
        self._lock = threading.Lock()

    def get(self, key: tuple) -> bytes | None:
        if self.ttl <= 0:
            return None
        with self._lock:
            hit = self._data.get(key)
            if hit and time.monotonic() - hit[0] < self.ttl:
                return hit[1]
            self._data.pop(key, None)
            return None

    def put(self, key: tuple, value: bytes) -> None:
        if self.ttl <= 0:
            return
        with self._lock:
            self._data[key] = (time.monotonic(), value)


class RelayServicer(service_pb2_grpc.GRPCInferenceServiceServicer):
    def __init__(
        self,
        gateway: Gateway,
        *,
        max_payload: int,
        default_timeout: float,
        cache_ttl: float,
    ):
        self.gateway = gateway
        self.max_payload = max_payload
        self.default_timeout = default_timeout
        self.cache = _TTLCache(cache_ttl)

    # -- relay core --

    def _relay_bytes(self, rpc: str, request, context: grpc.ServicerContext) -> bytes:
        body = request.SerializeToString()
        if len(body) > self.max_payload:
            raise RelayError(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"{rpc} request is {len(body)} bytes; the relay accepts at most "
                f"{self.max_payload}. Split the batch or use the file-transfer path.",
            )

        key = (rpc, body) if rpc in CACHED_RPCS else None
        if key is not None:
            cached = self.cache.get(key)
            if cached is not None:
                log.debug("%s served from cache", rpc)
                return cached

        # Honor the caller's gRPC deadline if it set one; otherwise use ours.
        remaining = context.time_remaining()
        budget = self.default_timeout if remaining is None else min(remaining, self.default_timeout)

        t0 = time.monotonic()
        raw = self.gateway.call(rpc, body, budget=budget, cancelled=lambda: not context.is_active())
        log.info(
            "%s ok: %d B in, %d B out, %.2fs",
            rpc, len(body), len(raw), time.monotonic() - t0,
        )
        if key is not None:
            self.cache.put(key, raw)
        return raw

    def _relay(self, rpc: str, request, context: grpc.ServicerContext):
        try:
            raw = self._relay_bytes(rpc, request, context)
        except RelayError as exc:
            log.warning("%s failed: %s", rpc, exc)
            context.abort(exc.code, exc.details)
        except Exception as exc:  # defensive: never leak a traceback as UNKNOWN
            log.exception("%s crashed", rpc)
            context.abort(grpc.StatusCode.INTERNAL, f"sidecar error: {exc}")
        return UNARY_RPCS[rpc].FromString(raw)

    # -- unary RPCs --

    def ServerLive(self, request, context):
        return self._relay("ServerLive", request, context)

    def ServerReady(self, request, context):
        return self._relay("ServerReady", request, context)

    def ServerMetadata(self, request, context):
        return self._relay("ServerMetadata", request, context)

    def ModelReady(self, request, context):
        return self._relay("ModelReady", request, context)

    def ModelMetadata(self, request, context):
        return self._relay("ModelMetadata", request, context)

    def ModelConfig(self, request, context):
        return self._relay("ModelConfig", request, context)

    def ModelStatistics(self, request, context):
        return self._relay("ModelStatistics", request, context)

    def ModelInfer(self, request, context):
        return self._relay("ModelInfer", request, context)

    # -- streaming: unrolled into sequential unary relays --

    def ModelStreamInfer(self, request_iterator, context):
        # Triton reports per-request failures on the stream via error_message
        # rather than terminating the stream, so we do the same.
        for request in request_iterator:
            try:
                raw = self._relay_bytes("ModelInfer", request, context)
                yield service_pb2.ModelStreamInferResponse(
                    infer_response=service_pb2.ModelInferResponse.FromString(raw)
                )
            except RelayError as exc:
                log.warning("ModelStreamInfer item failed: %s", exc)
                yield service_pb2.ModelStreamInferResponse(
                    error_message=f"{exc.code.name}: {exc.details}"
                )
            if not context.is_active():
                return


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--listen", default="127.0.0.1:8001",
        help="host:port to serve gRPC on (default: %(default)s; bind 127.0.0.1 "
             "unless you intend to share your token with everyone on the host)",
    )
    p.add_argument(
        "--gateway", default="https://inference-api.alcf.anl.gov/resource_server/sophia/triton/amsc-d3",
        help="gateway base URL, e.g. https://inference-api.alcf.anl.gov/resource_server/sophia/triton/amsc-d3"
    )
    p.add_argument("--max-workers", type=int, default=8, help="concurrent in-flight RPCs")
    p.add_argument(
        "--timeout", type=float, default=600.0,
        help="seconds to wait for a task when the client sets no deadline",
    )
    p.add_argument("--poll-interval", type=float, default=0.2, help="initial result poll interval")
    p.add_argument(
        "--max-payload", type=int, default=9_900_000,
        help="reject requests larger than this many serialized bytes "
             "(Globus Compute caps task payloads at 10 MB)",
    )
    p.add_argument("--cache-ttl", type=float, default=300.0, help="metadata/config cache TTL (0 disables)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        token = os.environ["INFERENCE_TOKEN"]
    except KeyError:
        log.error("INFERENCE_TOKEN is not set")
        return 2
    if not args.gateway:
        log.error("no gateway URL: pass --gateway or set INFERENCE_GATEWAY_URL")
        return 2

    gateway = Gateway(args.gateway, token, poll_interval=args.poll_interval)
    servicer = RelayServicer(
        gateway,
        max_payload=args.max_payload,
        default_timeout=args.timeout,
        cache_ttl=args.cache_ttl,
    )

    # Accept messages larger than our own limit so the size check above
    # produces a useful error instead of gRPC's generic one.
    msg_limit = max(64 << 20, args.max_payload * 2)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.max_workers),
        options=[
            ("grpc.max_receive_message_length", msg_limit),
            ("grpc.max_send_message_length", msg_limit),
        ],
    )
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(servicer, server)
    bound = server.add_insecure_port(args.listen)
    if bound == 0:
        log.error("could not bind %s", args.listen)
        return 1

    stop = threading.Event()

    def _on_signal(signum, _frame):
        log.info("received %s, draining", signal.Signals(signum).name)
        stop.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    server.start()
    log.info("Triton relay listening on %s -> %s", args.listen, gateway.base_url)
    stop.wait()
    server.stop(grace=30).wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
