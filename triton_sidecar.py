#!/usr/bin/env python3
"""Local gRPC sidecar for the Triton models served by the FIRST gateway.

Presents Triton's ``GRPCInferenceService`` on localhost so an unmodified
Triton gRPC client (C++ or Python) can point at it as if it were the server.
Each unary RPC is serialized to its protobuf bytes and POSTed to the gateway's
generic tasks route, which routes on the Triton model's name, and the
``*Response`` bytes are returned to the caller.  ``ModelStreamInfer`` is supported
by unrolling the stream into sequential unary relays.

Wire contract (see triton_relay.py, which serves the far end):

  POST {gateway}/tasks
      Authorization: Bearer <Globus access token>
      {"model": "{model_prefix}/{triton model}", "endpoint": "triton-rpc",
       "rpc": "<rpc>", "request_bytes": "<base64 {rpc}Request>"}
      -> 200 {"grpc_code": "OK", "response_bytes": "<base64 {rpc}Response>"}
      -> 200 {"grpc_code": "<grpc.StatusCode name>", "details": "..."}
      -> 4xx/5xx from the gateway, with Retry-After on 429 and cold-start 503

RPCs that name no model (ServerLive, ServerReady, ServerMetadata) go to
``--default-model``.

Authentication is handled by ``alcf_ai``'s ``AutoGlobusAuth`` flow: every
request derives a bearer token from the Globus refresh token stored under
``~/.globus`` by ``alcf-ai auth login``.  The access token is good for two
hours and the refresh token for thirty days, so a sidecar left running
overnight keeps working without anyone re-exporting a token.

Usage:
    uvx alcf-ai auth login        # once every 30 days
    python triton_sidecar.py --listen 127.0.0.1:8001

Then in the user's code:  InferenceServerGrpcClient::Create(&client, "localhost:8001")
"""

from __future__ import annotations

import argparse
import base64
import logging
import signal
import sys
import threading
import time
from concurrent import futures
from typing import Callable, Generator

import grpc
import httpx
from alcf_ai.client import AutoGlobusAuth
from tritonclient.grpc import service_pb2, service_pb2_grpc

log = logging.getLogger("triton_sidecar")

TASK_ENDPOINT = "triton-rpc"

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
# client's metadata handshake doesn't cost a gateway round trip each time.
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

# The model is busy or starting: wait for it.
RETRYABLE_STATUS = frozenset({429, 503})


class RelayError(Exception):
    """A failure that should surface to the gRPC caller as a status code."""

    def __init__(self, code: grpc.StatusCode, details: str):
        super().__init__(f"{code.name}: {details}")
        self.code = code
        self.details = details


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #


class SidecarGlobusAuth(AutoGlobusAuth):
    def __init__(self, cache_seconds: float = 60.0):
        self.cache_seconds = cache_seconds
        self._lock = threading.Lock()
        self._header: str | None = None
        self._fetched_at = 0.0

    def _authorize(self, request: httpx.Request, *, force: bool) -> None:
        with self._lock:
            if (
                self._header is not None
                and not force
                and time.monotonic() - self._fetched_at < self.cache_seconds
            ):
                request.headers["Authorization"] = self._header
                return
            # AutoGlobusAuth.auth_flow() refreshes if needed and stamps the
            # header on the request; we only want that first step here.
            flow = super().auth_flow(request)
            try:
                next(flow)
            finally:
                flow.close()
            self._header = request.headers["Authorization"]
            self._fetched_at = time.monotonic()

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        self._authorize(request, force=False)
        response = yield request
        if response.status_code != 401:
            return
        # Refresh once and give the request another chance
        log.warning("gateway returned 401, refreshing the access token")
        self._authorize(request, force=True)
        yield request

    def prime(self) -> None:
        """Acquire a token now, so that a required Globus login happens at startup"""
        self._authorize(httpx.Request("GET", "https://auth.globus.org"), force=True)


# --------------------------------------------------------------------------- #
# Gateway client
# --------------------------------------------------------------------------- #


class Gateway:
    """Client for the gateway's generic tasks route."""

    def __init__(
        self,
        base_url: str,
        auth: httpx.Auth,
        *,
        model_prefix: str,
        default_model: str,
        http_timeout: float = 30.0,
        max_retry_after: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_prefix = model_prefix
        self.default_model = default_model
        self.http_timeout = http_timeout
        self.max_retry_after = max_retry_after

        # Transport-level retries cover connections that were never
        # established, so they can't double-submit
        self.client = httpx.Client(
            auth=auth,
            transport=httpx.HTTPTransport(retries=3),
        )

    def close(self) -> None:
        self.client.close()

    def call(
        self,
        rpc: str,
        model: str,
        body: bytes,
        *,
        budget: float,
        cancelled: Callable[[], bool],
    ) -> bytes:
        """Relay one unary RPC.  ``budget`` is seconds allowed end-to-end."""
        deadline = time.monotonic() + budget
        payload = {
            "model": f"{self.model_prefix}/{model or self.default_model}",
            "endpoint": TASK_ENDPOINT,
            "rpc": rpc,
            "request_bytes": base64.b64encode(body).decode(),
        }
        while True:
            remaining = deadline - time.monotonic()
            try:
                resp = self.client.post(
                    f"{self.base_url}/tasks",
                    json=payload,
                    timeout=httpx.Timeout(self.http_timeout, read=max(remaining, 0.1)),
                )
            except httpx.TimeoutException:
                raise RelayError(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    f"{rpc} did not complete within the deadline",
                )
            except httpx.HTTPError as exc:
                raise RelayError(grpc.StatusCode.UNAVAILABLE, f"gateway unreachable: {exc}")

            if resp.status_code == 200:
                return self._parse_result(resp)
            if resp.status_code not in RETRYABLE_STATUS:
                raise self._error_from_response(resp)

            wait = self._retry_after(resp)
            if time.monotonic() + wait >= deadline:
                raise self._error_from_response(resp)
            log.info("gateway HTTP %d for %s; retrying in %.1fs", resp.status_code, rpc, wait)
            time.sleep(wait)
            if cancelled():
                raise RelayError(grpc.StatusCode.CANCELLED, "client cancelled")

    @staticmethod
    def _parse_result(resp: httpx.Response) -> bytes:
        try:
            result = resp.json()
            code = grpc.StatusCode[result["grpc_code"]]
            if code is grpc.StatusCode.OK:
                return base64.b64decode(result["response_bytes"], validate=True)
        except (ValueError, KeyError, TypeError) as exc:
            raise RelayError(
                grpc.StatusCode.INTERNAL,
                f"gateway returned an unexpected body: {exc!r}",
            )
        raise RelayError(code, str(result.get("details", "")))

    def _retry_after(self, resp: httpx.Response) -> float:
        """Honor Retry-After, clamped to something sane."""
        try:
            return min(max(float(resp.headers["Retry-After"]), 0.0), self.max_retry_after)
        except (KeyError, ValueError):  # missing, or HTTP-date form
            return 1.0

    @staticmethod
    def _error_from_response(resp: httpx.Response) -> RelayError:
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
        # ModelInferRequest uses model_name; the model-scoped rest use name.
        model = getattr(request, "model_name", None) or getattr(request, "name", "")
        raw = self.gateway.call(
            rpc, model, body, budget=budget, cancelled=lambda: not context.is_active()
        )
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
        "--gateway", default="https://inference-api.alcf.anl.gov/federated/v1",
        help="gateway base URL (default: %(default)s)",
    )
    p.add_argument(
        "--model-prefix", default="amsc-d3",
        help="catalog name of a Triton model is {prefix}/{name} (default: %(default)s)",
    )
    p.add_argument(
        "--default-model", default="snbamsc_2dcnn_u",
        help="Triton model that answers RPCs naming no model (default: %(default)s)",
    )
    p.add_argument("--max-workers", type=int, default=8, help="concurrent in-flight RPCs")
    p.add_argument(
        "--timeout", type=float, default=600.0,
        help="seconds to wait for a task when the client sets no deadline",
    )
    p.add_argument(
        "--max-payload", type=int, default=11_900_000,
        help="reject requests larger than this many serialized bytes "
             "(the gateway caps bodies at 16 MB, and base64 adds a third)",
    )
    p.add_argument("--cache-ttl", type=float, default=300.0, help="metadata/config cache TTL (0 disables)")
    p.add_argument(
        "--token-cache", type=float, default=60.0,
        help="seconds to reuse a Globus access token before re-checking it",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    level = args.log_level.upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # httpx logs a line per request; we already log one per RPC.
    # Keep it for --log-level DEBUG only.
    if level != "DEBUG":
        logging.getLogger("httpx").setLevel(logging.WARNING)

    if not args.gateway:
        log.error("no gateway URL: pass --gateway")
        return 2

    auth = SidecarGlobusAuth(cache_seconds=args.token_cache)
    try:
        auth.prime()
    except Exception as exc:
        log.error("could not obtain a Globus access token: %s", exc)
        log.error("log in with `uvx alcf-ai auth login`, then try again")
        return 2

    gateway = Gateway(
        args.gateway,
        auth,
        model_prefix=args.model_prefix,
        default_model=args.default_model,
    )
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
    gateway.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
