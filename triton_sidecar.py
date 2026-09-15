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
      Authorization: Bearer <Globus access token>
      -> 200 JSON {"task_id": "..."}   (any extra fields are ignored)

  GET  {gateway}/rpc/result/{task_id}
      -> 202                            task still pending
      -> 200 application/x-protobuf     serialized {rpc}Response
      -> 4xx/5xx JSON {"grpc_code": "<grpc.StatusCode name>", "details": "..."}

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

# Upstream hiccups worth retrying on the (idempotent) result GET.
RETRYABLE_STATUS = frozenset({502, 503, 504})


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
    """Submit-then-poll client for the REST gateway."""

    def __init__(
        self,
        base_url: str,
        auth: httpx.Auth,
        *,
        http_timeout: float = 30.0,
        poll_interval: float = 0.2,
        max_poll_interval: float = 2.0,
        max_transient_retries: int = 5,
    ):
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.max_poll_interval = max_poll_interval
        self.max_transient_retries = max_transient_retries

        # Transport-level retries cover connections that were never
        # established, so they can't double-submit
        self.client = httpx.Client(
            auth=auth,
            timeout=httpx.Timeout(http_timeout),
            headers={"Accept": "application/x-protobuf, application/json"},
            transport=httpx.HTTPTransport(retries=3),
            follow_redirects=True,  # requests did; APPEND_SLASH & friends
        )

    def close(self) -> None:
        self.client.close()

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

    def _submit(self, rpc: str, body: bytes) -> str:
        try:
            resp = self.client.post(
                f"{self.base_url}/rpc/{rpc}",
                content=body,
                headers={"Content-Type": "application/x-protobuf"},
            )
        except httpx.HTTPError as exc:
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
        transient = 0
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
                resp = self.client.get(url)
            except httpx.HTTPError as exc:
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
                # Deliberately ignoring Retry-After here: the gateway sends a
                # flat 2s on every pending task, which would floor each RPC at
                # 2s.  Our own backoff starts far tighter and converges on it.
                self._sleep(interval, deadline)
                interval = min(interval * 1.5, self.max_poll_interval)
                continue
            if resp.status_code in RETRYABLE_STATUS and transient < self.max_transient_retries:
                transient += 1
                log.warning(
                    "gateway HTTP %d polling task %s (retry %d/%d)",
                    resp.status_code, task_id, transient, self.max_transient_retries,
                )
                self._sleep(self._retry_after(resp, 0.5 * 2 ** transient), deadline)
                continue
            raise self._error_from_response(resp)

    @staticmethod
    def _sleep(seconds: float, deadline: float | None) -> None:
        if deadline is not None:
            seconds = min(seconds, max(0.0, deadline - time.monotonic()))
        time.sleep(seconds)

    @staticmethod
    def _retry_after(resp: httpx.Response, default: float) -> float:
        """Honor a Retry-After sent with an error, clamped to something sane."""
        raw = resp.headers.get("Retry-After")
        if raw is None:
            return default
        try:
            return min(max(float(raw), 0.0), 30.0)
        except ValueError:  # HTTP-date form; we don't need that precision
            return default

    @staticmethod
    def _error_from_response(resp: httpx.Response) -> RelayError:
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
    # httpx logs a line per request; we already log one per RPC, and each RPC
    # is a POST plus N polls.  Keep it for --log-level DEBUG only.
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

    gateway = Gateway(args.gateway, auth, poll_interval=args.poll_interval)
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
