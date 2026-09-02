#!/usr/bin/env python3
"""End-to-end smoke test for the Triton relay sidecar.

Uses the stock ``tritonclient.grpc`` client pointed at the sidecar, exactly as
a user would, and runs one inference per model with synthetic inputs.  Per
model it reports the readiness/metadata round trip, each inference's
end-to-end latency, and the output tensors received.

    python test_relay_client.py --url localhost:8001 --repeat 2

The first inference of each model includes any server-side readiness wait and
the sidecar's metadata fetch; ``--repeat`` shows steady-state latency.
Exit status is non-zero if any model fails.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype, triton_to_np_dtype

# --------------------------------------------------------------------------- #
# Synthetic inputs (same shapes/dtypes the .npz path used, unbatched)
# --------------------------------------------------------------------------- #

Rng = np.random.Generator


def _nugraph2(rng: Rng) -> dict[str, np.ndarray]:
    n_hits, n_sp = 90, 30
    i = np.arange(n_hits)
    return {
        "hit_table_hit_id": i.astype(np.int64),
        "hit_table_local_plane": (i // (n_hits // 3)).astype(np.int64),
        "hit_table_local_time": ((i % 30) * 2 + (i // 30) * 0.1).astype(np.float32),
        "hit_table_local_wire": ((i % 30) * 3 + (i // 30)).astype(np.int64),
        "hit_table_integral": (100.0 + (i % 11)).astype(np.float32),
        "hit_table_rms": (2.0 + (i % 5) * 0.1).astype(np.float32),
        "spacepoint_table_spacepoint_id": np.arange(n_sp, dtype=np.int64),
        "spacepoint_table_hit_id_u": np.arange(0, n_sp, dtype=np.int64),
        "spacepoint_table_hit_id_v": np.arange(n_sp, 2 * n_sp, dtype=np.int64),
        "spacepoint_table_hit_id_y": np.arange(2 * n_sp, 3 * n_sp, dtype=np.int64),
    }


INPUT_GENERATORS: dict[str, Callable[[Rng], dict[str, np.ndarray]]] = {
    "snbamsc_2dcnn_u": lambda rng: {
        "zero_padding2d_input": rng.random((1149, 128, 1), dtype=np.float32),
    },
    "snbamsc_2dcnn_v": lambda rng: {
        "zero_padding2d_input": rng.random((1148, 128, 1), dtype=np.float32),
    },
    "snbamsc_2dcnn_z": lambda rng: {
        "zero_padding2d_1_input": rng.random((480, 128, 1), dtype=np.float32),
    },
    "DoubleMetricLearning": lambda rng: {
        "FEATURES": rng.random((1, 44), dtype=np.float32),
    },
    "higgsInteractionNet": lambda rng: {
        "input_cpf": rng.random((30, 60), dtype=np.float32),
        "input_sv": rng.random((14, 5), dtype=np.float32),
    },
    "particlenet_AK4_PT": lambda rng: {
        "pf_points__0": rng.random((2, 100), dtype=np.float32),
        "pf_features__1": rng.random((20, 100), dtype=np.float32),
        "pf_mask__2": rng.random((1, 100), dtype=np.float32),
        "sv_points__3": rng.random((2, 10), dtype=np.float32),
        "sv_features__4": rng.random((11, 10), dtype=np.float32),
        "sv_mask__5": rng.random((1, 10), dtype=np.float32),
    },
    "nugraph2": _nugraph2,
}


# --------------------------------------------------------------------------- #
# Shape/dtype conformance (port of the old TritonHEPClient._prepare_input)
# --------------------------------------------------------------------------- #


def conform(name: str, data: np.ndarray, meta_shape: list[int], dtype: str,
            max_batch_size: int) -> np.ndarray:
    """Cast to the model's dtype and add a batch dim when the model batches."""
    target = triton_to_np_dtype(dtype)
    if data.dtype != target:
        data = data.astype(target)

    if max_batch_size > 0:
        unbatched = meta_shape[1:] if meta_shape and meta_shape[0] == -1 else meta_shape
        if data.ndim == len(unbatched):
            data = data[np.newaxis, ...]
        elif data.ndim != len(unbatched) + 1:
            raise ValueError(
                f"{name}: expected {len(unbatched)}D or {len(unbatched) + 1}D, "
                f"got shape {data.shape} (metadata shape {meta_shape})"
            )
    elif data.ndim != len(meta_shape):
        raise ValueError(
            f"{name}: expected {len(meta_shape)}D, got shape {data.shape} "
            f"(metadata shape {meta_shape})"
        )
    return data


# --------------------------------------------------------------------------- #
# Per-model test
# --------------------------------------------------------------------------- #


class ModelResult:
    def __init__(self, model: str):
        self.model = model
        self.ok = False
        self.error = ""
        self.ready_s = float("nan")   # is_model_ready + metadata + config
        self.infer_s: list[float] = []
        self.outputs: dict[str, tuple[tuple[int, ...], str]] = {}


def test_model(client: grpcclient.InferenceServerClient, model: str,
               raw_inputs: dict[str, np.ndarray], repeat: int,
               timeout: float) -> ModelResult:
    res = ModelResult(model)
    try:
        t0 = time.perf_counter()
        if not client.is_model_ready(model):
            raise RuntimeError("is_model_ready returned False")
        meta = client.get_model_metadata(model)
        config = client.get_model_config(model).config
        res.ready_s = time.perf_counter() - t0

        meta_inputs = {i.name: i for i in meta.inputs}
        missing = set(meta_inputs) - set(raw_inputs)
        extra = set(raw_inputs) - set(meta_inputs)
        if missing or extra:
            raise ValueError(
                f"input mismatch: missing={sorted(missing)} unexpected={sorted(extra)}"
            )

        infer_inputs = []
        for name, data in raw_inputs.items():
            mi = meta_inputs[name]
            data = conform(name, data, list(mi.shape), mi.datatype, config.max_batch_size)
            inp = grpcclient.InferInput(name, list(data.shape), np_to_triton_dtype(data.dtype))
            inp.set_data_from_numpy(data)
            infer_inputs.append(inp)
        requested = [grpcclient.InferRequestedOutput(o.name) for o in meta.outputs]

        for _ in range(repeat):
            t0 = time.perf_counter()
            result = client.infer(model, infer_inputs, outputs=requested, client_timeout=timeout)
            res.infer_s.append(time.perf_counter() - t0)

        for o in meta.outputs:
            arr = result.as_numpy(o.name)
            if arr is None:
                raise RuntimeError(f"output {o.name!r} missing from response")
            res.outputs[o.name] = (arr.shape, np_to_triton_dtype(arr.dtype))
        res.ok = True
    except (InferenceServerException, ValueError, RuntimeError) as exc:
        res.error = str(exc).strip().splitlines()[0] if str(exc) else type(exc).__name__
    return res


# --------------------------------------------------------------------------- #
# Reporting / entrypoint
# --------------------------------------------------------------------------- #


def _fmt_s(x: float) -> str:
    return "  n/a  " if x != x else f"{x:7.2f}"


def report(results: list[ModelResult]) -> None:
    w = max(len(r.model) for r in results)
    print(f"\n{'model':<{w}}  status  ready(s)  infer(s) [first, then steady]  outputs")
    print("-" * (w + 70))
    for r in results:
        status = "OK  " if r.ok else "FAIL"
        infer = ", ".join(f"{t:.2f}" for t in r.infer_s) or "-"
        detail = (
            "; ".join(f"{n}: {s} {d}" for n, (s, d) in r.outputs.items())
            if r.ok else r.error
        )
        print(f"{r.model:<{w}}  {status}    {_fmt_s(r.ready_s)}  {infer:<28}  {detail}")
    passed = sum(r.ok for r in results)
    print(f"\n{passed}/{len(results)} models passed")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", default="localhost:8001", help="sidecar address")
    p.add_argument("--models", nargs="*", default=None,
                   help="subset of models to test (default: all)")
    p.add_argument("--repeat", type=int, default=1, help="inferences per model")
    p.add_argument("--timeout", type=float, default=600.0,
                   help="per-inference client deadline in seconds")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    models = args.models or list(INPUT_GENERATORS)
    unknown = [m for m in models if m not in INPUT_GENERATORS]
    if unknown:
        p.error(f"no input generator for {unknown}; known: {list(INPUT_GENERATORS)}")

    client = grpcclient.InferenceServerClient(url=args.url)
    t0 = time.perf_counter()
    try:
        live = client.is_server_live()
    except InferenceServerException as exc:
        print(f"cannot reach sidecar at {args.url}: {exc}")
        return 2
    print(f"sidecar {args.url}: server_live={live} ({time.perf_counter() - t0:.2f}s)")

    rng = np.random.default_rng(args.seed)
    results = []
    for model in models:
        print(f"testing {model} ...", end=" ", flush=True)
        r = test_model(client, model, INPUT_GENERATORS[model](rng), args.repeat, args.timeout)
        print("ok" if r.ok else f"FAIL: {r.error}")
        results.append(r)

    report(results)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
