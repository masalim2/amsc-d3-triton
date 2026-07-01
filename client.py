"""gRPC inference client for Triton-served HEP models."""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import tritonclient.grpc as grpcclient

log = logging.getLogger(__name__)

MODEL_PORTS: dict[str, int] = {
    "snbamsc_2dcnn_u": 8501,
    "snbamsc_2dcnn_v": 8511,
    "snbamsc_2dcnn_z": 8521,
    "DoubleMetricLearning": 8531,
    "higgsInteractionNet": 8541,
    "particlenet_AK4_PT": 8551,
    "nugraph2": 8561,
}

TRITON_DTYPE_TO_NP = {
    "BOOL": np.bool_,
    "UINT8": np.uint8,
    "UINT16": np.uint16,
    "UINT32": np.uint32,
    "UINT64": np.uint64,
    "INT8": np.int8,
    "INT16": np.int16,
    "INT32": np.int32,
    "INT64": np.int64,
    "FP16": np.float16,
    "FP32": np.float32,
    "FP64": np.float64,
}


class TritonHEPClient:
    """gRPC client for Triton-served HEP models."""

    def __init__(self, hostname: str, timeout: float = 120):
        if not logging.root.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )

        self._clients: dict[str, grpcclient.InferenceServerClient] = {}
        self._urls: dict[str, str] = {}
        self._metadata: dict[str, dict] = {}

        for model_name, port in MODEL_PORTS.items():
            url = f"{hostname}:{port}"
            self._clients[model_name] = grpcclient.InferenceServerClient(url=url)
            self._urls[model_name] = url
            log.info("Registered client for %s at %s", model_name, url)

        self._wait_for_ready(timeout)
        self._discover_metadata()

    # -- startup helpers --

    def _wait_for_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        pending = set(self._clients)
        while pending:
            for name in list(pending):
                try:
                    if self._clients[name].is_server_ready():
                        log.info(
                            "Server ready: %s (%s)",
                            name,
                            self._urls[name],
                        )
                        pending.discard(name)
                except Exception:
                    pass
            if not pending:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Servers not ready after {timeout}s. "
                    f"Still waiting on: {pending}"
                )
            remaining = deadline - time.monotonic()
            log.info(
                "Waiting for %d server(s)... (%.0fs remaining)",
                len(pending),
                remaining,
            )
            time.sleep(min(2, max(0.1, remaining)))

    def _discover_metadata(self) -> None:
        for model_name, client in self._clients.items():
            try:
                meta = client.get_model_metadata(model_name)
                config = client.get_model_config(model_name)
                max_batch_size = config.config.max_batch_size

                inputs = {}
                for inp in meta.inputs:
                    inputs[inp.name] = {
                        "shape": list(inp.shape),
                        "dtype": inp.datatype,
                        "np_dtype": TRITON_DTYPE_TO_NP.get(inp.datatype, np.float32),
                    }

                outputs = {}
                for out in meta.outputs:
                    outputs[out.name] = {
                        "shape": list(out.shape),
                        "dtype": out.datatype,
                        "np_dtype": TRITON_DTYPE_TO_NP.get(out.datatype, np.float32),
                    }

                self._metadata[model_name] = {
                    "inputs": inputs,
                    "outputs": outputs,
                    "max_batch_size": max_batch_size,
                }
                log.info(
                    "Model %s: max_batch_size=%d, inputs=%s, outputs=%s",
                    model_name,
                    max_batch_size,
                    {n: (v["dtype"], v["shape"]) for n, v in inputs.items()},
                    {n: (v["dtype"], v["shape"]) for n, v in outputs.items()},
                )
            except Exception as exc:
                log.warning("Failed to get metadata for %s: %s", model_name, exc)

    # -- public properties --

    @property
    def available_models(self) -> list[str]:
        """Model names for which metadata was successfully discovered."""
        return list(self._metadata)

    def get_model_info(self, model_name: str) -> dict:
        """Return discovered metadata for *model_name*.

        The dict has keys ``"inputs"``, ``"outputs"`` (each mapping tensor name
        to ``{"shape", "dtype", "np_dtype"}``), and ``"max_batch_size"``.
        """
        if model_name not in self._metadata:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {self.available_models}"
            )
        return self._metadata[model_name]

    # -- generic inference --

    def _get_client(self, model_name: str) -> grpcclient.InferenceServerClient:
        if model_name not in self._clients:
            raise ValueError(
                f"No client for model '{model_name}'. "
                f"Available: {list(self._clients)}"
            )
        return self._clients[model_name]

    def _prepare_input(
        self, model_name: str, tensor_name: str, data: np.ndarray
    ) -> np.ndarray:
        """Cast *data* to the correct dtype and add a batch dimension if needed."""
        meta = self._metadata[model_name]
        input_info = meta["inputs"].get(tensor_name)
        if input_info is None:
            raise ValueError(
                f"Model '{model_name}' has no input '{tensor_name}'. "
                f"Available inputs: {list(meta['inputs'])}"
            )

        target_dtype = input_info["np_dtype"]
        meta_shape = input_info["shape"]
        max_batch = meta["max_batch_size"]

        data = np.asarray(data)
        if data.dtype != target_dtype:
            data = data.astype(target_dtype)

        if max_batch > 0:
            if meta_shape and meta_shape[0] == -1:
                meta_shape = meta_shape[1:]
            ndim_no_batch = len(meta_shape)
            if data.ndim == ndim_no_batch:
                data = data[np.newaxis, ...]
            elif data.ndim == ndim_no_batch + 1:
                pass
            else:
                raise ValueError(
                    f"Input '{tensor_name}' for model '{model_name}': "
                    f"expected {ndim_no_batch}D (unbatched) or "
                    f"{ndim_no_batch + 1}D (batched), got {data.ndim}D "
                    f"shape {data.shape}. Metadata shape (no batch): {meta_shape}"
                )
        else:
            expected_ndim = len(meta_shape)
            if data.ndim != expected_ndim:
                raise ValueError(
                    f"Input '{tensor_name}' for model '{model_name}': "
                    f"expected {expected_ndim}D, got {data.ndim}D "
                    f"shape {data.shape}. Metadata shape: {meta_shape}"
                )

        return data

    def infer(
        self,
        model_name: str,
        inputs: dict[str, np.ndarray],
        outputs: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Send an inference request with numpy arrays.

        Args:
            model_name: Name of the Triton model.
            inputs: ``{tensor_name: np.ndarray}`` for each model input.
                Arrays are cast and batch-padded automatically using the
                discovered model metadata.
            outputs: Output tensor names to retrieve.  ``None`` retrieves all.

        Returns:
            ``{output_name: np.ndarray}`` for each requested output.
        """
        client = self._get_client(model_name)
        meta = self.get_model_info(model_name)

        triton_inputs = []
        for name, data in inputs.items():
            data = self._prepare_input(model_name, name, data)
            triton_dtype = meta["inputs"][name]["dtype"]
            inp = grpcclient.InferInput(name, list(data.shape), triton_dtype)
            inp.set_data_from_numpy(data)
            triton_inputs.append(inp)

        output_names = list(meta["outputs"]) if outputs is None else outputs
        triton_outputs = [
            grpcclient.InferRequestedOutput(n) for n in output_names
        ]

        result = client.infer(
            model_name=model_name,
            inputs=triton_inputs,
            outputs=triton_outputs,
        )
        return {n: result.as_numpy(n) for n in output_names}

    def infer_files(
        self,
        model_name: str,
        input_path: str | os.PathLike,
        output_path: str | os.PathLike,
        outputs: list[str] | None = None,
    ) -> None:
        """Run inference reading/writing ``.npz`` files.

        Args:
            model_name: Name of the Triton model.
            input_path: Path to a ``.npz`` file whose keys are input tensor
                names.  A file-like object is also accepted.
            output_path: Path where the output ``.npz`` will be written.
                Keys are output tensor names.  A file-like object is also
                accepted.
            outputs: Output tensor names to retrieve.  ``None`` retrieves all.
        """
        with np.load(input_path) as data:
            inputs = {name: data[name] for name in data.files}
        result = self.infer(model_name, inputs, outputs=outputs)
        np.savez(output_path, **result)
