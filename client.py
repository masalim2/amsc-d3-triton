"""gRPC inference client for Triton-served HEP models."""

from __future__ import annotations

import logging
import time

import numpy as np
import tritonclient.grpc as grpcclient

log = logging.getLogger(__name__)

GPU_MODEL_PORT_OFFSETS: dict[str, int] = {
    "snbamsc_2dcnn_u": 0,
    "snbamsc_2dcnn_v": 10,
    "snbamsc_2dcnn_z": 20,
    "DoubleMetricLearning": 30,
    "higgsInteractionNet": 40,
    "particlenet_AK4_PT": 50,
    "nugraph2": 60,
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
    """gRPC client for Triton-served HEP models.

    Two usage modes:

    **Multi-model** (all GPU models on one node via
    ``launch-all-gpu-models-node-pbs.sh``)::

        client = TritonHEPClient("sophia-gpu-05.lab.alcf.anl.gov", base_port=8500)

    **Single-model** (one Triton server via ``launch-tritonserver-pbs.sh``)::

        client = TritonHEPClient("sophia-gpu-05.lab.alcf.anl.gov", grpc_port=8001)
    """

    def __init__(
        self,
        hostname: str,
        *,
        base_port: int | None = None,
        grpc_port: int = 8001,
        models: list[str] | None = None,
        timeout: float = 120,
    ):
        if not logging.root.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )

        self._clients: dict[str, grpcclient.InferenceServerClient] = {}
        self._urls: dict[str, str] = {}
        self._metadata: dict[str, dict] = {}

        if base_port is not None:
            port_map = GPU_MODEL_PORT_OFFSETS
            if models:
                unknown = set(models) - set(port_map)
                if unknown:
                    raise ValueError(
                        f"Unknown model(s): {unknown}. "
                        f"Known GPU models: {list(port_map)}"
                    )
                port_map = {m: port_map[m] for m in models}
            for model_name, offset in port_map.items():
                url = f"{hostname}:{base_port + offset + 1}"
                self._clients[model_name] = grpcclient.InferenceServerClient(url=url)
                self._urls[model_name] = url
                log.info("Registered client for %s at %s", model_name, url)
        else:
            url = f"{hostname}:{grpc_port}"
            self._clients["_default"] = grpcclient.InferenceServerClient(url=url)
            self._urls["_default"] = url
            log.info("Registered single client at %s", url)

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
                            name if name != "_default" else "single-model",
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
        if "_default" in self._clients:
            client = self._clients.pop("_default")
            url = self._urls.pop("_default")
            try:
                index = client.get_model_repository_index()
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot list models from {url}: {exc}"
                ) from exc
            for entry in index:
                name = entry.name
                state = getattr(entry, "state", "")
                if state and state != "READY":
                    log.debug("Skipping model %s (state=%s)", name, state)
                    continue
                self._clients[name] = client
                self._urls[name] = url
                log.info("Discovered model: %s", name)
            if not self._clients:
                raise RuntimeError(f"No READY models found on {url}")

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

    # -- helpers for model-specific methods --

    def _squeeze_batch(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim > 0 and arr.shape[0] == 1:
            return arr[0]
        return arr

    def _infer_single_output(
        self, model_name: str, inputs: dict[str, np.ndarray]
    ) -> np.ndarray:
        result = self.infer(model_name, inputs)
        out = next(iter(result.values()))
        return self._squeeze_batch(out)

    def _infer_squeeze_all(
        self, model_name: str, inputs: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        return {
            k: self._squeeze_batch(v)
            for k, v in self.infer(model_name, inputs).items()
        }

    # -- model-specific methods --

    def infer_snbamsc_2dcnn_u(self, data: np.ndarray) -> np.ndarray:
        """Infer on the U-plane 2D CNN.

        Args:
            data: Wire-tick image.  Accepted shapes: ``(wires, ticks)``,
                ``(wires, ticks, 1)``, or ``(1, wires, ticks, 1)``.
        """
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        return self._infer_single_output(
            "snbamsc_2dcnn_u", {"zero_padding2d_input": data}
        )

    def infer_snbamsc_2dcnn_v(self, data: np.ndarray) -> np.ndarray:
        """Infer on the V-plane 2D CNN.

        Args:
            data: Wire-tick image.  Accepted shapes: ``(wires, ticks)``,
                ``(wires, ticks, 1)``, or ``(1, wires, ticks, 1)``.
        """
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        return self._infer_single_output(
            "snbamsc_2dcnn_v", {"zero_padding2d_input": data}
        )

    def infer_snbamsc_2dcnn_z(self, data: np.ndarray) -> np.ndarray:
        """Infer on the Z-plane 2D CNN.

        Args:
            data: Wire-tick image.  Accepted shapes: ``(wires, ticks)``,
                ``(wires, ticks, 1)``, or ``(1, wires, ticks, 1)``.
        """
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        return self._infer_single_output(
            "snbamsc_2dcnn_z", {"zero_padding2d_1_input": data}
        )

    def infer_double_metric_learning(self, features: np.ndarray) -> np.ndarray:
        """Infer on the DoubleMetricLearning model.

        Args:
            features: Feature vector.  Accepted shapes: ``(44,)``,
                ``(1, 44)``, or ``(1, 1, 44)``.
        """
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features[np.newaxis, :]
        return self._infer_single_output(
            "DoubleMetricLearning", {"FEATURES": features}
        )

    def infer_higgs_interaction_net(
        self, input_cpf: np.ndarray, input_sv: np.ndarray
    ) -> np.ndarray:
        """Infer on the Higgs Interaction Network.

        Args:
            input_cpf: Charged-particle-flow candidates,
                shape ``(30, 60)`` or ``(1, 30, 60)``.
            input_sv: Secondary vertices,
                shape ``(14, 5)`` or ``(1, 14, 5)``.
        """
        return self._infer_single_output(
            "higgsInteractionNet",
            {"input_cpf": input_cpf, "input_sv": input_sv},
        )

    def infer_particlenet(
        self,
        *,
        pf_points: np.ndarray,
        pf_features: np.ndarray,
        pf_mask: np.ndarray,
        sv_points: np.ndarray,
        sv_features: np.ndarray,
        sv_mask: np.ndarray,
    ) -> np.ndarray:
        """Infer on ParticleNet AK4.

        All inputs accept shapes with or without a leading batch dimension.

        Args:
            pf_points: ``(2, 100)`` — particle-flow point coordinates.
            pf_features: ``(20, 100)`` — particle-flow features.
            pf_mask: ``(1, 100)`` — particle-flow mask.
            sv_points: ``(2, 10)`` — secondary-vertex point coordinates.
            sv_features: ``(11, 10)`` — secondary-vertex features.
            sv_mask: ``(1, 10)`` — secondary-vertex mask.
        """
        return self._infer_single_output(
            "particlenet_AK4_PT",
            {
                "pf_points__0": pf_points,
                "pf_features__1": pf_features,
                "pf_mask__2": pf_mask,
                "sv_points__3": sv_points,
                "sv_features__4": sv_features,
                "sv_mask__5": sv_mask,
            },
        )

    def infer_nugraph2(
        self,
        *,
        hit_table_hit_id: np.ndarray,
        hit_table_local_plane: np.ndarray,
        hit_table_local_time: np.ndarray,
        hit_table_local_wire: np.ndarray,
        hit_table_integral: np.ndarray,
        hit_table_rms: np.ndarray,
        spacepoint_table_spacepoint_id: np.ndarray,
        spacepoint_table_hit_id_u: np.ndarray,
        spacepoint_table_hit_id_v: np.ndarray,
        spacepoint_table_hit_id_y: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Infer on NuGraph2.

        All arrays are 1-D per event (variable length).  Integer columns
        (hit IDs, planes, wires, spacepoint cross-references) are cast to
        INT64; float columns (time, integral, RMS) are cast to FP32.  The
        hit and spacepoint tables must be self-consistent — use real or
        synthetic event data, not random values.

        Returns:
            ``{output_name: np.ndarray}`` for every model output.
        """
        return self._infer_squeeze_all(
            "nugraph2",
            {
                "hit_table_hit_id": hit_table_hit_id,
                "hit_table_local_plane": hit_table_local_plane,
                "hit_table_local_time": hit_table_local_time,
                "hit_table_local_wire": hit_table_local_wire,
                "hit_table_integral": hit_table_integral,
                "hit_table_rms": hit_table_rms,
                "spacepoint_table_spacepoint_id": spacepoint_table_spacepoint_id,
                "spacepoint_table_hit_id_u": spacepoint_table_hit_id_u,
                "spacepoint_table_hit_id_v": spacepoint_table_hit_id_v,
                "spacepoint_table_hit_id_y": spacepoint_table_hit_id_y,
            },
        )

    def infer_btagging(self, **inputs: np.ndarray) -> dict[str, np.ndarray]:
        """Infer on the BTagging ONNX model.

        Input tensor names and shapes are model-specific — call
        ``get_model_info("BTagging_network_8085e6c5717c")`` to discover them.

        Returns:
            ``{output_name: np.ndarray}`` for every model output.
        """
        return self._infer_squeeze_all("BTagging_network_8085e6c5717c", inputs)
