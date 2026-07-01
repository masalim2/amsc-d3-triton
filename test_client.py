"""Integration tests for TritonHEPClient.

Configure via environment variables:

    TRITON_HOSTNAME   Compute-node hostname (default: "localhost")
    TRITON_BASE_PORT  Base HTTP port for the multi-model launcher (default: 8500).
                      Set to "" to use single-model mode with TRITON_GRPC_PORT.
    TRITON_GRPC_PORT  gRPC port for single-model mode (default: 8001).
                      Only used when TRITON_BASE_PORT is empty.

Tests for models that are not available on the running server are skipped
automatically.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from client import TritonHEPClient

HOSTNAME = os.environ.get("TRITON_HOSTNAME", "localhost")
_base_port_str = os.environ.get("TRITON_BASE_PORT", "8500")
BASE_PORT: int | None = int(_base_port_str) if _base_port_str else None
GRPC_PORT: int = int(os.environ.get("TRITON_GRPC_PORT", "8001"))


def _require_model(client: TritonHEPClient, model_name: str) -> None:
    if model_name not in client.available_models:
        pytest.skip(f"Model '{model_name}' not available on server")


@pytest.fixture(scope="module")
def client() -> TritonHEPClient:
    try:
        if BASE_PORT is not None:
            return TritonHEPClient(HOSTNAME, base_port=BASE_PORT, timeout=30)
        return TritonHEPClient(HOSTNAME, grpc_port=GRPC_PORT, timeout=30)
    except Exception as exc:
        pytest.skip(f"Triton server not available: {exc}")


# ---------------------------------------------------------------------------
# snbamsc 2D CNN models
# ---------------------------------------------------------------------------


class TestSnbamsc2dCnnU:
    MODEL = "snbamsc_2dcnn_u"

    def test_basic(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        data = np.random.rand(1149, 128, 1).astype(np.float32)
        result = client.infer_snbamsc_2dcnn_u(data)
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 1

    def test_no_channel_dim(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        data = np.random.rand(1149, 128).astype(np.float32)
        result = client.infer_snbamsc_2dcnn_u(data)
        assert isinstance(result, np.ndarray)

    def test_with_batch_dim(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        data = np.random.rand(1, 1149, 128, 1).astype(np.float32)
        result = client.infer_snbamsc_2dcnn_u(data)
        assert isinstance(result, np.ndarray)

    def test_float64_coercion(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        data = np.random.rand(1149, 128, 1)  # float64
        result = client.infer_snbamsc_2dcnn_u(data)
        assert isinstance(result, np.ndarray)


class TestSnbamsc2dCnnV:
    MODEL = "snbamsc_2dcnn_v"

    def test_basic(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        data = np.random.rand(1148, 128, 1).astype(np.float32)
        result = client.infer_snbamsc_2dcnn_v(data)
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 1


class TestSnbamsc2dCnnZ:
    MODEL = "snbamsc_2dcnn_z"

    def test_basic(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        data = np.random.rand(480, 128, 1).astype(np.float32)
        result = client.infer_snbamsc_2dcnn_z(data)
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 1


# ---------------------------------------------------------------------------
# DoubleMetricLearning
# ---------------------------------------------------------------------------


class TestDoubleMetricLearning:
    MODEL = "DoubleMetricLearning"

    def test_basic(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        features = np.random.rand(1, 44).astype(np.float32)
        result = client.infer_double_metric_learning(features)
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 1

    def test_flat_vector(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        features = np.random.rand(44).astype(np.float32)
        result = client.infer_double_metric_learning(features)
        assert isinstance(result, np.ndarray)

    def test_with_batch(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        features = np.random.rand(1, 1, 44).astype(np.float32)
        result = client.infer_double_metric_learning(features)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# higgsInteractionNet
# ---------------------------------------------------------------------------


class TestHiggsInteractionNet:
    MODEL = "higgsInteractionNet"

    def test_basic(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        cpf = np.random.rand(30, 60).astype(np.float32)
        sv = np.random.rand(14, 5).astype(np.float32)
        result = client.infer_higgs_interaction_net(cpf, sv)
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 1

    def test_with_batch(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        cpf = np.random.rand(1, 30, 60).astype(np.float32)
        sv = np.random.rand(1, 14, 5).astype(np.float32)
        result = client.infer_higgs_interaction_net(cpf, sv)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# particlenet_AK4_PT
# ---------------------------------------------------------------------------


class TestParticleNet:
    MODEL = "particlenet_AK4_PT"

    def test_basic(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        result = client.infer_particlenet(
            pf_points=np.random.rand(2, 100).astype(np.float32),
            pf_features=np.random.rand(20, 100).astype(np.float32),
            pf_mask=np.random.rand(1, 100).astype(np.float32),
            sv_points=np.random.rand(2, 10).astype(np.float32),
            sv_features=np.random.rand(11, 10).astype(np.float32),
            sv_mask=np.random.rand(1, 10).astype(np.float32),
        )
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 1

    def test_with_batch(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        result = client.infer_particlenet(
            pf_points=np.random.rand(1, 2, 100).astype(np.float32),
            pf_features=np.random.rand(1, 20, 100).astype(np.float32),
            pf_mask=np.random.rand(1, 1, 100).astype(np.float32),
            sv_points=np.random.rand(1, 2, 10).astype(np.float32),
            sv_features=np.random.rand(1, 11, 10).astype(np.float32),
            sv_mask=np.random.rand(1, 1, 10).astype(np.float32),
        )
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# nugraph2
# ---------------------------------------------------------------------------


def _nugraph2_sample_event(n_hits: int = 90, n_sp: int = 30) -> dict:
    """Build a self-consistent synthetic NuGraph2 event."""
    return dict(
        hit_table_hit_id=np.arange(n_hits, dtype=np.int64),
        hit_table_local_plane=np.array(
            [i // (n_hits // 3) for i in range(n_hits)], dtype=np.int64
        ),
        hit_table_local_time=np.array(
            [float((i % 30) * 2 + (i // 30) * 0.1) for i in range(n_hits)],
            dtype=np.float32,
        ),
        hit_table_local_wire=np.array(
            [int((i % 30) * 3 + (i // 30)) for i in range(n_hits)],
            dtype=np.int64,
        ),
        hit_table_integral=np.array(
            [float(100.0 + (i % 11)) for i in range(n_hits)], dtype=np.float32
        ),
        hit_table_rms=np.array(
            [float(2.0 + (i % 5) * 0.1) for i in range(n_hits)],
            dtype=np.float32,
        ),
        spacepoint_table_spacepoint_id=np.arange(n_sp, dtype=np.int64),
        spacepoint_table_hit_id_u=np.arange(0, n_sp, dtype=np.int64),
        spacepoint_table_hit_id_v=np.arange(n_sp, 2 * n_sp, dtype=np.int64),
        spacepoint_table_hit_id_y=np.arange(
            2 * n_sp, 3 * n_sp, dtype=np.int64
        ),
    )


class TestNuGraph2:
    MODEL = "nugraph2"

    def test_basic(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        event = _nugraph2_sample_event()
        result = client.infer_nugraph2(**event)
        assert isinstance(result, dict)
        assert len(result) > 0
        for name, arr in result.items():
            assert isinstance(arr, np.ndarray), f"output '{name}' is not ndarray"

    def test_different_event_size(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        event = _nugraph2_sample_event(n_hits=60, n_sp=20)
        result = client.infer_nugraph2(**event)
        assert isinstance(result, dict)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# BTagging
# ---------------------------------------------------------------------------


class TestBTagging:
    MODEL = "BTagging_network_8085e6c5717c"

    def test_basic(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        info = client.get_model_info(self.MODEL)
        inputs = {}
        for name, spec in info["inputs"].items():
            shape = [
                max(1, d) for d in spec["shape"]
            ]
            if info["max_batch_size"] > 0:
                shape = [1, *shape]
            inputs[name] = np.random.rand(*shape).astype(spec["np_dtype"])
        result = client.infer_btagging(**inputs)
        assert isinstance(result, dict)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Generic infer() interface
# ---------------------------------------------------------------------------


class TestGenericInfer:
    def test_raw_infer(self, client: TritonHEPClient) -> None:
        if "DoubleMetricLearning" not in client.available_models:
            pytest.skip("DoubleMetricLearning not available")
        inputs = {"FEATURES": np.random.rand(1, 44).astype(np.float32)}
        result = client.infer("DoubleMetricLearning", inputs)
        assert isinstance(result, dict)
        assert len(result) > 0
        for name, arr in result.items():
            assert isinstance(arr, np.ndarray)
            assert arr.ndim >= 1

    def test_raw_infer_specific_outputs(self, client: TritonHEPClient) -> None:
        if "DoubleMetricLearning" not in client.available_models:
            pytest.skip("DoubleMetricLearning not available")
        info = client.get_model_info("DoubleMetricLearning")
        output_name = list(info["outputs"])[0]
        inputs = {"FEATURES": np.random.rand(1, 44).astype(np.float32)}
        result = client.infer(
            "DoubleMetricLearning", inputs, outputs=[output_name]
        )
        assert list(result.keys()) == [output_name]


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------


class TestInputCoercion:
    def test_float64_to_fp32(self, client: TritonHEPClient) -> None:
        if "DoubleMetricLearning" not in client.available_models:
            pytest.skip("DoubleMetricLearning not available")
        features = np.random.rand(44)  # float64, 1D
        result = client.infer_double_metric_learning(features)
        assert isinstance(result, np.ndarray)

    def test_int32_to_int64(self, client: TritonHEPClient) -> None:
        _require_model(client, "nugraph2")
        event = _nugraph2_sample_event()
        event["hit_table_hit_id"] = event["hit_table_hit_id"].astype(np.int32)
        result = client.infer_nugraph2(**event)
        assert isinstance(result, dict)

    def test_bad_shape_raises(self, client: TritonHEPClient) -> None:
        _require_model(client, "DoubleMetricLearning")
        with pytest.raises(ValueError, match="expected"):
            client.infer(
                "DoubleMetricLearning",
                {"FEATURES": np.random.rand(1, 2, 3, 44).astype(np.float32)},
            )

    def test_unknown_model_raises(self, client: TritonHEPClient) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            client.get_model_info("nonexistent_model")

    def test_unknown_input_raises(self, client: TritonHEPClient) -> None:
        if not client.available_models:
            pytest.skip("No models available")
        model = client.available_models[0]
        with pytest.raises(ValueError, match="no input"):
            client.infer(model, {"BOGUS_TENSOR": np.array([1.0])})
