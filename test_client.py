"""Integration tests for TritonHEPClient.

Configure via environment variables:

    TRITON_HOSTNAME   Compute-node hostname (default: "localhost")

Tests for models that are not available on the running server are skipped
automatically.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from client import TritonHEPClient

HOSTNAME = os.environ.get("TRITON_HOSTNAME", "localhost")


def _require_model(client: TritonHEPClient, model_name: str) -> None:
    if model_name not in client.available_models:
        pytest.skip(f"Model '{model_name}' not available on server")


@pytest.fixture(scope="module")
def client() -> TritonHEPClient:
    try:
        return TritonHEPClient(HOSTNAME, timeout=60)
    except Exception as exc:
        pytest.skip(f"Triton server not available: {exc}")


# ---------------------------------------------------------------------------
# Happy-path tests — one per model, all via infer()
# ---------------------------------------------------------------------------


class TestSnbamsc2dCnnU:
    MODEL = "snbamsc_2dcnn_u"

    def test_infer(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        data = np.random.rand(1149, 128, 1).astype(np.float32)
        result = client.infer(self.MODEL, {"zero_padding2d_input": data})
        assert isinstance(result, dict)
        assert len(list(result.values())) > 0
        for arr in result.values():
            assert isinstance(arr, np.ndarray)


class TestSnbamsc2dCnnV:
    MODEL = "snbamsc_2dcnn_v"

    def test_infer(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        data = np.random.rand(1148, 128, 1).astype(np.float32)
        result = client.infer(self.MODEL, {"zero_padding2d_input": data})
        assert isinstance(result, dict)
        assert len(list(result.values())) > 0
        for arr in result.values():
            assert isinstance(arr, np.ndarray)


class TestSnbamsc2dCnnZ:
    MODEL = "snbamsc_2dcnn_z"

    def test_infer(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        data = np.random.rand(480, 128, 1).astype(np.float32)
        result = client.infer(self.MODEL, {"zero_padding2d_1_input": data})
        assert isinstance(result, dict)
        assert len(list(result.values())) > 0
        for arr in result.values():
            assert isinstance(arr, np.ndarray)


class TestDoubleMetricLearning:
    MODEL = "DoubleMetricLearning"

    def test_infer(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        features = np.random.rand(1, 44).astype(np.float32)
        result = client.infer(self.MODEL, {"FEATURES": features})
        assert isinstance(result, dict)
        assert len(list(result.values())) > 0
        for arr in result.values():
            assert isinstance(arr, np.ndarray)


class TestHiggsInteractionNet:
    MODEL = "higgsInteractionNet"

    def test_infer(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        cpf = np.random.rand(30, 60).astype(np.float32)
        sv = np.random.rand(14, 5).astype(np.float32)
        result = client.infer(self.MODEL, {"input_cpf": cpf, "input_sv": sv})
        assert isinstance(result, dict)
        assert len(list(result.values())) > 0
        for arr in result.values():
            assert isinstance(arr, np.ndarray)


class TestParticleNet:
    MODEL = "particlenet_AK4_PT"

    def test_infer(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        result = client.infer(self.MODEL, {
            "pf_points__0": np.random.rand(2, 100).astype(np.float32),
            "pf_features__1": np.random.rand(20, 100).astype(np.float32),
            "pf_mask__2": np.random.rand(1, 100).astype(np.float32),
            "sv_points__3": np.random.rand(2, 10).astype(np.float32),
            "sv_features__4": np.random.rand(11, 10).astype(np.float32),
            "sv_mask__5": np.random.rand(1, 10).astype(np.float32),
        })
        assert isinstance(result, dict)
        assert len(list(result.values())) > 0
        for arr in result.values():
            assert isinstance(arr, np.ndarray)


class TestNuGraph2:
    MODEL = "nugraph2"

    def test_infer(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        n_hits, n_sp = 90, 30
        result = client.infer(self.MODEL, {
            "hit_table_hit_id": np.arange(n_hits, dtype=np.int64),
            "hit_table_local_plane": np.array(
                [i // (n_hits // 3) for i in range(n_hits)], dtype=np.int64
            ),
            "hit_table_local_time": np.array(
                [float((i % 30) * 2 + (i // 30) * 0.1) for i in range(n_hits)],
                dtype=np.float32,
            ),
            "hit_table_local_wire": np.array(
                [int((i % 30) * 3 + (i // 30)) for i in range(n_hits)],
                dtype=np.int64,
            ),
            "hit_table_integral": np.array(
                [float(100.0 + (i % 11)) for i in range(n_hits)],
                dtype=np.float32,
            ),
            "hit_table_rms": np.array(
                [float(2.0 + (i % 5) * 0.1) for i in range(n_hits)],
                dtype=np.float32,
            ),
            "spacepoint_table_spacepoint_id": np.arange(n_sp, dtype=np.int64),
            "spacepoint_table_hit_id_u": np.arange(0, n_sp, dtype=np.int64),
            "spacepoint_table_hit_id_v": np.arange(
                n_sp, 2 * n_sp, dtype=np.int64
            ),
            "spacepoint_table_hit_id_y": np.arange(
                2 * n_sp, 3 * n_sp, dtype=np.int64
            ),
        })
        assert isinstance(result, dict)
        assert len(result) > 0
        assert len(list(result.values())) > 0
        for arr in result.values():
            assert isinstance(arr, np.ndarray)


class TestBTagging:
    MODEL = "BTagging_network_8085e6c5717c"

    def test_infer(self, client: TritonHEPClient) -> None:
        _require_model(client, self.MODEL)
        info = client.get_model_info(self.MODEL)
        inputs = {}
        for name, spec in info["inputs"].items():
            shape = [max(1, d) for d in spec["shape"]]
            if info["max_batch_size"] > 0:
                shape = [1, *shape]
            inputs[name] = np.random.rand(*shape).astype(spec["np_dtype"])
        result = client.infer(self.MODEL, inputs)
        assert len(list(result.values())) > 0
        assert isinstance(result, dict)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# infer_files()
# ---------------------------------------------------------------------------


class TestInferFiles:
    MODEL = "DoubleMetricLearning"

    def test_roundtrip(self, client: TritonHEPClient, tmp_path) -> None:
        _require_model(client, self.MODEL)
        inp = tmp_path / "input.npz"
        out = tmp_path / "output.npz"
        np.savez(inp, FEATURES=np.random.rand(1, 44).astype(np.float32))
        client.infer_files(self.MODEL, inp, out)
        with np.load(out) as data:
            assert len(data.files) > 0
            assert len(list(data.values())) > 0
            for arr in data.values():
                assert isinstance(arr, np.ndarray)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
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
