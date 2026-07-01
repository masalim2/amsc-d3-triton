"""Generate sample .npz input files for each Triton HEP model.

Shapes and tensor names match test_client.py exactly.
"""

from pathlib import Path

import numpy as np

OUTDIR = Path(__file__).parent / "sample_inputs"


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)

    # snbamsc_2dcnn_u
    np.savez(
        OUTDIR / "snbamsc_2dcnn_u.npz",
        zero_padding2d_input=rng.random((1149, 128, 1), dtype=np.float32),
    )

    # snbamsc_2dcnn_v
    np.savez(
        OUTDIR / "snbamsc_2dcnn_v.npz",
        zero_padding2d_input=rng.random((1148, 128, 1), dtype=np.float32),
    )

    # snbamsc_2dcnn_z
    np.savez(
        OUTDIR / "snbamsc_2dcnn_z.npz",
        zero_padding2d_1_input=rng.random((480, 128, 1), dtype=np.float32),
    )

    # DoubleMetricLearning
    np.savez(
        OUTDIR / "DoubleMetricLearning.npz",
        FEATURES=rng.random((1, 44), dtype=np.float32),
    )

    # higgsInteractionNet
    np.savez(
        OUTDIR / "higgsInteractionNet.npz",
        input_cpf=rng.random((30, 60), dtype=np.float32),
        input_sv=rng.random((14, 5), dtype=np.float32),
    )

    # particlenet_AK4_PT
    np.savez(
        OUTDIR / "particlenet_AK4_PT.npz",
        **{
            "pf_points__0": rng.random((2, 100), dtype=np.float32),
            "pf_features__1": rng.random((20, 100), dtype=np.float32),
            "pf_mask__2": rng.random((1, 100), dtype=np.float32),
            "sv_points__3": rng.random((2, 10), dtype=np.float32),
            "sv_features__4": rng.random((11, 10), dtype=np.float32),
            "sv_mask__5": rng.random((1, 10), dtype=np.float32),
        },
    )

    # nugraph2
    n_hits, n_sp = 90, 30
    np.savez(
        OUTDIR / "nugraph2.npz",
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
            [float(100.0 + (i % 11)) for i in range(n_hits)],
            dtype=np.float32,
        ),
        hit_table_rms=np.array(
            [float(2.0 + (i % 5) * 0.1) for i in range(n_hits)],
            dtype=np.float32,
        ),
        spacepoint_table_spacepoint_id=np.arange(n_sp, dtype=np.int64),
        spacepoint_table_hit_id_u=np.arange(0, n_sp, dtype=np.int64),
        spacepoint_table_hit_id_v=np.arange(n_sp, 2 * n_sp, dtype=np.int64),
        spacepoint_table_hit_id_y=np.arange(2 * n_sp, 3 * n_sp, dtype=np.int64),
    )

    for f in sorted(OUTDIR.glob("*.npz")):
        with np.load(f) as data:
            keys = ", ".join(f"{k}{data[k].shape}" for k in data.files)
            print(f"  {f.name}: {keys}")


if __name__ == "__main__":
    main()
