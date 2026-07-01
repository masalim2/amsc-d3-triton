# Sample Inputs

Pre-built `.npz` files for each Triton HEP model, with shapes matching the
model metadata exactly. Regenerate with `python generate_sample_inputs.py`.

## CLI usage

After authenticating (`alcf-ai auth login`), submit any sample input:

```bash
# snbamsc_2dcnn_u
alcf-ai d3-triton submit snbamsc_2dcnn_u sample_inputs/snbamsc_2dcnn_u.npz

# snbamsc_2dcnn_v
alcf-ai d3-triton submit snbamsc_2dcnn_v sample_inputs/snbamsc_2dcnn_v.npz

# snbamsc_2dcnn_z
alcf-ai d3-triton submit snbamsc_2dcnn_z sample_inputs/snbamsc_2dcnn_z.npz

# DoubleMetricLearning
alcf-ai d3-triton submit DoubleMetricLearning sample_inputs/DoubleMetricLearning.npz

# higgsInteractionNet
alcf-ai d3-triton submit higgsInteractionNet sample_inputs/higgsInteractionNet.npz

# particlenet_AK4_PT
alcf-ai d3-triton submit particlenet_AK4_PT sample_inputs/particlenet_AK4_PT.npz

# nugraph2
alcf-ai d3-triton submit nugraph2 sample_inputs/nugraph2.npz
```

To stage out results to a Globus collection, add `--to-collection-id <UUID>`.
