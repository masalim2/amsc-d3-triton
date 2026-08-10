from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from alcf_ai import InferenceClient
from alcf_ai.auth import STAGING_COLLECTION_ROOT
from rich import print

client = InferenceClient()

sample_dir = Path("/datascience/msalim/test-staging-area/amsc-d3-triton/sample_inputs")
models = [
  "snbamsc_2dcnn_u",
  "snbamsc_2dcnn_v",
  "snbamsc_2dcnn_z",
  "DoubleMetricLearning",
  "higgsInteractionNet",
  "particlenet_AK4_PT",
  "nugraph2",
]

collection_id = "05d2c76a-e867-4f67-aa57-76edeb0beda0"


def run_inference(model_name: str, input_path: Path) -> dict:
    """Stage in an input file, run Triton inference, and stage out the result."""
    # Stage the input file in from the source collection:
    stagein = client.stage_in(
        input_path, Path(input_path.name), from_collection_id=collection_id
    )
    remote_input = STAGING_COLLECTION_ROOT + str(stagein.destination_path)
    remote_output = remote_input.rsplit(".", 1)[0] + ".output.npz"

    # Submit the inference request and poll for completion:
    resp = client.d3_triton.submit(
        model_name=model_name,
        input_path=remote_input,
        output_path=remote_output,
    )
    result = client.d3_triton.poll_task_result(resp.task_id)

    # Stage the result back to the source collection:
    output_filename = Path(result["output_path"]).name
    local_output = input_path.with_suffix(".output.npz")
    client.stage_out(collection_id, Path(output_filename), local_output)

    return result


with ThreadPoolExecutor(max_workers=8) as pool:
    # Example: one input file per model
    inputs = [sample_dir / f"{model}.npz" for model in models]
    # Submit all stage_in / inference / stage_out pipelines to run in parallel:
    futures = {
        pool.submit(run_inference, model, input_path): model
        for model, input_path in zip(models, inputs)
    }
    for future in as_completed(futures):
        model = futures[future]
        result = future.result()
        print(f"[green]{model}[/green] completed: {result}")
