"""Register the Triton HEP inference function with Globus Compute."""

from __future__ import annotations

from typing import TypedDict

from globus_compute_sdk import Client


class Payload(TypedDict):
    model_name: str
    input_path: str
    output_path: str
    outputs: list[str] | None


def submit(payload: Payload):
    from pathlib import Path

    from triton_hep_client import TritonHEPClient

    model_name = payload["model_name"]
    input_path = str(Path(payload["input_path"]).resolve())
    output_path = str(Path(payload["output_path"]).resolve())
    outputs = payload.get("outputs")

    client = TritonHEPClient("localhost")
    client.infer_files(model_name, input_path, output_path, outputs=outputs)

    return {
        "model_name": model_name,
        "output_path": output_path,
    }


gcc = Client()
func_uuid = gcc.register_function(submit)
print(f"Registered {submit.__name__}: {func_uuid=}")
