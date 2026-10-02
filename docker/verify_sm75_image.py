# SPDX-License-Identifier: Apache-2.0
"""Verify and record the fixed runtime contract for the SM75 image."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import torch

EXPECTED_PACKAGES = {
    "vllm": "0.30.0",
    "flashinfer-python": "0.7.0",
    "flashinfer-cubin": "0.7.0",
    "transformers": "5.15.1",
}


def main() -> None:
    installed = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
    for name, expected in EXPECTED_PACKAGES.items():
        if installed[name].partition("+")[0] != expected:
            raise RuntimeError(f"Expected {name} {expected}, found {installed[name]}")
    try:
        jit_cache_version = importlib.metadata.version("flashinfer-jit-cache")
    except importlib.metadata.PackageNotFoundError:
        jit_cache_version = None
    if jit_cache_version is not None:
        raise RuntimeError(
            "flashinfer-jit-cache must be absent on SM75; found "
            f"{jit_cache_version}"
        )
    if torch.__version__.partition("+")[0] != "2.13.0":
        raise RuntimeError(f"Expected Torch 2.13.0, found {torch.__version__}")
    if torch.version.cuda != "12.9":
        raise RuntimeError(f"Expected CUDA 12.9, found {torch.version.cuda}")

    from vllm.config.model import str_dtype_to_torch_dtype
    from vllm.entrypoints.serve.utils.api_utils import redact_sensitive_args

    redacted = redact_sensitive_args(
        {"api_key": ["build-secret"], "hf_token": "build-token", "model": "ok"}
    )
    if redacted != {
        "api_key": "***",
        "hf_token": "***",
        "model": "ok",
    }:
        raise RuntimeError(f"Sensitive argument redaction failed: {redacted}")
    if str_dtype_to_torch_dtype("float16") is not torch.float16:
        raise RuntimeError("String HF dtype override was not normalized")

    result = {
        "packages": installed,
        "flashinfer_jit_cache": "absent",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "target_compute_capability": "7.5",
    }
    output = Path("/opt/vllm-turing/evidence/runtime-contract.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
