# mypy: disable-error-code="import"
"""Triton kernel utility functions."""

import os
import warnings

import torch


def supports_triton() -> bool:
    if "USE_TRITON" in os.environ:
        return os.environ["USE_TRITON"] == "1"

    if not torch.cuda.is_available():
        return False

    try:
        import triton

        assert triton is not None
        return True
    except (ImportError, ModuleNotFoundError):
        if torch.cuda.is_available():
            warnings.warn("Triton is not installed, but CUDA is available; install with `pip install triton`")
        return False
