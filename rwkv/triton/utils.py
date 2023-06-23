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


def largest_div_power_of_2(n: int, init_k: int = 32) -> int:
    k = init_k
    while k * 2 <= n:
        k *= 2
    return k


def get_block_size_c(chans: int) -> int:
    if chans < 32:
        return 32
    if chans < 64:
        return 64
    return 128
