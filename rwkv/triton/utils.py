"""Triton kernel utility functions."""

import torch


def supports_triton() -> bool:
    if not torch.cuda.is_available():
        return False

    try:
        import triton

        assert triton is not None
        return True
    except ImportError:
        return False
