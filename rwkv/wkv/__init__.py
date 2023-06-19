"""Implements the WKV part of the RWKV model.

This provides a few different implementations of the WKV algorithm, which
is used to compute the output of the model.
"""

import functools
import logging
import warnings
from typing import Callable, Literal

import torch
from torch import Tensor

from .eps import initial_state_with_eps, wkv_with_eps
from .log import initial_state_log_space, wkv_log_space
from .vanilla import initial_state_vanilla, wkv_vanilla

logger = logging.getLogger(__name__)

WkvImpl = Literal["triton", "vanilla", "eps", "log"]

# The WKV function takes the arguments (w, u, k, v, state) and returns the
# tuple (wkv, state). The state should be a single tensor.
WkvFn = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]]
WkvInitState = Tensor


@functools.lru_cache
def supports_triton() -> bool:
    return torch.cuda.is_available()


def get_wkv_fn(emb_dim: int, impl: WkvImpl | None = None) -> tuple[WkvFn, WkvInitState]:
    """Returns the WKV function to use and the hidden state.

    The function takes the

    Args:
        emb_dim: The embedding dimension.
        impl: The implementation to use.

    Returns:
        The WKV function to use.
    """
    if impl is None or impl == "triton":
        try:
            from rwkv.triton.wkv_kernel import initial_state_triton, triton_wkv

            return triton_wkv, initial_state_triton(emb_dim)

        except ImportError:
            if impl is None:
                warnings.warn("Triton is not available, falling back to vanilla implementation.")
                impl = "vanilla"
            else:
                raise

    match impl:
        case "vanilla":
            return wkv_vanilla, initial_state_vanilla(emb_dim)
        case "log":
            return wkv_log_space, initial_state_log_space(emb_dim)
        case "eps":
            return wkv_with_eps, initial_state_with_eps(emb_dim)
        case _:
            raise ValueError(f"Unknown implementation: {impl}")
