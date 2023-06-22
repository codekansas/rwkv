"""Implements the WKV part of the RWKV model.

This provides a few different implementations of the WKV algorithm, which
is used to compute the output of the model.
"""

import logging
import os
import warnings
from typing import Callable, Literal, cast, get_args

from torch import Tensor

from rwkv.triton.utils import supports_triton
from rwkv.wkv.eps import initial_state_with_eps, wkv_with_eps
from rwkv.wkv.log import initial_state_log_space, wkv_log_space
from rwkv.wkv.vanilla import initial_state_vanilla, wkv_vanilla

logger = logging.getLogger(__name__)

WkvImpl = Literal["vanilla", "eps", "log"]

# The WKV function takes the arguments (w, u, k, v, state) and returns the
# tuple (wkv, state). The state should be a single tensor.
WkvFn = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]]
WkvInitState = Tensor


def get_default_impl() -> WkvImpl:
    if "WKV_IMPL" in os.environ:
        assert (wkv_impl := os.environ["WKV_IMPL"]) in get_args(WkvImpl)
        return cast(WkvImpl, wkv_impl)

    warnings.warn("WKV_IMPL environment variable not set; using default")
    return "log"


def get_wkv_fn(emb_dim: int, impl: WkvImpl | None = None, use_triton: bool = True) -> tuple[WkvFn, WkvInitState]:
    """Returns the WKV function to use and the hidden state.

    The function takes the

    Args:
        emb_dim: The embedding dimension.
        impl: The implementation to use.
        use_triton: Whether to use the Triton implementation if available.

    Returns:
        The WKV function to use.
    """
    if impl is None:
        impl = get_default_impl()

    match impl:
        case "vanilla":
            if use_triton and supports_triton():
                from rwkv.triton.wkv.vanilla import wkv_triton_vanilla

                return wkv_triton_vanilla, initial_state_vanilla(emb_dim)

            return wkv_vanilla, initial_state_vanilla(emb_dim)
        case "log":
            if use_triton and supports_triton():
                from rwkv.triton.wkv.log import wkv_triton_log_space

                return wkv_triton_log_space, initial_state_log_space(emb_dim)

            return wkv_log_space, initial_state_log_space(emb_dim)
        case "eps":
            if use_triton and supports_triton():
                from rwkv.triton.wkv.eps import wkv_triton_with_eps

                return wkv_triton_with_eps, initial_state_with_eps(emb_dim)

            return wkv_with_eps, initial_state_with_eps(emb_dim)
        case _:
            raise ValueError(f"Unknown implementation: {impl}")
