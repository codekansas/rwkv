"""Implements the WKV part of the RWKV model.

This provides a few different implementations of the WKV algorithm, which
is used to compute the output of the model.
"""

import functools
import logging
from typing import Callable, Literal

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

WkvImpl = Literal["vanilla", "eps", "log", "triton"]


@functools.lru_cache
def supports_triton() -> bool:
    return torch.cuda.is_available()


def _wkv_vanilla(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    """Runs the core WKV computation.

    Args:
        w: The decay tensor, with shape (D)
        u: The output multiplier tensor, with shape (D)
        k: The K tensor, with shape (B, T, D)
        v: The V tensor, with shape (B, T, D)
        state: The state tensor, with shape (B, 2, D), consisting of the
            alpha and beta tensors, each with shape (B, 1, D)

    Returns:
        The WKV tensor, with shape (B, T, D), and the next state, with shape
        (B, 2, D), consisting of the next alpha and beta tensors, each with
        shape (B, 1, D)
    """
    assert w.dim() == u.dim() == 1
    assert k.dim() == v.dim() == state.dim()

    alpha, beta = state.chunk(2, dim=1)

    _, tsz, _ = k.shape

    ew = torch.exp(-torch.exp(w))

    wkvs = []

    for t in range(tsz):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]
        euk = torch.exp(u + kt)
        wkv = (alpha + euk * vt) / (beta + euk)
        wkvs.append(wkv)

        ek = torch.exp(kt)
        alpha = ew * alpha + ek * vt
        beta = ew * beta + ek

    return torch.cat(wkvs, 1), torch.cat((alpha, beta), dim=1)


def _wkv_with_eps(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    """Runs the core WKV computation.

    Args:
        w: The decay tensor, with shape (D)
        u: The output multiplier tensor, with shape (D)
        k: The K tensor, with shape (B, T, D)
        v: The V tensor, with shape (B, T, D)
        state: The last state, with shape (B, 3, D), consisting of the last
            alpha, beta and epsilon tensors, each with shape (B, 1, D)

    Returns:
        The WKV tensor, with shape (B, T, D), and the next state, with shape
        (B, 3, D), consisting of the next alpha, beta and epsilon tensors,
        each with shape (B, 1, D)
    """
    assert w.dim() == u.dim() == 1
    assert k.dim() == v.dim() == state.dim() == state.dim() == 3

    alpha, beta, eps = state.chunk(3, dim=1)

    _, tsz, _ = k.shape

    w = -torch.exp(w)  # (D)

    wkvs = []

    for t in range(tsz):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]
        ukt = u + kt
        tau = torch.maximum(ukt, eps)
        e1 = torch.exp(eps - tau)
        e2 = torch.exp(ukt - tau)
        wkv = (e1 * alpha + e2 * vt) / (e1 * beta + e2)
        wkvs.append(wkv)

        eps = torch.maximum(w, kt)
        e1 = torch.exp(w - eps)
        e2 = torch.exp(kt - eps)
        alpha = e1 * alpha + e2 * vt
        beta = e1 * beta + e2

    return torch.cat(wkvs, 1), torch.cat((alpha, beta, eps), dim=1)


def _wkv_log_space(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    """Runs the core WKV computation.

    Args:
        w: The decay tensor, with shape (D)
        u: The output multiplier tensor, with shape (D)
        k: The K tensor, with shape (B, T, D)
        v: The V tensor, with shape (B, T, D)
        state: The last state, with shape (B, 3, D), consisting of the last
            alpha, beta and epsilon tensors, each with shape (B, 1, D)

    Returns:
        The WKV tensor, with shape (B, T, D), and the next state, with shape
        (B, 3, D), consisting of the next alpha, beta and epsilon tensors,
        each with shape (B, 1, D)
    """
    assert w.dim() == u.dim() == 1
    assert k.dim() == v.dim() == state.dim()

    log_alpha_plus, log_alpha_minus, log_beta = state.chunk(3, dim=1)

    _, tsz, _ = k.shape

    w = -torch.exp(w)  # (D)
    wkvs = []

    for t in range(tsz):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]
        v_plus = torch.clamp(vt, min=0) + 1e-9
        v_minus = torch.clamp(-vt, min=0) + 1e-9
        log_v_plus = torch.log(v_plus)
        log_v_minus = torch.log(v_minus)

        log_wkv_plus = torch.logaddexp(u + kt + log_v_plus, log_alpha_plus) - torch.logaddexp(u + kt, log_beta)
        log_wkv_minus = torch.logaddexp(u + kt + log_v_minus, log_alpha_minus) - torch.logaddexp(u + kt, log_beta)

        wkv = torch.exp(log_wkv_plus) - torch.exp(log_wkv_minus)
        wkvs.append(wkv)

        log_alpha_plus = torch.logaddexp(w + log_alpha_plus, kt + log_v_plus)
        log_alpha_minus = torch.logaddexp(w + log_alpha_minus, kt + log_v_minus)
        log_beta = torch.logaddexp(w + log_beta, kt)

    return torch.cat(wkvs, 1), torch.cat((log_alpha_plus, log_alpha_minus, log_beta), dim=1)


def get_wkv_fn(
    emb_dim: int,
    impl: WkvImpl = "triton",
) -> tuple[Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]], Tensor]:
    """Returns the WKV function to use and the hidden state.

    The function takes the

    Args:
        emb_dim: The embedding dimension.
        impl: The implementation to use. Can be ``"vanilla"``, ``"log"``,
            ``"eps"``, or ``"triton"``.

    Returns:
        The WKV function to use.
    """
    if impl == "triton":
        if supports_triton():
            from rwkv.triton.wkv_kernel import triton_wkv

            return triton_wkv, torch.full((1, 3, emb_dim), float("-inf"))

        logger.warning("Triton implementation is not available; falling back to log implementation")
        impl = "log"

    match impl:
        case "vanilla":
            return _wkv_vanilla, torch.zeros(1, 3, emb_dim)
        case "log":
            return _wkv_log_space, torch.full((1, 3, emb_dim), float("-inf"))
        case "eps":
            return _wkv_with_eps, torch.zeros(1, 1, emb_dim)
        case _:
            raise ValueError(f"Unknown implementation: {impl}")
