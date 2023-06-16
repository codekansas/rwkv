# mypy: disable-error-code="override"
"""Provides a numerically-stable implementation of the WKV computation.

This implementation follows the official implementation.
"""

import torch
from torch import Tensor
from torch.autograd.function import Function, FunctionCtx, once_differentiable


def wkv_with_eps_forward(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    assert w.dim() == u.dim() == 1
    assert k.dim() == v.dim() == state.dim() == state.dim() == 3

    alpha, beta, eps = state.chunk(3, dim=1)

    _, tsz, _ = k.shape

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


def wkv_with_eps_backward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    grad_wkv: Tensor,
    grad_state: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    raise NotImplementedError


class WkvWithEps(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        w: Tensor,
        u: Tensor,
        k: Tensor,
        v: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ctx.save_for_backward(w, u, k, v, state)
        return wkv_with_eps_forward(w, u, k, v, state)

    @staticmethod
    @once_differentiable
    def backward(
        ctx: FunctionCtx,
        grad_wkv: Tensor,
        grad_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        w, u, k, v, state = ctx.saved_tensors
        return wkv_with_eps_backward(w, u, k, v, state, grad_wkv, grad_state)


def initial_state_with_eps(emb_dim: int) -> Tensor:
    return torch.zeros(1, 3, emb_dim)


def wkv_with_eps(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    """Runs the core WKV computation.

    Args:
        w: The decay tensor, with shape (D)
        u: The output multiplier tensor, with shape (D)
        k: The K tensor, with shape (B, T, D)
        v: The V tensor, with shape (B, T, D)
        state: The state tensor, with shape (B, 3, D), consisting of the
            alpha, beta and eps tensors, each with shape (B, 1, D)

    Returns:
        The WKV tensor, with shape (B, T, D), and the next state, with shape
        (B, 3, D), consisting of the next alpha, beta and eps tensors, each
        with shape (B, 1, D)
    """
    return WkvWithEps.apply(w, u, k, v, state)
