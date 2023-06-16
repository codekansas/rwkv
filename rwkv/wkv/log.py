# mypy: disable-error-code="override"
"""Provides a numerically-stable implementation of the WKV computation.

This implementation uses log-space state variables, verses the original
implementation which offsets the exponents.
"""

import torch
from torch import Tensor
from torch.autograd.function import Function, FunctionCtx, once_differentiable


def wkv_log_space_forward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    assert w.dim() == u.dim() == 1
    assert k.dim() == v.dim() == state.dim()

    log_alpha_plus, log_alpha_minus, log_beta = state.chunk(3, dim=1)

    _, tsz, _ = k.shape

    wkvs = []

    for t in range(tsz):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]
        v_plus = torch.clamp(vt, min=0) + eps
        v_minus = torch.clamp(-vt, min=0) + eps
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


def wkv_log_space_backward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    grad_wkv: Tensor,
    grad_state: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    raise NotImplementedError


class WkvLogSpace(Function):
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
        return wkv_log_space_forward(w, u, k, v, state)

    @staticmethod
    @once_differentiable
    def backward(
        ctx: FunctionCtx,
        grad_wkv: Tensor,
        grad_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        w, u, k, v, state = ctx.saved_tensors
        return wkv_log_space_backward(w, u, k, v, state, grad_wkv, grad_state)


def initial_state_log_space(emb_dim: int) -> Tensor:
    return torch.full((1, 3, emb_dim), float("-inf"))


def wkv_log_space(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    """Runs the core WKV computation.

    Args:
        w: The decay tensor, with shape (D)
        u: The output multiplier tensor, with shape (D)
        k: The K tensor, with shape (B, T, D)
        v: The V tensor, with shape (B, T, D)
        state: The state tensor, with shape (B, 3, D), consisting of the
            alpha plus, alpha minus and beta tensors, each with shape (B, 1, D)

    Returns:
        The WKV tensor, with shape (B, T, D), and the next state, with shape
        (B, 2, D), consisting of the next alpha plus, alpha minus and beta
        tensors, each with shape (B, 1, D)
    """
    return WkvLogSpace.apply(w, u, k, v, state)
