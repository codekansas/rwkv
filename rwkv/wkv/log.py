# mypy: disable-error-code="override"
"""Provides a numerically-stable implementation of the WKV computation.

This implementation uses log-space state variables, verses the original
implementation which offsets the exponents.
"""

import math
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
    bsz, tsz, chans = k.shape

    assert w.shape == u.shape == (chans,)
    assert v.shape == (bsz, tsz, chans)
    assert state.shape == (bsz, 3, 1, chans)

    ln_alpha_p, ln_alpha_m, ln_beta = state[:, :, -1].chunk(3, dim=1)

    log_eps = math.log(eps)

    wkvs = []
    ln_alpha_ps = [ln_alpha_p]
    ln_alpha_ms = [ln_alpha_m]
    ln_betas = [ln_beta]

    def logaddexp(a: Tensor, b: Tensor) -> Tensor:
        max_av = torch.maximum(a, b)
        return max_av + torch.log(torch.exp(a - max_av) + torch.exp(b - max_av))

    def logsubexp(a: Tensor, b: Tensor) -> Tensor:
        max_av = torch.maximum(torch.maximum(a, b), torch.full_like(a, log_eps))
        return max_av + torch.log(torch.exp(a - max_av) - torch.exp(b - max_av))

    for t in range(tsz):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]
        v_plus = torch.clamp(vt, min=0) + eps
        v_minus = torch.clamp(-vt, min=0) + eps
        ln_v_p = torch.log(v_plus)
        ln_v_m = torch.log(v_minus)

        ln_alpha_pn = torch.minimum(ln_alpha_p, ln_alpha_m) - eps
        ln_alpha_p = logsubexp(ln_alpha_p, ln_alpha_pn)
        ln_alpha_m = logsubexp(ln_alpha_m, ln_alpha_pn)

        ln_wkv_p = logaddexp(u + kt + ln_v_p, ln_alpha_p) - logaddexp(u + kt, ln_beta)
        ln_wkv_m = logaddexp(u + kt + ln_v_m, ln_alpha_m) - logaddexp(u + kt, ln_beta)

        wkv = torch.exp(ln_wkv_p) - torch.exp(ln_wkv_m)
        wkvs.append(wkv)

        ln_alpha_p = logaddexp(w + ln_alpha_p, kt + ln_v_p)
        ln_alpha_m = logaddexp(w + ln_alpha_m, kt + ln_v_m)
        ln_beta = logaddexp(w + ln_beta, kt)

        ln_alpha_ps.append(ln_alpha_p)
        ln_alpha_ms.append(ln_alpha_m)
        ln_betas.append(ln_beta)

    ln_alpha_p = torch.stack(ln_alpha_ps, dim=2)
    ln_alpha_m = torch.stack(ln_alpha_ms, dim=2)
    ln_beta = torch.stack(ln_betas, dim=2)

    return torch.cat(wkvs, 1), torch.cat((ln_alpha_p, ln_alpha_m, ln_beta), dim=1)


def wkv_log_space_backward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    grad_wkv: Tensor,
    grad_state: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    assert w.shape == u.shape == (chans,)
    assert v.shape == (bsz, tsz, chans)
    assert state.shape == (bsz, 3, tsz + 1, chans)
    assert grad_wkv.shape == (bsz, tsz, chans)
    assert grad_state.shape == (bsz, 3, 1, chans)

    ln_alpha_p, ln_alpha_m, log_beta = state.chunk(3, dim=1)

    grad_w = torch.zeros_like(w)
    grad_u = torch.zeros_like(u)
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)

    for t in reversed(range(tsz)):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]
        ln_alpha_p_prev, ln_alpha_m_prev, ln_beta_prev = ln_alpha_p[:, :, t], ln_alpha_m[:, :, t], log_beta[:, :, t]


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
        wkv, state_out = wkv_log_space_forward(w, u, k, v, state)
        ctx.save_for_backward(w, u, k, v, state_out)
        return wkv, state_out[:, :, -1:]

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
    return torch.full((1, 3, 1, emb_dim), float("-inf"))


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
