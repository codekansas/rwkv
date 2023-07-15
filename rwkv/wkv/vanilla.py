# mypy: disable-error-code="override"
"""Provides a vanilla implementation of the WKV computation.

This implementation is not numerically stable, it is provided mainly for
educational purposes.
"""

import torch
from torch import Tensor
from torch.autograd.function import Function, FunctionCtx, once_differentiable


@torch.jit.script
def wkv_vanilla_forward(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    assert w.shape == u.shape == (chans,)
    assert v.shape == (bsz, tsz, chans)
    assert state.shape == (bsz, 2, 1, chans)

    alpha, beta = state[:, :, -1].chunk(2, dim=1)  # (B, 1, D), (B, 1, D)

    ew = torch.exp(-w)

    wkvs = []
    alphas = [alpha]
    betas = [beta]

    for t in range(tsz):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]
        euk = torch.exp(u + kt)
        wkv = (alpha + euk * vt) / (beta + euk)
        wkvs.append(wkv)

        ek = torch.exp(kt)
        alpha = ew * alpha + ek * vt
        beta = ew * beta + ek

        alphas.append(alpha)
        betas.append(beta)

    alpha = torch.stack(alphas, dim=2)
    beta = torch.stack(betas, dim=2)

    return torch.cat(wkvs, 1), torch.cat((alpha, beta), dim=1)


@torch.jit.script
def wkv_vanilla_backward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    grad_wkv: Tensor,
    grad_state: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    assert w.shape == u.shape == (chans,), f"{w.shape}, {u.shape} != {(chans,)}"
    assert v.shape == (bsz, tsz, chans), f"{v.shape} != {(bsz, tsz, chans)}"
    assert state.shape == (bsz, 2, tsz, chans), f"{state.shape} != {(bsz, 2, tsz, chans)}"
    assert grad_wkv.shape == (bsz, tsz, chans), f"{grad_wkv.shape} != {(bsz, tsz, chans)}"
    assert grad_state.shape == (bsz, 2, 1, chans), f"{grad_state.shape} != {(bsz, 2, 1, chans)}"

    alpha, beta = state.chunk(2, dim=1)  # (B, 1, T + 1, D), (B, 1, T + 1, D)
    grad_alpha, grad_beta = grad_state[:, :, 0].chunk(2, dim=1)  # (B, 1, D), (B, 1, D)

    ew = torch.exp(-w)

    grad_w = torch.zeros_like(w)
    grad_u = torch.zeros_like(u)
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)

    for t in range(tsz - 1, -1, -1):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]
        alpha_prev, beta_prev = alpha[:, :, t], beta[:, :, t]
        euk = torch.exp(u + kt)
        ek = torch.exp(kt)

        denom = beta_prev + euk
        denom_sq = denom * denom

        grad_wkvt = grad_wkv[:, t : t + 1]

        # Backpropagates wkv gradients.
        grad_uk = grad_wkvt * euk * (beta_prev * vt - alpha_prev) / denom_sq
        grad_u += grad_uk.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_uk
        grad_v[:, t : t + 1] += grad_wkvt * euk / denom

        grad_alpha_wkv = grad_wkvt / denom
        grad_beta_wkv = -grad_wkvt * (euk * vt + alpha_prev) / denom_sq

        # Backpropagates alpha gradients.
        grad_w += (grad_alpha * ew * alpha_prev).flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_alpha * ek * vt
        grad_v[:, t : t + 1] += grad_alpha * ek

        # Backpropagates beta gradients.
        grad_w += (grad_beta * ew * beta_prev).flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_beta * ek

        # Computes gradients for alpha and beta.
        grad_alpha = grad_alpha * ew + grad_alpha_wkv
        grad_beta = grad_beta * ew + grad_beta_wkv

    return -grad_w, grad_u, grad_k, grad_v, torch.stack((grad_alpha, grad_beta), dim=1)


class WkvVanilla(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        w: Tensor,
        u: Tensor,
        k: Tensor,
        v: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        wkv, state_out = wkv_vanilla_forward(w, u, k, v, state)
        ctx.save_for_backward(w, u, k, v, state_out[:, :, :-1])
        return wkv, state_out[:, :, -1:]

    @staticmethod
    @once_differentiable
    def backward(
        ctx: FunctionCtx,
        grad_wkv: Tensor,
        grad_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        w, u, k, v, state = ctx.saved_tensors
        return wkv_vanilla_backward(w, u, k, v, state, grad_wkv, grad_state)


def initial_state_vanilla(emb_dim: int) -> Tensor:
    return torch.zeros(1, 2, 1, emb_dim)


def wkv_vanilla(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    """Runs the core WKV computation.

    Args:
        w: The decay tensor, with shape (D)
        u: The output multiplier tensor, with shape (D)
        k: The K tensor, with shape (B, T, D)
        v: The V tensor, with shape (B, T, D)
        state: The state tensor, with shape (B, 2, T, D), consisting of the
            alpha and beta tensors, each with shape (B, 1, T, D)

    Returns:
        The WKV tensor, with shape (B, T, D), and the next state, with shape
        (B, 2, 1, D), consisting of the next alpha and beta tensors, each with
        shape (B, 1, 1, D)
    """
    return WkvVanilla.apply(w, u, k, v, state)
