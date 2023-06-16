"""Implements the WKV part of the RWKV model."""

from typing import Callable

import torch
from torch import Tensor


def _wkv_with_eps(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
    eps: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Runs the core WKV computation.

    Args:
        w: The decay tensor, with shape (D)
        u: The output multiplier tensor, with shape (D)
        k: The K tensor, with shape (B, T, D)
        v: The V tensor, with shape (B, T, D)
        alpha: The last numerator, with shape (B, 1, D)
        beta: The last denominator, with shape (B, 1, D)
        eps: The epsilon tensor, with shape (B, 1, D)

    Returns:
        The WKV tensor, with shape (B, T, D), and the next alpha, beta and
        epsilon tensors, each with shape (B, 1, D)
    """
    assert w.dim() == u.dim() == 1
    assert k.dim() == v.dim() == alpha.dim() == beta.dim() == 3

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

    return torch.cat(wkvs, 1), alpha, beta, eps


def _wkv_vanilla(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Runs the core WKV computation.

    Args:
        w: The decay tensor, with shape (D)
        u: The output multiplier tensor, with shape (D)
        k: The K tensor, with shape (B, T, D)
        v: The V tensor, with shape (B, T, D)
        alpha: The last numerator, with shape (B, 1, D)
        beta: The last denominator, with shape (B, 1, D)

    Returns:
        The WKV tensor, with shape (B, T, D), and the next alpha and beta
        tensors, each with shape (B, 1, D)
    """
    assert w.dim() == u.dim() == 1
    assert k.dim() == v.dim() == alpha.dim() == beta.dim() == 3

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

    return torch.cat(wkvs, 1), alpha, beta


def _wkv_log_space(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    log_alpha_plus: Tensor,
    log_alpha_minus: Tensor,
    log_beta: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Runs the core WKV computation.

    Args:
        w: The decay tensor, with shape (D)
        u: The output multiplier tensor, with shape (D)
        k: The K tensor, with shape (B, T, D)
        v: The V tensor, with shape (B, T, D)
        log_alpha_plus: The last positive numerator part, with shape (B, 1, D)
        log_alpha_minus: The last negative numerator part, with shape (B, 1, D)
        log_beta: The last denominator, with shape (B, 1, D)

    Returns:
        The WKV tensor, with shape (B, T, D), and the next alpha plus, alpha
        minus and beta tensors, each with shape (B, 1, D)
    """
    assert w.dim() == u.dim() == 1
    assert k.dim() == v.dim() == log_alpha_plus.dim() == log_alpha_minus.dim() == log_beta.dim() == 3

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

    return torch.cat(wkvs, 1), log_alpha_plus, log_alpha_minus, log_beta


def get_wkv_fn() -> Callable:
    """Returns the WKV function to use.

    The function takes six tensors as input, and returns three tensors as
    output. The input tensors are ``w``, ``u``, ``k``, ``v``, ``alpha``, and
    ``beta``, and the output tensors are ``out``, ``alpha``, and ``beta``.

    Returns:
        The WKV function to use.
    """
    if torch.cuda.is_available():
        from rwkv.triton.wkv_kernel import triton_wkv

        return triton_wkv

    return _wkv_log_space
