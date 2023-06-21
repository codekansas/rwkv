# mypy: disable-error-code="import, no-untyped-def, override"
# ruff: noqa: ANN001, ANN201, ANN202, N803, N806
"""Defines Triton kernels for numerically-stable RWKV forward and backward passes."""

from typing import cast

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import Function, FunctionCtx, once_differentiable


@triton.jit
def wkv_triton_with_eps_forward_kernel(
    # W
    w_ptr,
    w_s_c,
    # U
    u_ptr,
    u_s_c,
    # K
    k_ptr,
    k_s_b,
    k_s_t,
    k_s_c,
    # V
    v_ptr,
    v_s_b,
    v_s_t,
    v_s_c,
    # State
    state_ptr,
    state_s_b,
    state_s_abe,
    state_s_c,
    # WKV
    wkv_ptr,
    wkv_s_b,
    wkv_s_t,
    wkv_s_c,
    # Output state
    state_out_ptr,
    state_out_s_b,
    state_out_s_abe,
    state_out_s_t,
    state_out_s_c,
    # Params
    chans,
    tsz,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Parallelize over the batch dimension.
    b_idx = tl.program_id(0)

    cs = tl.arange(0, BLOCK_SIZE_C)
    cmask = cs < chans

    # Pointers to the batch (and possibly channel) for the input tensors.
    k_ptr = k_ptr + b_idx * k_s_b
    v_ptr = v_ptr + b_idx * v_s_b
    alpha_ptr = state_ptr + b_idx * state_s_b
    beta_ptr = state_ptr + b_idx * state_s_b + state_s_abe
    eps_ptr = state_ptr + b_idx * state_s_b + 2 * state_s_abe

    # Pointers to the batch (and possibly channel) for the output tensors.
    wkv_ptr = wkv_ptr + b_idx * wkv_s_b
    alpha_out_ptr = state_out_ptr + b_idx * state_out_s_b
    beta_out_ptr = state_out_ptr + b_idx * state_out_s_b + state_out_s_abe
    eps_out_ptr = state_out_ptr + b_idx * state_out_s_b + 2 * state_out_s_abe

    # Loads parameters.
    alpha = tl.load(alpha_ptr + cs * state_s_c, mask=cmask)
    beta = tl.load(beta_ptr + cs * state_s_c, mask=cmask)
    eps = tl.load(eps_ptr + cs * state_s_c, mask=cmask)
    w = tl.load(w_ptr + cs * w_s_c, mask=cmask)
    u = tl.load(u_ptr + cs * u_s_c, mask=cmask)

    for t in range(tsz):
        kt = tl.load(k_ptr + t * k_s_t + cs * k_s_c, mask=cmask)
        vt = tl.load(v_ptr + t * v_s_t + cs * v_s_c, mask=cmask)

        ukt = u + kt
        tau = tl.maximum(ukt, eps)
        e1 = tl.exp(eps - tau)
        e2 = tl.exp(ukt - tau)
        wkv = (e1 * alpha + e2 * vt) / (e1 * beta + e2)
        tl.store(wkv_ptr + t * wkv_s_t + cs * wkv_s_c, wkv, mask=cmask)

        w_eps = w + eps
        eps = tl.maximum(w_eps, kt)
        e1 = tl.exp(w_eps - eps)
        e2 = tl.exp(kt - eps)
        alpha = e1 * alpha + e2 * vt
        beta = e1 * beta + e2
        tl.store(alpha_out_ptr + t * state_out_s_t + cs * state_out_s_c, alpha, mask=cmask)
        tl.store(beta_out_ptr + t * state_out_s_t + cs * state_out_s_c, beta, mask=cmask)
        tl.store(eps_out_ptr + t * state_out_s_t + cs * state_out_s_c, eps, mask=cmask)


def wkv_triton_with_eps_forward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
) -> tuple[Tensor, Tensor]:
    (bsz, tsz, chans), device, dtype = k.shape, k.device, k.dtype

    # Checks tensor shapes.
    assert v.shape == (bsz, tsz, chans), f"{v.shape} != {(bsz, tsz, chans)}"
    assert state.shape == (bsz, 3, 1, chans), f"{state.shape} != {(bsz, 3, 1, chans)}"
    assert w.shape == (chans,), f"{w.shape} != {(chans,)}"
    assert u.shape == (chans,), f"{u.shape} != {(chans,)}"

    # Checks tensor dtypes and devices.
    for t in (v, state, w, u):
        assert t.dtype == dtype and t.device == device, f"{t.dtype} != {dtype} or {t.device} != {device}"

    # New tensors to output.
    wkvs = k.new_empty(bsz, tsz, chans)
    state_out = k.new_empty(bsz, 3, tsz, chans)

    wkv_triton_with_eps_forward_kernel[(bsz, chans)](
        # W
        w,
        w.stride(0),
        # U
        u,
        u.stride(0),
        # K
        k,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        # V
        v,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        # State
        state,
        state.stride(0),
        state.stride(1),
        state.stride(3),
        # WKV
        wkvs,
        wkvs.stride(0),
        wkvs.stride(1),
        wkvs.stride(2),
        # Output state
        state_out,
        state_out.stride(0),
        state_out.stride(1),
        state_out.stride(2),
        state_out.stride(3),
        # Params
        chans,
        tsz,
        BLOCK_SIZE_C=min(triton.next_power_of_2(chans), 32),
    )

    state_out = torch.cat((state, state_out), dim=2)

    return wkvs, state_out


@triton.jit
def wkv_with_eps_triton_backward_kernel(
    # W
    w_ptr,
    w_s_c,
    # U
    u_ptr,
    u_s_c,
    # K
    k_ptr,
    k_s_b,
    k_s_t,
    k_s_c,
    # V
    v_ptr,
    v_s_b,
    v_s_t,
    v_s_c,
    # State
    state_ptr,
    state_s_b,
    state_s_abe,
    state_s_t,
    state_s_c,
    # WKV grad
    gwkv_ptr,
    gwkv_s_b,
    gwkv_s_t,
    gwkv_s_c,
    # Output state grad
    gstate_out_ptr,
    gstate_out_s_b,
    gstate_out_s_abe,
    gstate_out_s_c,
    # W grad
    gw_ptr,
    gw_s_c,
    # U grad
    gu_ptr,
    gu_s_c,
    # K grad
    gk_ptr,
    gk_s_b,
    gk_s_t,
    gk_s_c,
    # V grad
    gv_ptr,
    gv_s_b,
    gv_s_t,
    gv_s_c,
    # State grad
    gstate_ptr,
    gstate_s_b,
    gstate_s_abe,
    gstate_s_c,
    # Params
    tsz,
    chans,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Parallelize over the batch dimension.
    b_idx = tl.program_id(0)

    cs = tl.arange(0, BLOCK_SIZE_C)
    cmask = cs < chans

    # Pointers to the batch (and possibly channel) for the input tensors.
    k_ptr = k_ptr + b_idx * k_s_b
    v_ptr = v_ptr + b_idx * v_s_b
    alpha_ptr = state_ptr + b_idx * state_s_b
    beta_ptr = state_ptr + b_idx * state_s_b + state_s_abe
    eps_ptr = state_ptr + b_idx * state_s_b + 2 * state_s_abe

    # Pointers to the batch (and possibly channel) for the output tensors.
    gk_ptr = gk_ptr + b_idx * gk_s_b
    gv_ptr = gv_ptr + b_idx * gv_s_b

    # Pointers to gradients which were recieved by the function.
    gwkv_ptr = gwkv_ptr + b_idx * gwkv_s_b
    galpha_out_ptr = gstate_out_ptr + b_idx * gstate_out_s_b
    gbeta_out_ptr = gstate_out_ptr + b_idx * gstate_out_s_b + gstate_out_s_abe
    geps_out_ptr = gstate_out_ptr + b_idx * gstate_out_s_b + 2 * gstate_out_s_abe

    # Loads parameters.
    galpha = tl.load(galpha_out_ptr + gstate_out_s_c * cs, mask=cmask)
    gbeta = tl.load(gbeta_out_ptr + gstate_out_s_c * cs, mask=cmask)
    geps = tl.load(geps_out_ptr + gstate_out_s_c * cs, mask=cmask)
    w = tl.load(w_ptr + w_s_c * cs, mask=cmask)
    u = tl.load(u_ptr + u_s_c * cs, mask=cmask)

    # Gradient accumulators.
    gw = tl.zeros_like(w)
    gu = tl.zeros_like(u)

    alpha_prev = tl.load(alpha_ptr + tsz * state_s_t + state_s_c * cs, mask=cmask)
    beta_prev = tl.load(beta_ptr + tsz * state_s_t + state_s_c * cs, mask=cmask)
    eps_prev = tl.load(eps_ptr + tsz * state_s_t + state_s_c * cs, mask=cmask)

    for t in range(tsz):
        tc = tsz - t - 1

        kt = tl.load(k_ptr + tc * k_s_t + k_s_c * cs, mask=cmask)
        vt = tl.load(v_ptr + tc * v_s_t + v_s_c * cs, mask=cmask)

        alpha_curr = alpha_prev
        beta_curr = beta_prev
        eps_curr = eps_prev

        alpha_prev = tl.load(alpha_ptr + tc * state_s_t + state_s_c * cs, mask=cmask)
        beta_prev = tl.load(beta_ptr + tc * state_s_t + state_s_c * cs, mask=cmask)
        eps_prev = tl.load(eps_ptr + tc * state_s_t + state_s_c * cs, mask=cmask)

        ukt = u + kt
        tau = tl.maximum(ukt, eps_prev)
        e1 = tl.exp(eps_prev - tau)
        e2 = tl.exp(ukt - tau)

        euke = tl.exp(ukt + eps_prev - 2 * tau)

        denom = e1 * beta_prev + e2
        denom_sq = denom * denom

        gwkvt = tl.load(gwkv_ptr + tc * gwkv_s_t + gwkv_s_c * cs, mask=cmask)

        # Backpropagates wkv gradients.
        guk = gwkvt * e2 * (e1 * beta_prev * vt - e1 * alpha_prev) / denom_sq
        gu += guk
        gk = guk
        gv = gwkvt * e2 / denom

        galpha_wkv = gwkvt * e1 / denom
        gbeta_wkv = -gwkvt * e1 * (e2 * vt + e1 * alpha_prev) / denom_sq
        geps_wkv = gwkvt * euke * (alpha_prev - vt * beta_prev) / (e1 * beta_prev + e2) ** 2

        e1 = tl.exp(w + eps_prev - eps_curr)
        e2 = tl.exp(kt - eps_curr)

        # Backpropagates alpha gradients.
        galpha_we = galpha * e1 * alpha_prev
        gw += galpha_we
        gk += galpha * e2 * vt
        gv += galpha * e2
        geps += galpha * -alpha_curr

        # Backpropagates beta gradients.
        gbeta_we = gbeta * e1 * beta_prev
        gw += gbeta_we
        gk += gbeta * e2
        geps += gbeta * -beta_curr

        # Backpropagates epsilon gradients.
        geps_mask = w + eps_prev > kt
        geps_we = tl.where(geps_mask, geps, tl.zeros_like(geps))
        gw += geps_we
        gk += tl.where(geps_mask, tl.zeros_like(geps), geps)

        # Stores the gradients for k and v.
        tl.store(gk_ptr + tc * gk_s_t + gk_s_c * cs, gk, mask=cmask)
        tl.store(gv_ptr + tc * gv_s_t + gv_s_c * cs, gv, mask=cmask)

        # Computes new gradients for alpha and beta.
        galpha = galpha * e1 + galpha_wkv
        gbeta = gbeta * e1 + gbeta_wkv
        geps = galpha_we + gbeta_we + geps_we + geps_wkv

    # Stores final gradients for alpha and beta.
    galpha_ptr = gstate_ptr + b_idx * gstate_s_b
    gbeta_ptr = gstate_ptr + b_idx * gstate_s_b + gstate_s_abe
    geps_ptr = gstate_ptr + b_idx * gstate_s_b + 2 * gstate_s_abe
    tl.store(galpha_ptr + gstate_s_c * cs, galpha, mask=cmask)
    tl.store(gbeta_ptr + gstate_s_c * cs, gbeta, mask=cmask)
    tl.store(geps_ptr + gstate_s_c * cs, geps, mask=cmask)

    # Stores final gradients for w and u.
    tl.atomic_add(gw_ptr + gw_s_c * cs, gw, mask=cmask)
    tl.atomic_add(gu_ptr + gu_s_c * cs, gu, mask=cmask)


def wkv_triton_with_eps_backward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    grad_wkv: Tensor,
    grad_state: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    (bsz, tsz, chans), device, dtype = k.shape, k.device, k.dtype

    # Checks tensor shapes.
    assert v.shape == (bsz, tsz, chans), f"{v.shape} != {(bsz, tsz, chans)}"
    assert state.shape == (bsz, 3, tsz, chans), f"{state.shape} != {(bsz, 3, tsz, chans)}"
    assert w.shape == (chans,), f"{w.shape} != {(chans,)}"
    assert u.shape == (chans,), f"{u.shape} != {(chans,)}"
    assert grad_wkv.shape == (bsz, tsz, chans)
    assert grad_state.shape == (bsz, 3, 1, chans)

    # Checks tensor dtypes and devices.
    for t in (v, state, w, u, grad_wkv, grad_state):
        assert t.dtype == dtype and t.device == device, f"{t.dtype} != {dtype} or {t.device} != {device}"

    # New tensors to output.
    gw = torch.zeros_like(w)
    gu = torch.zeros_like(u)
    gk = torch.empty_like(k)
    gv = torch.empty_like(v)
    gstate = k.new_empty(bsz, 3, 1, chans)

    wkv_with_eps_triton_backward_kernel[(bsz,)](
        # W
        w,
        w.stride(0),
        # U
        u,
        u.stride(0),
        # K
        k,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        # V
        v,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        # State
        state,
        state.stride(0),
        state.stride(1),
        state.stride(2),
        state.stride(3),
        # WKV grad
        grad_wkv,
        grad_wkv.stride(0),
        grad_wkv.stride(1),
        grad_wkv.stride(2),
        # Output state grad
        grad_state,
        grad_state.stride(0),
        grad_state.stride(1),
        grad_state.stride(3),
        # W grad
        gw,
        gw.stride(0),
        # U grad
        gu,
        gu.stride(0),
        # K grad
        gk,
        gk.stride(0),
        gk.stride(1),
        gk.stride(2),
        # V grad
        gv,
        gv.stride(0),
        gv.stride(1),
        gv.stride(2),
        # State grad
        gstate,
        gstate.stride(0),
        gstate.stride(1),
        gstate.stride(3),
        # Params
        tsz,
        chans,
        BLOCK_SIZE_C=min(triton.next_power_of_2(chans), 32),
    )

    return gw, gu, gk, gv, gstate


class WKVTritonFunction(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        w: Tensor,
        u: Tensor,
        k: Tensor,
        v: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        wkv, state_out = wkv_triton_with_eps_forward(w, u, k, v, state)
        ctx.save_for_backward(w, u, k, v, state_out[:, :, :-1])
        return wkv, state_out[:, :, -1:]

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, gwkv: Tensor, gstate: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        w, u, k, v, state = cast(tuple[Tensor, ...], ctx.saved_tensors)
        gw, gu, gk, gv, gstate = wkv_triton_with_eps_backward(w, u, k, v, state, gwkv, gstate)
        return gw, gu, gk, gv, gstate


def wkv_triton_with_eps(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    return WKVTritonFunction.apply(w, u, k, v, state)
