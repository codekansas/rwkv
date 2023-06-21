# mypy: disable-error-code="import, no-untyped-def, override"
# ruff: noqa: ANN001, ANN201, ANN202, N803, N806
"""Defines Triton kernels for the vanilla RWKV forward and backward passes.

This kernel is used to make the WKV computation in the attention layer run
faster while using less memory. It requires that ``triton`` is installed, which
in turn requires a ``triton``-compatible GPU and CUDA version.
"""

from typing import cast

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import Function, FunctionCtx, once_differentiable


@triton.jit
def wkv_triton_vanilla_forward_kernel(
    w_ptr,
    u_ptr,
    k_ptr,
    v_ptr,
    alpha_ptr,
    beta_ptr,
    chans,
    tsz,
    wkv_ptr,
    alpha_out_ptr,
    beta_out_ptr,
):
    # Parallelize over the batch and channel dimensions.
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)

    chans_tsz = chans * tsz

    # Pointers to the batch (and possibly channel) for the input tensors.
    k_ptr = k_ptr + b_idx * chans_tsz + c_idx
    v_ptr = v_ptr + b_idx * chans_tsz + c_idx
    alpha_ptr = alpha_ptr + b_idx * chans + c_idx
    beta_ptr = beta_ptr + b_idx * chans + c_idx
    w_ptr = w_ptr + c_idx
    u_ptr = u_ptr + c_idx

    # Pointers to the batch (and possibly channel) for the output tensors.
    wkv_ptr = wkv_ptr + b_idx * chans_tsz + c_idx
    alpha_out_ptr = alpha_out_ptr + b_idx * chans_tsz + c_idx
    beta_out_ptr = beta_out_ptr + b_idx * chans_tsz + c_idx

    # Loads parameters.
    alpha = tl.load(alpha_ptr)
    beta = tl.load(beta_ptr)
    w = tl.load(w_ptr)
    u = tl.load(u_ptr)

    ew = tl.exp(w)

    for t in range(tsz):
        tc = t * chans

        kt = tl.load(k_ptr + tc)
        vt = tl.load(v_ptr + tc)

        euk = tl.exp(u + kt)

        wkv = (alpha + euk * vt) / (beta + euk)
        tl.store(wkv_ptr + tc, wkv)

        ek = tl.exp(kt)
        alpha = ew * alpha + ek * vt
        beta = ew * beta + ek

        tl.store(alpha_out_ptr + tc, alpha)
        tl.store(beta_out_ptr + tc, beta)


def wkv_triton_vanilla_forward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
) -> tuple[Tensor, Tensor]:
    (bsz, tsz, chans), device, dtype = k.shape, k.device, k.dtype

    # Performs tensor checks.
    for t in (k, v):
        assert t.shape == (bsz, tsz, chans), f"{t.shape} != {(bsz, tsz, chans)}"
        assert t.stride(0) == tsz * chans, f"{t.stride(0)} != {tsz * chans}"
        assert t.stride(1) == chans, f"{t.stride(1)} != {chans}"
        assert t.size(2) == 1 or t.stride(2) == 1, f"{t.stride(2)} != 1"
    assert state.shape == (bsz, 2, 1, chans), f"{state.shape} != {(bsz, 2, 1, chans)}"
    assert state.stride(0) == chans * 2, f"{state.stride(0)} != {chans * 2}"
    assert state.stride(1) == chans, f"{state.stride(1)} != {chans}"
    assert state.stride(2) == chans, f"{state.stride(2)} != 1"
    assert state.stride(3) == 1, f"{state.stride(3)} != 1"
    for t in (w, u):
        assert t.shape == (chans,), f"{t.shape} != {(chans,)}"
        assert t.stride(0) == 1, f"{t.stride(0)} != 1"
    for t in (v, state, w, u):
        assert t.dtype == dtype and t.device == device, f"{t.dtype} != {dtype} or {t.device} != {device}"

    alpha, beta = state[:, :, -1].chunk(2, dim=1)  # (B, 1, D), (B, 1, D)
    alpha, beta = alpha.contiguous(), beta.contiguous()

    # New tensors to output.
    wkvs = k.new_empty(bsz, tsz, chans)
    alpha_out = k.new_empty(bsz, tsz, chans)
    beta_out = k.new_empty(bsz, tsz, chans)

    wkv_triton_vanilla_forward_kernel[(bsz, chans)](
        w,
        u,
        k,
        v,
        alpha,
        beta,
        chans,
        tsz,
        wkvs,
        alpha_out,
        beta_out,
    )

    state_out = torch.stack([torch.cat([alpha, alpha_out], dim=1), torch.cat([beta, beta_out], dim=1)], dim=1)

    return wkvs, state_out


@triton.jit
def wkv_vanilla_triton_backward_kernel(
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
    state_s_ab,
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
    gstate_out_s_ab,
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
    gstate_s_ab,
    gstate_s_c,
    # Params
    tsz,
):
    # Parallelize over the batch and channel dimensions.
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)

    # Pointers to the batch (and possibly channel) for the input tensors.
    k_ptr = k_ptr + b_idx * k_s_b + c_idx * k_s_c
    v_ptr = v_ptr + b_idx * v_s_b + c_idx * v_s_c
    alpha_ptr = state_ptr + b_idx * state_s_b + c_idx * state_s_c
    beta_ptr = state_ptr + b_idx * state_s_b + state_s_ab + c_idx * state_s_c
    w_ptr = w_ptr + c_idx * w_s_c
    u_ptr = u_ptr + c_idx * u_s_c

    # Pointers to the batch (and possibly channel) for the output tensors.
    gw_ptr = gw_ptr + c_idx * gw_s_c
    gu_ptr = gu_ptr + c_idx * gu_s_c
    gk_ptr = gk_ptr + b_idx * gk_s_b + c_idx * gk_s_c
    gv_ptr = gv_ptr + b_idx * gv_s_b + c_idx * gv_s_c

    # Pointers to gradients which were recieved by the function.
    gwkv_ptr = gwkv_ptr + b_idx * gwkv_s_b + c_idx * gwkv_s_c
    galpha_out_ptr = gstate_out_ptr + b_idx * gstate_out_s_b + c_idx * gstate_out_s_c
    gbeta_out_ptr = gstate_out_ptr + b_idx * gstate_out_s_b + gstate_out_s_ab + c_idx * gstate_out_s_c

    # Loads parameters.
    galpha = tl.load(galpha_out_ptr)
    gbeta = tl.load(gbeta_out_ptr)
    w = tl.load(w_ptr)
    u = tl.load(u_ptr)

    ew = tl.exp(w)

    # Gradient accumulators.
    gw = 0.0
    gu = 0.0

    for t in range(tsz):
        tc = tsz - t - 1

        kt = tl.load(k_ptr + tc * k_s_t)
        vt = tl.load(v_ptr + tc * v_s_t)
        alpha_prev = tl.load(alpha_ptr + tc * state_s_t)
        beta_prev = tl.load(beta_ptr + tc * state_s_t)
        euk = tl.exp(u + kt)
        ek = tl.exp(kt)

        denom = beta_prev + euk
        denom_sq = denom * denom

        gwkvt = tl.load(gwkv_ptr + tc * gwkv_s_t)

        gk = 0.0
        gv = 0.0

        # Backpropagates wkv gradients.
        guk = gwkvt * euk * (beta_prev * vt - alpha_prev) / denom_sq
        gu += guk
        gk += guk
        gv += gwkvt * euk / denom

        galpha_wkv = gwkvt / denom
        gbeta_wkv = -gwkvt * (euk * vt + alpha_prev) / denom_sq

        # Backpropagates alpha gradients.
        gw += galpha * ew * alpha_prev
        gk += galpha * ek * vt
        gv += galpha * ek

        # Backpropagates beta gradients.
        gw += gbeta * ew * beta_prev
        gk += gbeta * ek

        # Stores the gradients for k and v.
        tl.store(gk_ptr + tc * gk_s_t, gk)
        tl.store(gv_ptr + tc * gv_s_t, gv)

        # Computes new gradients for alpha and beta.
        galpha = galpha * ew + galpha_wkv
        gbeta = gbeta * ew + gbeta_wkv

    # Stores final gradients for alpha and beta.
    tl.store(gstate_ptr + b_idx * gstate_s_b + c_idx * gstate_s_c, galpha)
    tl.store(gstate_ptr + b_idx * gstate_s_b + gstate_s_ab + c_idx * gstate_s_c, gbeta)

    # Stores final gradients for w and u.
    tl.atomic_add(gw_ptr, gw)
    tl.atomic_add(gu_ptr, gu)


def wkv_triton_vanilla_backward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    grad_wkv: Tensor,
    grad_state: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    (bsz, tsz, chans), device, dtype = k.shape, k.device, k.dtype

    w = w.contiguous()
    u = u.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    state = state.contiguous()
    grad_wkv = grad_wkv.contiguous()
    grad_state = grad_state.contiguous()

    # Checks tensor shapes.
    assert v.shape == (bsz, tsz, chans), f"{v.shape} != {(bsz, tsz, chans)}"
    assert state.shape == (bsz, 2, tsz, chans), f"{state.shape} != {(bsz, 2, tsz, chans)}"
    assert w.shape == (chans,), f"{w.shape} != {(chans,)}"
    assert u.shape == (chans,), f"{u.shape} != {(chans,)}"
    assert grad_wkv.shape == (bsz, tsz, chans)
    assert grad_state.shape == (bsz, 2, 1, chans)

    # Checks tensor dtypes and devices.
    for t in (v, state, w, u, grad_wkv, grad_state):
        assert t.dtype == dtype and t.device == device, f"{t.dtype} != {dtype} or {t.device} != {device}"

    # New tensors to output.
    gw = torch.zeros_like(w, memory_format=torch.contiguous_format)
    gu = torch.zeros_like(u, memory_format=torch.contiguous_format)
    gk = torch.empty_like(k, memory_format=torch.contiguous_format)
    gv = torch.empty_like(v, memory_format=torch.contiguous_format)
    gstate = k.new_empty(bsz, 2, 1, chans)

    wkv_vanilla_triton_backward_kernel[(bsz, chans)](
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
        wkv, state_out = wkv_triton_vanilla_forward(w, u, k, v, state)
        ctx.save_for_backward(w, u, k, v, state_out[:, :, :-1])
        return wkv, state_out[:, :, -1:]

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, gwkv: Tensor, gstate: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        w, u, k, v, state = cast(tuple[Tensor, ...], ctx.saved_tensors)
        gw, gu, gk, gv, gstate = wkv_triton_vanilla_backward(w, u, k, v, state, gwkv, gstate)
        return gw, gu, gk, gv, gstate


def wkv_triton_vanilla(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    return WKVTritonFunction.apply(w, u, k, v, state)
