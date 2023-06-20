# mypy: disable-error-code="import, no-untyped-def, override"
# ruff: noqa: ANN001, ANN201, ANN202, N803, N806
"""Defines Triton kernels for the vanilla RWKV forward and backward passes.

This kernel is used to make the WKV computation in the attention layer run
faster while using less memory. It requires that ``triton`` is installed, which
in turn requires a ``triton``-compatible GPU and CUDA version.
"""

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
    bsz, tsz, chans = k.shape

    alpha, beta = state[:, :, -1].chunk(2, dim=1)  # (B, 1, D), (B, 1, D)

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
    w_ptr,
    u_ptr,
    k_ptr,
    v_ptr,
    alpha_ptr,
    beta_ptr,
    chans,
    tsz,
    gwkv_ptr,
    galpha_out_ptr,
    gbeta_out_ptr,
    gw_ptr,
    gu_ptr,
    gk_ptr,
    gv_ptr,
    galpha_ptr,
    gbeta_ptr,
):
    # Parallelize over the batch and channel dimensions.
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)

    chans_tsz = chans * tsz

    # Pointers to the batch (and possibly channel) for the input tensors.
    k_ptr = k_ptr + b_idx * chans_tsz + c_idx
    v_ptr = v_ptr + b_idx * chans_tsz + c_idx
    alpha_ptr = alpha_ptr + b_idx * chans_tsz + c_idx
    beta_ptr = beta_ptr + b_idx * chans_tsz + c_idx
    w_ptr = w_ptr + c_idx
    u_ptr = u_ptr + c_idx

    # Pointers to the batch (and possibly channel) for the output tensors.
    gw_ptr = gw_ptr + c_idx
    gu_ptr = gu_ptr + c_idx
    gk_ptr = gk_ptr + b_idx * chans_tsz + c_idx
    gv_ptr = gv_ptr + b_idx * chans_tsz + c_idx

    # Pointers to gradients which were recieved by the function.
    gwkv_ptr = gwkv_ptr + b_idx * chans_tsz + c_idx
    galpha_out_ptr = galpha_out_ptr + b_idx * chans + c_idx
    gbeta_out_ptr = gbeta_out_ptr + b_idx * chans + c_idx

    # Loads parameters.
    galpha = tl.load(galpha_out_ptr)
    gbeta = tl.load(gbeta_out_ptr)
    w = tl.load(w_ptr)
    ew = tl.exp(w)
    u = tl.load(u_ptr)

    # Gradient accumulators.
    gw = 0.0
    gu = 0.0

    for t in range(tsz):
        tc = (tsz - t - 1) * chans
        kt = tl.load(k_ptr + tc)
        vt = tl.load(v_ptr + tc)
        alpha_prev = tl.load(alpha_ptr + tc)
        beta_prev = tl.load(beta_ptr + tc)
        euk = tl.exp(u + kt)
        ek = tl.exp(kt)

        denom = beta_prev + euk
        denom_sq = denom * denom

        gwkvt = tl.load(gwkv_ptr + tc)

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
        tl.store(gk_ptr + tc, gk)
        tl.store(gv_ptr + tc, gv)

        # Computes new gradients for alpha and beta.
        galpha = galpha * ew + galpha_wkv
        gbeta = gbeta * ew + gbeta_wkv

    # Stores final gradients for alpha and beta.
    tl.store(galpha_ptr + b_idx * chans + c_idx, galpha)
    tl.store(gbeta_ptr + b_idx * chans + c_idx, gbeta)

    # Stores final gradients for w and u.
    tl.store(gw_ptr, gw)
    tl.store(gu_ptr, gu)


def wkv_triton_vanilla_backward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    grad_wkv: Tensor,
    grad_state: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    alpha, beta = state.chunk(2, dim=1)  # (B, 1, T + 1, D), (B, 1, T + 1, D)
    galpha_out, gbeta_out = grad_state[:, :, 0].chunk(2, dim=1)  # (B, 1, D), (B, 1, D)

    # New tensors to output.
    gw = torch.zeros_like(w)
    gu = torch.zeros_like(u)
    gk = torch.zeros_like(k)
    gv = torch.zeros_like(v)
    galpha = k.new_empty(bsz, 1, chans)
    gbeta = k.new_empty(bsz, 1, chans)

    wkv_vanilla_triton_backward_kernel[(bsz, chans)](
        w,
        u,
        k,
        v,
        alpha,
        beta,
        chans,
        tsz,
        grad_wkv,
        galpha_out,
        gbeta_out,
        gw,
        gu,
        gk,
        gv,
        galpha,
        gbeta,
    )

    return gw, gu, gk, gv, torch.stack((galpha, gbeta), dim=1)


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

        wkv, state_out = wkv_triton_vanilla_forward(w, u, k, v, state)

        ctx.save_for_backward(w, u, k, v, state_out[:, :, :-1])

        return wkv, state_out[:, :, -1:]

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, gwkv: Tensor, gstate: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        (bsz, tsz, chans), device, dtype = gwkv.shape, gwkv.device, gwkv.dtype

        # Performs tensor checks.
        for t in (gwkv, gstate):
            assert t.shape == (bsz, tsz, chans), f"{t.shape} != {(bsz, tsz, chans)}"
            assert t.stride(0) == tsz * chans, f"{t.stride(0)} != {tsz * chans}"
            assert t.stride(1) == chans, f"{t.stride(1)} != {chans}"
            assert t.size(2) == 1 or t.stride(2) == 1, f"{t.stride(2)} != 1"

        w, u, k, v, state = ctx.saved_tensors
        gw, gu, gk, gv, gstate = wkv_triton_vanilla_backward(w, u, k, v, state, gwkv, gstate)
        return gw, gu, gk, gv, gstate


def wkv_triton_vanilla(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    return WKVTritonFunction.apply(w, u, k, v, state)
