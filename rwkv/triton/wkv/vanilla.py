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
    out_ptr,
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
    out_ptr = out_ptr + b_idx * chans_tsz + c_idx
    alpha_out_ptr = alpha_out_ptr + b_idx * chans_tsz + c_idx
    beta_out_ptr = beta_out_ptr + b_idx * chans_tsz + c_idx

    # Loads parameters.
    alpha = tl.load(alpha_ptr).to(tl.float32)
    beta = tl.load(beta_ptr).to(tl.float32)
    w = tl.load(w_ptr).to(tl.float32)
    u = tl.load(u_ptr).to(tl.float32)

    ew = tl.exp(w)

    for t in range(tsz):
        tc = t * chans

        kt = tl.load(k_ptr + tc).to(tl.float32)
        vt = tl.load(v_ptr + tc).to(tl.float32)

        euk = tl.exp(u + kt)

        wkv = (alpha + euk * vt) / (beta + euk)
        tl.store(out_ptr + tc, wkv)

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
    out_ptr,
    alpha_out_ptr,
    beta_out_ptr,
    chans,
    tsz,
    gout_ptr,
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
    out_ptr = out_ptr + b_idx * chans_tsz + c_idx
    alpha_out_ptr = alpha_out_ptr + b_idx * chans + c_idx
    beta_out_ptr = beta_out_ptr + b_idx * chans + c_idx
    w_ptr = w_ptr + c_idx
    u_ptr = u_ptr + c_idx
    gout_ptr = gout_ptr + b_idx * chans_tsz + c_idx
    galpha_out_ptr = galpha_out_ptr + b_idx * chans + c_idx
    gbeta_out_ptr = gbeta_out_ptr + b_idx * chans + c_idx

    # Pointers to the batch (and possibly channel) for the output tensors.
    gw_ptr = gw_ptr + c_idx
    gu_ptr = gu_ptr + c_idx
    gk_ptr = gk_ptr + b_idx * chans_tsz + c_idx
    gv_ptr = gv_ptr + b_idx * chans_tsz + c_idx
    galpha_ptr = galpha_ptr + b_idx * chans + c_idx
    gbeta_ptr = gbeta_ptr + b_idx * chans + c_idx

    # Loads parameters.
    tl.load(alpha_out_ptr).to(tl.float32)
    tl.load(beta_out_ptr).to(tl.float32)
    galpha = tl.load(galpha_out_ptr).to(tl.float32)
    gbeta = tl.load(gbeta_out_ptr).to(tl.float32)
    w = -tl.exp(tl.load(w_ptr).to(tl.float32))
    tl.exp(w)
    u = tl.load(u_ptr).to(tl.float32)

    for t in range(tsz - 1, -1, -1):
        tl.load(gout_ptr + t * chans).to(tl.float32)
        kt = tl.load(k_ptr + t * chans).to(tl.float32)
        tl.load(v_ptr + t * chans).to(tl.float32)
        tl.exp(kt)
        tl.exp(u + kt)

    tl.store(galpha_ptr, galpha)
    tl.store(gbeta_ptr, gbeta)


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
    gw = k.new_empty(chans)
    gu = k.new_empty(chans)
    gk = k.new_empty(bsz, tsz, chans)
    gv = k.new_empty(bsz, tsz, chans)
    galpha = k.new_empty(bsz, 1, chans)
    gbeta = k.new_empty(bsz, 1, chans)
    geps = k.new_empty(bsz, 1, chans)

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
        geps,
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
            assert t.shape == (bsz, tsz, chans)
            assert t.stride(0) == tsz * chans
            assert t.stride(1) == chans
            assert t.size(2) == 1 or t.stride(2) == 1
        assert state.shape == (bsz, 2, 1, chans)
        assert state.stride(0) == chans * 2
        assert state.stride(1) == chans
        assert state.stride(2) == 1
        for t in (w, u):
            assert t.shape == (chans,)
            assert t.stride(0) == 1
        for t in (v, state, w, u):
            assert t.dtype == dtype and t.device == device

        wkv, state_out = wkv_triton_vanilla_forward(w, u, k, v, state)

        ctx.save_for_backward(w, u, k, v, state_out)

        return wkv, state_out[:, :, -1]

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, gwkv: Tensor, gstate: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        w, u, k, v, state = ctx.saved_tensors
        gw, gu, gk, gv, gstate = wkv_triton_vanilla_backward(w, u, k, v, state, gwkv, gstate)
        return gw, gu, gk, gv, gstate


def wkv_triton_vanilla(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    return WKVTritonFunction.apply(w, u, k, v, state)
