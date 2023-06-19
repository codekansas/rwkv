# mypy: disable-error-code="import, no-untyped-def, override"
# ruff: noqa: ANN001, ANN201, ANN202, N803, N806
"""Defines a Triton kernel for the RWKV forward and backward passes.

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
def _forward_kernel(
    w_ptr,
    u_ptr,
    k_ptr,
    v_ptr,
    alpha_ptr,
    beta_ptr,
    eps_ptr,
    chans,
    tsz,
    out_ptr,
    alpha_out_ptr,
    beta_out_ptr,
    eps_out_ptr,
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
    alpha_out_ptr = alpha_out_ptr + b_idx * chans + c_idx
    beta_out_ptr = beta_out_ptr + b_idx * chans + c_idx

    # Loads parameters.
    alpha = tl.load(alpha_ptr).to(tl.float32)
    beta = tl.load(beta_ptr).to(tl.float32)
    w = -tl.exp(tl.load(w_ptr).to(tl.float32))
    u = tl.load(u_ptr).to(tl.float32)

    ew = tl.exp(w)

    for t in range(tsz):
        tc = t * chans

        kt = tl.load(k_ptr + tc).to(tl.float32)
        vt = tl.load(v_ptr + tc).to(tl.float32)

        euk = tl.exp(u + kt)

        out = (alpha + euk * vt) / (beta + euk)
        tl.store(out_ptr + tc, out)

        ek = tl.exp(kt)
        alpha = ew * alpha + ek * vt
        beta = ew * beta + ek

    tl.store(alpha_out_ptr, alpha)
    tl.store(beta_out_ptr, beta)


def _forward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
    eps: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    # New tensors to output.
    out = k.new_empty(bsz, tsz, chans)
    alpha_out = k.new_empty(bsz, 1, chans)
    beta_out = k.new_empty(bsz, 1, chans)
    eps_out = k.new_empty(bsz, 1, chans)

    _forward_kernel[(bsz, chans)](
        w,
        u,
        k,
        v,
        alpha,
        beta,
        eps,
        chans,
        tsz,
        out,
        alpha_out,
        beta_out,
        eps_out,
    )

    return out, alpha_out, beta_out, eps_out


@triton.jit
def _backward_kernel(
    w_ptr,
    u_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    alpha_out_ptr,
    beta_out_ptr,
    eps_out_ptr,
    chans,
    tsz,
    gout_ptr,
    galpha_out_ptr,
    gbeta_out_ptr,
    geps_out_ptr,
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


def _backward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    alpha_out: Tensor,
    beta_out: Tensor,
    eps_out: Tensor,
    gout: Tensor,
    galpha_out: Tensor,
    gbeta_out: Tensor,
    geps_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    # New tensors to output.
    gw = k.new_empty(chans)
    gu = k.new_empty(chans)
    gk = k.new_empty(bsz, tsz, chans)
    gv = k.new_empty(bsz, tsz, chans)
    galpha = k.new_empty(bsz, 1, chans)
    gbeta = k.new_empty(bsz, 1, chans)
    geps = k.new_empty(bsz, 1, chans)

    _backward_kernel[(bsz, chans)](
        w,
        u,
        k,
        v,
        out,
        alpha_out,
        beta_out,
        eps_out,
        chans,
        tsz,
        gout,
        galpha_out,
        gbeta_out,
        geps_out,
        gw,
        gu,
        gk,
        gv,
        galpha,
        gbeta,
        geps,
    )

    return gw, gu, gk, gv, galpha, gbeta


class _WKV(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        w: Tensor,
        u: Tensor,
        k: Tensor,
        v: Tensor,
        alpha: Tensor,
        beta: Tensor,
        eps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        (bsz, tsz, chans), device, dtype = k.shape, k.device, k.dtype

        # Performs tensor checks.
        for t in (k, v):
            assert t.shape == (bsz, tsz, chans)
            assert t.stride(0) == tsz * chans
            assert t.stride(1) == chans
            assert t.size(2) == 1 or t.stride(2) == 1
        for t in (alpha, beta):
            assert t.shape == (bsz, 1, chans)
            assert t.stride(0) == chans
            assert t.stride(1) == chans
            assert t.stride(2) == 1
        for t in (w, u):
            assert t.shape == (chans,)
            assert t.stride(0) == 1
        for t in (v, alpha, beta, w, u):
            assert t.dtype == dtype and t.device == device

        out, alpha_out, beta_out, eps_out = _forward(
            w,
            u,
            k,
            v,
            alpha,
            beta,
            eps,
        )

        ctx.save_for_backward(w, u, k, v, out, alpha_out, beta_out, eps_out)

        return out, alpha_out, beta_out, eps_out

    @staticmethod
    @once_differentiable
    def backward(
        ctx: FunctionCtx,
        gout: Tensor,
        galpha_out: Tensor,
        gbeta_out: Tensor,
        geps_out: Tensor,
    ) -> tuple[Tensor, ...]:
        w, u, k, v, out, alpha_out, beta_out, eps_out = ctx.saved_tensors
        gw, gu, gk, gv, ga, gb, ge = _backward(
            w,
            u,
            k,
            v,
            out,
            alpha_out,
            beta_out,
            eps_out,
            gout,
            galpha_out,
            gbeta_out,
            geps_out,
        )
        return gw, gu, gk, gv, ga, gb, ge


def triton_wkv(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    return _WKV.apply(w, u, k, v, state)


def initial_state_triton(emb_dim: int) -> Tensor:
    return torch.zeros(1, 1, emb_dim, dtype=torch.float32, device="cuda")
