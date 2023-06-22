"""Tests the WKV computation using epsilon offset for numerical stability."""

from typing import cast

import pytest
import torch
from torch import Tensor

from rwkv.wkv.eps import initial_state_with_eps, wkv_with_eps, wkv_with_eps_forward


def _get_dummy_tensors(bsz: int, tsz: int, chans: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, ...]:
    # w = -torch.exp(torch.rand(chans, dtype=dtype, device=device))
    w = torch.rand(chans, dtype=dtype, device=device)
    u = torch.rand(chans, dtype=dtype, device=device)
    k = torch.randn(bsz, tsz, chans, dtype=dtype, device=device)
    v = torch.randn(bsz, tsz, chans, dtype=dtype, device=device)
    return w, u, k, v


def _copy_with_grad(*t: Tensor) -> tuple[Tensor, ...]:
    return tuple(t_i.detach().clone().requires_grad_(True) for t_i in t)


def _get_grads(*t: Tensor) -> tuple[Tensor | None, ...]:
    return tuple(cast(Tensor, t_i.grad) for t_i in t)


def test_eps_wkv() -> None:
    bsz, tsz, chans = 2, 7, 16
    device, dtype = torch.device("cpu"), torch.float64

    w, u, k, v = _get_dummy_tensors(bsz, tsz, chans, device, dtype)
    state = initial_state_with_eps(chans).repeat_interleave(bsz, dim=0).to(device, dtype)

    # Runs in full mode.
    out_full, _ = wkv_with_eps(w, u, k, v, state)

    # Runs in iterative mode.
    out_parts: list[Tensor] = []
    for t in range(tsz):
        out_part, state = wkv_with_eps(w, u, k[:, t : t + 1], v[:, t : t + 1], state)
        out_parts.append(out_part)
    out_partial = torch.cat(out_parts, dim=1)

    assert torch.allclose(out_full, out_partial)


@pytest.mark.parametrize("mode", ["state", "wkv", "both"])
def test_gradients_eps_wkv(mode: str) -> None:
    bsz, tsz, chans = 2, 7, 16
    device, dtype = torch.device("cpu"), torch.float64

    w, u, k, v = _get_dummy_tensors(bsz, tsz, chans, device, dtype)
    state = initial_state_with_eps(chans).repeat_interleave(bsz, dim=0).to(device, dtype)

    def backprop(wkv_out: Tensor, state_out: Tensor) -> None:
        if mode == "both":
            (wkv_out.sum() + state_out.sum()).backward()
        elif mode == "wkv":
            wkv_out.sum().backward()
        elif mode == "state":
            state_out.sum().backward()
        else:
            raise ValueError(f"Invalid mode: {mode}")

    # Uses autograd to compute the gradients.
    wt, ut, kt, vt, statet = _copy_with_grad(w, u, k, v, state)
    wkv_ref, state_out_ref = wkv_with_eps_forward(wt, ut, kt, vt, statet)
    state_out_ref = state_out_ref[:, :, -1:]
    backprop(wkv_ref, state_out_ref)
    wgr, ugr, kgr, vgr, stategr = _get_grads(wt, ut, kt, vt, statet)

    # Uses the manual gradient computation to compute the gradients.
    wt, ut, kt, vt, statet = _copy_with_grad(w, u, k, v, state)
    wkv_man, state_out_man = wkv_with_eps(wt, ut, kt, vt, statet)
    backprop(wkv_man, state_out_man)
    wgm, ugm, kgm, vgm, stategm = _get_grads(wt, ut, kt, vt, statet)

    for gr, gm in zip((wgr, ugr, kgr, vgr, stategr), (wgm, ugm, kgm, vgm, stategm)):
        if gr is not None and gm is not None:
            assert torch.allclose(gr, gm)
