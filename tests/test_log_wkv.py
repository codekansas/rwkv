"""Tests the WKV computation using log-space state variables for stability."""

from typing import cast

import pytest
import torch
from torch import Tensor

from rwkv.wkv.log import initial_state_log_space, wkv_log_space, wkv_log_space_forward


def _get_dummy_tensors(bsz: int, tsz: int, chans: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, ...]:
    w = -torch.exp(torch.rand(chans, dtype=dtype, device=device))
    u = torch.rand(chans, dtype=dtype, device=device)
    k = torch.randn(bsz, tsz, chans, dtype=dtype, device=device)
    v = torch.randn(bsz, tsz, chans, dtype=dtype, device=device)
    return w, u, k, v


def _copy_with_grad(*t: Tensor) -> tuple[Tensor, ...]:
    return tuple(t_i.detach().clone().requires_grad_(True) for t_i in t)


def _get_grads(*t: Tensor) -> tuple[Tensor | None, ...]:
    return tuple(cast(Tensor, t_i.grad) for t_i in t)


def test_log_wkv() -> None:
    bsz, tsz, chans = 2, 7, 16
    device, dtype = torch.device("cpu"), torch.float32

    w, u, k, v = _get_dummy_tensors(bsz, tsz, chans, device, dtype)
    state = initial_state_log_space(chans).repeat_interleave(bsz, dim=0).to(device, dtype)

    # Runs in full mode.
    out_full, _ = wkv_log_space(w, u, k, v, state)

    # Runs in iterative mode.
    out_parts: list[Tensor] = []
    for t in range(tsz):
        out_part, state = wkv_log_space(w, u, k[:, t : t + 1], v[:, t : t + 1], state)
        out_parts.append(out_part)
    out_partial = torch.cat(out_parts, dim=1)

    assert torch.allclose(out_full, out_partial, atol=1e-5)


@pytest.mark.parametrize("mode", ["state", "wkv", "both"])
def test_log_wkv_gradients(mode: str) -> None:
    bsz, tsz, chans = 2, 7, 16
    device, dtype = torch.device("cpu"), torch.float32

    w, u, k, v = _get_dummy_tensors(bsz, tsz, chans, device, dtype)
    state = initial_state_log_space(chans).repeat_interleave(bsz, dim=0).to(device, dtype)

    # Uses autograd to compute the gradients.
    wt, ut, kt, vt, statet = _copy_with_grad(w, u, k, v, state)
    wkv_ref, state_out_ref = wkv_log_space_forward(wt, ut, kt, vt, statet)
    state_out_ref = state_out_ref[:, :, -1:]
    wkv_grad = torch.zeros_like(wkv_ref) if mode == "state" else torch.rand_like(wkv_ref)
    state_out_grad = torch.zeros_like(state_out_ref) if mode == "wkv" else torch.rand_like(state_out_ref)
    torch.autograd.backward((wkv_ref, state_out_ref), (wkv_grad, state_out_grad))
    wgr, ugr, kgr, vgr, stategr = _get_grads(wt, ut, kt, vt, statet)

    # Uses the manual gradient computation to compute the gradients.
    wt, ut, kt, vt, statet = _copy_with_grad(w, u, k, v, state)
    wkv_man, state_out_man = wkv_log_space(wt, ut, kt, vt, statet)
    torch.autograd.backward((wkv_man, state_out_man), (wkv_grad, state_out_grad))
    wgm, ugm, kgm, vgm, stategm = _get_grads(wt, ut, kt, vt, statet)

    for gr, gm, gname in [
        (wgr, wgm, "w"),
        (ugr, ugm, "u"),
        (kgr, kgm, "k"),
        (vgr, vgm, "v"),
        (stategr, stategm, "state"),
    ]:
        if gr is None or gm is None:
            continue
        assert torch.allclose(gr, gm, atol=1e-5), f"Gradient {gname} mismatch"
