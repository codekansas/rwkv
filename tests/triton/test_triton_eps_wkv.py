"""Tests the numerically stable WKV Triton kernels."""

import pytest
import torch
from torch import Tensor

from rwkv.wkv.eps import initial_state_with_eps, wkv_with_eps_backward, wkv_with_eps_forward


def _get_dummy_tensors(bsz: int, tsz: int, chans: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, ...]:
    w = torch.rand(chans, dtype=dtype, device=device)
    u = torch.rand(chans, dtype=dtype, device=device)
    k = torch.randn(bsz, tsz, chans, dtype=dtype, device=device)
    v = torch.randn(bsz, tsz, chans, dtype=dtype, device=device)
    return w, u, k, v


@pytest.mark.has_triton()
def test_triton_with_eps_wkv() -> None:
    from rwkv.triton.wkv.eps import wkv_triton_with_eps_backward, wkv_triton_with_eps_forward

    bsz, tsz, chans = 2, 7, 16
    device, dtype = torch.device("cuda"), torch.float64

    w, u, k, v = _get_dummy_tensors(bsz, tsz, chans, device, dtype)
    state = initial_state_with_eps(chans).repeat_interleave(bsz, dim=0).to(device, dtype)

    wkv_ref, state_out_ref = wkv_with_eps_forward(w, u, k, v, state)
    wkv, state_out = wkv_triton_with_eps_forward(w, u, k, v, state)

    assert torch.allclose(wkv_ref, wkv)
    assert torch.allclose(state_out_ref, state_out)

    grad_wkv = torch.randn_like(wkv)
    grad_state = torch.randn_like(state_out[:, :, -1:])

    # state_out_ref, state_out = state_out_ref[:, :, :-1], state_out[:, :, :-1]
    dw_ref, du_ref, dk_ref, dv_ref, dstate_ref = wkv_with_eps_backward(w, u, k, v, state_out_ref, grad_wkv, grad_state)
    dw, du, dk, dv, dstate = wkv_triton_with_eps_backward(w, u, k, v, state_out, grad_wkv, grad_state)

    for a, b, name in [
        (dw_ref, dw, "dw"),
        (du_ref, du, "du"),
        (dk_ref, dk, "dk"),
        (dv_ref, dv, "dv"),
        (dstate_ref, dstate, "dstate"),
    ]:
        assert torch.allclose(a, b), f"{name} is not close!"


if __name__ == "__main__":
    # python -m tests.triton.test_triton_with_eps_wkv
    test_triton_with_eps_wkv()
