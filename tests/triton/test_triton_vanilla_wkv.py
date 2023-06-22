"""Tests the vanilla WKV Triton kernels."""

import pytest
import torch
from torch import Tensor

from rwkv.wkv.vanilla import initial_state_vanilla, wkv_vanilla_backward, wkv_vanilla_forward


def _get_dummy_tensors(bsz: int, tsz: int, chans: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, ...]:
    w = -torch.exp(torch.rand(chans, dtype=dtype, device=device))
    u = torch.rand(chans, dtype=dtype, device=device)
    k = torch.randn(bsz, tsz, chans, dtype=dtype, device=device)
    v = torch.randn(bsz, tsz, chans, dtype=dtype, device=device)
    return w, u, k, v


@pytest.mark.has_triton()
@pytest.mark.parametrize("tsz", [1, 4])
def test_triton_vanilla_wkv(tsz: int) -> None:
    from rwkv.triton.wkv.vanilla import wkv_triton_vanilla_backward, wkv_triton_vanilla_forward

    bsz, chans = 2, 768
    device, dtype = torch.device("cuda"), torch.float32

    w, u, k, v = _get_dummy_tensors(bsz, tsz, chans, device, dtype)
    state = initial_state_vanilla(chans).repeat_interleave(bsz, dim=0).to(device, dtype)

    wkv_ref, state_out_ref = wkv_vanilla_forward(w, u, k, v, state)
    wkv, state_out = wkv_triton_vanilla_forward(w, u, k, v, state)

    assert torch.allclose(wkv_ref, wkv, atol=1e-5)
    assert torch.allclose(state_out_ref, state_out, atol=1e-5)

    grad_wkv = torch.randn_like(wkv)
    grad_state = torch.randn_like(state_out[:, :, -1:])

    state_out_ref, state_out = state_out_ref[:, :, :-1], state_out[:, :, :-1]
    dw_ref, du_ref, dk_ref, dv_ref, dstate_ref = wkv_vanilla_backward(w, u, k, v, state_out_ref, grad_wkv, grad_state)
    dw, du, dk, dv, dstate = wkv_triton_vanilla_backward(w, u, k, v, state_out, grad_wkv, grad_state)

    for a, b, name in [
        (dw_ref, dw, "dw"),
        (du_ref, du, "du"),
        (dk_ref, dk, "dk"),
        (dv_ref, dv, "dv"),
        (dstate_ref, dstate, "dstate"),
    ]:
        assert torch.allclose(a, b, atol=1e-5), f"{name} is not close!"


if __name__ == "__main__":
    # python -m tests.triton.test_triton_vanilla_wkv
    test_triton_vanilla_wkv(4)
