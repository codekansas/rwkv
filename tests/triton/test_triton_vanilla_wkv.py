"""Tests the vanilla WKV Triton kernels."""

import pytest
import torch
from torch import Tensor

from rwkv.triton.wkv.vanilla import wkv_triton_vanilla_backward, wkv_triton_vanilla_forward
from rwkv.wkv.vanilla import initial_state_vanilla, wkv_vanilla_backward, wkv_vanilla_forward


def _get_dummy_tensors(bsz: int, tsz: int, chans: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, ...]:
    w = torch.rand(chans, dtype=dtype, device=device)
    u = torch.rand(chans, dtype=dtype, device=device)
    k = torch.randn(bsz, tsz, chans, dtype=dtype, device=device)
    v = torch.randn(bsz, tsz, chans, dtype=dtype, device=device)
    return w, u, k, v


@pytest.mark.has_triton()
@pytest.mark.parametrize("mode", ["state", "wkv", "both"])
def test_triton_vanilla_wkv(mode: str) -> None:
    bsz, tsz, chans = 2, 7, 16
    device, dtype = torch.device("cuda"), torch.float32

    tsz = 1
    torch.manual_seed(1338)  # TODO: Remove later

    w, u, k, v = _get_dummy_tensors(bsz, tsz, chans, device, dtype)
    state = initial_state_vanilla(chans).repeat_interleave(bsz, dim=0).to(device, dtype)

    wkv_ref, state_out_ref = wkv_vanilla_forward(w, u, k, v, state)
    wkv, state_out = wkv_triton_vanilla_forward(w, u, k, v, state)

    assert torch.allclose(wkv_ref, wkv)
    assert torch.allclose(state_out_ref, state_out)

    grad_wkv = torch.zeros_like(wkv) if mode == "state" else torch.randn_like(wkv)
    grad_state = torch.zeros_like(state_out[:, :, -1:]) if mode == "wkv" else torch.randn_like(state_out[:, :, -1:])

    dw_ref, du_ref, dk_ref, dv_ref, dstate_ref = wkv_vanilla_backward(w, u, k, v, state_out_ref, grad_wkv, grad_state)
    dw, du, dk, dv, dstate = wkv_triton_vanilla_backward(w, u, k, v, state_out, grad_wkv, grad_state)

    # breakpoint()

    # for a, b in [(dw_ref, dw), (du_ref, du), (dk_ref, dk), (dv_ref, dv), (dstate_ref, dstate)]:
    #     assert torch.allclose(a, b)


if __name__ == "__main__":
    # python -m tests.triton.test_triton_vanilla_wkv
    test_triton_vanilla_wkv("both")
