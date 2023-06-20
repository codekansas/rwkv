"""Tests that the various WKV implementations match."""

import pytest
import torch
from torch import Tensor

from rwkv.wkv import WkvImpl, get_wkv_fn

IMPLS: list[tuple[WkvImpl, WkvImpl]] = [("vanilla", "log"), ("vanilla", "eps"), ("eps", "log"),  ]
TRITON_IMPLS: list[tuple[WkvImpl, WkvImpl]] = [("vanilla", "triton-vanilla"), ("log", "triton-log"), ("eps", "triton-eps"),]


def _get_dummy_tensors(bsz: int, tsz: int, chans: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, ...]:
    w = torch.rand(chans, dtype=dtype, device=device)
    u = torch.rand(chans, dtype=dtype, device=device)
    k = torch.randn(bsz, tsz, chans, dtype=dtype, device=device)
    v = torch.randn(bsz, tsz, chans, dtype=dtype, device=device)
    return w, u, k, v


@pytest.mark.parametrize("impls", IMPLS)
def test_wkv_matches(impls: tuple[WkvImpl, WkvImpl]) -> None:
    bsz, tsz, chans = 2, 7, 16
    device, dtype = torch.device("cpu"), torch.float32
    w, u, k, v = _get_dummy_tensors(bsz, tsz, chans, device, dtype)

    impl_a, impl_b = impls
    wkv_fn_a, state_a = get_wkv_fn(emb_dim=chans, impl=impl_a)
    wkv_fn_b, state_b = get_wkv_fn(emb_dim=chans, impl=impl_b)
    state_a = state_a.repeat_interleave(bsz, dim=0).to(device, dtype)
    state_b = state_b.repeat_interleave(bsz, dim=0).to(device, dtype)

    wkv_a, _ = wkv_fn_a(w, u, k, v, state_a)
    wkv_b, _ = wkv_fn_b(w, u, k, v, state_b)

    assert torch.allclose(wkv_a, wkv_b)


@pytest.mark.parametrize("impls", TRITON_IMPLS)
def test_triton_wkv_matches(impls: tuple[WkvImpl, WkvImpl]) -> None:
    bsz, tsz, chans = 2, 7, 16
    device, dtype = torch.device("cuda"), torch.float32
    w, u, k, v = _get_dummy_tensors(bsz, tsz, chans, device, dtype)

    impl_a, impl_b = impls
    wkv_fn_a, state_a = get_wkv_fn(emb_dim=chans, impl=impl_a)
    wkv_fn_b, state_b = get_wkv_fn(emb_dim=chans, impl=impl_b)
    state_a = state_a.repeat_interleave(bsz, dim=0).to(device, dtype)
    state_b = state_b.repeat_interleave(bsz, dim=0).to(device, dtype)

    wkv_a, _ = wkv_fn_a(w, u, k, v, state_a)
    wkv_b, _ = wkv_fn_b(w, u, k, v, state_b)

    assert torch.allclose(wkv_a, wkv_b)


if __name__ == "__main__":
    # python -m tests.test_wkv
    test_wkv_matches(("vanilla", "log"))
