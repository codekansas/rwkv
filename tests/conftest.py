# mypy: disable-error-code="import"
"""Pytest configuration file."""

import functools

import pytest
import torch
from _pytest.python import Function, Metafunc
from ml.utils.random import set_random_seed as set_random_seed_ml


@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    set_random_seed_ml(1337)


@functools.lru_cache()
def has_gpu() -> bool:
    return torch.cuda.is_available()


@functools.lru_cache()
def has_multi_gpu() -> bool:
    return has_gpu() and torch.cuda.device_count() > 1


@functools.lru_cache()
def has_mps() -> bool:
    return torch.backends.mps.is_available()


@functools.lru_cache()
def has_triton() -> bool:
    if not has_gpu():
        return False

    try:
        import triton

        assert triton is not None
        return True

    except Exception:
        return False


def pytest_runtest_setup(item: Function) -> None:
    for mark in item.iter_markers():
        if mark.name == "has_gpu" and not has_gpu():
            pytest.skip("Skipping because this test requires a GPU and none is available")
        if mark.name == "multi_gpu" and not has_multi_gpu():
            pytest.skip("Skipping because this test requires multiple GPUs but <= 1 are available")
        if mark.name == "has_mps" and not has_mps():
            pytest.skip("Skipping because this test requires an MPS device and none is available")
        if mark.name == "has_triton" and not has_triton():
            pytest.skip("Skipping because this test requires Triton and none is available")


def pytest_collection_modifyitems(items: list[Function]) -> None:
    items.sort(key=lambda x: x.get_closest_marker("slow") is not None)


def pytest_generate_tests(metafunc: Metafunc) -> None:
    if "device" in metafunc.fixturenames:
        torch_devices = [torch.device("cpu")]
        if has_gpu():
            torch_devices.append(torch.device("cuda"))
        if has_mps():
            torch_devices.append(torch.device("mps"))
        metafunc.parametrize("device", torch_devices)
