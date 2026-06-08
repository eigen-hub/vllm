# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

import pytest

import vllm.utils.deep_gemm as deep_gemm
from vllm.platforms.interface import DeviceCapability


@dataclass
class FakeCudaPlatform:
    capabilities: tuple[DeviceCapability, ...]

    def is_cuda(self) -> bool:
        return True

    def device_count(self) -> int:
        return len(self.capabilities)

    def get_device_capability(self, device_id: int = 0) -> DeviceCapability:
        return self.capabilities[device_id]

    def support_deep_gemm(self) -> bool:
        return deep_gemm._capability_supports_deep_gemm(self.capabilities[0])


@pytest.fixture(autouse=True)
def clear_deep_gemm_support_caches():
    deep_gemm._cuda_device_capability.cache_clear()
    deep_gemm.is_deep_gemm_e8m0_used.cache_clear()
    deep_gemm._USE_DSV4_REF_KERNELS = None
    deep_gemm._DSV4_REF_KERNELS_SYNCED = False

    yield

    deep_gemm._cuda_device_capability.cache_clear()
    deep_gemm.is_deep_gemm_e8m0_used.cache_clear()
    deep_gemm._USE_DSV4_REF_KERNELS = None
    deep_gemm._DSV4_REF_KERNELS_SYNCED = False


@pytest.mark.parametrize(
    ("capabilities", "current_device", "expected"),
    [
        ((DeviceCapability(8, 0), DeviceCapability(9, 0)), 0, False),
        ((DeviceCapability(8, 0), DeviceCapability(9, 0)), 1, True),
        ((DeviceCapability(9, 0), DeviceCapability(10, 0)), 0, True),
        ((DeviceCapability(9, 0), DeviceCapability(10, 0)), 1, True),
        ((DeviceCapability(8, 0), DeviceCapability(8, 0)), 0, False),
    ],
)
def test_deep_gemm_uses_current_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
    capabilities: tuple[DeviceCapability, ...],
    current_device: int,
    expected: bool,
):
    monkeypatch.setattr(deep_gemm, "current_platform", FakeCudaPlatform(capabilities))
    monkeypatch.setattr(deep_gemm, "_current_cuda_device_id", lambda: current_device)
    monkeypatch.setattr(deep_gemm, "has_deep_gemm", lambda: True)
    monkeypatch.setattr(deep_gemm.envs, "VLLM_USE_DEEP_GEMM", True)

    assert deep_gemm.is_deep_gemm_supported() is expected


def test_deep_gemm_rechecks_when_current_cuda_device_changes(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        deep_gemm,
        "current_platform",
        FakeCudaPlatform((DeviceCapability(8, 0), DeviceCapability(9, 0))),
    )
    current_device = 0
    monkeypatch.setattr(deep_gemm, "_current_cuda_device_id", lambda: current_device)
    monkeypatch.setattr(deep_gemm, "has_deep_gemm", lambda: True)
    monkeypatch.setattr(deep_gemm.envs, "VLLM_USE_DEEP_GEMM", True)

    assert deep_gemm.is_deep_gemm_supported() is False

    current_device = 1

    assert deep_gemm.is_deep_gemm_supported() is True


def test_dsv4_reference_sync_keeps_deep_gemm_current_device_local(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        deep_gemm,
        "current_platform",
        FakeCudaPlatform((DeviceCapability(9, 0),)),
    )
    monkeypatch.setattr(deep_gemm, "_current_cuda_device_id", lambda: 0)
    monkeypatch.setattr(deep_gemm, "has_deep_gemm", lambda: True)
    monkeypatch.setattr(deep_gemm.envs, "VLLM_USE_DEEP_GEMM", True)

    assert deep_gemm.is_deep_gemm_supported() is True

    deep_gemm.sync_dsv4_reference_kernels(True)

    assert deep_gemm.use_dsv4_reference_kernels() is True
    assert deep_gemm.is_deep_gemm_supported() is True


@pytest.mark.parametrize(
    ("func", "args"),
    [
        (
            deep_gemm.fp8_fp4_mqa_logits,
            (None, None, None, None, None, False),
        ),
        (
            deep_gemm.get_paged_mqa_logits_metadata,
            (None, 64, 1),
        ),
        (
            deep_gemm.fp8_fp4_paged_mqa_logits,
            (None, None, None, None, None, None, 1, False),
        ),
    ],
)
def test_deep_gemm_mqa_wrappers_fail_closed_on_unsupported_current_device(
    monkeypatch: pytest.MonkeyPatch,
    func: Any,
    args: tuple[Any, ...],
):
    monkeypatch.setattr(
        deep_gemm,
        "current_platform",
        FakeCudaPlatform((DeviceCapability(8, 0),)),
    )
    monkeypatch.setattr(deep_gemm, "_current_cuda_device_id", lambda: 0)
    monkeypatch.setattr(deep_gemm, "has_deep_gemm", lambda: True)
    monkeypatch.setattr(deep_gemm.envs, "VLLM_USE_DEEP_GEMM", True)
    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)

    with pytest.raises(RuntimeError, match="DeepGEMM backend is not available"):
        func(*args)
