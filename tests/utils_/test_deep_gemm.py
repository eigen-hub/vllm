# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

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
    deep_gemm.is_deep_gemm_supported.cache_clear()
    deep_gemm._visible_cuda_capabilities.cache_clear()
    deep_gemm._visible_cuda_devices_support_deep_gemm.cache_clear()

    yield

    deep_gemm.is_deep_gemm_supported.cache_clear()
    deep_gemm._visible_cuda_capabilities.cache_clear()
    deep_gemm._visible_cuda_devices_support_deep_gemm.cache_clear()


@pytest.mark.parametrize(
    ("capabilities", "expected"),
    [
        ((DeviceCapability(9, 0), DeviceCapability(9, 0)), True),
        ((DeviceCapability(8, 0), DeviceCapability(9, 0)), False),
        ((DeviceCapability(9, 0), DeviceCapability(10, 0)), False),
        ((DeviceCapability(8, 0), DeviceCapability(8, 0)), False),
    ],
)
def test_deep_gemm_requires_uniform_supported_cuda_visible_set(
    monkeypatch: pytest.MonkeyPatch,
    capabilities: tuple[DeviceCapability, ...],
    expected: bool,
):
    monkeypatch.setattr(deep_gemm, "current_platform", FakeCudaPlatform(capabilities))
    monkeypatch.setattr(deep_gemm, "has_deep_gemm", lambda: True)
    monkeypatch.setattr(deep_gemm.envs, "VLLM_USE_DEEP_GEMM", True)

    assert deep_gemm.is_deep_gemm_supported() is expected
