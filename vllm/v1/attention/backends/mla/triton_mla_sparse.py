# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-Triton sparse MLA backend for GPUs without sparse FlashMLA/FlashInfer."""

from typing import ClassVar

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import AttentionBackend, AttentionCGSupport
from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
    XPUMLASparseImpl,
    XPUMLASparseMetadata,
    XPUMLASparseMetadataBuilder,
)
from vllm.v1.attention.ops.mqa_logits_triton import (
    warmup_fp8_mqa_logits_triton,
    warmup_fp8_paged_mqa_logits_triton,
)
from vllm.v1.attention.ops.triton_sparse_mla_kernel import (
    _DIM_QK_V3,
    _DIM_QK_V4,
    KV_SPLITS_CANDIDATES,
    triton_sparse_mla_attention,
)

# DeepSeek-V3.2 / GLM-5.1 indexer shape, the only model family this backend
# serves. Used only for autotune priming. If a future model differs, the kernel
# re-tunes on first real use, matching the no-warmup behavior.
_INDEXER_NUM_HEADS = 64
_INDEXER_HEAD_DIM = 128


class TritonMLASparseMetadataBuilder(XPUMLASparseMetadataBuilder):
    """Metadata builder with CUDA cudagraph support for Triton sparse MLA."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class TritonMLASparseImpl(XPUMLASparseImpl):
    """XPU sparse impl override using the split-KV Triton kernel on CUDA."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Cache device SM count so the hot path avoids a device lookup.
        self._sm_count: int | None = None
        if self.topk_indices_buffer is not None:
            device_id = self.topk_indices_buffer.device.index
            if device_id is None:
                device_id = torch.cuda.current_device()
            self._sm_count = num_compute_units(device_id)
        self._warmup_autotune(kwargs["indexer"])

    def _warmup_autotune(self, indexer) -> None:
        """Prime Triton autotune caches at init to avoid first-request sweeps."""
        if self.topk_indices_buffer is None:
            return
        device = self.topk_indices_buffer.device
        topk = self.topk_indices_buffer.shape[-1]
        indices = torch.zeros(1, 1, topk, dtype=torch.int32, device=device)
        # Warm both V3.2/GLM-5 and DeepSeek V4 sparse MLA dimensions.
        for dim_qk in (_DIM_QK_V3, _DIM_QK_V4):
            q_warm = torch.empty(
                1, self.num_heads, dim_qk, dtype=torch.bfloat16, device=device
            )
            kv_warm = torch.empty(64, 1, dim_qk, dtype=torch.bfloat16, device=device)
            for splits in KV_SPLITS_CANDIDATES:
                triton_sparse_mla_attention(
                    q_warm,
                    kv_warm,
                    indices,
                    sm_scale=self.softmax_scale,
                    num_kv_splits=splits,
                    sm_count=self._sm_count,
                )

        indexer_num_heads = getattr(indexer, "n_head", _INDEXER_NUM_HEADS)
        indexer_head_dim = getattr(indexer, "head_dim", _INDEXER_HEAD_DIM)
        warmup_fp8_mqa_logits_triton(
            num_heads=indexer_num_heads,
            head_dim=indexer_head_dim,
            device=device,
        )
        cfg = get_current_vllm_config_or_none()
        if cfg is not None:
            warmup_fp8_paged_mqa_logits_triton(
                num_heads=indexer_num_heads,
                head_dim=indexer_head_dim,
                block_size=cfg.cache_config.block_size,
                device=device,
            )

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: XPUMLASparseMetadata,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )
        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output = triton_sparse_mla_attention(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            sm_scale=self.softmax_scale,
            sm_count=self._sm_count,
        )
        return output[:, : self.num_heads, :]


class TritonMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE"

    @staticmethod
    def get_metadata_cls() -> type[XPUMLASparseMetadata]:
        return XPUMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type[TritonMLASparseMetadataBuilder]:
        return TritonMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[TritonMLASparseImpl]:
        return TritonMLASparseImpl

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [_DIM_QK_V3, _DIM_QK_V4]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True
