# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the Triton sparse MLA kernel with DeepSeek V4 (d_qk=512).

Compares Triton kernel output against a reference Python implementation.
Tests both V4 (512) and V3.2 (576) dimensions to ensure backward compatibility.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_sparse_mla_kernel import (
    _DIM_QK_V3,
    _DIM_QK_V4,
    triton_sparse_mla_attention,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton sparse MLA kernel requires CUDA/ROCm",
)


def _merge_two_lse(
    lse0: torch.Tensor, lse1: torch.Tensor | None, s_q: int, h_q: int
) -> torch.Tensor:
    """Merge two LSE tensors for attention sink handling.

    This is the same logic used in the reference implementation.
    """
    if lse1 is None:
        return lse0
    return torch.logsumexp(
        torch.stack([lse0.view(s_q, h_q), lse1.broadcast_to(s_q, h_q)], dim=0),
        dim=0,
    )


def reference_mla_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference Python implementation for sparse MLA prefill.

    This implements the same algorithm as _ref_sparse_attn_prefill in
    deepseek_v4_attention.py, adapted for standalone testing.

    Args:
        q: [s_q, h_q, d_qk] query tensor
        kv: [s_kv, 1, d_qk] key-value tensor
        indices: [s_q, 1, topk] index tensor (may contain -1 for invalid)
        sm_scale: softmax scale
        d_v: value dimension
        topk_length: [s_q] actual valid topk per token (optional)
        attn_sink: [h_q] attention sink values (optional)

    Returns:
        o: [s_q, h_q, d_v] output in bf16
        o_fp32: [s_q, h_q, d_v] output in fp32
        max_logits: [s_q, h_q] max logits per token/head
        lse: [s_q, h_q] log-sum-exp per token/head
    """
    s_q, h_q, d_qk = q.shape
    s_kv, _, _ = kv.shape
    _, _, topk = indices.shape

    indices = indices.clone().squeeze(1)

    if topk_length is not None:
        mask = torch.arange(topk, device=topk_length.device).unsqueeze(0).broadcast_to(
            s_q, topk
        ) >= topk_length.unsqueeze(1)
        indices[mask] = -1

    invalid_mask = (indices < 0) | (indices >= s_kv)
    indices[invalid_mask] = 0

    q_fp32 = q.float()
    gathered_kv = (
        kv.index_select(dim=0, index=indices.flatten())
        .reshape(s_q, topk, d_qk)
        .float()
    )

    P = q_fp32 @ gathered_kv.transpose(1, 2)
    P *= sm_scale
    P[invalid_mask.unsqueeze(1).broadcast_to(P.shape)] = float("-inf")

    orig_lse = torch.logsumexp(P, dim=-1)
    max_logits = P.max(dim=-1).values

    lse_for_o = _merge_two_lse(orig_lse, attn_sink, s_q, h_q)
    lse_for_o = lse_for_o.clone()
    lse_for_o[lse_for_o == float("-inf")] = float("+inf")
    s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
    out = s_for_o @ gathered_kv[..., :d_v]

    lonely_q_mask = orig_lse == float("-inf")
    orig_lse[lonely_q_mask] = float("+inf")
    return (out.to(kv.dtype), out, max_logits, orig_lse)


@pytest.fixture(scope="module")
def kv_cache_v4():
    """KV cache for V4 (d_qk=512)."""
    torch.manual_seed(0)
    return torch.randn(32768, 1, _DIM_QK_V4, dtype=torch.bfloat16, device="cuda")


@pytest.fixture(scope="module")
def kv_cache_v3():
    """KV cache for V3.2 (d_qk=576)."""
    torch.manual_seed(0)
    return torch.randn(32768, 1, _DIM_QK_V3, dtype=torch.bfloat16, device="cuda")


def _assert_triton_matches_reference(
    num_tokens: int,
    num_heads: int,
    topk: int,
    kv_cache: torch.Tensor,
    dim_qk: int,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
) -> None:
    """Compare Triton kernel output against reference implementation."""
    torch.manual_seed(0)
    d_v = 512

    q = torch.randn(num_tokens, num_heads, dim_qk, dtype=torch.bfloat16, device="cuda")

    indices = torch.full(
        (num_tokens, 1, topk), -1, dtype=torch.int32, device="cuda"
    )
    for t in range(num_tokens):
        valid_count = min(t + 1, topk)
        if valid_count > 0:
            idx = torch.randint(0, kv_cache.shape[0] // 2, (valid_count,), device="cuda")
            indices[t, 0, :valid_count] = idx

    sm_scale = dim_qk**-0.5

    triton_out = triton_sparse_mla_attention(
        q,
        kv_cache,
        indices,
        sm_scale=sm_scale,
        num_kv_splits=1,
    )

    ref_out, ref_out_fp32, ref_max_logits, ref_lse = reference_mla_sparse_prefill(
        q, kv_cache, indices, sm_scale, d_v, topk_length, attn_sink
    )

    torch.testing.assert_close(
        triton_out.float(),
        ref_out.float(),
        atol=1e-2,
        rtol=1e-2,
    )


class TestV4Dimension:
    """Tests for DeepSeek V4 dimension (d_qk=512)."""

    @pytest.mark.parametrize("num_tokens,num_heads", [(1, 16), (1, 128), (8, 32)])
    @pytest.mark.parametrize("topk", [128, 1024, 2048])
    def test_v4_correctness(self, num_tokens, num_heads, topk, kv_cache_v4):
        _assert_triton_matches_reference(
            num_tokens,
            num_heads,
            topk,
            kv_cache_v4,
            _DIM_QK_V4,
        )

    @pytest.mark.parametrize("num_tokens", [1, 8, 32])
    def test_v4_various_lengths(self, num_tokens, kv_cache_v4):
        _assert_triton_matches_reference(
            num_tokens,
            num_heads=64,
            topk=1024,
            kv_cache=kv_cache_v4,
            dim_qk=_DIM_QK_V4,
        )


class TestV3BackwardCompatibility:
    """Tests to ensure V3.2 (d_qk=576) still works after V4 changes."""

    @pytest.mark.parametrize("num_tokens,num_heads", [(1, 16), (8, 32), (32, 64)])
    @pytest.mark.parametrize("topk", [128, 1024])
    def test_v3_correctness(self, num_tokens, num_heads, topk, kv_cache_v3):
        _assert_triton_matches_reference(
            num_tokens,
            num_heads,
            topk,
            kv_cache_v3,
            _DIM_QK_V3,
        )


class TestTopkLength:
    """Tests for variable topk per token (topk_length parameter)."""

    def test_v4_with_topk_length(self, kv_cache_v4):
        """Test when different tokens have different valid topk counts."""
        num_tokens = 8
        num_heads = 32
        topk = 2048

        torch.manual_seed(0)
        q = torch.randn(
            num_tokens, num_heads, _DIM_QK_V4, dtype=torch.bfloat16, device="cuda"
        )
        indices = torch.full(
            (num_tokens, 1, topk), -1, dtype=torch.int32, device="cuda"
        )

        for t in range(num_tokens):
            valid_count = (t + 1) * 100
            idx = torch.randint(0, kv_cache_v4.shape[0], (valid_count,), device="cuda")
            indices[t, 0, :valid_count] = idx

        topk_length = torch.tensor(
            [(t + 1) * 100 for t in range(num_tokens)],
            dtype=torch.int32,
            device="cuda",
        )

        sm_scale = _DIM_QK_V4**-0.5

        triton_out = triton_sparse_mla_attention(
            q, kv_cache_v4, indices, sm_scale=sm_scale, num_kv_splits=1
        )

        ref_out, _, _, _ = reference_mla_sparse_prefill(
            q, kv_cache_v4, indices, sm_scale, d_v=512, topk_length=topk_length
        )

        torch.testing.assert_close(
            triton_out.float(),
            ref_out.float(),
            atol=1e-2,
            rtol=1e-2,
        )


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_v4_all_invalid_indices(self, kv_cache_v4):
        """Regression: all indices are -1 (should not produce NaN)."""
        num_tokens = 5
        num_heads = 16
        topk = 2048

        torch.manual_seed(0)
        q = torch.randn(
            num_tokens, num_heads, _DIM_QK_V4, dtype=torch.bfloat16, device="cuda"
        )
        indices = torch.full(
            (num_tokens, 1, topk), -1, dtype=torch.int32, device="cuda"
        )

        out = triton_sparse_mla_attention(
            q, kv_cache_v4, indices, sm_scale=0.0417, num_kv_splits=1
        )

        assert not torch.isnan(out).any(), "Output contains NaN with all -1 indices"
        assert not torch.isinf(out).any(), "Output contains inf with all -1 indices"

    def test_v4_partial_valid_indices(self, kv_cache_v4):
        """Test when only some topk slots are valid."""
        num_tokens = 3
        num_heads = 32
        topk = 1024

        torch.manual_seed(0)
        q = torch.randn(
            num_tokens, num_heads, _DIM_QK_V4, dtype=torch.bfloat16, device="cuda"
        )
        indices = torch.full(
            (num_tokens, 1, topk), -1, dtype=torch.int32, device="cuda"
        )

        for t in range(num_tokens):
            idx = torch.arange(t + 1, device="cuda").long()
            indices[t, 0, : len(idx)] = idx

        out = triton_sparse_mla_attention(
            q, kv_cache_v4, indices, sm_scale=0.1, num_kv_splits=1
        )

        assert out.shape == (num_tokens, num_heads, 512)
        assert not torch.isnan(out).any()

    def test_v4_single_token(self, kv_cache_v4):
        """Test decode-like case (single token)."""
        _assert_triton_matches_reference(
            num_tokens=1,
            num_heads=64,
            topk=128,
            kv_cache=kv_cache_v4,
            dim_qk=_DIM_QK_V4,
        )


class TestSplitKV:
    """Tests for split-KV path (num_kv_splits > 1)."""

    @pytest.mark.parametrize("num_kv_splits", [2, 4, 8])
    def test_v4_split_kv_matches_single_pass(self, num_kv_splits, kv_cache_v4):
        """Split-KV output should match single-pass."""
        torch.manual_seed(0)
        num_tokens = 32
        num_heads = 64
        topk = 2048

        q = torch.randn(
            num_tokens, num_heads, _DIM_QK_V4, dtype=torch.bfloat16, device="cuda"
        )
        indices = torch.randint(
            0, kv_cache_v4.shape[0], (num_tokens, 1, topk), dtype=torch.int32, device="cuda"
        )

        out_ref = triton_sparse_mla_attention(
            q, kv_cache_v4, indices, sm_scale=0.1, num_kv_splits=1
        )
        out_split = triton_sparse_mla_attention(
            q, kv_cache_v4, indices, sm_scale=0.1, num_kv_splits=num_kv_splits
        )

        torch.testing.assert_close(
            out_split.float(),
            out_ref.float(),
            atol=5e-2,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("num_kv_splits", [1, 2, 4])
    def test_v4_auto_split_heuristic(self, num_kv_splits, kv_cache_v4):
        """Auto split heuristic should work."""
        torch.manual_seed(0)
        num_tokens = 64
        num_heads = 128
        topk = 4096

        q = torch.randn(
            num_tokens, num_heads, _DIM_QK_V4, dtype=torch.bfloat16, device="cuda"
        )
        indices = torch.randint(
            0, kv_cache_v4.shape[0], (num_tokens, 1, topk), dtype=torch.int32, device="cuda"
        )

        out = triton_sparse_mla_attention(
            q, kv_cache_v4, indices, sm_scale=0.1, num_kv_splits=num_kv_splits
        )

        assert out.shape == (num_tokens, num_heads, 512)
        assert not torch.isnan(out).any()


class TestLongSequence:
    """Tests for long sequence handling (relevant for 1M context)."""

    def test_v4_long_prefill(self, kv_cache_v4):
        """Test with long sequence (simulating prefill)."""
        num_tokens = 256
        num_heads = 64
        topk = 4096

        torch.manual_seed(0)
        q = torch.randn(
            num_tokens, num_heads, _DIM_QK_V4, dtype=torch.bfloat16, device="cuda"
        )
        indices = torch.randint(
            0, kv_cache_v4.shape[0], (num_tokens, 1, topk), dtype=torch.int32, device="cuda"
        )

        out = triton_sparse_mla_attention(
            q, kv_cache_v4, indices, sm_scale=0.1, num_kv_splits=1
        )

        assert out.shape == (num_tokens, num_heads, 512)
        assert not torch.isnan(out).any()

    def test_v4_very_long_topk(self, kv_cache_v4):
        """Test with very large topk (like 1M context)."""
        num_tokens = 16
        num_heads = 32
        topk = 16384

        torch.manual_seed(0)
        q = torch.randn(
            num_tokens, num_heads, _DIM_QK_V4, dtype=torch.bfloat16, device="cuda"
        )
        indices = torch.randint(
            0, min(32768, kv_cache_v4.shape[0]), (num_tokens, 1, topk), dtype=torch.int32, device="cuda"
        )

        out = triton_sparse_mla_attention(
            q, kv_cache_v4, indices, sm_scale=0.1, num_kv_splits=1
        )

        assert out.shape == (num_tokens, num_heads, 512)
        assert not torch.isnan(out).any()
