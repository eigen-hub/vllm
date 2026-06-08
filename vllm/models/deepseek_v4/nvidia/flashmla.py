# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
    compute_fp8_einsum_recipe,
    deep_gemm_fp8_o_proj,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)
from vllm.triton_utils import LOG2E, tl, triton
from vllm.utils.deep_gemm import use_dsv4_reference_kernels_current_device
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


class DeepseekV4FlashMLAAttention(DeepseekV4Attention):
    """FlashMLA sparse MLA attention layer for DeepSeek V4 (CUDA)."""

    backend_cls = DeepseekV4FlashMLABackend

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe()
        self._use_reference_kernels = use_dsv4_reference_kernels_current_device()

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self._use_reference_kernels:
            # SM80 reference path: BF16 inverse RoPE via Triton kernel + direct
            # wo_a Linear call. Avoids fused_inv_rope_fp8_quant which uses
            # tl.float8e4nv (unsupported on SM80).
            o_rotated = _apply_inv_rope_to_o(
                o,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.rope_head_dim,
            )
            if self.n_local_groups > 1:
                z = self._apply_wo_a_bmm(o_rotated)
            else:
                z = self.wo_a(o_rotated.flatten(1))
                if isinstance(z, tuple):
                    z = z[0]
            return self.wo_b(z)
        return deep_gemm_fp8_o_proj(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a,
            self.wo_b,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
            einsum_recipe=self._einsum_recipe,
            tma_aligned_scales=self._tma_aligned_scales,
        )

    def _ensure_wo_a_bmm_weight(self, ref: torch.Tensor) -> None:
        if getattr(self, '_wo_a_bmm_weight', None) is not None:
            return
        k_dim = self.wo_a.input_size_per_partition
        n_total = self.wo_a.output_size_per_partition
        n_groups = self.n_local_groups
        n_per_group = n_total // n_groups
        eye = torch.eye(k_dim, dtype=ref.dtype, device=ref.device)
        with torch.no_grad():
            w_t = self.wo_a(eye)
        if isinstance(w_t, tuple):
            w_t = w_t[0]
        self._wo_a_bmm_weight = (
            w_t.view(k_dim, n_groups, n_per_group).permute(1, 0, 2).contiguous()
        )

    def _apply_wo_a_bmm(self, o_rotated: torch.Tensor) -> torch.Tensor:
        self._ensure_wo_a_bmm_weight(o_rotated)
        assert self._wo_a_bmm_weight is not None
        n_groups = self.n_local_groups
        num_tokens = o_rotated.shape[0]
        k_per_group = self._wo_a_bmm_weight.shape[1]
        n_per_group = self._wo_a_bmm_weight.shape[2]
        x = (
            o_rotated.reshape(num_tokens, n_groups, k_per_group)
            .transpose(0, 1)
            .contiguous()
        )
        out = torch.bmm(x, self._wo_a_bmm_weight)
        return out.transpose(0, 1).reshape(num_tokens, n_groups * n_per_group)

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        # FP8 decode kernel only supports h_q = 64 or 128.
        if num_heads > 128:
            raise ValueError(
                f"DeepseekV4 FlashMLA does not support {num_heads} heads "
                "(FP8 decode kernel requires h_q in {64, 128})."
            )
        return 64 if num_heads <= 64 else 128

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        # Get SWA and indexer metadata from forward context
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: no real metadata. Reserve the same bf16
            # gather workspace _forward_prefill would; the dequantize / topk
            # / sparse_fwd kernels are skipped this step.
            swa_only = self.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        # SWA-only layers (compress_ratio <= 1) don't have their own KV cache
        # allocation, so self.kv_cache may be empty after profiling cleanup.
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        # We treat queries in the same seq as different queries
        # and later we only attend by generated indices.
        # q arrives pre-padded to self.padded_heads by the outer wrapper.
        q = q.unsqueeze(1)

        # Prepare SWA cache (num_blocks, swa_block_size, 1, head_bytes)
        # Use unsqueeze to preserve strides (handles padded blocks correctly)
        swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)
        # Reshape KV cache to (num_blocks, block_size, 1, head_bytes)
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        # One FlashMLASchedMeta per layer type, shared across all same-type
        # layers within this decode step. The first forward call per type
        # triggers the in-kernel planner (allocating tile_scheduler_metadata
        # and num_splits via PyTorch's graph-aware allocator so CUDA graph
        # capture reuses the same addresses on replay); subsequent same-type
        # layers see have_initialized=True and skip the planner.
        if self.compress_ratio <= 1:
            tile_metadata = swa_metadata.tile_sched_swaonly
        elif self.compress_ratio == 4:
            tile_metadata = swa_metadata.tile_sched_c4a
        elif self.compress_ratio == 128:
            tile_metadata = swa_metadata.tile_sched_c128a
        else:
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio}; "
                "expected 1, 4, or 128."
            )
        if self._use_reference_kernels:
            # SM80 decode path: gather KV from paged cache + Triton sparse attn
            assert swa_cache is not None
            extra_block_size = 0
            extra_indices_flat = None
            extra_topk = 0
            if not swa_only:
                extra_block_size = attn_metadata.block_size // self.compress_ratio
                extra_indices_flat = topk_indices.reshape(num_decode_tokens, -1)
                extra_topk = extra_indices_flat.shape[1]
            swa_topk = swa_indices.shape[-1]
            total_topk = swa_topk + extra_topk
            gathered = torch.empty(
                (num_decode_tokens, total_topk, self.head_dim),
                dtype=torch.bfloat16, device=q.device,
            )
            invalid_mask = torch.empty(
                (num_decode_tokens, total_topk), dtype=torch.bool, device=q.device,
            )
            torch.ops.vllm.deepseek_v4_gather_sm80(
                gathered, invalid_mask,
                self.swa_cache_layer.kv_cache,
                swa_metadata.block_size,
                swa_indices.reshape(num_decode_tokens, -1),
                swa_lens,
                None if swa_only else kv_cache.squeeze(-2),
                extra_block_size,
                extra_indices_flat,
                topk_lens,
                self.nope_head_dim, self.rope_head_dim, self.head_dim,
            )
            torch.ops.vllm.deepseek_v4_sparse_decode_sm80(
                output.unsqueeze(1),
                q.to(torch.bfloat16).contiguous(),
                gathered.unsqueeze(1), invalid_mask.unsqueeze(1),
                self.attn_sink[:q.shape[1]],
                self.scale, self.head_dim,
            )
        else:
            assert tile_metadata is not None, (
                "swa_metadata missing tile_sched entry for "
                f"compress_ratio={self.compress_ratio}; "
                "DeepseekSparseSWAMetadataBuilder.build_tile_scheduler did not "
                "allocate one for this layer type."
            )
            out, _ = flash_mla_with_kvcache(
                q=q,
                k_cache=swa_cache,
                block_table=None,
                head_dim_v=512,
                tile_scheduler_metadata=tile_metadata,
                cache_seqlens=None,
                is_fp8_kvcache=True,
                indices=swa_indices,
                topk_length=swa_lens,
                softmax_scale=self.scale,
                attn_sink=self.attn_sink,
                extra_k_cache=kv_cache if not swa_only else None,
                extra_indices_in_kvcache=topk_indices,
                extra_topk_length=topk_lens,
                out=output.unsqueeze(1),
            )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
            # Compressed region must fit the full compressed pool (seq_len //
            # compress_ratio), not just top_k. top_k bounds how many indices
            # the indexer selects, not the pool size it indexes into.
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            # NOTE(woosuk): topk_indices will not be used for SWA-only layers.
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        chunk_size_const = self.PREFILL_CHUNK_SIZE
        num_chunks = (num_prefills + chunk_size_const - 1) // chunk_size_const

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((chunk_size_const, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size_const
            chunk_end = min(chunk_start + chunk_size_const, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
            )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
            )
        if self._use_reference_kernels:
            out_chunk = triton_sparse_mla_attention(
                q[query_start:query_end],
                kv.view(-1, 1, q.shape[-1]),
                combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                attn_sink=self.attn_sink[: q[query_start:query_end].shape[1]],
            )
            output[query_start:query_end].copy_(out_chunk)
        else:
            flash_mla_sparse_fwd(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
                out=output[query_start:query_end],
            )

# ---------------------------------------------------------------------------
# SM80 reference kernel helpers
# ---------------------------------------------------------------------------

@triton.jit
def _inv_rope_bf16_kernel(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope: tl.constexpr,
    nope_dim: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_t >= T:
        return
    pos = tl.load(positions_ptr + pid_t)
    base_cs = cos_sin_cache_ptr + pos * rope_dim
    r = tl.arange(0, half_rope)
    cos_v = tl.load(base_cs + r).to(tl.float32)
    sin_v = tl.load(base_cs + half_rope + r).to(tl.float32)
    base_row = o_ptr + (pid_t * H + pid_h) * D + nope_dim
    even = tl.load(base_row + 2 * r).to(tl.float32)
    odd = tl.load(base_row + 2 * r + 1).to(tl.float32)
    new_even = even * cos_v + odd * sin_v
    new_odd = odd * cos_v - even * sin_v
    tl.store(base_row + 2 * r, new_even.to(tl.bfloat16))
    tl.store(base_row + 2 * r + 1, new_odd.to(tl.bfloat16))


def _apply_inv_rope_to_o(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    if not o.is_contiguous():
        o = o.contiguous()
    out = o.clone()
    num_tokens, num_heads, head_dim = out.shape
    nope_dim = head_dim - rope_dim
    half_rope = rope_dim // 2

    _inv_rope_bf16_kernel[(num_tokens, num_heads)](
        out,
        positions.to(torch.int64).contiguous(),
        cos_sin_cache,
        num_tokens,
        H=num_heads,
        D=head_dim,
        rope_dim=rope_dim,
        half_rope=half_rope,
        nope_dim=nope_dim,
    )
    return out


def _dsv4_sm80_o_proj_fake(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    return torch.empty_like(o)


from vllm.utils.torch_utils import direct_register_custom_op

direct_register_custom_op(
    op_name="dsv4_sm80_o_proj",
    op_func=_apply_inv_rope_to_o,
    mutates_args=[],
    fake_impl=_dsv4_sm80_o_proj_fake,
)

# ---------------------------------------------------------------------------
# SM80 FP8 einsum fallback
# ---------------------------------------------------------------------------

import math


def _decode_e8m0_scales(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.float8_e8m0fnu:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )
        return _upcast_e8m0_to_fp32(scale).contiguous()
    return scale.to(torch.float32)


def _expand_last_dim_scales(scale: torch.Tensor, last_dim: int) -> torch.Tensor:
    scale = _decode_e8m0_scales(scale)
    block = math.ceil(last_dim / scale.shape[-1])
    return torch.repeat_interleave(scale, block, dim=-1)[..., :last_dim]


def _expand_2d_block_scales(
    scale: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    scale = _decode_e8m0_scales(scale)
    row_blocks, col_blocks = scale.shape[-2:]
    row_block = math.ceil(rows / row_blocks)
    col_block = math.ceil(cols / col_blocks)
    scale = torch.repeat_interleave(scale, row_block, dim=-2)[..., :rows, :]
    scale = torch.repeat_interleave(scale, col_block, dim=-1)[..., :, :cols]
    return scale


def _deepseek_v4_fp8_einsum_fallback(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
) -> None:
    if equation != "bhr,hdr->bhd":
        raise RuntimeError(f"Unsupported fallback equation: {equation}")

    num_groups = a.shape[1]
    hidden_dim = a.shape[2]
    output_dim = b.shape[0] // num_groups

    if b.shape[0] % num_groups != 0:
        raise RuntimeError(
            f"Cannot reshape weight of shape {tuple(b.shape)} into "
            f"({num_groups}, {output_dim}, {hidden_dim})."
        )

    a_deq = (a.to(torch.float32) * _expand_last_dim_scales(a_scale, hidden_dim)).to(
        torch.bfloat16
    )
    b_deq = b.view(num_groups, output_dim, hidden_dim).to(torch.float32)
    b_scale_deq = _expand_2d_block_scales(
        b_scale.view(num_groups, -1, b_scale.shape[-1]),
        output_dim,
        hidden_dim,
    )
    b_deq = (b_deq * b_scale_deq).to(torch.bfloat16)
    out.copy_(torch.einsum(equation, a_deq, b_deq).to(out.dtype))


def deepseek_v4_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    from vllm.utils.deep_gemm import use_dsv4_reference_kernels_current_device
    if use_dsv4_reference_kernels_current_device():
        _deepseek_v4_fp8_einsum_fallback(a, a_scale, b, b_scale, out, equation)
        return
    from vllm.utils.deep_gemm import fp8_einsum
    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))


direct_register_custom_op(
    op_name="deepseek_v4_fp8_einsum",
    op_func=deepseek_v4_fp8_einsum,
    mutates_args=["out"],
    fake_impl=lambda a, a_s, b, b_s, out, eq, recipe: None,
)

# ---------------------------------------------------------------------------
# SM80 Triton sparse attention for decode
# ---------------------------------------------------------------------------

@triton.jit
def _dsv4_sm80_sparse_attn_split_kernel(
    q_ptr,
    kv_ptr,
    invalid_mask_ptr,
    acc_split_ptr,
    max_split_ptr,
    sum_split_ptr,
    n_tokens,
    total_topk,
    sm_scale_log2,
    H: tl.constexpr,
    D: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    SPLIT_T: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_split = tl.program_id(1)
    pid_h = tl.program_id(2)

    if pid_t >= n_tokens:
        return

    chunk_size = (total_topk + SPLIT_T - 1) // SPLIT_T
    n_start_chunk = pid_split * chunk_size
    n_end_chunk = tl.minimum(n_start_chunk + chunk_size, total_topk)

    head_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_off < H
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    q = tl.load(
        q_ptr + pid_t * H * D + head_off[:, None] * D + d_off[None, :],
        mask=head_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    e_max = tl.zeros((BLOCK_H,), dtype=tl.float32) - 1.0e30
    e_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_DV), dtype=tl.float32)

    n_iter = (chunk_size + BLOCK_N - 1) // BLOCK_N
    for n_block in range(n_iter):
        n_start = n_start_chunk + n_block * BLOCK_N
        n_off = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_off < n_end_chunk

        invalid_u8 = tl.load(
            invalid_mask_ptr + pid_t * total_topk + n_off,
            mask=n_mask,
            other=1,
        )
        valid = (invalid_u8 == 0) & n_mask

        kv = tl.load(
            kv_ptr + pid_t * total_topk * D + n_off[:, None] * D + d_off[None, :],
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(kv))
        qk *= sm_scale_log2
        qk = tl.where(head_mask[:, None] & valid[None, :], qk, -1.0e30)

        n_e_max = tl.maximum(tl.max(qk, axis=1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(tl.float16), kv.to(tl.float16))
        e_sum = e_sum * re_scale + tl.sum(p, axis=1)
        e_max = n_e_max

    dv_off = tl.arange(0, BLOCK_DV)
    dv_mask = dv_off < D_V
    base_acc = (
        pid_t * SPLIT_T * H * D_V
        + pid_split * H * D_V
        + head_off[:, None] * D_V
        + dv_off[None, :]
    )
    tl.store(
        acc_split_ptr + base_acc,
        acc,
        mask=head_mask[:, None] & dv_mask[None, :],
    )
    base_ms = pid_t * SPLIT_T * H + pid_split * H + head_off
    tl.store(max_split_ptr + base_ms, e_max, mask=head_mask)
    tl.store(sum_split_ptr + base_ms, e_sum, mask=head_mask)


@triton.jit
def _dsv4_sm80_sparse_attn_combine_kernel(
    acc_split_ptr,
    max_split_ptr,
    sum_split_ptr,
    attn_sink_ptr,
    out_ptr,
    n_tokens,
    has_sink: tl.constexpr,
    H: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    SPLIT_T: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_t >= n_tokens:
        return

    head_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_off < H

    e_max_global = tl.zeros((BLOCK_H,), dtype=tl.float32) - 1.0e30
    for s in range(SPLIT_T):
        m = tl.load(
            max_split_ptr + pid_t * SPLIT_T * H + s * H + head_off,
            mask=head_mask,
            other=-1.0e30,
        )
        e_max_global = tl.maximum(e_max_global, m)

    sink_log2 = tl.zeros((BLOCK_H,), dtype=tl.float32)
    if has_sink:
        sink = tl.load(attn_sink_ptr + head_off, mask=head_mask, other=0.0)
        sink_log2 = sink * 1.4426950408889634
        e_max_global = tl.maximum(e_max_global, sink_log2)

    dv_off = tl.arange(0, BLOCK_DV)
    dv_mask = dv_off < D_V
    acc_global = tl.zeros((BLOCK_H, BLOCK_DV), dtype=tl.float32)
    sum_global = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for s in range(SPLIT_T):
        m_s = tl.load(
            max_split_ptr + pid_t * SPLIT_T * H + s * H + head_off,
            mask=head_mask,
            other=-1.0e30,
        )
        sum_s = tl.load(
            sum_split_ptr + pid_t * SPLIT_T * H + s * H + head_off,
            mask=head_mask,
            other=0.0,
        )
        scale = tl.exp2(m_s - e_max_global)
        sum_global += scale * sum_s

        base_acc = (
            pid_t * SPLIT_T * H * D_V
            + s * H * D_V
            + head_off[:, None] * D_V
            + dv_off[None, :]
        )
        acc_s = tl.load(
            acc_split_ptr + base_acc,
            mask=head_mask[:, None] & dv_mask[None, :],
            other=0.0,
        )
        acc_global += scale[:, None] * acc_s

    if has_sink:
        sum_global += tl.exp2(sink_log2 - e_max_global)

    sum_safe = tl.where(sum_global > 0, sum_global, 1.0)
    out = (acc_global / sum_safe[:, None]).to(tl.bfloat16)
    tl.store(
        out_ptr + pid_t * H * D_V + head_off[:, None] * D_V + dv_off[None, :],
        out,
        mask=head_mask[:, None] & dv_mask[None, :],
    )


def _dsv4_sm80_sparse_attn_decode_triton(
    q: torch.Tensor,
    gathered_kv: torch.Tensor,
    invalid_mask: torch.Tensor,
    attn_sink: torch.Tensor | None,
    sm_scale: float,
    head_dim_v: int,
) -> torch.Tensor:
    n_tokens, h, d = q.shape
    _, t, d_kv = gathered_kv.shape
    assert d_kv == d
    assert invalid_mask.shape == (n_tokens, t)

    block_d = triton.next_power_of_2(d)
    block_dv = triton.next_power_of_2(head_dim_v)
    block_h = 16
    block_n = 32
    n_tiles = (t + block_n - 1) // block_n
    split_t = max(1, min(16, n_tiles))

    out = torch.empty((n_tokens, h, head_dim_v), dtype=torch.bfloat16, device=q.device)
    invalid_u8 = invalid_mask.to(torch.uint8)

    acc_split = torch.empty(
        (n_tokens, split_t, h, head_dim_v),
        dtype=torch.float32,
        device=q.device,
    )
    max_split = torch.empty(
        (n_tokens, split_t, h), dtype=torch.float32, device=q.device
    )
    sum_split = torch.empty_like(max_split)

    grid_split = (n_tokens, split_t, triton.cdiv(h, block_h))
    _dsv4_sm80_sparse_attn_split_kernel[grid_split](
        q,
        gathered_kv,
        invalid_u8,
        acc_split,
        max_split,
        sum_split,
        n_tokens,
        t,
        sm_scale * LOG2E,
        H=h,
        D=d,
        D_V=head_dim_v,
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        BLOCK_DV=block_dv,
        SPLIT_T=split_t,
        num_warps=4,
    )

    grid_combine = (n_tokens, triton.cdiv(h, block_h))
    _dsv4_sm80_sparse_attn_combine_kernel[grid_combine](
        acc_split,
        max_split,
        sum_split,
        attn_sink if attn_sink is not None else q.new_zeros(h),
        out,
        n_tokens,
        has_sink=(attn_sink is not None),
        H=h,
        D_V=head_dim_v,
        BLOCK_H=block_h,
        BLOCK_DV=block_dv,
        SPLIT_T=split_t,
        num_warps=4,
    )
    return out


def _sparse_decode_sm80_op(
    out: torch.Tensor,
    q: torch.Tensor,
    gathered_kv: torch.Tensor,
    invalid_mask: torch.Tensor,
    attn_sink: torch.Tensor | None,
    sm_scale: float,
    head_dim_v: int,
) -> None:
    # The call site passes 4D (num_tokens, 1, H, D) while the Triton kernel
    # expects 3D (num_tokens, H, D). Squeeze the singleton batch dim.
    out.copy_(
        _dsv4_sm80_sparse_attn_decode_triton(
            q.squeeze(1), gathered_kv.squeeze(1), invalid_mask.squeeze(1),
            attn_sink, sm_scale, head_dim_v,
        ).unsqueeze(1)
    )


direct_register_custom_op(
    op_name="deepseek_v4_sparse_decode_sm80",
    op_func=_sparse_decode_sm80_op,
    mutates_args=["out"],
    fake_impl=lambda out, q, g, m, s, scale, hdv: None,
)


def _gather_sm80_op(
    gathered: torch.Tensor,
    invalid_mask: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    swa_block_size: int,
    swa_indices: torch.Tensor,
    swa_topk_length: torch.Tensor | None,
    extra_kv_cache: torch.Tensor | None,
    extra_block_size: int,
    extra_indices: torch.Tensor | None,
    extra_topk_length: torch.Tensor | None,
    nope_dim: int,
    rope_dim: int,
    head_dim: int,
) -> None:
    from vllm.models.deepseek_v4.common.ops.sm80_gather import (
        gather_dequant_two_scopes_with_mask,
    )
    gather_dequant_two_scopes_with_mask(
        swa_kv_cache, swa_block_size,
        swa_indices, swa_topk_length,
        extra_kv_cache, extra_block_size,
        extra_indices, extra_topk_length,
        nope_dim, rope_dim, head_dim,
        gathered, invalid_mask,
    )


direct_register_custom_op(
    op_name="deepseek_v4_gather_sm80",
    op_func=_gather_sm80_op,
    mutates_args=["gathered", "invalid_mask"],
    fake_impl=lambda g, m, *a: None,
)


def triton_sparse_mla_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor | None = None,
) -> torch.Tensor:
    """Optimized sparse attention for SM80 prefill/decode chunks.
    Delegates to triton_sparse_mla_kernel for best performance.
    q: (M, H, D), kv: (N, 1, D), indices: (M, 1, topk).
    """
    from vllm.v1.attention.ops.triton_sparse_mla_kernel import (
        triton_sparse_mla_attention as _kernel_attn,
    )
    return _kernel_attn(
        q, kv, indices, sm_scale=sm_scale, attn_sink=attn_sink,
    )
