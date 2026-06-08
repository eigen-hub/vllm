# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SM80-safe gather+dequantize kernels for DeepSeek V4 KV cache.

Replaces ``tl.float8e4nv`` with software ``_decode_e4m3fn`` from
``vllm.v1.attention.ops.mqa_logits_triton``.

Exports:
- ``_gather_with_mask_kernel``: Triton JIT kernel
- ``gather_dequant_two_scopes_with_mask``: Python launcher
- ``_dequantize_and_gather_k_cache_sm80``: SM80 dequant gather
- ``_dequant_gather_sm80_op_fake``: Fake op for Dynamo
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.mqa_logits_triton import _decode_e4m3fn


@triton.jit
def _gather_with_mask_kernel(
    cache_ptr,
    indices_ptr,
    topk_lens_ptr,
    out_ptr,
    mask_ptr,
    n_blocks,
    block_stride: tl.constexpr,
    cache_block_size: tl.constexpr,
    fp8_dim: tl.constexpr,
    rope_bytes: tl.constexpr,
    head_dim: tl.constexpr,
    topk_in: tl.constexpr,
    total_topk: tl.constexpr,
    slot_offset: tl.constexpr,
    n_quant_blocks: tl.constexpr,
    quant_block: tl.constexpr,
    HAS_TOPK_LENS: tl.constexpr,
):
    """Gather + per-row dequantize into a merged buffer. SM80-safe."""
    pid = tl.program_id(0)
    flat_idx = tl.load(indices_ptr + pid)
    token_id = pid // topk_in
    slot_in = pid % topk_in
    dst_slot = slot_offset + slot_in
    dst_row = token_id * total_topk + dst_slot
    cache_capacity = n_blocks * cache_block_size
    is_padding = flat_idx == -1
    is_in_range = (flat_idx >= 0) & (flat_idx < cache_capacity)
    if HAS_TOPK_LENS:
        topk_len = tl.load(topk_lens_ptr + token_id).to(tl.int64)
        is_beyond = slot_in.to(tl.int64) >= topk_len
    else:
        is_beyond = False
    invalid = is_padding | is_beyond | (~is_in_range)
    tl.store(mask_ptr + dst_row, invalid)
    safe_idx = tl.where(is_in_range, flat_idx, 0)
    block_idx = safe_idx // cache_block_size
    pos_in_block = safe_idx % cache_block_size
    token_data_size = fp8_dim + rope_bytes
    scale_dim = n_quant_blocks + 1
    block_base = cache_ptr + block_idx.to(tl.int64) * block_stride
    token_data = block_base + pos_in_block * token_data_size
    token_scale = (
        block_base + cache_block_size * token_data_size + pos_in_block * scale_dim
    )
    out_row = out_ptr + dst_row * head_dim
    for qb in tl.static_range(n_quant_blocks):
        offsets = qb * quant_block + tl.arange(0, quant_block)
        offset_mask = offsets < fp8_dim
        x_u8 = tl.load(token_data + offsets, mask=offset_mask, other=0)
        x_dequant = _decode_e4m3fn(x_u8)
        scale_u8 = tl.load(token_scale + qb)
        scale = tl.exp2(scale_u8.to(tl.float32) - 127.0)
        x_bf16 = (x_dequant * scale).to(tl.bfloat16)
        x_bf16 = tl.where(is_in_range, x_bf16, tl.zeros_like(x_bf16))
        tl.store(out_row + offsets, x_bf16, mask=offset_mask)
    bf16_src = (token_data + fp8_dim).to(tl.pointer_type(tl.bfloat16))
    out_bf16 = out_row + fp8_dim
    for j in tl.static_range(rope_bytes // (2 * 16)):
        chunk_offsets = j * 16 + tl.arange(0, 16)
        bf16_vals = tl.load(bf16_src + chunk_offsets)
        bf16_vals = tl.where(is_in_range, bf16_vals, tl.zeros_like(bf16_vals))
        tl.store(out_bf16 + chunk_offsets, bf16_vals)


def gather_dequant_two_scopes_with_mask(
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
    gathered: torch.Tensor,
    invalid_mask: torch.Tensor,
) -> None:
    """Gather SWA + extra KV entries into one buffer with per-row invalid mask."""
    assert nope_dim % 64 == 0
    assert head_dim == nope_dim + rope_dim
    n_tokens = gathered.shape[0]
    if n_tokens == 0:
        return
    swa_indices_flat = swa_indices.reshape(n_tokens, -1)
    swa_topk = swa_indices_flat.shape[1]
    if extra_indices is not None and extra_kv_cache is not None:
        extra_indices_flat = extra_indices.reshape(n_tokens, -1)
        extra_topk = extra_indices_flat.shape[1]
    else:
        extra_indices_flat = None
        extra_topk = 0
    total_topk = swa_topk + extra_topk
    assert gathered.shape == (n_tokens, total_topk, head_dim)
    assert invalid_mask.shape == (n_tokens, total_topk)
    device = swa_kv_cache.device
    n_quant_blocks = nope_dim // 64

    def _to_i64(t):
        return t if t.dtype == torch.int64 else t.to(torch.int64)

    swa_cache_u8 = (
        swa_kv_cache.view(torch.uint8)
        if swa_kv_cache.dtype != torch.uint8 else swa_kv_cache
    )
    swa_lens = swa_topk_length
    has_swa_lens = swa_lens is not None
    if has_swa_lens:
        swa_lens = swa_lens.reshape(n_tokens)
        if swa_lens.dtype != torch.int32:
            swa_lens = swa_lens.to(torch.int32)
    else:
        swa_lens = torch.empty(0, dtype=torch.int32, device=device)

    _gather_with_mask_kernel[(n_tokens * swa_topk,)](
        swa_cache_u8, _to_i64(swa_indices_flat).reshape(-1), swa_lens,
        gathered, invalid_mask, swa_cache_u8.shape[0],
        block_stride=swa_cache_u8.stride(0), cache_block_size=swa_block_size,
        fp8_dim=nope_dim, rope_bytes=rope_dim * 2, head_dim=head_dim,
        topk_in=swa_topk, total_topk=total_topk, slot_offset=0,
        n_quant_blocks=n_quant_blocks, quant_block=64, HAS_TOPK_LENS=has_swa_lens,
    )
    if extra_topk > 0:
        assert extra_indices_flat is not None and extra_kv_cache is not None
        extra_cache_u8 = (
            extra_kv_cache.view(torch.uint8)
            if extra_kv_cache.dtype != torch.uint8 else extra_kv_cache
        )
        extra_lens = extra_topk_length
        has_extra_lens = extra_lens is not None
        if has_extra_lens:
            extra_lens = extra_lens.reshape(n_tokens)
            if extra_lens.dtype != torch.int32:
                extra_lens = extra_lens.to(torch.int32)
        else:
            extra_lens = torch.empty(0, dtype=torch.int32, device=device)
        _gather_with_mask_kernel[(n_tokens * extra_topk,)](
            extra_cache_u8, _to_i64(extra_indices_flat).reshape(-1), extra_lens,
            gathered, invalid_mask, extra_cache_u8.shape[0],
            block_stride=extra_cache_u8.stride(0), cache_block_size=extra_block_size,
            fp8_dim=nope_dim, rope_bytes=rope_dim * 2, head_dim=head_dim,
            topk_in=extra_topk, total_topk=total_topk, slot_offset=swa_topk,
            n_quant_blocks=n_quant_blocks, quant_block=64, HAS_TOPK_LENS=has_extra_lens,
        )


def _dequant_gather_sm80_op_fake(
    out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
):
    return None





@triton.jit
def _dequantize_and_gather_k_kernel_sm80(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    max_blocks_per_seq: tl.constexpr,
    fp8_dim: tl.constexpr,
    bf16_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    block_stride: tl.constexpr,
    output_dim: tl.constexpr,
    fp8_max: tl.constexpr,
    n_quant_blocks: tl.constexpr,
):
    """SM80 variant: uses _decode_e4m3fn instead of tl.float8e4nv."""
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len

    for i in range(worker_id, gather_len, num_workers):
        pos = start_pos + i
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size
        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)
        cache_block_ptr = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride
        token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
        token_scale_ptr = (
            cache_block_ptr
            + cache_block_size * token_data_size
            + pos_in_block * scale_dim
        )
        token_fp8_ptr = token_data_ptr
        token_bf16_ptr = token_data_ptr + fp8_dim
        output_row_ptr = out_ptr + batch_idx * out_stride0 + (offset + i) * out_stride1

        for qblock_idx in tl.static_range(n_quant_blocks):
            qblock_start = qblock_idx * quant_block
            if qblock_start < fp8_dim:
                offsets = qblock_start + tl.arange(0, quant_block)
                mask = offsets < fp8_dim
                x_uint8 = tl.load(token_fp8_ptr + offsets, mask=mask, other=0)
                x_float = _decode_e4m3fn(x_uint8)
                encoded_scale = tl.load(token_scale_ptr + qblock_idx)
                exponent = encoded_scale.to(tl.float32) - 127.0
                scale = tl.exp2(exponent)
                x_dequant = x_float * scale
                tl.store(output_row_ptr + offsets, x_dequant.to(tl.bfloat16), mask=mask)

        bf16_output_offset = fp8_dim
        bf16_cache_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))
        for j in tl.static_range(bf16_dim // 16):
            chunk_offsets = j * 16 + tl.arange(0, 16)
            bf16_vals = tl.load(bf16_cache_ptr + chunk_offsets)
            tl.store(output_row_ptr + bf16_output_offset + chunk_offsets, bf16_vals)


def _dequantize_and_gather_k_cache_sm80(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    """SM80 Triton dispatch -- uses _decode_e4m3fn instead of tl.float8e4nv."""
    TOKEN_FP8_DIM = 448
    TOKEN_BF16_DIM = 64
    TOKEN_SCALE_DIM = 8
    QUANT_BLOCK_SIZE = 64
    TOKEN_DATA_SIZE = TOKEN_FP8_DIM + TOKEN_BF16_DIM * 2

    num_reqs = seq_lens.shape[0]
    NUM_WORKERS = 128
    _dequantize_and_gather_k_kernel_sm80[(num_reqs, NUM_WORKERS)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        offset,
        gather_lens,
        max_blocks_per_seq=block_table.shape[-1],
        fp8_dim=TOKEN_FP8_DIM,
        bf16_dim=TOKEN_BF16_DIM,
        scale_dim=TOKEN_SCALE_DIM,
        quant_block=QUANT_BLOCK_SIZE,
        cache_block_size=block_size,
        token_data_size=TOKEN_DATA_SIZE,
        block_stride=k_cache.stride(0),
        output_dim=512,
        fp8_max=448.0,
        n_quant_blocks=7,
    )
