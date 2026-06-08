# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch import Tensor

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _rmsnorm_nw_kernel(
    x_ptr,
    out_ptr,
    stride_row,
    D,
    eps,
    RBLOCK: tl.constexpr,
):
    """Weight-free RMSNorm Triton kernel: out = x * rsqrt(mean(x², -1) + eps)."""
    row = tl.program_id(0)
    cols = tl.arange(0, RBLOCK)
    mask = cols < D

    x = tl.load(
        x_ptr + row * stride_row + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    var = tl.sum(x * x, 0) / D
    rstd = tl.rsqrt(var + eps)

    out = (x * rstd).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * D + cols, out, mask=mask, eviction_policy="evict_first")


def rmsnorm_nw(x: Tensor, eps: float) -> Tensor:
    """Weight-free RMSNorm over the last dimension.

    Treats *x* as ``[num_rows, D]`` where ``num_rows = product(shape[:-1])``.
    Returns a contiguous tensor with the same shape and dtype as *x*.
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    x_2d = x.reshape(-1, D)
    num_rows = x_2d.shape[0]

    out = torch.empty_like(x_2d)
    RBLOCK = triton.next_power_of_2(D)

    _rmsnorm_nw_kernel[(num_rows,)](
        x_2d,
        out,
        x_2d.stride(0),
        D,
        eps,
        RBLOCK=RBLOCK,
        num_warps=1 if RBLOCK <= 512 else (4 if RBLOCK <= 4096 else 8),
    )
    return out.view(orig_shape)


@triton.jit
def _hc_head_reduce_store_kernel(
    pre_ptr,
    x_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    pre_stride_t: tl.constexpr,
    pre_stride_m: tl.constexpr,
    x_stride_t: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_h: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden_size

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for mix_idx in tl.static_range(0, hc_mult):
        pre = tl.load(pre_ptr + token_idx * pre_stride_t + mix_idx * pre_stride_m).to(
            tl.float32
        )
        x = tl.load(
            x_ptr
            + token_idx * x_stride_t
            + mix_idx * x_stride_m
            + offsets * x_stride_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += pre * x

    tl.store(
        out_ptr + token_idx * out_stride_t + offsets * out_stride_h,
        acc,
        mask=mask,
    )


def hc_head_reduce_triton_kernel(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> None:
    x_flat = x.flatten(-2)
    x_normed = rmsnorm_nw(x_flat, norm_eps)
    mixes = F.linear(x_normed.float(), hc_fn)
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps

    hidden_size = x.shape[-1]
    hc_mult = x.shape[-2]
    block_h = 1024
    _hc_head_reduce_store_kernel[(x.shape[0], (hidden_size + block_h - 1) // block_h)](
        pre,
        x,
        out,
        hidden_size,
        hc_mult,
        pre.stride(0),
        pre.stride(1),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
        num_warps=4,
    )


def _hc_head_triton(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    """Fill pre-allocated `out` (T, H) in-place with the hc_head result."""
    if hs_flat.shape[0] == 0:
        return

    hc_head_reduce_triton_kernel(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        out,
        rms_eps,
        hc_eps,
    )
    return


direct_register_custom_op(
    op_name="hc_head_triton",
    op_func=_hc_head_triton,
    mutates_args=["out"],
)


# ---------------------------------------------------------------------------
# SM80 Triton reference kernels for MHC pre/post blocks
# ---------------------------------------------------------------------------

@triton.jit
def _mhc_pre_fused_kernel(
    mixes_ptr,
    residual_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    pre_mix_ptr,
    post_mix_ptr,
    comb_mix_ptr,
    n_tokens,
    hc: tl.constexpr,
    h: tl.constexpr,
    hc_mult3: tl.constexpr,
    BLOCK_H: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= n_tokens:
        return

    total = hc * h
    res_base = residual_ptr + pid * total
    sqrsum = tl.zeros((), dtype=tl.float32)
    for blk_start in range(0, total, BLOCK_H):
        offs = blk_start + tl.arange(0, BLOCK_H)
        mask = offs < total
        vals = tl.load(res_base + offs, mask=mask, other=0.0).to(tl.float32)
        sqrsum += tl.sum(vals * vals)

    rsqrt_val = 1.0 / tl.sqrt(sqrsum / total + rms_eps)

    s0 = tl.load(hc_scale_ptr + 0)
    s1 = tl.load(hc_scale_ptr + 1)
    s2 = tl.load(hc_scale_ptr + 2)

    hc_idx = tl.arange(0, hc)
    pre_mixes = tl.load(mixes_ptr + pid * hc_mult3 + hc_idx).to(tl.float32)
    pre_base = tl.load(hc_base_ptr + hc_idx).to(tl.float32)
    pre_mix = tl.sigmoid(pre_mixes * rsqrt_val * s0 + pre_base) + hc_pre_eps
    tl.store(pre_mix_ptr + pid * hc + hc_idx, pre_mix)

    post_mixes = tl.load(mixes_ptr + pid * hc_mult3 + hc + hc_idx).to(tl.float32)
    post_base = tl.load(hc_base_ptr + hc + hc_idx).to(tl.float32)
    post_mix = tl.sigmoid(post_mixes * rsqrt_val * s1 + post_base) * hc_post_mult_value
    tl.store(post_mix_ptr + pid * hc + hc_idx, post_mix)

    rows = tl.arange(0, hc)[:, None]
    cols = tl.arange(0, hc)[None, :]
    comb_off = pid * hc_mult3 + 2 * hc + rows * hc + cols
    comb_mixes = tl.load(mixes_ptr + comb_off).to(tl.float32)
    comb_base = tl.load(hc_base_ptr + 2 * hc + rows * hc + cols).to(tl.float32)
    cm = comb_mixes * rsqrt_val * s2 + comb_base

    row_max = tl.max(cm, axis=1)
    cm = cm - row_max[:, None]
    cm = tl.exp(cm)
    row_sum = tl.sum(cm, axis=1)
    cm = cm / row_sum[:, None] + hc_sinkhorn_eps

    col_sum = tl.sum(cm, axis=0)
    cm = cm / (col_sum[None, :] + hc_sinkhorn_eps)

    for _ in range(sinkhorn_iters - 1):
        row_sum = tl.sum(cm, axis=1)
        cm = cm / (row_sum[:, None] + hc_sinkhorn_eps)
        col_sum = tl.sum(cm, axis=0)
        cm = cm / (col_sum[None, :] + hc_sinkhorn_eps)

    out_off = comb_mix_ptr + pid * hc * hc + rows * hc + cols
    tl.store(out_off, cm)


@triton.jit
def _mhc_layer_input_kernel(
    residual_ptr,
    pre_mix_ptr,
    out_ptr,
    n_tokens,
    hc: tl.constexpr,
    h: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_hb = tl.program_id(1).to(tl.int64)

    h_off = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_off < h

    pre = tl.load(pre_mix_ptr + pid_t * hc + tl.arange(0, hc)).to(tl.float32)
    res_off = (
        residual_ptr + pid_t * hc * h + tl.arange(0, hc)[:, None] * h + h_off[None, :]
    )
    res = tl.load(res_off, mask=h_mask[None, :], other=0.0).to(tl.float32)
    out = tl.sum(pre[:, None] * res, axis=0).to(tl.bfloat16)

    tl.store(out_ptr + pid_t * h + h_off, out, mask=h_mask)


@triton.jit
def _mhc_post_per_token(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    out_ptr,
    n_tokens,
    hc: tl.constexpr,
    h: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_hb = tl.program_id(1).to(tl.int64)

    h_off = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_off < h

    a_off = (
        a_ptr
        + pid_t * hc * hc
        + tl.arange(0, hc)[:, None] * hc
        + tl.arange(0, hc)[None, :]
    )
    a = tl.load(a_off).to(tl.float32)
    c = tl.load(c_ptr + pid_t * hc + tl.arange(0, hc)).to(tl.float32)
    d = tl.load(d_ptr + pid_t * h + h_off, mask=h_mask, other=0.0).to(tl.float32)

    b_off = b_ptr + pid_t * hc * h + tl.arange(0, hc)[:, None] * h + h_off[None, :]
    b = tl.load(b_off, mask=h_mask[None, :], other=0.0).to(tl.float32)

    a_t = tl.trans(a)
    mixed = tl.sum(a_t[:, :, None] * b[None, :, :], axis=1)
    result = (mixed + c[:, None] * d[None, :]).to(tl.bfloat16)

    out_off = out_ptr + pid_t * hc * h + tl.arange(0, hc)[:, None] * h + h_off[None, :]
    tl.store(out_off, result, mask=h_mask[None, :])


def mhc_pre_triton(
    mixes: torch.Tensor,
    residual_flat: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_iters: int,
    hc: int,
    h: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_tokens, hc_mult3 = mixes.shape
    assert hc_mult3 == hc * 2 + hc * hc
    assert residual_flat.is_contiguous()
    assert mixes.is_contiguous()

    pre_mix = torch.empty(n_tokens, hc, dtype=torch.float32, device=mixes.device)
    post_mix = torch.empty(n_tokens, hc, dtype=torch.float32, device=mixes.device)
    comb_mix = torch.empty(n_tokens, hc, hc, dtype=torch.float32, device=mixes.device)

    _mhc_pre_fused_kernel[(n_tokens,)](
        mixes,
        residual_flat,
        hc_scale,
        hc_base,
        pre_mix,
        post_mix,
        comb_mix,
        n_tokens=n_tokens,
        hc=hc,
        h=h,
        hc_mult3=hc_mult3,
        BLOCK_H=1024,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_iters=sinkhorn_iters,
        num_warps=4,
    )

    layer_input = torch.empty(n_tokens, h, dtype=torch.bfloat16, device=mixes.device)
    block_h = 256 if h >= 256 else 128
    grid = (n_tokens, triton.cdiv(h, block_h))
    _mhc_layer_input_kernel[grid](
        residual_flat,
        pre_mix,
        layer_input,
        n_tokens=n_tokens,
        hc=hc,
        h=h,
        BLOCK_H=block_h,
        num_warps=4,
    )

    return post_mix, comb_mix, layer_input


def mhc_post_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    outer_shape = residual.shape[:-2]
    hc = residual.shape[-2]
    h = residual.shape[-1]
    n_tokens = residual.numel() // (hc * h)

    res = residual.reshape(n_tokens, hc, h).contiguous()
    comb = comb_res_mix.reshape(n_tokens, hc, hc).to(torch.float32).contiguous()
    post = post_layer_mix.reshape(n_tokens, hc).to(torch.float32).contiguous()
    x_flat = x.reshape(n_tokens, h).contiguous()

    out = torch.empty_like(res)
    block_h = 256 if h >= 256 else 128
    grid = (n_tokens, triton.cdiv(h, block_h))
    _mhc_post_per_token[grid](
        comb,
        res,
        post,
        x_flat,
        out,
        n_tokens,
        hc=hc,
        h=h,
        BLOCK_H=block_h,
    )
    return out.view(*outer_shape, hc, h)
