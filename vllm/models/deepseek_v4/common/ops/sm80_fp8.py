# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SM80 software FP8 E4M3 encode/decode.

``tl.float8e4nv`` cannot be compiled on SM80 GPUs. These kernels use bit
manipulation to encode/decode the E4M3 format in software.

Exports:
- ``_encode_e4m3fn_sw``: Triton JIT function, encode fp32 -> E4M3 uint8
- ``_decode_e4m3fn``: from ``vllm.v1.attention.ops.mqa_logits_triton``
"""

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.mqa_logits_triton import _decode_e4m3fn


@triton.jit
def _encode_e4m3fn_sw(x: tl.tensor):
    """Software FP8 E4M3 encoder for SM80.

    Implements the same rounding as torch.float8_e4m3fn using bit manipulation.
    Handles normal, subnormal, zero, and NaN inputs.
    """
    bits = x.to(tl.uint32, bitcast=True)
    sign = (bits >> 31) & 1
    abs_bits = bits & 0x7FFFFFFF
    exp_fp32 = (abs_bits >> 23).to(tl.int32)
    mant_fp32 = abs_bits & 0x7FFFFF
    is_zero = abs_bits == 0
    is_inf_or_nan = exp_fp32 == 0xFF
    is_nan = is_inf_or_nan & (mant_fp32 != 0)
    exp_fp8 = exp_fp32 - 120
    mant_extracted = mant_fp32 >> 20
    round_bit = (mant_fp32 >> 19) & 1
    sticky = (mant_fp32 & 0x7FFFF) != 0
    odd = (mant_extracted & 1) == 1
    round_up = round_bit & (sticky.to(tl.uint32) | odd.to(tl.uint32))
    mant_rounded = mant_extracted + round_up
    carry = mant_rounded == 8
    exp_after = exp_fp8 + carry.to(tl.int32)
    mant_after = tl.where(carry, 0, mant_rounded)
    packed_normal = ((exp_after.to(tl.uint32) & 0xF) << 3) | (mant_after & 0x7)
    impl_mant = (tl.full((), 1, tl.uint32) << 23) | mant_fp32
    sub_shift = (141 - exp_fp32).to(tl.uint32)
    safe_shift = tl.minimum(sub_shift, 31)
    sub_m_int = impl_mant >> safe_shift
    sub_round_bit = tl.where(
        safe_shift >= 1,
        (impl_mant >> (safe_shift - 1)) & 1,
        tl.zeros_like(impl_mant),
    )
    sticky_mask = tl.where(
        safe_shift >= 2,
        (tl.full((), 1, tl.uint32) << (safe_shift - 1)) - 1,
        tl.zeros_like(impl_mant),
    )
    sub_sticky = (impl_mant & sticky_mask) != 0
    sub_odd = (sub_m_int & 1) == 1
    sub_round_up = sub_round_bit & (sub_sticky.to(tl.uint32) | sub_odd.to(tl.uint32))
    sub_m_rounded = sub_m_int + sub_round_up
    sub_promotes = sub_m_rounded == 8
    sub_packed = tl.where(
        sub_promotes, tl.full((), 0x08, tl.uint32), sub_m_rounded & 0x7
    )
    over_max_finite = (exp_after >= 16) | ((exp_after == 15) & (mant_after == 7))
    packed_normal = tl.where(over_max_finite, 0x7E, packed_normal)
    is_subnormal = exp_fp8 <= 0
    encoded = tl.where(is_subnormal, sub_packed, packed_normal)
    encoded = tl.where(is_zero, tl.zeros_like(encoded), encoded)
    encoded = tl.where(is_nan, tl.full((), 0x7F, tl.uint32), encoded)
    encoded = encoded | (sign << 7)
    return encoded.to(tl.uint8)
