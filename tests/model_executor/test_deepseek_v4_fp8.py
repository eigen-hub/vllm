import torch
import numpy as np
from vllm import _custom_ops as ops
from vllm.v1.attention.ops.deepseek_v4_ops import gather_dequant_two_scopes_with_mask
from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max

def test_fp8_roundtrip():
    device = "cuda"
    torch.set_default_device(device)
    
    # 1. Setup deterministic input
    # Include normals, subnormals, and edge cases
    # FP8 E4M3FN max is 448.0.
    # Smallest normal is 2^-6 * 1.0 = 0.015625
    # Smallest subnormal is 2^-6 * (1/8) = 2^-9 = 0.001953125
    input_vals = [
        0.0, -0.0,
        0.001, 0.002, 0.005,  # Subnormals
        0.015, 0.02, 0.1, 1.0, 10.0, 100.0, 440.0, 448.0, 450.0, # Normals and clamping
        -1.0, -448.0,
    ]
    head_dim = 512
    nope_dim = 448
    rope_dim = 64
    num_tokens = len(input_vals)
    
    # Fill a tensor with these values repeated
    k = torch.zeros((num_tokens, head_dim), dtype=torch.bfloat16)
    for i, val in enumerate(input_vals):
        k[i, :nope_dim] = val
        k[i, nope_dim:] = val # RoPE part is just bf16, but we test it anyway
        
    # 2. Encode using C++ indexer_k_quant_and_cache
    block_size = 64
    num_blocks = (num_tokens + block_size - 1) // block_size
    # KV cache layout: [num_blocks, block_size, 584] for DeepseekV4 SWA
    # 584 = 448 (NoPE FP8) + 128 (RoPE BF16) + 8 (Scales)
    kv_cache = torch.zeros((num_blocks, block_size, 584), dtype=torch.uint8)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int32)
    
    # We need to use the actual C++ kernel
    # Note: indexer_k_quant_and_cache in vLLM main handles the quantization
    # into the paged cache format.
    ops.indexer_k_quant_and_cache(
        k,
        kv_cache,
        slot_mapping,
        64, # quant_block_size
        "fp8",
    )
    
    # 3. Decode using Triton gather_dequant_two_scopes_with_mask
    # We need to simulate the metadata for the Triton kernel
    swa_indices = torch.arange(num_tokens, dtype=torch.int32).view(num_tokens, 1)
    swa_topk_length = torch.ones(num_tokens, dtype=torch.int32)
    
    # gather_dequant_two_scopes_with_mask(
    #     swa_kv_cache, swa_block_size, swa_indices, swa_topk_length,
    #     extra_kv_cache, extra_block_size, extra_indices, extra_topk_length,
    #     nope_dim, rope_dim, head_dim
    # )
    gathered_kv, invalid_mask = gather_dequant_two_scopes_with_mask(
        swa_kv_cache=kv_cache,
        swa_block_size=block_size,
        swa_indices=swa_indices,
        swa_topk_length=swa_topk_length,
        extra_kv_cache=None,
        extra_block_size=0,
        extra_indices=None,
        extra_topk_length=None,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        head_dim=head_dim,
    )
    
    # 4. Compare
    # The output should be BF16
    # Note: FP8 quantization is lossy, so we compare against the theoretical
    # round-trip value or just check for large discrepancies.
    # Actually, we should check if they match bit-for-bit with what we expect
    # from a software-only implementation.
    
    print(f"{'Input':>10} | {'Output':>10} | {'Diff':>10}")
    print("-" * 35)
    for i, val in enumerate(input_vals):
        out_val = gathered_kv[i, 0, 0].item()
        print(f"{val:10.4f} | {out_val:10.4f} | {abs(val - out_val):10.4f}")

    # Check for subnormal handling specifically
    # mant=1 -> 2^-9 = 0.001953125
    # Let's see if 0.002 rounds to 0.001953125 or 0.0
    
if __name__ == "__main__":
    if torch.cuda.is_available():
        test_fp8_roundtrip()
    else:
        print("CUDA not available")
