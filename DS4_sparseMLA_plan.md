# Achievement of SM80 Parity for DeepSeek V4 Sparse MLA

This plan addresses the numerical regressions and architectural gaps in the `vllm-main` branch for DeepSeek V4 Sparse MLA on SM80 (A100) hardware. We will reconcile the orchestration logic with the `vllm-triton` reference branch and ensure cluster-wide synchronization of kernel dispatch decisions.

## User Review Required

> [!IMPORTANT]
> This change introduces a cluster-wide barrier (`sync_dsv4_reference_kernels`) during model initialization to ensure all TP ranks agree on whether to use reference Triton kernels. This is necessary for heterogeneous clusters (e.g., mix of H100 and A100) where local capability checks would otherwise cause rank divergence.

> [!TIP]
> Setting `DSV4_GATHER_VERIFY=1` in the environment will enable side-by-side verification of the fused gathering kernel against a PyTorch reference during decode.

## Proposed Changes

### [Component] Model Execution & Synchronization

#### [MODIFY] [deepseek_v4.py](file:///c:/Users/liuex/projects/vllm-main/vllm/model_executor/models/deepseek_v4.py)
- Import `sync_dsv4_reference_kernels` from `vllm.utils.deep_gemm`.
- Call `sync_dsv4_reference_kernels()` in `DeepseekV4Model.__init__` to broadcast the dispatch decision to all TP ranks.

---

### [Component] Attention Layer Orchestration

#### [MODIFY] [deepseek_v4_attention.py](file:///c:/Users/liuex/projects/vllm-main/vllm/model_executor/layers/deepseek_v4_attention.py)
- **Initialization**: Add `self._arange_cache` to `DeepseekV4MLAAttention.__init__` to optimize mask generation.
- **New Methods**:
    - Port `_get_arange` for cached index generation.
    - Port `_gather_dequant_blocked_k_at_indices_pytorch` (reference gather for verification).
    - Port `_verify_gather` to perform bitwise comparison when `DSV4_GATHER_VERIFY=1`.
    - Port `_ref_sparse_attn_decode_gather` to centralize the SM80 decode gathering and verification pipeline.
    - Port `_ref_sparse_attn_prefill` to provide a robust PyTorch-based fallback for sparse attention on platforms without FlashMLA.
- **Orchestration Update**:
    - Update `_forward_decode` to delegate to `_ref_sparse_attn_decode_gather` when `self._use_reference_kernels` is True.
    - Update `_forward_prefill` to wrap the Triton/FlashMLA calls in a `try-except` block, falling back to `_ref_sparse_attn_prefill` on failure (critical for SM80 stability).

## Verification Plan

### Automated Tests
- Run Sparse MLA regression tests on SM80:
  ```bash
  .venv/bin/python -m pytest tests/v1/attention/test_deepseek_v4.py -v
  ```
- Run with verification enabled:
  ```bash
  DSV4_GATHER_VERIFY=1 .venv/bin/python -m pytest tests/v1/attention/test_deepseek_v4.py -v
  ```

### Manual Verification
- Deploy on a heterogeneous cluster (H100 + A100) and verify that all ranks arrive at the same `USE_DSV4_REF_KERNELS` flag value and generate identical outputs.
