# UnsafeCloneEliminationPass Debugging Findings

## Summary

`UnsafeCloneEliminationPass` (registered in `vllm/compilation/passes/pass_manager.py:186`) is the root cause of a `torch.compile` garbage-output bug on mixed SM80+SM90 GPU configurations (2× A100 + 2× H100). Disabling this pass fixes the bug while keeping all other compilation pipeline changes intact.

## Hardware

- 4 GPUs, TP=4, EP enabled
- GPU 0,1: NVIDIA A100 80GB PCIe (SM80)
- GPU 2,3: NVIDIA H100 PCIe (SM90)
- CUDA 12.8, Driver 570.133.07
- Model: DeepSeek V4 FP8 (`dsv4/`)

## Symptoms

| Mode | Output | `finish_reason` |
|------|--------|-----------------|
| `torch.compile` with `UnsafeCloneEliminationPass` | Garbage (e.g. `"mountain historical society indiana\n\n### Born..."`) | `length` |
| `torch.compile` without `UnsafeCloneEliminationPass` | Correct (e.g. `"Hello! How can I help you today?"`) | `stop` |
| `--enforce-eager` | Correct | `stop` |

## The Pass

**File**: `vllm/compilation/passes/ir/clone_elimination.py`

```python
class UnsafeCloneEliminationPass(VllmInductorPass):
```

Removes `torch.ops.aten.clone.default` nodes from the FX graph after vLLM IR lowering. The pass is explicitly documented as **unsafe** — "it does not (yet) take aliasing into account." It only supports known vLLM patterns and makes no guarantee of soundness on general graphs.

### Removal Logic

For each `clone` node in the graph:
1. If the clone **is written to** (by an in-place op) and the original is used after the write → **preserve** the clone (safety check)
2. If the original is a **non-donated graph input** → **preserve** the clone (safety check)
3. Otherwise → **remove** the clone, replacing all uses with the original

### The Bug

On SM80, reference kernels produce different IR lowering patterns than SM90. Specifically:
- SM80 uses reference kernels (not TileLang JIT kernels) for certain operations
- These reference kernels may involve different mutation patterns or tensor aliasing
- `UnsafeCloneEliminationPass` removes clone nodes that are actually needed for correctness on SM80, because the alias-analysis heuristics in the pass don't account for the IR patterns produced by SM80-specific kernel selection

The result is that cloned tensors share storage with their originals when they shouldn't, causing downstream ops to read corrupted data after in-place mutations.

## Bisect Results

Three compilation pipeline changes differentiate vllm-main from vllm-git:

| # | Change | File | Description |
|---|--------|------|-------------|
| A | `VllmIRInplaceFunctionalizationPass` | `backends.py:934` | Pre-grad in-place functionalization pass |
| B | **`UnsafeCloneEliminationPass`** | **`pass_manager.py:186`** | **Post-grad clone elimination (ROOT CAUSE)** |
| C | `func_impl_fn` + `run_functional_passes=False` | `lowering_pass.py:68` | IR lowering: clones activations for in-place impls, skips DCE/functional cleanup |

Combinations tested (each with cache cleared between runs):

| A | B | C | Result |
|---|---|---|---|
| OFF | OFF | OFF | ✅ Correct |
| ON | ON | OFF | ❌ Garbage |
| OFF | ON | OFF | ❌ Garbage |
| ON | **OFF** | OFF | ✅ Correct |
| ON | **OFF** | ON | ✅ **Correct — minimal fix** |

Only B (`UnsafeCloneEliminationPass`) needs to be disabled. A and C are safe to leave at their original vllm-main defaults.

## Root Cause Analysis

The pass is called "unsafe" for good reason. It removes clone nodes based on a heuristic that doesn't fully model aliasing:

1. **Aliasing not tracked**: The pass only checks `donated_input_ids` and write-ordering. It doesn't track whether tensors alias each other through views, slices, or transposes. On SM80, reference kernels may produce such aliased tensors where clones are semantically required.

2. **SM80-specific IR patterns**: The kernel dispatch (`use_dsv4_reference_kernels`) selects different implementations on SM80 vs SM90. These implementations lower to different FX graph patterns, specifically around tensor mutation and clone placement. The pass's heuristics were likely tuned against SM90 patterns and don't generalize to SM80.

3. **The `func_impl_fn` interaction**: The IR lowering (Change C) uses `func_impl_fn` which clones activation tensors for in-place implementations to ensure functional semantics. `UnsafeCloneEliminationPass` then removes some of these clones. On SM80, this removal is incorrect because the SM80 reference kernels have different aliasing requirements that the clone was protecting.

## Fix Applied

```python
# vllm/compilation/passes/pass_manager.py:187-189
# DISABLED: UnsafeCloneEliminationPass causes incorrect torch.compile
# output on mixed SM80+SM90. Root cause of SM80 garbage output bug.
self.clone_elimination = None  # UnsafeCloneEliminationPass(config)
```

All other passes (including `VllmIRInplaceFunctionalizationPass` and the IR lowering change) remain at their original vllm-main defaults. No other files were modified.

## open Questions

1. **Does the bug also reproduce on pure SM80 (4× A100)?** Not tested. The mixed SM80+SM90 configuration with EP may exacerbate the issue.
2. **What specific IR pattern causes the incorrect clone removal?** Requires dumping the FX graph before/after the pass on SM80 to identify the specific node.
3. **Can the pass be fixed instead of disabled?** The pass needs proper aliasing support (tracking views, slices, etc.) to be sound on all kernel patterns. The `TODO(luka)` in the source notes this as an open problem.
4. **Performance impact of disabling?** Without clone elimination, compiled graphs may have extra `clone()` calls, potentially increasing memory usage or runtime. This has not been benchmarked.
