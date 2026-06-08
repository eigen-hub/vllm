# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

# --- Monkey-patch CUTLASS DSL GPU arch detection ------------------------------
#
# CUTLASS DSL's `detect_gpu_arch()` calls
# `get_compute_capability_major_minor(device_id=0)`, hardcoding physical GPU 0
# on every worker process.  On heterogeneous nodes (e.g. A100 + H100) this
# causes ALL workers to compile CUTLASS DSL kernels for GPU 0's architecture,
# making the JIT execution engine (gen_jit_engine) disagree with the per-kernel
# `--gpu-arch` compile option, which produces a segfault in TVMFFIFunctionCall.
#
# We patch the runtime function to use the current worker's assigned GPU
# (torch.cuda.current_device()), which is correct because gpu_worker.py calls
# torch.accelerator.set_device_index() before model construction.
#
# The patch must be applied BEFORE any code that creates a CUTLASS DSL module
# calls `cute.compile(kernel, ...)`.  Importing `cutlass.base_dsl.runtime.cuda`
# is safe: it does not trigger DSL EnvironmentVarManager initialisation.
# ------------------------------------------------------------------------------

if torch.cuda.is_available():
    # Patch detect_gpu_arch to use the current worker's assigned GPU
    # instead of hardcoded device_id=0.
    #
    # IMPORTANT: importing cutlass.base_dsl.env_manager triggers the full
    # cutlass package init chain (cutlass/__init__.py:63 ->
    # cutlass.cute -> cute/__init__.py:225 -> CuTeDSL._get_dsl()),
    # which eagerly creates the CuTeDSL singleton with envar.arch set
    # via detect_gpu_arch("CUTE_DSL") BEFORE this patch runs.
    # The singleton's envar.arch therefore reflects device 0's GPU
    # (A100 -> sm_80), which mismatches the per-kernel --gpu-arch
    # option on H100 workers and disables the JIT execution engine.
    #
    # We cannot fix envar.arch eagerly at import time because the worker
    # process has not yet called torch.accelerator.set_device_index(),
    # so torch.cuda.current_device() still returns device 0 (A100) on
    # all workers.  Instead we wrap cute.compile below to correct the
    # singleton's envar.arch lazily at call time, by which point the
    # worker has bound itself to the correct GPU.
    import cutlass.base_dsl.env_manager as _env_mgr

    def _patched_detect_gpu_arch(prefix: str) -> str:
        cur_dev = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(cur_dev)
        suffix = "a" if major >= 9 else ""
        return f"sm_{major}{minor}{suffix}"

    _env_mgr.detect_gpu_arch = _patched_detect_gpu_arch

    # Clear LRU caches so subsequent env lookups use the patched default.
    _env_mgr.get_str_env_var.cache_clear()

from cutlass import BFloat16, Float32, Int64, Uint32, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.nvgpu import cpasync

# LoadCacheMode lives in cpasync but is not re-exported from cutlass.cute.nvgpu.
# dequant_gather_k_cutedsl.py:117 references cute.nvgpu.LoadCacheMode.
import cutlass.cute.nvgpu as _cute_nvgpu

_cute_nvgpu.LoadCacheMode = cpasync.LoadCacheMode

from cutlass.cutlass_dsl import T, dsl_user_op

# ---------------------------------------------------------------------------
# Wrap cute.compile to correct the singleton's envar.arch at call time
# (after the worker has bound itself to the correct GPU).
#
# Imports inside the if-guard above are resolved by the time we reach here,
# so _patched_detect_gpu_arch is always defined when CUDA is available.
# The no-CUDA branch is unreachable for vLLM, but we keep the wrapper
# defined for safety.
# ---------------------------------------------------------------------------

_GS_CORRECTED_SINGLETON: set = set()


def _ensure_correct_dsl_arch():
    if not torch.cuda.is_available():
        return
    from cutlass.base_dsl.dsl import DSLSingletonMeta as _DSLSingletonMeta

    for _cls, _inst in _DSLSingletonMeta._instances.items():
        if _cls in _GS_CORRECTED_SINGLETON:
            continue
        _c = _patched_detect_gpu_arch("CUTE_DSL")
        _inst.envar.arch = _c
        _inst.envar.enable_tvm_ffi = True
        _GS_CORRECTED_SINGLETON.add(_cls)


_orig_compile = cute.compile


def _patched_compile(*args, **kwargs):
    _ensure_correct_dsl_arch()
    return _orig_compile(*args, **kwargs)


cute.compile = _patched_compile


def cute_arch_from_device(device: torch.device | int | None = None) -> str:
    if not torch.cuda.is_available():
        return ""

    if isinstance(device, int):
        device_id = device
    else:
        device_id = torch.device(device).index if device is not None else None
        if device_id is None:
            device_id = torch.cuda.current_device()

    major, minor = torch.cuda.get_device_capability(device_id)
    # Add the "a" suffix for SM90+ to match CUTLASS DSL's detect_gpu_arch
    # convention, which enables architecture-specific features (TMA, WGMMA).
    # Even kernels that do not directly use TMA may still rely on the
    # compiler pass pipeline changes that the "a" suffix activates.
    suffix = "a" if major >= 9 else ""
    return f"sm_{major}{minor}{suffix}"


def cute_compile_options(*, gpu_arch: str = "", enable_tvm_ffi: bool = True) -> str:
    options: list[str] = []
    if enable_tvm_ffi:
        options.append("--enable-tvm-ffi")
    if gpu_arch:
        options.extend(["--gpu-arch", gpu_arch])
    return " ".join(options)


# https://github.com/NVIDIA/cutlass/blob/v4.3.2/include/cute/arch/copy_sm90_desc.hpp#L193-L197
EVICT_NORMAL = Int64(0x1000000000000000)
EVICT_FIRST = Int64(0x12F0000000000000)
EVICT_LAST = Int64(0x14F0000000000000)


@dsl_user_op
def recast_val(x, dtype, *, loc=None, ip=None):
    return dtype(llvm.bitcast(dtype.mlir_type, x.ir_value(loc=loc, ip=ip)))


def simple_tma_copy(atom, src, dst, mbar=None, cache_policy=None):
    """A simple helper that wraps group_modes() and tma_partition()
    NOTE: this should be called WITHOUT cute.elect_one()
    """
    if isinstance(atom.op, cpasync.CopyBulkTensorTileG2SOp):
        gmem = src
        smem = dst
    elif isinstance(atom.op, cpasync.CopyBulkTensorTileS2GOp):
        smem = src
        gmem = dst
    else:
        raise ValueError

    s_part, g_part = cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(smem, 0),
        cute.group_modes(gmem, 0),
    )

    if isinstance(atom.op, cpasync.CopyBulkTensorTileG2SOp):
        cute.copy(atom, g_part, s_part, tma_bar_ptr=mbar, cache_policy=cache_policy)
    elif isinstance(atom.op, cpasync.CopyBulkTensorTileS2GOp):
        cute.copy(atom, s_part, g_part, cache_policy=cache_policy)
    else:
        raise ValueError


# can't find the equivalent in nvvm
@dsl_user_op
def fence_before_tma_store(*, loc=None, ip=None):
    llvm.inline_asm(
        T.i32(),
        [],
        "mov.u32 $0, 0;\n\t"
        "fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_bf16(
    a: cute.TensorSSA, b: cute.TensorSSA, c: cute.TensorSSA, *, loc=None, ip=None
):
    if a.element_type == BFloat16:
        a = cute.recast_tensor(a, Uint32)
    if b.element_type == BFloat16:
        b = cute.recast_tensor(b, Uint32)

    mlir_ty = Float32.mlir_type
    out = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_ty] * 4),
        [a[i].ir_value(loc=loc, ip=ip) for i in range(4)]
        + [b[i].ir_value(loc=loc, ip=ip) for i in range(2)]
        + [c[i].ir_value(loc=loc, ip=ip) for i in range(4)],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, "
        "{$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([4], mlir_ty, loc=loc),
        [llvm.extractvalue(mlir_ty, out, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 4, Float32)


@dsl_user_op
def _bf16x2_abs(a: Uint32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip)],
        "abs.bf16x2 $0, $1;",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint32(out)


@dsl_user_op
def _bf16x2_max(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "max.bf16x2 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint32(out)


@dsl_user_op
def _bf16x2_mul(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "mul.rn.bf16x2 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint32(out)


# ---------------------------------------------------------------------------
# Monkey-patch JIT-time _Pointer (cutlass.cute.core._Pointer) to support
# subscript load/store via ptr[index] / ptr[index] = value.
#
# The runtime _Pointer (cutlass.cute.runtime._Pointer) already supports this
# via the __cuda_array_interface__ protocol, but the JIT-time _Pointer used
# during MLIR IR generation (core._Pointer) lacks __getitem__/__setitem__.
#
# Several CUTLASS DSL kernels in vLLM (e.g. sparse_attn_compress_cutedsl.py)
# use tensor.iterator[flat_idx] to bypass the tensor's layout and directly
# access the underlying flat buffer.  Without this patch those kernels fail
# during compilation with:
#   TypeError: '_Pointer' object is not subscriptable
#
# The patch creates a 1-element tensor view at (pointer + index) and loads
# or stores through it.  MLIR optimisation passes collapse the redundant ops.
# ---------------------------------------------------------------------------

from cutlass.cute.core import _Pointer as _CorePointer
from cutlass.cute.tensor import make_tensor
from cutlass.cute.core import make_layout


@dsl_user_op
def _core_pointer_getitem(self, index, *, loc=None, ip=None):
    offset_ptr = self + index
    view = make_tensor(offset_ptr, make_layout((1,), loc=loc, ip=ip), loc=loc, ip=ip)
    return view.__getitem__(0, loc=loc, ip=ip)


@dsl_user_op
def _core_pointer_setitem(self, index, value, *, loc=None, ip=None):
    offset_ptr = self + index
    view = make_tensor(offset_ptr, make_layout((1,), loc=loc, ip=ip), loc=loc, ip=ip)
    view.__setitem__(0, value, loc=loc, ip=ip)


_CorePointer.__getitem__ = _core_pointer_getitem
_CorePointer.__setitem__ = _core_pointer_setitem

# Add absf to cutlass.cute.math
import cutlass.cute.math as _cute_math


@dsl_user_op
def _absf(x: Float32, *, loc=None, ip=None) -> Float32:
    out = llvm.inline_asm(
        T.f32(),
        [x.ir_value(loc=loc, ip=ip)],
        "abs.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(out)


_cute_math.absf = _absf


# ---------------------------------------------------------------------------
# Add fmin (float32 minimum) to cutlass.cute.arch.
#
# cutlass.cute.arch provides fmax but not fmin.  The sparse_attn_compress
# kernel uses fmin for FP8 quantisation clipping.  We add it via inline PTX
# "min.f32".
# ---------------------------------------------------------------------------

import cutlass.cute.arch as _cute_arch


@dsl_user_op
def _fmin(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    out = llvm.inline_asm(
        T.f32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "min.f32 $0, $1, $2;",
        "=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(out)


_cute_arch.fmin = _fmin
