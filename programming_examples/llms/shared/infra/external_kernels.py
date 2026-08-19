# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""External C++ kernel compilation utilities.

Compiles all external .o files from source to avoid relying on stale
pre-compiled artifacts. Each function checks if the .o exists and skips
recompilation if so (delete the .o to force recompile).

Compiled .o files are placed in CWD (build_peano/) where aiecc finds them
via its link_with search path.
"""

import os
import shutil
import subprocess
from pathlib import Path


def _get_peano_clang():
    """Find the Peano clang++ compiler."""
    peano_dir = os.environ.get("PEANO_INSTALL_DIR", "")
    if peano_dir:
        return os.path.join(peano_dir, "bin", "clang++")
    raise RuntimeError("PEANO_INSTALL_DIR not set")


def _get_aie_include_dir():
    """Find the AIE API include directory (for aie_api/aie.hpp)."""
    # Primary: locate via aie-opt on PATH. Matches the convention used by
    # every other Makefile in this repo (AIEOPT_DIR = $(dir $(which aie-opt))/..)
    # and works for both local source builds and CI's mlir_aie wheel install.
    aie_opt = shutil.which("aie-opt")
    if aie_opt:
        p = Path(aie_opt).resolve().parent.parent / "include"
        if (p / "aie_api" / "aie.hpp").exists():
            return str(p)
    # Explicit override: MLIR_AIE_INSTALL_DIR env var (useful in git worktrees
    # where the local-dev relative path below resolves to the worktree root
    # rather than the main repo root).
    mlir_aie_dir = os.environ.get("MLIR_AIE_INSTALL_DIR", "")
    if mlir_aie_dir:
        p = Path(mlir_aie_dir) / "include"
        if (p / "aie_api" / "aie.hpp").exists():
            return str(p)
    # Fallback: explicit local dev install path.
    p = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "my_install"
        / "mlir-aie"
        / "install"
        / "include"
    )
    if (p / "aie_api" / "aie.hpp").exists():
        return str(p)
    raise RuntimeError(
        "Cannot find aie_api/aie.hpp include directory "
        "(no aie-opt on PATH, no MLIR_AIE_INSTALL_DIR, no my_install/mlir-aie/install)"
    )


_PEANO_FLAGS = [
    "-O2",
    "-std=c++20",
    "--target=aie2p-none-unknown-elf",
    "-DNDEBUG",
    # Short-circuit aie_api's ADF graph headers: aie.hpp -> aie_adf.hpp (guarded
    # by __AIE_API_AIE_ADF_HPP__) -> adf/stream.hpp -> #include <adf.h>. adf.h is
    # a Vitis-only header absent from the Peano include path, so without this
    # guard the compile fails with "'adf.h' file not found". These compute
    # kernels don't use the ADF stream API; the XRT kernel tests pass the same
    # define.
    "-D__AIE_API_AIE_ADF_HPP__",
    "-Wno-parentheses",
    "-Wno-attributes",
    "-Wno-macro-redefined",
    "-Wno-empty-body",
]


def _compile_kernel(src_path, output_name, extra_flags=None, force=False):
    """Compile a C++ kernel to .o using Peano clang++.

    Args:
        src_path: Path to the .cc source file
        output_name: Name of the output .o file (placed in CWD)
        extra_flags: Additional compiler flags (e.g., -D defines)
        force: If True, recompile even if .o exists
    """
    if not force and Path(output_name).exists():
        return

    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"Kernel source not found: {src}")

    clang = _get_peano_clang()
    include_dir = _get_aie_include_dir()

    cmd = [clang] + _PEANO_FLAGS + [f"-I{include_dir}"]
    if extra_flags:
        cmd.extend(extra_flags)
    cmd.extend(["-c", str(src), "-o", output_name])

    print(f"  Compiling {output_name} from {src.name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Filter warnings, only show errors
        errors = [l for l in result.stderr.split("\n") if "error" in l.lower()]
        raise RuntimeError(f"Failed to compile {output_name}: {' '.join(errors[:3])}")


# ---------------------------------------------------------------------------
# Individual kernel compilation functions
# ---------------------------------------------------------------------------

_PROJ_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent
)  # programming_examples/


def compile_silu_and_mul():
    """Compile silu_and_mul.o from programming_examples/silu_and_mul/silu_and_mul.cc."""
    src = _PROJ_ROOT / "silu_and_mul" / "silu_and_mul.cc"
    include_dir = _get_aie_include_dir()
    utils_header = Path(include_dir) / "aie_kernels" / "aie_kernel_utils.h"
    extra = []
    if utils_header.exists():
        extra = [f"-include", str(utils_header)]
    _compile_kernel(src, "silu_and_mul.o", extra_flags=extra)


def compile_gemm_mm(
    tile_m=64,
    tile_n=128,
    tile_k_l1=32,
    sym_suffix="",
    out_name="mm.o",
    gen_init=False,
    gen_with_acc=False,
):
    """Compile mm.o from matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc.

    The hand-tuned Peano -O2 vectorized GEMM microkernel (external path), ~1.5-1.65x
    faster than direct-codegen on large shapes (kernel_registry/details/GEMM_bf16_in_fp32_out.md).
    DIM_M/DIM_N/DIM_K are baked in at compile time and MUST match the tile_m/tile_n/
    tile_k_l1 passed to the GEMM module builder. Exposes op_has_no_registered_library_name
    (f32-C matmul), zero_f32_mn, f32_to_bf16_mn.

    sym_suffix / out_name: to link SEVERAL mm.o variants into ONE ELF, the symbols
    must not collide -- and two GEMMs at the same tile_m but different tile_n are
    two different micro-kernels, because DIM_N is baked in here. PREFER
    `compile_gemm_mm_variant` below, which derives both names from (tile_m, tile_n)
    via the one authority for them, `gemm_builder.gemm_variant_names`
    (-> sym_suffix "_m64n128", out_name "mm_m64n128.o"). Spelling them by hand is
    how an object gets built at the wrong DIM_N and still links: see Phase D2 in
    `transformer_layer/README.md` for what that returns. Default empty suffix /
    "mm.o" keeps the original names for single-variant ELFs (back-compat).

    gen_init / gen_with_acc: emit the two opt-in kernel families used by the
    staged transformer-layer pipelines. Both default OFF, and with both off the
    object is byte-identical to what this function produced before they existed
    -- that is deliberate, because ten shipped LLM deployments link this source.

      gen_init      -> matmul_init_<in>_<out>(A, B, C)
                       Zero-then-multiply: C is overwritten, not accumulated
                       into, so it replaces the zero_* call before the first
                       K-tile. Calling it for a later K-tile silently discards
                       the partial sums.
      gen_with_acc  -> matmul_with_acc_<in>_<out>(A, B, Acc, C)
                       Partials read from an explicit Acc buffer, result written
                       to a distinct C. Acc and C must share the tile layout and
                       element type.

    Both respect sym_suffix.
    """
    src = _PROJ_ROOT / "matrix_multiplication" / "bf16_in_fp32_out" / "mm_aie2p.cc"
    extra = [
        "-DBIT_WIDTH=8",
        f"-DDIM_M={tile_m}",
        f"-DDIM_N={tile_n}",
        f"-DDIM_K={tile_k_l1}",
        f"-DDIM_N_DIV_4={tile_n // 4}",
        f"-DDIM_M_DIV_4={tile_m // 4}",
        f"-DDIM_N_DIV_8={tile_n // 8}",
        f"-DDIM_M_DIV_8={tile_m // 8}",
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
    ]
    if sym_suffix:
        extra.append(f"-DSYM_SUFFIX={sym_suffix}")
    if gen_init:
        extra.append("-DGENERATE_MATMUL_INIT_KERNELS")
    if gen_with_acc:
        extra.append("-DGENERATE_MATMUL_WITH_ACC_KERNELS")
    _compile_kernel(src, out_name, extra_flags=extra, force=True)


def compile_gemm_mm_variant(tile_m, tile_n, tile_k_l1=32, **kwargs):
    """`compile_gemm_mm` with the symbol suffix and object name DERIVED, not given.

    The names come from `shared.builders.gemm_builder.gemm_variant_names`, which
    is also where every GEMM module's `sym_suffix` / `link_with_name` come from.
    Call this rather than spelling `sym_suffix=` and `out_name=` yourself: the
    two must agree per (tile_m, tile_n), and when they do not, the failure is
    either an unresolved symbol at link time or -- worse, and what Phase D2
    actually measured -- an object built at the wrong `-DDIM_N` that links
    cleanly and returns zeros for part of every output tile.

    Any remaining keyword (`gen_init`, `gen_with_acc`, ...) is passed straight
    through.
    """
    from shared.builders.gemm_builder import gemm_variant_names

    sym_suffix, out_name = gemm_variant_names(tile_m, tile_n)
    compile_gemm_mm(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k_l1=tile_k_l1,
        sym_suffix=sym_suffix,
        out_name=out_name,
        **kwargs,
    )


def compile_rope():
    """Compile rope.o from programming_examples/rope_halfsplit/rope_halfsplit.cc.

    Uses rope_halfsplit.cc (half-split rotation matching HuggingFace Llama)
    instead of upstream rope.cc (interleaved rotation). Same function name
    (@rope) and signature, so no MLIR changes needed. The kernel lives in the
    standalone rope_halfsplit registry example; llama links the same source.
    """
    src = _PROJ_ROOT / "rope_halfsplit" / "rope_halfsplit.cc"
    _compile_kernel(src, "rope.o")


def compile_attn_npu2(
    head_dim=64, lkp=None, lqp_tile=None, force=False, causal_row_helpers=False
):
    """Compile attn_npu2.o (FlashAttention kernel) from source.

    The attn_npu2.cc defines are PER-TILE, not per-launch (see the canonical
    Makefile): ``lqp`` = tile_size_q (= lqp_launch / num_q_tiles), ``lkp`` =
    K/V chunk size per tile, ``dk``/``dv`` = the K/V dimension TILE (= lkp),
    and ``dk_full``/``dv_full`` = the full head_dim. The matmul microkernels
    are instantiated with these tile shapes, so they MUST match the L1 buffer
    shapes the Python builder emits or the kernel hangs (ERT_CMD_STATE_TIMEOUT).

    head_dim=64 (llama32_1b seq-first): lkp == head_dim, so the legacy
    "everything = head_dim" defaults are correct.

    head_dim=128 (head-first path): the kernel tiles dk/dv into dv_chunks=2
    slices of lkp=64, and tile_size_q=64 (lqp_launch=256 / num_q_tiles=4). So
    pass lkp=64, lqp_tile=64; dk_full/dv_full stay at head_dim (128).

    Args:
        head_dim: full head dimension (-> dk_full / dv_full).
        lkp: K/V chunk size per tile (= dk/dv tile). Defaults to head_dim
            (legacy hd==lkp behavior).
        lqp_tile: Q tile size (tile_size_q). Defaults to lkp.
        force: recompile even if attn_npu2.o exists (needed when the same CWD
            previously built a different-shaped .o, e.g. hd=64 then hd=128).
        causal_row_helpers: also emit copy_O_tile_rows / store_row_value /
            copy_row_values. Off by default. copy_O_tile_rows is numerically a
            no-op by design -- it exists so a KV block entirely above the causal
            diagonal, which runs no matmul, still completes the O tile's DMA
            write. Without it that design hangs on ERT_CMD_STATE_TIMEOUT.
    """
    if lkp is None:
        lkp = head_dim
    if lqp_tile is None:
        lqp_tile = lkp
    src = _PROJ_ROOT / "flash_attention" / "kernel_fusion_based" / "attn_npu2.cc"
    extra = [
        "-DBIT_WIDTH=8",
        f"-Dlqp={lqp_tile}",
        f"-Dlkp={lkp}",
        f"-Ddk={lkp}",
        f"-Ddk_full={head_dim}",
        f"-Ddv={lkp}",
        f"-Ddv_full={head_dim}",
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        "-DROUND_CONV_EVEN",
    ]
    if causal_row_helpers:
        extra.append("-DCAUSAL_ROW_HELPERS")
    _compile_kernel(src, "attn_npu2.o", extra_flags=extra, force=force)
    # Also create attn.o copy (some link_with attributes use "attn.o").
    # Refresh whenever attn_npu2.o exists so a force-rebuild (different tile
    # shape) doesn't leave a stale attn.o behind.
    if Path("attn_npu2.o").exists():
        shutil.copy2("attn_npu2.o", "attn.o")


def compile_mv(tile_m=8):
    """Compile mv.o (standard GEMV kernel) from source."""
    src = _PROJ_ROOT / "matrix_vector_multiplication" / "bf16" / "mv.cc"
    _compile_kernel(src, "mv.o", extra_flags=[f"-DDIM_M_OUTPUT={tile_m}"])


def compile_mv_int4_bf16(m_tile=8, k_chunk=2048, gs=128):
    """Compile mv_int4_bf16.o (int4-AWQ GEMV micro-kernel) from source.

    Produces a config-tagged object and stages it as the canonical
    `mv_int4_bf16.o` (the name link_with attributes expect). The int4 GEMM
    prefill compiles the same .cc with DIM_M=16 to a different config-tagged
    name (`mv_int4_bf16_matmul.o`), so the two variants don't clobber each
    other in CWD across sessions; the last-staged canonical .o is whichever
    variant the current compile needs.

    `[2026-08-19]` The tag carries the GROUP SIZE too: `_compile_kernel`
    skips an existing .o by NAME, and DIM_GS is baked in at compile time --
    a gs=32 build (GGUF q4_0, SmolLM2) after a gs=128 one (AWQ) in the same
    CWD would silently reuse the wrong kernel, the same
    same-name-different-content class `compile_gemm_mm` was bitten by. The
    canonical copy runs on every call, so the right variant is staged even
    when the tagged object is reused.
    """
    src = _PROJ_ROOT / "matrix_vector_multiplication" / "int4_awq" / "mv_int4_bf16.cc"
    tagged = f"mv_int4_bf16_gemv_gs{gs}.o"
    _compile_kernel(
        src,
        tagged,
        extra_flags=[
            f"-DDIM_M={m_tile}",
            f"-DDIM_K={k_chunk}",
            f"-DDIM_GS={gs}",
        ],
    )
    shutil.copy2(tagged, "mv_int4_bf16.o")


def compile_mv_bf16():
    """Compile mv_bf16.o for the 2-tile matvec+add primitive used by
    o_gemv_ffn stages 1 and 3."""
    src = _PROJ_ROOT / "matrix_vector_multiplication" / "bf16_cascade" / "mv_bf16.cc"
    _compile_kernel(src, "mv_bf16.o")


def compile_attn_decode_npu2(head_dim=64):
    """Compile attn_decode_npu2.o (RoPE helpers for the fused decode kernel)."""
    src = _PROJ_ROOT / "attention_decode" / "attn_decode_npu2.cc"
    _compile_kernel(
        src,
        "attn_decode_npu2.o",
        extra_flags=[
            f"-DDIM_N={head_dim}",
            f"-DHEAD_SIZE={head_dim}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ],
    )


# ---------------------------------------------------------------------------
# Transformer-layer execution-study kernels
#
# These four live behind opt-in -D flags and are not built by
# compile_all_external_kernels(); the transformer_layer example drives them
# explicitly. Keeping them opt-in is what lets the shared LLM path keep
# compiling byte-identical objects.
# ---------------------------------------------------------------------------

_TL_KERNELS = _PROJ_ROOT / "transformer_layer" / "kernels"

# encoder.cc and addnorm_ffn.cc do `#include "zero.cc"`, which resolves against
# this directory. They deliberately reuse the GEMM example's zero.cc rather than
# carrying a fourth copy of it.
_ZERO_CC_DIR = _PROJ_ROOT / "matrix_multiplication" / "bf16_in_fp32_out"


def _tl_dim_flags(tile_m, tile_k, tile_n):
    return [f"-DDIM_M={tile_m}", f"-DDIM_K={tile_k}", f"-DDIM_N={tile_n}"]


def compile_encoder(
    tile_m=64,
    tile_k=64,
    tile_n=64,
    build_ffn=True,
    build_addnorm=True,
    out_name="encoder.o",
):
    """Compile encoder.o from transformer_layer/kernels/encoder.cc.

    The encoder-block kernels backing the `ffn` and `addnorm` operators.

    build_ffn / build_addnorm select which half of the file is emitted. They are
    independent, and BOTH DEFAULT TO TRUE here even though the source emits
    nothing when neither is defined -- an object with no symbols links silently
    and fails much later at dispatch, so this wrapper refuses to produce one.

    tile_m / tile_k / tile_n bake in the FFN matmul shape. The source
    static_asserts DIM_M % 16, DIM_N % 16 and DIM_K % 8 under build_ffn; the
    addnorm entry points ignore them entirely and take cols/rows at runtime.

    FOOTGUN: encoder.o and addnorm_ffn.o both define ffn_gelu_bf16 and
    ffn_eltwise_add_bf16_vector, so they cannot be linked into one ELF as-is.
    """
    if not build_ffn and not build_addnorm:
        raise ValueError(
            "compile_encoder: at least one of build_ffn / build_addnorm must be "
            "set, otherwise encoder.o exports no symbols"
        )
    extra = _tl_dim_flags(tile_m, tile_k, tile_n) + [
        f"-I{_ZERO_CC_DIR}",
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
    ]
    if build_ffn:
        extra.append("-DBUILD_FFN")
    if build_addnorm:
        extra.append("-DBUILD_ADDNORM")
    _compile_kernel(_TL_KERNELS / "encoder.cc", out_name, extra_flags=extra, force=True)


def compile_addnorm_ffn(
    tile_m=64,
    tile_k=64,
    tile_n=64,
    build_ffn=True,
    build_addnorm=True,
    pre_add=False,
    out_name="addnorm_ffn.o",
):
    """Compile addnorm_ffn.o from transformer_layer/kernels/addnorm_ffn.cc.

    One source covering what iron carried as two near-identical files
    (addnorm_ffn.cc and addnorm_ffn_addnorm.cc), selected by `pre_add`:

      pre_add=False (default)  statistics over `input`;
                               out = gamma * norm(input) + residual
      pre_add=True             statistics over `input + residual`;
                               out1 = gamma * norm(input + residual), and the
                               2-output form's out2 carries the raw pre-add sum
                               forward as the next block's residual stream

    Getting this backwards produces a plausible-looking activation that is
    wrong by one residual add, which survives a shape check and shows up only as
    drift in a per-layer cosine comparison.

    tile_m / tile_k / tile_n bake in the FFN matmul shape. The static_asserts
    here are tighter than encoder.cc's: up_proj needs DIM_N % 32 and down_proj
    needs DIM_K % 32.
    """
    if not build_ffn and not build_addnorm:
        raise ValueError(
            "compile_addnorm_ffn: at least one of build_ffn / build_addnorm "
            "must be set, otherwise addnorm_ffn.o exports no symbols"
        )
    extra = _tl_dim_flags(tile_m, tile_k, tile_n) + [
        f"-I{_ZERO_CC_DIR}",
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
    ]
    if build_ffn:
        extra.append("-DBUILD_FFN")
    if build_addnorm:
        extra.append("-DBUILD_ADDNORM")
    if pre_add:
        extra.append("-DADDNORM_PRE_ADD")
    _compile_kernel(
        _TL_KERNELS / "addnorm_ffn.cc", out_name, extra_flags=extra, force=True
    )


def compile_layer_norm(vec_len=16, out_name="layer_norm.o"):
    """Compile layer_norm.o from programming_examples/layer_norm/layer_norm.cc.

    Exposes layer_norm, layer_norm_rows and add_layer_norm_rows -- the
    multi-row forms the layer_norm example's direct-codegen builder does not
    cover. `cols` must be a multiple of vec_len; there is no scalar tail, so a
    non-multiple silently drops the remainder.
    """
    src = _PROJ_ROOT / "layer_norm" / "layer_norm.cc"
    _compile_kernel(src, out_name, extra_flags=[f"-DLN_VEC_LEN={vec_len}"], force=True)


def compile_softmax_streaming(vec_len=64, out_name="softmax_streaming.o"):
    """Compile softmax_streaming.o from programming_examples/softmax/softmax.cc.

    Emits the two-pass streaming family (init_softmax_scale_buffer,
    partial_softmax_rows_bf16, normalize_softmax_rows_bf16,
    copy_softmax_scale_bf16) alongside the file's existing single-shot
    softmax_bf16. `vec_len` must divide the row width.

    Written to a distinct out_name rather than softmax.o so a design that links
    both the single-shot and the streaming object does not pick up two
    definitions of softmax_bf16.
    """
    src = _PROJ_ROOT / "softmax" / "softmax.cc"
    _compile_kernel(
        src,
        out_name,
        extra_flags=["-DSOFTMAX_STREAMING", f"-DSM_VEC_LEN={vec_len}"],
        force=True,
    )


def compile_all_external_kernels(head_dim=64, quant="bf16", int4_gs=128):
    """Compile all external C++ kernels from source.

    Call this before kernel compilation to ensure all .o files are fresh.
    Each kernel is only compiled if its .o doesn't already exist.
    Delete build_peano/*.o to force recompilation.

    Args:
        head_dim: attention head dimension (RoPE / attn kernel macros).
        quant: "bf16" (default) or "awq". When "awq" the int4-AWQ GEMV
            micro-kernel (`mv_int4_bf16.o`) is also built so the int4
            decode ELFs can link it. bf16 GEMV objects are still built
            so mixed paths (e.g. bf16 prefill + int4 decode) keep working.
    """
    compile_silu_and_mul()
    compile_rope()
    compile_attn_npu2(head_dim=head_dim)
    compile_attn_decode_npu2(head_dim=head_dim)
    compile_mv()
    compile_mv_bf16()
    if quant == "awq":
        # `int4_gs` is the checkpoint's group size (AWQ 128, GGUF q4_0 32).
        # It MUST reach this call: this sweep runs inside EVERY
        # compile_and_cache via prepare_air_project, so a default here
        # restages the canonical mv_int4_bf16.o at gs=128 AFTER any earlier
        # gs=32 staging and immediately BEFORE aiecc links -- the first
        # SmolLM2 int4 build linked the wrong group size exactly this way
        # (devq 372/376: k/v from the decode ELF uncorrelated with host).
        compile_mv_int4_bf16(gs=int4_gs)
