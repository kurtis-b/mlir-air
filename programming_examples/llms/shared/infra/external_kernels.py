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


def compile_gelu_and_mul():
    """Compile gelu_and_mul.o from programming_examples/gelu_and_mul/gelu_and_mul.cc.

    The GELU-tanh twin of silu_and_mul.o, for models whose GLU uses
    gelu_pytorch_tanh (Gemma3) rather than SiLU."""
    src = _PROJ_ROOT / "gelu_and_mul" / "gelu_and_mul.cc"
    include_dir = _get_aie_include_dir()
    utils_header = Path(include_dir) / "aie_kernels" / "aie_kernel_utils.h"
    extra = []
    if utils_header.exists():
        extra = [f"-include", str(utils_header)]
    _compile_kernel(src, "gelu_and_mul.o", extra_flags=extra)


def compile_gemm_mm(
    tile_m=64,
    tile_n=128,
    tile_k_l1=32,
    sym_suffix="",
    out_name="mm.o",
    bfp16=True,
    tile_k_l1_pad=0,
):
    """Compile mm.o from matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc.

    The hand-tuned Peano -O2 vectorized GEMM microkernel (external path), ~1.5-1.65x
    faster than direct-codegen on large shapes (kernel_registry/details/GEMM_bf16_in_fp32_out.md).
    DIM_M/DIM_N/DIM_K are baked in at compile time and MUST match the tile_m/tile_n/
    tile_k_l1 passed to the GEMM module builder. Exposes op_has_no_registered_library_name
    (f32-C matmul), zero_f32_mn, f32_to_bf16_mn.

    sym_suffix / out_name: to link SEVERAL mm.o variants into ONE ELF, the
    symbols must not collide -- and two GEMMs at the same tile_m but different
    tile_n are two different micro-kernels, because DIM_N is baked in here.
    PREFER `compile_gemm_mm_variant` below, which derives both names from
    (tile_m, tile_n) via the one authority for them,
    `gemm_builder.gemm_variant_names` (-> "_m64" / "mm_m64.o" at tile_n=128,
    "_m64n64" / "mm_m64n64.o" otherwise). Spelling them by hand is how an
    object gets built at the wrong DIM_N and still links. Default empty suffix
    / "mm.o" keeps the original names for single-variant ELFs (back-compat).
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
    ]
    # BFP16 block-float emulation (shared 8-elem exponent) trades matmul accuracy
    # for throughput. It is the validated aie2p GEMM path for every model here.
    # bfp16=False was tried for the SigLIP vision encoder (to raise precision) but
    # the native aie2p bf16 8x8x8 mmul in mm_aie2p.cc uses a DIFFERENT C_block
    # accumulator layout than the BFP16 branch, so it produces WRONG results at
    # these tiles (block-vs-oracle 0.28-0.90) — NOT a usable precision knob without
    # rewriting the microkernel. Left as a flag (default True = unchanged) purely
    # to document the experiment. Keep bfp16=True.
    if bfp16:
        extra.append("-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16")
    # tile_k_l1_pad > tile_k_l1: B carries trailing bias rows the reduction
    # skips. DIM_K_PAD is what makes `extract_bias_from_b` (compute herd) and
    # the `f32_to_bf16_bias{,_gelu}_mn` drain casts see the padded stride.
    if tile_k_l1_pad and tile_k_l1_pad != tile_k_l1:
        extra.append(f"-DDIM_K_PAD={tile_k_l1_pad}")
    if sym_suffix:
        extra.append(f"-DSYM_SUFFIX={sym_suffix}")
    _compile_kernel(src, out_name, extra_flags=extra, force=True)


def compile_gemm_mm_variant(tile_m, tile_n, tile_k_l1=32, **kwargs):
    """`compile_gemm_mm` with the symbol suffix and object name DERIVED, not given.

    The names come from `shared.builders.gemm_builder.gemm_variant_names`,
    which is also where every GEMM module's `sym_suffix` / `link_with_name`
    come from (bare `_m64` / `mm_m64.o` at tile_n=128, `_m64n64` /
    `mm_m64n64.o` otherwise). Call this rather than spelling `sym_suffix=` and
    `out_name=` yourself: the two must agree per (tile_m, tile_n), and when
    they do not, the failure is either an unresolved symbol at link time or --
    worse, and what the study actually measured -- an object built at the
    wrong `-DDIM_N` that links cleanly and returns zeros for part of every
    output tile.

    Any remaining keyword (`bfp16`, `tile_k_l1_pad`, ...) is passed straight
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
    head_dim=64,
    lkp=None,
    lqp_tile=None,
    force=False,
    bfp16=True,
    dk_tile=None,
    dv_tile=None,
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
        lkp: K/V chunk size per tile -- how much of the K/V sequence a tile
            holds at once. Defaults to head_dim (legacy hd==lkp behavior).
            It also supplies the DEFAULT d tile, but the two are independent:
            pass dk_tile/dv_tile to set the d tile separately.
        lqp_tile: Q tile size (tile_size_q). Defaults to lkp.
        dk_tile / dv_tile: the K/V dimension TILE. Default to lkp (the kernel
            slices d into lkp-wide chunks). The temporal-causal kernel keeps d
            WHOLE instead, so it passes dk_tile = dv_tile = head_dim. The
            head-first path widens dv_tile alone, past lkp but short of
            head_dim, to shrink the dv_chunks launch axis (and with it the K
            re-streaming that axis costs). Both must match the builder.
        force: recompile even if attn_npu2.o exists (needed when the same CWD
            previously built a different-shaped .o, e.g. hd=64 then hd=128).
    """
    if lkp is None:
        lkp = head_dim
    if lqp_tile is None:
        lqp_tile = lkp
    if dk_tile is None:
        dk_tile = lkp
    if dv_tile is None:
        dv_tile = lkp
    src = _PROJ_ROOT / "flash_attention" / "kernel_fusion_based" / "attn_npu2.cc"
    _flags = [
        "-DBIT_WIDTH=8",
        f"-Dlqp={lqp_tile}",
        f"-Dlkp={lkp}",
        f"-Ddk={dk_tile}",
        f"-Ddk_full={head_dim}",
        f"-Ddv={dv_tile}",
        f"-Ddv_full={head_dim}",
        "-DROUND_CONV_EVEN",
    ]
    # BFP16 block-float emulation of the Q@Kᵀ / P@V matmuls trades accuracy for
    # throughput. It is fine for the llama/qwen backbones, but on the SigLIP
    # vision encoder it injects a SYSTEMATIC per-channel bias into the attention
    # output (~81% of the FA error is the shared column-mean), which survives
    # LayerNorm centering and accumulates coherently across the 12 layers'
    # residual-cancellation blocks (post_ln cosine 0.945). bfp16=False selects the
    # native aie2p bf16 8x8x8 mmul (accauto=accfloat accumulate) — still fully on
    # NPU, higher precision.
    if bfp16:
        _flags.append("-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16")
    _compile_kernel(src, "attn_npu2.o", extra_flags=_flags, force=force)
    # Also create attn.o copy (some link_with attributes use "attn.o").
    # Refresh whenever attn_npu2.o exists so a force-rebuild (different tile
    # shape) doesn't leave a stale attn.o behind.
    if Path("attn_npu2.o").exists():
        shutil.copy2("attn_npu2.o", "attn.o")


def compile_mv(tile_m=8):
    """Compile mv.o (standard GEMV kernel) from source."""
    src = _PROJ_ROOT / "matrix_vector_multiplication" / "bf16" / "mv.cc"
    _compile_kernel(src, "mv.o", extra_flags=[f"-DDIM_M_OUTPUT={tile_m}"])


def compile_mv_heads(head_dim=128):
    """Compile `mv_heads_hd{head_dim}.o`: the GEMV rows kernel plus the in-core
    per-head QK-norm + RoPE epilogue (doc 57 O1 second half, 2026-08-22) from
    programming_examples/matrix_vector_multiplication/bf16/mv_heads.cc.

    HEAD_DIM is baked in (the per-head loops unroll; no float division), so
    the object is head_dim-tagged like the GEMM variants and the builder's
    `link_with` names the tagged object directly -- no canonical copy to
    go stale between models of different head_dim.
    """
    src = _PROJ_ROOT / "matrix_vector_multiplication" / "bf16" / "mv_heads.cc"
    name = mv_heads_object_name(head_dim)
    # Rebuild when the source is newer than the object: a stale same-name .o
    # in a build cwd otherwise links silently (or fails on a renamed symbol
    # several frames away, as it did on 2026-08-23).
    out = Path(name)
    stale = out.exists() and out.stat().st_mtime < src.stat().st_mtime
    _compile_kernel(src, name, extra_flags=[f"-DHEAD_DIM={head_dim}"], force=stale)
    return name


def mv_heads_object_name(head_dim=128):
    return f"mv_heads_hd{head_dim}.o"


def compile_mv_int4_bf16(m_tile=8, k_chunk=2048, gs=128, r=None):
    """Compile mv_int4_bf16.o (int4-AWQ GEMV micro-kernel) from source.

    Produces the config-tagged
    `mv_int4_bf16_gemv_m{m_tile}_gs{gs}_k{k_chunk}_r{r}.o` and stages it as
    the canonical `mv_int4_bf16.o` (the name link_with attributes expect). The int4 GEMM prefill compiles the same .cc with
    DIM_M=16 to a different config-tagged name (`mv_int4_bf16_matmul.o`),
    so the two variants don't clobber each other in CWD across sessions;
    the last-staged canonical .o is whichever variant the current compile
    needs.

    The tag carries EVERY define this function bakes in (as `compile_mv_heads`
    tags head_dim): `_compile_kernel` skips an existing .o by NAME, so an
    untagged name hands every later caller the first variant built in that
    CWD -- a gs=32 build (GGUF q4_0) after a gs=128 one (AWQ) silently
    reused the gs=128 kernel. The canonical copy runs on every call, so the
    right variant is staged even when the tagged object is reused. Adding a
    -D below without adding it to the name re-opens the same hole.

    `r` is the inner vector width of the GEMV bodies (queue item 23: at r=64
    the weight load fuses into one `vldb.unpack`, 47 -> 34 vector ops per
    gs=128 group, measured at study commit 20fad8c8). Default None computes
    the kernel's own DIM_R predicate (64 where gs % 64 == 0, else 32) and
    passes it explicitly; the kernel static_asserts the two agree. `r` is
    baked in exactly like DIM_GS and DIM_K, so the tag carries it: without
    that, a control arm rebuilt at r=32 in a CWD that already held an r=64
    object would silently link the r=64 kernel and be reported as the
    control -- an A/B whose two arms are the same bytes. `MV_INT4_GEMV_R` in
    the environment is the only override besides the `r=` argument, and it
    exists for exactly that A/B.
    """
    if r is None:
        env_r = os.environ.get("MV_INT4_GEMV_R")
        if env_r:
            try:
                r = int(env_r)
            except ValueError:
                raise ValueError(
                    f"MV_INT4_GEMV_R={env_r!r} is not an integer -- refusing to "
                    "guess an inner vector width"
                )
        else:
            # Must match mv_int4_bf16.cc's DIM_R predicate exactly.
            r = 64 if gs % 64 == 0 else 32
    if r not in (32, 64):
        raise ValueError(f"inner vector width r={r} is not 32 or 64")
    if gs % r:
        raise ValueError(
            f"inner vector width r={r} does not divide group size gs={gs}; the "
            "kernel static_asserts this and the build would fail late"
        )
    src = _PROJ_ROOT / "matrix_vector_multiplication" / "int4_awq" / "mv_int4_bf16.cc"
    tagged = f"mv_int4_bf16_gemv_m{m_tile}_gs{gs}_k{k_chunk}_r{r}.o"
    _compile_kernel(
        src,
        tagged,
        extra_flags=[
            f"-DDIM_M={m_tile}",
            f"-DDIM_K={k_chunk}",
            f"-DDIM_GS={gs}",
            f"-DDIM_R={r}",
        ],
    )
    shutil.copy2(tagged, "mv_int4_bf16.o")
    return tagged


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


def compile_all_external_kernels(
    head_dim=64, quant="bf16", int4_gs=128, int4_k_chunk=2048
):
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
        int4_gs: the checkpoint's group size baked into that micro-kernel
            (AWQ 128, GGUF q4_0 32); only read when quant == "awq".
        int4_k_chunk: DIM_K baked into that micro-kernel; only read when
            quant == "awq".
    """
    compile_silu_and_mul()
    compile_rope()
    compile_attn_npu2(head_dim=head_dim)
    compile_attn_decode_npu2(head_dim=head_dim)
    compile_mv()
    compile_mv_bf16()
    if quant == "awq":
        # `int4_gs` MUST reach this call: this sweep runs inside EVERY
        # compile_and_cache via prepare_air_project, so a default here
        # restages the canonical mv_int4_bf16.o at gs=128 AFTER any earlier
        # gs=32 staging and immediately BEFORE aiecc links -- the study
        # branch's first SmolLM2 int4 build linked the wrong group size
        # exactly this way. `int4_k_chunk` rides the same channel (doc 56
        # H2b: qwen's int4 cascade is DIM_K=1024; a default here would
        # restage llama's 2048 kernel right before aiecc links the qwen ELF).
        compile_mv_int4_bf16(gs=int4_gs, k_chunk=int4_k_chunk)
