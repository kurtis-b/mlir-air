# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compile-only gate for the transformer-layer AIE2P device kernels.

Builds every kernel Phase A of the transformer-layer execution studies ports,
then checks each resulting object actually contains code and actually exports
the entry points its builder will call.

WHY THE SYMBOL CHECK EXISTS
    Peano is happy to emit a valid, well-formed, completely empty object file.
    encoder.cc and addnorm_ffn.cc gate everything they define behind
    ``-DBUILD_FFN`` / ``-DBUILD_ADDNORM``; miss the flag and you get a 700-byte
    .o that links without complaint and fails at dispatch, far from the cause.
    So "it compiled" is not the gate -- "it compiled, is larger than
    MIN_OBJECT_BYTES, and defines these named symbols" is.

FOOTGUNS
    - Objects land in the current working directory, matching
      ``external_kernels``' convention (aiecc finds them via its link_with
      search path). Run this from a build directory, not from the source tree.
    - ``encoder.o`` and ``addnorm_ffn.o`` both define ``ffn_gelu_bf16`` and
      ``ffn_eltwise_add_bf16_vector``. They are built here as separate objects
      and are NOT linkable into a single ELF without renaming one set.
    - The two ``addnorm_ffn`` variants differ only in whether the residual is
      folded in before the normalization statistics. Both are built here, to
      distinct object names, precisely because they are easy to confuse.
    - This is compile-only. Nothing here touches the NPU, and nothing here
      says anything about numerical correctness.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parent.parent  # programming_examples/
sys.path.insert(0, str(_PROJ_ROOT / "llms" / "shared" / "infra"))

import external_kernels as ek  # noqa: E402

# An object holding real code is several KB. The empty-but-valid objects this
# check exists to catch are a few hundred bytes.
MIN_OBJECT_BYTES = 1024


def _llvm_nm():
    """Locate llvm-nm, preferring Peano's so it can read aie2p objects."""
    peano_dir = os.environ.get("PEANO_INSTALL_DIR", "")
    if peano_dir:
        candidate = Path(peano_dir) / "bin" / "llvm-nm"
        if candidate.exists():
            return str(candidate)
    found = shutil.which("llvm-nm")
    if found:
        return found
    raise RuntimeError(
        "Cannot find llvm-nm (no PEANO_INSTALL_DIR/bin/llvm-nm, none on PATH)"
    )


def defined_symbols(obj_path):
    """Return the set of globally defined symbols in `obj_path`."""
    result = subprocess.run(
        [_llvm_nm(), "--defined-only", str(obj_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    symbols = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        # "<addr> <type> <name>"; global text symbols are type T.
        if len(parts) == 3 and parts[1] == "T":
            symbols.add(parts[2])
    return symbols


def check_object(obj_name, expected_symbols):
    """Assert the object is non-trivial and exports `expected_symbols`."""
    obj_path = Path(obj_name)
    if not obj_path.exists():
        raise RuntimeError(f"{obj_name}: not produced")

    size = obj_path.stat().st_size
    if size < MIN_OBJECT_BYTES:
        raise RuntimeError(
            f"{obj_name}: only {size} bytes, under the {MIN_OBJECT_BYTES}-byte "
            "floor -- almost certainly compiled with no feature flag set"
        )

    present = defined_symbols(obj_path)
    missing = sorted(set(expected_symbols) - present)
    if missing:
        raise RuntimeError(f'{obj_name}: missing extern "C" symbols {missing}')

    print(f"  {obj_name}: {size} bytes, {len(expected_symbols)} symbols checked")


# Each entry: (label, build callable, object name, symbols that must be present)
def build_plan(tile_m, tile_k, tile_n):
    return [
        (
            "GEMM microkernel, init + with_acc families",
            lambda: ek.compile_gemm_mm(
                tile_m=64,
                tile_n=128,
                tile_k_l1=32,
                out_name="mm_transformer.o",
                gen_init=True,
                gen_with_acc=True,
            ),
            "mm_transformer.o",
            [
                "op_has_no_registered_library_name",
                "matmul_init_bf16_f32",
                "matmul_with_acc_bf16_f32",
                "zero_f32_mn",
                "f32_to_bf16_mn",
            ],
        ),
        (
            "Streaming softmax",
            lambda: ek.compile_softmax_streaming(),
            "softmax_streaming.o",
            [
                "softmax_bf16",
                "init_softmax_scale_buffer",
                "partial_softmax_rows_bf16",
                "normalize_softmax_rows_bf16",
                "copy_softmax_scale_bf16",
            ],
        ),
        (
            "Multi-row LayerNorm",
            lambda: ek.compile_layer_norm(),
            "layer_norm.o",
            ["layer_norm", "layer_norm_rows", "add_layer_norm_rows"],
        ),
        (
            "FlashAttention with causal row helpers",
            lambda: ek.compile_attn_npu2(
                head_dim=64, causal_row_helpers=True, force=True
            ),
            "attn_npu2.o",
            [
                "apply_causal_mask",
                "copy_O_tile_rows",
                "store_row_value",
                "copy_row_values",
            ],
        ),
        (
            "Encoder block (FFN + AddNorm)",
            lambda: ek.compile_encoder(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n),
            "encoder.o",
            [
                "ffn_zero_bf16_up_proj",
                "ffn_zero_bf16_down_proj",
                "ffn_matmul_init_bf16_bf16_up_proj",
                "ffn_matmul_bf16_bf16_up_proj",
                "ffn_matmul_init_bf16_bf16_down_proj",
                "ffn_matmul_with_acc_bf16_bf16_down_proj",
                "ffn_gelu_bf16",
                "ffn_eltwise_add_bf16_vector",
                "ln_passThroughTile_in",
                "fused_add_layer_norm_1outs",
                "fused_layer_norm_1outs",
                "ln_mul_weights_1outs",
                "ln_mul_add_1outs",
                "fused_add_layer_norm_2outs",
                "ln_zero_bf16",
                "ln_zero_f32",
                "ln_calc_sum_sumsq",
            ],
        ),
        (
            "AddNorm+FFN, post-add residual",
            lambda: ek.compile_addnorm_ffn(
                tile_m=tile_m,
                tile_k=tile_k,
                tile_n=tile_n,
                pre_add=False,
                out_name="addnorm_ffn.o",
            ),
            "addnorm_ffn.o",
            [
                "zero_bf16_up_proj",
                "zero_bf16_down_proj",
                "matmul_bf16_bf16_up_proj_half_inps",
                "matmul_with_acc_bf16_bf16_down_proj",
                "ffn_passThroughTile_out",
                "ffn_gelu_bf16",
                "ffn_eltwise_add_bf16_vector",
                "fused_add_layer_norm_1outs",
                "fused_add_layer_norm_2outs",
                "ln_passThroughTile_out",
                "ln_passThroughTile_in",
            ],
        ),
        (
            "AddNorm+FFN, pre-add residual",
            lambda: ek.compile_addnorm_ffn(
                tile_m=tile_m,
                tile_k=tile_k,
                tile_n=tile_n,
                pre_add=True,
                out_name="addnorm_ffn_pre_add.o",
            ),
            "addnorm_ffn_pre_add.o",
            [
                "matmul_bf16_bf16_up_proj_half_inps",
                "fused_add_layer_norm_1outs",
                "fused_add_layer_norm_2outs",
            ],
        ),
    ]


def check_pre_add_variants_differ():
    """The two addnorm_ffn objects must not be identical.

    If -DADDNORM_PRE_ADD ever stops reaching the two fused_add_layer_norm
    templates, both variants silently become the post-add form and every
    pre-add design starts computing a subtly wrong activation. Comparing the
    objects is the cheapest way to notice.
    """
    post = Path("addnorm_ffn.o").read_bytes()
    pre = Path("addnorm_ffn_pre_add.o").read_bytes()
    if post == pre:
        raise RuntimeError(
            "addnorm_ffn.o and addnorm_ffn_pre_add.o are byte-identical: "
            "-DADDNORM_PRE_ADD is not reaching the fused_add_layer_norm bodies"
        )
    print("  addnorm_ffn pre-add / post-add variants differ, as expected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-m", type=int, default=64)
    parser.add_argument("--tile-k", type=int, default=64)
    parser.add_argument("--tile-n", type=int, default=64)
    args = parser.parse_args()

    plan = build_plan(args.tile_m, args.tile_k, args.tile_n)
    for label, build, obj_name, symbols in plan:
        print(f"{label}:")
        build()
        check_object(obj_name, symbols)

    check_pre_add_variants_differ()

    print(f"Built and checked {len(plan)} kernel objects.")
    print("Compilation passed.")


if __name__ == "__main__":
    main()
