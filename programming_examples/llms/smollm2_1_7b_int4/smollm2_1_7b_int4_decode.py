# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolLM2-1.7B int4 (GGUF q4_0) Decode on MLIR-AIR (NPU2).

Thin over `llama32_1b_int4_decode`: the per-token block runner and the CPU
KV-cache attention are imported unchanged (both are config-generic, and MHA
is just their `group_size = n_heads // n_kv_heads = 1` case). What is
model-specific is kernel COMPILATION, for two reasons:

* the shapes: emb 2048, kv_dim 2048 (full MHA — wk/wv square), hidden 8192,
  24 layers;
* the group size: this checkpoint is q4_0, `gs = 32` (the AWQ example runs
  128). `DIM_GS` is baked into the device micro-kernel at compile time, so
  `compile_mv_int4_bf16(gs=32)` must stage the canonical `mv_int4_bf16.o`
  before ANY of the int4 decode ELFs link — the rms builder restages it at
  build time, and this module also stages it explicitly first so the
  ordering is stated rather than inherited.

Usage (standalone decode-only smoke):
    cd build_peano
    python3 ../smollm2_1_7b_int4_decode.py --compile-only
For full e2e (prefill + decode + chat), use `smollm2_1_7b_int4_inference.py`.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LLMS_DIR = os.path.dirname(_THIS_DIR)
_LLAMA_INT4 = os.path.join(_LLMS_DIR, "llama32_1b_int4")
_SMOLLM2_BF16 = os.path.join(_LLMS_DIR, "smollm2_1_7b")
# Insert in reverse priority order so the LAST insert wins sys.path[0].
# `_LLAMA_INT4` must beat the bf16 dirs so `multi_launch_builder` resolves
# to the int4 package (the bf16 sibling has a same-named one); `_THIS_DIR`
# stays first for this model's own modules.
for _p in (_LLMS_DIR, _SMOLLM2_BF16, _LLAMA_INT4, _THIS_DIR):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from smollm2_1_7b_weights import LlamaConfig  # noqa: E402
from llama32_1b_int4_decode import (  # noqa: E402,F401
    decode_attention_cpu,
    run_decode_block,
)
from shared.infra.cache import KernelCache  # noqa: E402
from shared.infra.backend_presets import (  # noqa: E402
    RGR_INT4_BACKEND,
    OGF_INT4_BACKEND,
    LM_GEMV_BACKEND,
)

#: q4_0's block size IS the group size. Everything downstream — the packed
#: BOs the loader built, the ELF builders' tile maths, and the device
#: micro-kernel's DIM_GS — must agree on it.
GS = 32


def compile_decode_kernels(cache, config):
    """Compile the 3 int4 decode kernels at SmolLM2 shapes and gs=32."""
    from shared.infra.external_kernels import (
        compile_all_external_kernels,
        compile_mv_int4_bf16,
    )

    compile_all_external_kernels(head_dim=config.head_dim, quant="awq", int4_gs=GS)
    # Belt and braces: the staging that actually decides what aiecc links is
    # the per-compile sweep INSIDE compile_and_cache (prepare_air_project),
    # which is why the two int4 ELF compiles below carry `int4_gs` in their
    # backend kwargs -- the study branch's first build, without it, linked
    # the gs=128 default because that sweep restaged the canonical right
    # before the link, after every explicit staging here had run.
    compile_mv_int4_bf16(m_tile=8, k_chunk=2048, gs=GS)

    emb_dim = config.emb_dim
    kv_dim = config.n_kv_heads * config.head_dim

    print(f"\n{'='*60}")
    print(f"Compiling int4 decode kernels (SmolLM2, gs={GS})...")
    print(f"{'='*60}\n")

    from multi_launch_builder.rms_qkv_int4_rope_multi import (
        build_rms_qkv_int4_rope_module,
    )

    cache.compile_and_cache(
        "rms_qkv_int4_rope",
        build_rms_qkv_int4_rope_module(
            emb_dim=emb_dim,
            kv_dim=kv_dim,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            gs=GS,
        ),
        {"verbose": cache.verbose, "int4_gs": GS, **RGR_INT4_BACKEND},
    )

    from multi_launch_builder.o_gemv_ffn_int4_multi import (
        build_o_gemv_ffn_int4_module,
    )

    cache.compile_and_cache(
        "o_gemv_ffn_int4",
        build_o_gemv_ffn_int4_module(
            emb_dim=emb_dim, hidden_dim=config.hidden_dim, gs=GS
        ),
        {"verbose": cache.verbose, "int4_gs": GS, **OGF_INT4_BACKEND},
    )

    # LM head stays bf16 and tied to embed_table (Q6_K never consumed).
    from shared.builders.lm_head_gemv_multi import build_lm_head_gemv_module

    cache.compile_and_cache(
        "lm_head_gemv",
        build_lm_head_gemv_module(emb_dim),
        {"verbose": cache.verbose, **LM_GEMV_BACKEND},
    )

    cache._save_manifest()
    print(f"\nAll {len(cache.artifacts)} decode kernels compiled.")


def _main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Compile the SmolLM2 int4 decode kernels. This entry "
        "NEVER dispatches -- it only fills the kernel cache."
    )
    ap.add_argument(
        "--compile-only",
        action="store_true",
        help="accepted for symmetry with the other model entries, and "
        "redundant: this entry only ever compiles.",
    )
    ap.add_argument("--cache-dir", type=str, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.compile_only and args.verbose:
        print(
            "[smollm2-int4-decode] --compile-only is redundant here: this "
            "entry only compiles"
        )

    config = LlamaConfig()
    cache = KernelCache(cache_dir=args.cache_dir, verbose=args.verbose)
    compile_decode_kernels(cache, config)


if __name__ == "__main__":
    _main()
