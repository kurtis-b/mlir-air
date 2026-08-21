# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""`DeviceCaps` -- the XDNA2 bounds the planner reasons with, lifted, not invented.

Every number here is read from a named place in the tree (the study's
`mapping_space.py` / `profiles.py`, the builders' asserts, doc 57's
measurements). `test_plan.py` cross-checks the study-sourced ones against the
study modules when they are importable, so a bound that moves there moves here.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class DeviceCaps:
    # Array geometry (NPU2 / Strix): 8 columns x 4 compute rows, one memtile per column.
    columns: int = 8
    core_rows: int = 4
    l2_bytes_per_column: int = 512 * 1024     # mapping_space / matvec.py L2_CAPACITY
    l1_bytes: int = 64 * 1024                 # per core; profiles.SOFTMAX_L1_BYTES is the same number
    max_feed_channels: int = 6                # mapping_space.MAX_FEED_CHANNELS
    max_placeable_herd_x: int = 4             # mapping_space.MAX_PLACEABLE_HERD_X (study's GEMM families)
    # DMA descriptor limits.
    bd_repeat_cap: int = 255                  # qwen3_0_6b_decode: n_part/32 - 1 <= 255
    gemv_rows_per_bd_repeat: int = 32         # ... the 32 in that formula (8-row tiles x 4 m_input? -- read as written)
    # Study bounds (profiles.py), each a builder refusal before aircc.
    fa_parallel_seq: int = 256                # profiles.FA_PARALLEL_SEQ
    attn_gemm_seq_multiple: int = 512         # profiles.ATTN_GEMM_SEQ_MULTIPLE
    softmax_l1_bytes: int = 64 * 1024         # profiles.SOFTMAX_L1_BYTES
    softmax_scale_bands: int = 4              # profiles.SOFTMAX_SCALE_BANDS
    softmax_itemsize: int = 2                 # profiles.SOFTMAX_ITEMSIZE
    fused_plane_stride_cap: int = 2 ** 20     # profiles.FUSED_PLANE_STRIDE_CAP
    fused_seq_min: int = 256                  # profiles.FUSED_SEQ_MIN
    # Model-driver bounds (llms/*/ARCHITECTURE.md): the lean fused O+FFN forms.
    lean_form_emb_max: int = 2560             # "emb=1024 (< 2560)"
    lean_form_hidden_multiple: int = 512      # "hidden=3072 is divisible by 512"
    # FlashAttention variants by head_dim (llms/shared/fa_headfirst.py, attn_npu2_seqfirst.py).
    headfirst_fa_head_dim: int = 128          # host transposes around it (seq-first dk_chunks>1 bug)
    seqfirst_fa_head_dim: int = 64
    # Measured dispatch constants (doc 57 section 1.5, devq 450-451).
    launch_boundary_us: float = 107.0         # per air.launch configuration in a multi-launch ELF
    run_fixed_us: float = 146.0               # per xrt.run (submission + wait + small syncs)
    gemv_stream_gbs: float = 32.0             # LM-head class GEMV, boundary-diluted (doc 57 section 1.1)

    @property
    def gemv_max_rows_per_launch(self):
        """Largest M one bf16 GEMV launch may stream: (rows/32 - 1) <= repeat cap."""
        return (self.bd_repeat_cap + 1) * self.gemv_rows_per_bd_repeat

    def softmax_fits_l1(self, seq):
        need = (3 * seq + self.softmax_scale_bands) * self.softmax_itemsize
        return need <= self.softmax_l1_bytes

    def fused_seq_range(self, emb, ladder):
        return self.fused_seq_min, max(s for s in ladder if s * emb <= self.fused_plane_stride_cap)

    def as_dict(self):
        return asdict(self)


NPU2_CAPS = DeviceCaps()
