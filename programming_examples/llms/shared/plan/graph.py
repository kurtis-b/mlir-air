# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""`ModelGraph` -- the analog of the ggml graph (doc 56 section 3.2).

A small typed DAG, not a ggml clone and not direct builder calls. Tensors carry
shape as a function of (M, kv_len), logical / storage / compute / accumulator
dtypes, layout, lifetime and storage class; nodes carry op, inputs/outputs,
attributes, a phase predicate and the repeated-layer marker. The model is one
repeated block template plus embedding, final norm, LM head and the recurrent
KV state. Golden JSON for Qwen3-0.6B and Llama-3.2-1B is written from their
ARCHITECTURE.md and pinned by test_plan.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

# Shape symbols. A shape is a tuple of ints and/or these strings.
M = "M"            # query rows of the phase: prompt (chunk) length in prefill, 1 in decode
KV = "kv_len"      # keys visible to the phase's queries
VOCAB = "vocab"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    n_layers: int
    emb_dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_dim: int
    vocab_size: int
    qk_norm: bool = False
    tied_embeddings: bool = False
    rope_theta: float = 10000.0
    eps: float = 1e-6
    activation: str = "swiglu"
    weight_dtype: str = "bf16"
    rope_convention: str = "halfsplit"  # HF Llama/Qwen rotate (d[i], d[i+hd/2])
    # Driver facts the planner reproduces rather than derives (doc 56 H0 gate):
    lm_head_rows_per_launch: int = 0    # the shipped _LM_N_PART (0 = derive from the BD repeat cap)

    @property
    def q_dim(self):
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self):
        return self.n_kv_heads * self.head_dim


@dataclass(frozen=True)
class Tensor:
    id: str
    shape: tuple
    logical_dtype: str = "bf16"
    storage_dtype: str = "bf16"
    compute_dtype: str = "bf16"
    accum_dtype: str = "f32"
    layout: str = "row_major"          # row_major | head_first | seq_first | packed_int4
    storage_class: str = "activation"  # weight | activation | kv_state | scratch
    lifetime: str = "phase"            # resident (weights, kv) | phase | node (intermediate)


@dataclass(frozen=True)
class Node:
    id: str
    op: str
    inputs: tuple
    outputs: tuple
    attrs: tuple = ()          # tuple of (key, value) -- hashable
    phase: str = "both"        # prefill | decode | both
    repeated: bool = True      # True: one instance per layer (ids carry "L" as the layer symbol)

    def attr(self, key, default=None):
        for k, v in self.attrs:
            if k == key:
                return v
        return default


@dataclass
class ModelGraph:
    spec: ModelSpec
    tensors: dict = field(default_factory=dict)
    nodes: list = field(default_factory=list)

    def add_tensor(self, t):
        if t.id in self.tensors:
            raise ValueError(f"duplicate tensor {t.id}")
        self.tensors[t.id] = t
        return t.id

    def add_node(self, n):
        for tid in n.inputs + n.outputs:
            if tid not in self.tensors:
                raise ValueError(f"node {n.id} names unknown tensor {tid}")
        self.nodes.append(n)
        return n

    def shape_of(self, tid, M_=1, kv_len=1):
        sub = {M: M_, KV: kv_len, VOCAB: self.spec.vocab_size}
        return tuple(sub.get(d, d) if isinstance(d, str) else d for d in self.tensors[tid].shape)

    def nbytes(self, tid, M_=1, kv_len=1):
        itemsize = {"bf16": 2, "f16": 2, "f32": 4, "int4": 0.5, "uint8": 1}[self.tensors[tid].storage_dtype]
        n = 1
        for d in self.shape_of(tid, M_, kv_len):
            n *= d
        return int(n * itemsize)

    def phase_nodes(self, phase):
        return [n for n in self.nodes if n.phase in ("both", phase)]

    def producers(self, tid):
        return [n for n in self.nodes if tid in n.outputs]

    def to_json(self):
        return json.dumps(
            {
                "spec": asdict(self.spec),
                "tensors": [asdict(t) for t in self.tensors.values()],
                "nodes": [asdict(n) for n in self.nodes],
            },
            indent=1,
            default=list,
        )

    @classmethod
    def from_json(cls, text):
        d = json.loads(text)
        g = cls(ModelSpec(**d["spec"]))
        for t in d["tensors"]:
            t["shape"] = tuple(t["shape"])
            g.add_tensor(Tensor(**t))
        for n in d["nodes"]:
            n["inputs"], n["outputs"] = tuple(n["inputs"]), tuple(n["outputs"])
            n["attrs"] = tuple(tuple(a) for a in n["attrs"])
            g.add_node(Node(**n))
        return g


def decoder_graph(spec):
    """The repeated pre-norm decoder block + embed / final norm / LM head / KV state.

    Layer-repeated tensor and node ids carry the literal "L"; the planner
    instantiates them per layer. Weights are stored as the GEMM/GEMV builders
    consume them: (in, out) for prefill GEMMs, transposed (out, in) for decode
    GEMVs -- the graph records the logical (in, out) shape and leaves the
    transposition to lowering.
    """
    g = ModelGraph(spec)
    E, Q, KVd, H, HD = spec.emb_dim, spec.q_dim, spec.kv_dim, spec.hidden_dim, spec.head_dim
    W = dict(storage_class="weight", lifetime="resident", storage_dtype=spec.weight_dtype)
    A = dict(storage_class="activation", lifetime="node")

    # --- once ---
    g.add_tensor(Tensor("embed_table", (VOCAB, E), **W))
    g.add_tensor(Tensor("token_ids", (M,), logical_dtype="i32", storage_dtype="f32", **A))
    g.add_tensor(Tensor("x_0", (M, E), **A))
    g.add_node(Node("embed", "embed_lookup", ("token_ids", "embed_table"), ("x_0",), phase="both", repeated=False))
    g.add_tensor(Tensor("final_norm_w", (E,), **W))
    g.add_tensor(Tensor("x_final", (M, E), **A))
    g.add_tensor(Tensor("x_final_normed", (1, E), **A))
    g.add_node(Node("final_norm", "rms_norm", ("x_final", "final_norm_w"), ("x_final_normed",),
                    attrs=(("eps", spec.eps), ("rows", "last")), phase="both", repeated=False))
    g.add_tensor(Tensor("lm_head_w", (E, VOCAB), **W))
    g.add_tensor(Tensor("logits", (1, VOCAB), storage_dtype="f32", **A))
    g.add_node(Node("lm_head", "matmul", ("x_final_normed", "lm_head_w"), ("logits",),
                    attrs=(("M_rows", 1), ("tied", spec.tied_embeddings)), phase="both", repeated=False))

    # --- KV state (recurrent, resident) ---
    g.add_tensor(Tensor("k_cache_L", ("ctx", spec.n_kv_heads, HD), storage_class="kv_state", lifetime="resident", layout="head_first"))
    g.add_tensor(Tensor("v_cache_L", ("ctx", spec.n_kv_heads, HD), storage_class="kv_state", lifetime="resident", layout="head_first"))

    # --- block template ---
    for name, shape in (("attn_norm_w_L", (E,)), ("wq_L", (E, Q)), ("wk_L", (E, KVd)), ("wv_L", (E, KVd)),
                        ("wo_L", (Q, E)), ("ffn_norm_w_L", (E,)), ("w_gate_L", (E, H)), ("w_up_L", (E, H)), ("w_down_L", (H, E))):
        g.add_tensor(Tensor(name, shape, **W))
    if spec.qk_norm:
        g.add_tensor(Tensor("q_norm_w_L", (HD,), **W))
        g.add_tensor(Tensor("k_norm_w_L", (HD,), **W))
    g.add_tensor(Tensor("rope_lut", ("ctx", HD), **W))
    for name, shape in (("x_in_L", (M, E)), ("normed_L", (M, E)), ("q_L", (M, Q)), ("k_L", (M, KVd)), ("v_L", (M, KVd)),
                        ("q_n_L", (M, Q)), ("k_n_L", (M, KVd)), ("q_roped_L", (M, Q)), ("k_roped_L", (M, KVd)),
                        ("attn_out_L", (M, Q)), ("o_L", (M, E)), ("x_mid_L", (M, E)), ("ffn_normed_L", (M, E)),
                        ("gate_L", (M, H)), ("up_L", (M, H)), ("swiglu_L", (M, H)), ("down_L", (M, E)), ("x_out_L", (M, E))):
        g.add_tensor(Tensor(name, shape, **A))

    n = g.add_node
    n(Node("attn_norm_L", "rms_norm", ("x_in_L", "attn_norm_w_L"), ("normed_L",), attrs=(("eps", spec.eps),)))
    n(Node("q_proj_L", "matmul", ("normed_L", "wq_L"), ("q_L",)))
    n(Node("k_proj_L", "matmul", ("normed_L", "wk_L"), ("k_L",)))
    n(Node("v_proj_L", "matmul", ("normed_L", "wv_L"), ("v_L",)))
    if spec.qk_norm:
        n(Node("q_norm_L", "rms_norm_per_head", ("q_L", "q_norm_w_L"), ("q_n_L",), attrs=(("eps", spec.eps), ("head_dim", HD))))
        n(Node("k_norm_L", "rms_norm_per_head", ("k_L", "k_norm_w_L"), ("k_n_L",), attrs=(("eps", spec.eps), ("head_dim", HD))))
        q_pre, k_pre = "q_n_L", "k_n_L"
    else:
        q_pre, k_pre = "q_L", "k_L"
    n(Node("rope_q_L", "rope", (q_pre, "rope_lut"), ("q_roped_L",), attrs=(("convention", spec.rope_convention), ("head_dim", HD))))
    n(Node("rope_k_L", "rope", (k_pre, "rope_lut"), ("k_roped_L",), attrs=(("convention", spec.rope_convention), ("head_dim", HD))))
    n(Node("kv_append_L", "kv_append", ("k_roped_L", "v_L"), ("k_cache_L", "v_cache_L")))
    n(Node("attention_L", "attention", ("q_roped_L", "k_cache_L", "v_cache_L"), ("attn_out_L",),
           attrs=(("causal", True), ("n_heads", spec.n_heads), ("n_kv_heads", spec.n_kv_heads), ("head_dim", HD),
                  ("scale", HD ** -0.5))))
    n(Node("o_proj_L", "matmul", ("attn_out_L", "wo_L"), ("o_L",)))
    n(Node("residual_1_L", "add", ("x_in_L", "o_L"), ("x_mid_L",)))
    n(Node("ffn_norm_L", "rms_norm", ("x_mid_L", "ffn_norm_w_L"), ("ffn_normed_L",), attrs=(("eps", spec.eps),)))
    n(Node("gate_proj_L", "matmul", ("ffn_normed_L", "w_gate_L"), ("gate_L",)))
    n(Node("up_proj_L", "matmul", ("ffn_normed_L", "w_up_L"), ("up_L",)))
    n(Node("swiglu_L", "swiglu", ("gate_L", "up_L"), ("swiglu_L",)))
    n(Node("down_proj_L", "matmul", ("swiglu_L", "w_down_L"), ("down_L",)))
    n(Node("residual_2_L", "add", ("x_mid_L", "down_L"), ("x_out_L",)))
    return g


# The two golden models (doc 56 H0), from their ARCHITECTURE.md.
QWEN3_0_6B = ModelSpec(
    name="qwen3_0_6b", hf_id="Qwen/Qwen3-0.6B", n_layers=28, emb_dim=1024, n_heads=16, n_kv_heads=8,
    head_dim=128, hidden_dim=3072, vocab_size=151936, qk_norm=True, tied_embeddings=True,
    rope_theta=1_000_000.0, eps=1e-6, lm_head_rows_per_launch=0,
    # `[2026-08-23]` derived: the driver ships the planner's own partitioning
    # (qwen3_0_6b_decode.lm_head_parts, which applies the same BD-repeat rule).
    # `[2026-08-26]` queue item 28: that rule now runs at LM_HEAD_HERD_ROWS = 4
    # core rows (plan.py), so it yields 2 x 65536 + 20864 -- 3 launches, not 10.
    # No field is added here on purpose: `asdict(spec)` is inside the plan's
    # hashed body, so a new spec field would move EVERY model's plan sha,
    # including models this item does not touch.
)
LLAMA32_1B = ModelSpec(
    name="llama32_1b", hf_id="meta-llama/Llama-3.2-1B-Instruct", n_layers=16, emb_dim=2048, n_heads=32,
    n_kv_heads=8, head_dim=64, hidden_dim=8192, vocab_size=128256, qk_norm=False, tied_embeddings=True,
    rope_theta=500_000.0, eps=1e-5, lm_head_rows_per_launch=16384,    # lm_head_gemv_multi default n_part
)
# `[2026-08-26]` doc 56 H2a (queue item 17): the int4 sibling. Same architecture
# as LLAMA32_1B; what differs is the checkpoint (AMD's AWQ uint4 asym gs=128,
# bf16 lm_head) and the precision plan the driver ships (`w4_decode`: int4
# storage for the decode GEMVs, bf16 compute/accumulate, bf16 prefill weights
# resident separately -- doc 56 section 3.5). `weight_dtype` stays bf16: it is
# the GRAPH's storage default (embed / norms / lm_head / the dequant prefill
# copy ARE bf16); the decode stages' int4 storage is the w4_decode plan's,
# expressed in `fuse` (plan.py), not a per-tensor rewrite H2b will do properly.
LLAMA32_1B_INT4 = ModelSpec(
    name="llama32_1b_int4", hf_id="amd/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead",
    n_layers=16, emb_dim=2048, n_heads=32, n_kv_heads=8, head_dim=64, hidden_dim=8192,
    vocab_size=128256, qk_norm=False, tied_embeddings=True, rope_theta=500_000.0, eps=1e-5,
    lm_head_rows_per_launch=16384,    # the driver's _LM_N_PART (8 x 16384, bf16 head)
)
