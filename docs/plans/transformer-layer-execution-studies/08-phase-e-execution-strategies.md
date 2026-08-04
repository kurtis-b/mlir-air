# 08 — Phase E: The Four Execution Strategies

Build the four points of the taxonomy defined in
[03-measurement-model.md](03-measurement-model.md), over one shared golden model.

## Layout

```
programming_examples/transformer_layer/pattern/
├── reference.py      # shared torch golden model (encoder_bert + decoder_gpt2)
├── offload/          # host owns the layer; 8 GEMMs offloaded individually
├── runlist/          # fine-grained operator sequence, one runlist
├── coarse/           # few fused kernels, one runlist  (iron's `hybrid`)
└── fused/            # MLIR-level fusion via stitch_elf
```

Each strategy directory holds its builder module, a `README.md`, and tests. Per convention rule
8, the per-strategy `reference.py` re-export shims are **not** ported — import the shared one.

Per convention rule 7, the directory is `coarse`; only the CSV value is `hybrid`.

## The shared reference

`pattern/reference.py` (172 lines, pure torch) ports verbatim. It provides `encoder_bert` and
`decoder_gpt2` variants and an `include_output=False` escape hatch so the 16384-token ladder
stays tractable. It is the correctness anchor for all four modes, and was already used as the
Phase D gate.

## Build order

**`offload` first.** It is host-torch plus 8 single-GEMM dispatches — the least new machinery,
and it produces the first real CSV row, which exercises the whole measurement path end to end
before the harder modes land.

The eight offloaded GEMMs:

```
q_proj  k_proj  v_proj  attn_scores  attn_output  output_proj  up_proj  down_proj
```

Everything between them — reshapes, softmax, scaling, masking, normalization, residuals — stays
in torch on the host. Port `_blocked_attention` and `_resolve_query_block_size`: above a scratch
threshold the operator switches to blocked attention over query blocks, because long sequences
cannot materialize the full attention score matrix. iron's constants are
`MAX_ATTENTION_SCRATCH_BUFFER_BYTES = 3 GiB` and `MIN_BLOCKED_QUERY_BLOCK_SIZE = 256`.

Note `offload` shares its query-blocking logic with `runlist` in iron; keep that sharing so both
block attention identically.

**`runlist` and `coarse` next.** Both depend on Phase B's runlist aggregation.

- `runlist` — iron uses 29 kernels and 42 runlist entries over fine-grained operators (GEMM,
  transpose, softmax, elementwise-mul, causal-mask, GeLU, LayerNorm, add-and-norm,
  elementwise-add).
- `coarse` — iron uses 12 runlist entries over 5–6 fused kernels (`qkv_proj`, `mha_out_proj`,
  `ffn`, `addnorm`, `layer_norm`, `elementwise_add`).

**`fused` last.** A thin wrapper over existing `stitch_elf` usage — this is MLIR-AIR's
production mechanism, and measuring it is what makes the port additive rather than duplicative.

## What is not ported

iron reaches one-xclbin-many-kernels through `aiecc --xclbin-input` incremental merge, chaining
`XclbinArtifact.xclbin_input` with `--xclbin-instance-name` and `--xclbin-kernel-id 0x801,
0x802, …` so each new kernel merges into the incoming xclbin.

MLIR-AIR does not need this. Multiple independently-compiled ELFs bind as modules into one
`hw_context` (Phase B). Do not reproduce the merge.

## Instrumentation

Every mode reports the full **dispatch vector** from
[03-measurement-model.md](03-measurement-model.md) — host submissions, runlist entries, AIR
launches, herd launches, sync boundaries, bytes transferred.

Critically, there must be **one implementation** of each field that all four modes call. A
per-mode reimplementation of "what counts as a submission" would make the comparison
meaningless.

## Work items

1. Port `pattern/reference.py` verbatim (already done as part of Phase D).
2. `offload/` — host-torch layer with 8 GEMM dispatches, shared query blocking.
3. `runlist/` — fine-grained operator sequence over the Phase B aggregation.
4. `coarse/` — fused-kernel sequence over the same.
5. `fused/` — `stitch_elf` wrapper.
6. Wire the shared dispatch-vector instrumentation into all four.
7. Per-strategy `README.md` explaining what boundary it isolates and what it costs.
8. Equivalence tests across all four against the shared reference.

## Gate

Two conditions:

1. **Equivalence** — all four modes agree with the torch reference on identical weights.
2. **Distinguishability** — the dispatch vector differs across them as the taxonomy predicts:
   `offload` with 8 submissions and 8 sync boundaries; `runlist` with 1 submission and many
   entries; `coarse` with 1 submission and few entries; `fused` with few submissions and many
   AIR launches at near-zero intermediate sync.

If the vectors do not separate the modes, the taxonomy is not measuring what it claims and the
measurement model needs revisiting before Phase F consumes it.

## Risks

- **`offload` is intrinsically noisy** — roughly ten times the run-to-run drift of the other
  modes, and an XRT version change alone has moved it 19–39% at `seq_len >= 4096` while leaving
  the others within 0.6%. This is a property of host-mediated dispatch, not a bug. The
  comparator's wider tolerances for it must be preserved.
- If Phase B's multi-ELF runlist assumption failed, `runlist` and `coarse` cannot be built as
  specified and this phase must be rescoped. **It did not** — one `hw_context` per ELF and one
  runlist across them is bit-identical to sequential dispatch and measurably faster
  ([05a §5](05a-phase-b-runlist-spike-result.md)). What is still unmeasured is how many
  concurrent `hw_context`s NPU2 grants: three is confirmed, `runlist` wants 29. If that binds it
  binds loudly — `xrt.hw_context` raises at load time rather than returning wrong numbers — so
  the rescope would be forced by an exception, not discovered in the results.
