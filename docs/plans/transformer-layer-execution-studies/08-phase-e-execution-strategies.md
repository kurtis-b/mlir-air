# 08 — Phase E: The Four Execution Strategies

Build the four points of the taxonomy defined in
[03-measurement-model.md](03-measurement-model.md), over one shared golden model.

`[2026-08-05]` Re-anchored against what Phase D actually produced. Four things below decide how
this phase starts; read them before planning anything.

## What Phase D left you, and what it did not

Phase D built one `encoder_bert` layer at `baseline_768`, `seq = 4096`, matching an FP32 torch
oracle over its whole output with zero mismatches. Do not rebuild any of it; the example's own
`programming_examples/transformer_layer/README.md` is the authoritative inventory.

| Piece | Where | What it means here |
|---|---|---|
| The shared golden model | `pattern/reference.py` | **Import it.** Both workload variants, `fuse_qkv_weight()`, per-boundary helpers, `WEIGHT_DRAW_ORDER`, and the `include_output=False` escape hatch |
| Its composition check | `pattern/test_reference.py` | Seven host-only tests pinning erf-vs-tanh GeLU, post-add-vs-pre-add residual, and QKV column order — the three substitutions a numerical comparison would survive |
| A working `coarse` | `builders/block.py` | `block_config()`, `run_block()`, `describe_block()`, `BLOCK_BOUNDARIES`, over four `KernelCache.run_sequence` calls |
| The dispatch vector | `KernelCache.run_sequence` returns one; the block records four | The instrumentation already exists and is already proven on hardware |

### `coarse` is mostly built

`builders/block.py` **is** a fused-operator sequence over one runlist, which is what this document
calls `coarse`. This phase's job there is to give it a strategy directory and route it through the
shared instrumentation — not to write it again. Budget accordingly: `coarse` is the cheapest of the
four now, not the second-hardest.

### The dispatch vector Phase D measured, and the surprise in it

The block records one vector per sequence. In order — qkv+mha, norm 1, ffn, norm 2:

```
host_submissions  runlist_entries  air_launches  herd_launches  sync_boundaries      bytes
       1                2               6             10              9          80,216,064
       1               64               1             64            193          18,875,904
       1                1               4              8              7          84,934,656
       1               64               1             64            193          18,875,904
```

**The two 64-entry rows are the normalization points, and they are 64 dispatches each, not one.**
`build_addnorm_module` requires `rows == herd_x * rows_per_call`, which at `cols = 768` caps a call
at 64 of the layer's 4096 rows (`builders/block.py::norm_rows` derives it; do not carry a row count
across widths). So `coarse`'s dispatch numbers are dominated by `addnorm`, not by the GEMMs.

That is a real result about where the cost sits, and the taxonomy has to be able to explain it. It
also means the `coarse`-versus-`runlist` distinction is far narrower than iron's "12 entries versus
42" suggests: **128 of `coarse`'s 131 entries — 98% — are one operator's row blocking**, before any
fine-grained mode is built at all.

## The ladder is blocked, and clearing it is this phase's job

**Everything so far runs at `seq = 4096` only.** Two collisions, one root cause, in
`llms/shared/builders/gemm_builder.py`: symbol and object names are minted from the GEMM *method*
alone and ignore `tile_n`.

| Where it bites | Symptom |
|---|---|
| `stitch_elf` | Two same-method GEMMs at different `tile_n` declare `f32_to_bf16_mn_<suffix>` twice with different memref types — `redefinition of symbol named ...`. This is why `build_ffn_module` builds at no `baseline_768` point except 4096, where the registry happens to put its two projections on different methods. |
| `compile_gemm_mm` | The object is named from the method (`mm_m32.o` / `mm_m64.o`) while `tile_n` is baked in as `-DDIM_N`, so the FFN's up-projection (`drain`, 128) and the o-projection (`drain`, 96) **write the same file** and one silently gets the other's micro-kernel. D2 works around it by interleaving inside `builders/block.py`; any caller that builds several of these together without interleaving hits it again, silently and with no diagnostic. |

One `(method, tile_n)`-aware naming change closes both. Phases C and D were forbidden from
touching that file. **Phase E needs the sequence ladder, so Phase E is the phase that makes the
change** — and that pulls the cross-deployment regression rule into its gate: `make verify` over
all ten shipped LLM deployments, serialized under `flock`. See
[13](13-verification-and-acceptance.md#the-cross-deployment-regression-rule). `gate-c4.sh` is the
model for a gate with that second leg.

Do this **first**. Every mode below wants more than one sequence length, and until it lands there
is exactly one point on the ladder to measure.

## The tolerance has no headroom

The layer's `atol` sits at the hard `1e-1` ceiling with a 1.35x margin over its measured
`atol_required` of 7.4e-2. The cause is output scale rather than error — `mean_rel_L1` is 1.7e-2,
in line with the per-operator rows — but this phase compares **four** modes against that same
oracle. `rtol` is pinned at `1.6e-2` and the driver rejects any `atol` above `1e-1`.

If a mode does not fit, that is a finding to report, not a tolerance to widen. A mode that needs
more room than the block did is telling you its arithmetic differs from the block's, which is
either a defect or a genuine property of that execution boundary — and either is a result.

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

`[2026-08-05]` **It exists. Import it, do not port it.**
`programming_examples/transformer_layer/pattern/reference.py` is the FP32 re-expression Phase D2
produced: iron's structure and its load-bearing `WEIGHT_DRAW_ORDER`, computed in FP32 from
bf16-rounded inputs rather than iron's bf16-everywhere. It provides `encoder_bert` and
`decoder_gpt2` variants, `fuse_qkv_weight()`, per-boundary helpers, and the `include_output=False`
escape hatch that keeps the 16384-token ladder tractable. It is the correctness anchor for all four
modes and it already gated Phase D.

`pattern/test_reference.py` is its independence check: seven host-only tests pinning the
composition against a straight-line transcription. Keep them passing — they are what stops the
oracle drifting into agreement with the device, and they cover the three substitutions a purely
numerical comparison would survive.

## Build order

`[2026-08-05]` **Before any of them: the `gemm_builder.py` naming fix above**, with the ten-model
regression check. Everything below wants more than one sequence length.

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
  `ffn`, `addnorm`, `layer_norm`, `elementwise_add`). **`[2026-08-05]` This one is already built**
  as `builders/block.py`; it needs a directory and the shared instrumentation, not a rewrite. Its
  measured shape is 4 sequences and 131 runlist entries, not 12, because each of the two
  normalization points is 64 dispatches — see the table above before treating iron's count as the
  target.

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

0. **`[2026-08-05]` Unblock the ladder**: `(method, tile_n)`-aware symbol *and* object naming in
   `llms/shared/builders/gemm_builder.py`, with `make verify` over all ten shipped models. Nothing
   below can walk the sequence ladder until this lands, and `builders/block.py`'s interleaving
   workaround can then be removed.
1. Import the FP32 `pattern/reference.py` Phase D produced. Do not re-port iron's bf16 original,
   and keep `pattern/test_reference.py` passing.
2. `offload/` — host-torch layer with 8 GEMM dispatches, shared query blocking.
3. `runlist/` — fine-grained operator sequence over the Phase B aggregation.
4. `coarse/` — **give `builders/block.py` a strategy directory and the shared instrumentation.**
   It already is this mode; do not write a second one.
5. `fused/` — `stitch_elf` wrapper.
6. Wire the shared dispatch-vector instrumentation into all four. The vector already exists —
   `KernelCache.run_sequence` returns one and the block records four — so this is unification, not
   invention.
7. Per-strategy `README.md` explaining what boundary it isolates and what it costs.
8. Equivalence tests across all four against the shared reference, at the pinned `rtol = 1.6e-2`
   and an `atol` no greater than `1e-1`.

## Gate

Two conditions:

1. **Equivalence** — all four modes agree with the torch reference on identical weights.
2. **Distinguishability** — the dispatch vector differs across them as the taxonomy predicts:
   `offload` with 8 submissions and 8 sync boundaries; `runlist` with 1 submission and many
   entries; `coarse` with 1 submission and few entries; `fused` with few submissions and many
   AIR launches at near-zero intermediate sync.

If the vectors do not separate the modes, the taxonomy is not measuring what it claims and the
measurement model needs revisiting before Phase F consumes it.

`[2026-08-05]` **Calibrate "few" against what `coarse` actually measures**, not against iron's
counts: 4 submissions and 131 runlist entries for the layer, because the two normalization points
are 64 dispatches each. If `runlist` is expected to have "many" entries and `coarse` "few", those
words have to survive `coarse` already having 131 — otherwise the predicted separation is between
a number and itself. Decide what the vector's discriminating fields actually are before building
the modes, because that decision is the gate.

This is also the natural shape for the driver's objective check: the four modes' recorded vectors
either differ in the predicted directions or they do not, and that is checkable from the artifacts
without trusting anything the session writes.

## Risks

- **`offload` is intrinsically noisy** — roughly ten times the run-to-run drift of the other
  modes, and an XRT version change alone has moved it 19–39% at `seq_len >= 4096` while leaving
  the others within 0.6%. This is a property of host-mediated dispatch, not a bug. The
  comparator's wider tolerances for it must be preserved.
- If Phase B's multi-ELF runlist assumption failed, `runlist` and `coarse` cannot be built as
  specified and this phase must be rescoped. **It did not** — one `hw_context` per ELF and one
  runlist across them is bit-identical to sequential dispatch and measurably faster
  ([05a §5](05a-phase-b-runlist-spike-result.md)).
- **The `hw_context` ceiling is 32 on this device — measured, not assumed.** `runlist` wants 29,
  so it fits with three to spare. Probed by opening contexts while holding every prior one until
  XRT refused: contexts 1-32 succeeded, 33 failed with
  `RuntimeError: DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-2)`. That confirms the failure
  is loud and at load time, so a future overrun surfaces as an exception rather than as wrong
  numbers.

  Two caveats on the margin. The probe cycled **4 distinct ELFs** to reach 32 contexts, so what
  is demonstrated is a limit on concurrent *contexts*, not on 29 *distinct* ELFs; if the ceiling
  turns out to depend on per-ELF resources rather than context count alone, 29 distinct designs
  could bind sooner. And three spare is thin — anything else holding a context on the device
  concurrently (another example, a stray process, a future mode wanting one more kernel) eats the
  margin. Re-probe with the real 29 artifacts before relying on it.
