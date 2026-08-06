# 08 — Phase E: The Four Execution Strategies

Build the four points of the taxonomy defined in
[03-measurement-model.md](03-measurement-model.md), over one shared golden model.

`[2026-08-05]` Re-anchored against what Phase D actually produced. Four things below decide how
this phase starts; read them before planning anything.

## `[2026-08-05]` Outcome: the four modes, measured

All five sub-phases passed gate, objective and tamper checks — 24 of 60 invocations, ~8.5 hours.
The layer computes identically in all four modes (full 4096×768 output, zero mismatches, ten clean
per-boundary stages each) and the dispatch vectors are:

| mode | submissions | entries | air launches | herd launches | sync boundaries | bytes |
|---|---|---|---|---|---|---|
| `offload` | 6 | 6 | 7 | 19 | 19 | 139,984,896 |
| `runlist` | 5 | 391 | 14 | 404 | 403 | 165,347,328 |
| `coarse` | 4 | 131 | 12 | 146 | 402 | 202,902,528 |
| `fused` | 1 | 3 | 16 | 24 | 19 | 184,025,088 |

`fused` is the clearest result in the table: it collapses `coarse`'s 402 sync boundaries to 19 and
its 131 runlist entries to 3, while carrying *more* AIR launches (16 against 12). That is precisely
the signature the taxonomy predicts for MLIR-level fusion — the work moves below the runlist, into
the ELF — and it is the one mode whose numbers were not predicted in advance by anything.

Both non-gating predictions also held (`fused` entries below `coarse`, `fused` AIR launches at or
above it).

### Three of this plan's four predicted vectors were wrong

[03](03-measurement-model.md) predicted `offload` at 8 submissions and 8 sync boundaries,
`runlist` at 1 submission with ~29 entries, and `coarse` at 1 submission with ~6 entries. Measured:
6/19, 5/391, and 4/131. The causes are all recorded and all real — attention leaving the device for
`offload`, a whole-BO dispatch argument forcing multiple submissions, and `build_addnorm_module`'s
64-row L1 cap dominating everything. **Phase F consumes these numbers; take them from this table,
not from 03's prose.**

### One gating clause stopped being a test, and that is a finding

**`runlist.runlist_entries > coarse.runlist_entries` is now true by construction, not by
measurement**, and `pattern/runlist/README.md` says so in as many words.

The sequence matters. E4's first structure measured 13 entries over 2 runlists — below `coarse` —
and the session's first response was the one [08d](08d-phase-e4-runlist.md) asks for: a commit
documenting why no faithful decomposition exceeds 131. Review round 2 then raised a blocking
finding titled *"The implementation knowingly fails the mandatory E4 ordinal gate"* — citing the
gate criteria rather than a defect — and the fix restructured the mode to 391.

The restructuring is defensible on its merits, and arguably better science: the streaming structure
varied operator granularity *and* dispatch schedule at once, and at the normalization points it was
64× **coarser** than `coarse`. Holding the schedule fixed (band size imported from
`builders.block.norm_rows`, never tuned) and subdividing each unit isolates granularity, which is
the variable being compared. The arithmetic is unchanged.

But once `runlist` is *defined* as "`coarse`'s schedule, subdivided", the inequality cannot fail.
Two lessons:

- **A gate criterion stated in the gate description is visible to the reviewer, and a reviewer will
  optimize for it.** `phase_gate_description` for E2–E5 spells out the ordinal thresholds *and*
  says "report the number; do not inflate the decomposition". The reviewer acted on the threshold
  and ignored the caveat. Either withhold the numeric criterion from the description, or make the
  caveat a blocking instruction to the reviewer rather than advice to the session.
- **Clause 3 should be replaced** for Phase F's purposes. `entries` ratio at a *fixed* schedule is
  a definition; what would actually discriminate is a field neither mode controls by construction —
  `herd_launches` (404 against 146) is the candidate, since it counts executed work rather than
  dispatch packaging.

### `runlist` cannot be one runlist on this hardware

Its premise — "1 submission over many entries" — is not reachable. It measured **5 submissions**,
because a host stage between the projections and the output projection forces at least two before
banding restages add more, and because re-executing one GEMM ELF inside a single runlist corrupts,
which forces per-projection ELFs against the 32-context ceiling. Recorded in
`pattern/runlist/README.md`.

## This document is the overview; the work is five sub-phases

`[2026-08-05]` Split, for the reason C and D were and one more: `PL_STEP_TIMEOUT` caps an implement
session at three hours, and E1 carries the ten-model regression check that cost C4's gate hours.
**Each sub-phase has its own document, and that document is its session's entire task list** — the
implement prompt injects it whole, so five sessions sharing this one would each try to do
everything.

| Sub-phase | Document | What it lands |
|---|---|---|
| E1 | [08a](08a-phase-e1-unblock-the-ladder.md) | The `(method, tile_n)` naming fix, a second ladder point, the two over-cap module splits |
| E2 | [08b](08b-phase-e2-coarse-and-instrumentation.md) | `coarse` as a strategy directory, and the artifact contract the other three are measured against |
| E3 | [08c](08c-phase-e3-offload.md) | `offload` |
| E4 | [08d](08d-phase-e4-runlist.md) | `runlist`, plus the two operators that do not exist yet |
| E5 | [08e](08e-phase-e5-fused-and-distinguishability.md) | `fused`, and the four-mode distinguishability gate |

### Four decisions taken before any code, because the harness enforces them

1. **`coarse` wraps `builders/block.py`**, it does not re-home it. See [08b](08b-phase-e2-coarse-and-instrumentation.md).
2. **`pattern/<mode>/`**, with a separate `KernelCache` directory per mode — a shared one lets two
   modes trade fingerprint-matching ELFs and misattribute a dispatch vector.
3. **Distinguishability is ordinal over driver-summed totals, never absolute thresholds.** Four
   gating clauses, two predictions recorded but not halting. See §Gate.
4. **`offload`'s attention stays in host torch**, six GEMM dispatches rather than eight. See
   §Build order — the option this document previously offered is not available.

### Two claims this document made that the code does not support

- **`stitch_elf` and `compile_gemm_mm` are not in `gemm_builder.py`.** They are
  `llms/shared/infra/stitching.py:318` and `llms/shared/infra/external_kernels.py:133`. The fix
  still lands in `gemm_builder.py` — it holds the suffix/object table and `gemm_method_spec()` is
  its only selector — and the blast radius is *smaller* than feared: `gemm_method_spec` has no
  external callers, and the one file importing it by name
  (`llms/qwen25_0_5b/qwen25_0_5b_prefill.py:61`) never calls it.
- **A `transpose` example does exist**, in three variants under `data_transfer_transpose/`
  (`dma/`, `channel/`, `dma_bf16/`, the last with a `transpose.cc`). What does not exist is a
  builder or a registry row. `elementwise_mul` genuinely does not exist in any form.

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
also means the `coarse`-versus-`runlist` distinction is far narrower than iron's real 5 versus 16 (not the "12 versus
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

**It also blocks `fused` outright, not just the ladder.** A whole-layer `stitch_elf` has to
co-link the layer's four projection GEMMs, and the block's own resolved specs are:

```
qkv       fused-cast  tile_n = 96        ffn_down  fused-cast  tile_n = 96
ffn_up    drain       tile_n = 128       o_proj    drain       tile_n = 96
```

Two `drain` GEMMs at different `tile_n` in one module is exactly the redefinition `stitch_elf`
rejects. So mode 4 cannot be built at `seq = 4096` either — at any sequence length — until the
naming fix lands. Do this **first**: it is a hard prerequisite for one of the four modes and for
every mode's second data point.

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

`[2026-08-05]` **What Phase D actually built does not match that tree**, and the difference is a
decision this phase has to take rather than drift into. `pattern/` holds `__init__.py`,
`reference.py` and `test_reference.py` — the oracle and its own tests, no strategy code. The
assembled layer is `builders/block.py` with `builders/block_cache.py` beside it, its tests are
enrolled as `.lit` files at the example root, and there is one example-level `README.md` rather
than one per strategy. Decide whether `coarse/` re-homes `builders/block.py` or wraps it, and
whether four per-strategy READMEs are worth it against the example README the sessions have been
maintaining. Say which in the phase's own document before writing code.

Note also that D2's split — oracle in `pattern/`, builder in `builders/` — is a deliberate
departure from [convention rule 4](02-porting-conventions.md) ("the reference oracle lives in the
same file as the builder"), because one oracle is shared by four modes. That exception is recorded
in the example's README but not in rule 4, so a reviewer applying the checklist to Phase E's code
will flag it. Expect that, and answer it rather than restructuring around it.

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

`[2026-08-05]` **Two corrections to that ordering.** After Phase D, `coarse` has the least new
machinery — it is built and gated on hardware, and `offload` is not. And `offload` has a
prerequisite nothing else does, below.

The eight offloaded GEMMs:

```
q_proj  k_proj  v_proj  attn_scores  attn_output  output_proj  up_proj  down_proj
```

`[2026-08-05]` **Two of those eight do not resolve, and this is `offload`'s alone.** At
`baseline_768`, `seq = 4096`:

```
attn_scores   4096 x   64 x 4096   ->  gemm_config() raises
attn_output   4096 x 4096 x   64   ->  gemm_config() raises
```

Neither shape is in `kernel_registry` — there is no `K = 64` or `N = 64` bf16-out row anywhere in
it (minimum K is 512, minimum N is 128 across all 69 shapes). [06](06-phase-c-operators.md) says
"the attention GEMMs go through FlashAttention rather than `gemm_builder` and need no GEMM registry
row", which is true for `coarse` and `fused`, where attention is inside `mha_out_proj`. It is
**false for `offload`**, the one mode whose whole premise is dispatching them as standalone GEMMs.

**`[2026-08-05]` Decided: attention stays in host torch, and `offload` dispatches six GEMMs.**

```
q_proj  k_proj  v_proj  output_proj  up_proj  down_proj
```

This document previously offered two options — sweep the two shapes into the registry, or route
attention through FlashAttention. **The first is not available.** `sweep/sweep_families.py` derives
K and N from `FAMILY_HIDDEN × ROLE_KN_MULTIPLES`, with a minimum hidden of 512, and M from
`SEQ_LADDER`; no `--family` can stage a 64 in the K or N position. `attn_scores` would additionally
need `K = 64` against a minimum `tile_k_l2` of 256, which does not tile.

Host torch is chosen over a FlashAttention dispatch because it is what this document already
prescribes for everything between the GEMMs, and because it is numerically the safest of the three
against an `atol` with no headroom — host FP32 attention lands closer to the FP32 oracle than the
device path does. The cost is honest and must be recorded: `offload` becomes a *hybrid* boundary
rather than a pure per-GEMM device implementation. The artifact carries `attention_path` and the
mode's README says so, and the distinguishability gate asks for an ordering rather than the number
eight. See [08c](08c-phase-e3-offload.md).

One rule that decision makes load-bearing: **the mode computes and the oracle checks, and they may
not share arithmetic.** `offload` does more host math than any other mode, and calling
`pattern/reference.py`'s per-boundary helpers to do it would compare a value against itself.

Everything between them — reshapes, softmax, scaling, masking, normalization, residuals — stays
in torch on the host. Port `_blocked_attention` and `_resolve_query_block_size`: above a scratch
threshold the operator switches to blocked attention over query blocks, because long sequences
cannot materialize the full attention score matrix. iron's constants are
`MAX_ATTENTION_SCRATCH_BUFFER_BYTES = 3 GiB` and `MIN_BLOCKED_QUERY_BLOCK_SIZE = 256`.

Note `offload` shares its query-blocking logic with `runlist` in iron; keep that sharing so both
block attention identically.

**`runlist` and `coarse` next.** Both depend on Phase B's runlist aggregation.

- `runlist` — **`[2026-08-05]` iron uses 12 kernels and 16 runlist entries** (encoder; 13/17 decoder), not the 29/42 earlier drafts of this document claimed. The operator families are right (GEMM,
  transpose, softmax, elementwise-mul, causal-mask, GeLU, LayerNorm, add-and-norm,
  elementwise-add). **`[2026-08-05]` Two of those operators do not exist in MLIR-AIR at all**:
  there is no `transpose` or `elementwise_mul` builder or example anywhere in
  `programming_examples/`. [01](01-port-inventory.md) lists `transpose/design.py` and
  `elementwise_mul/design.py` among the iron files needing the same treatment as the rest, but
  assigns them to no phase. They are `runlist`'s, and they are new device work rather than
  re-expression — the only new device work left in this phase. Budget for it, and re-derive the
  entry count at `baseline_768` rather than carrying iron's 42 across.
- `coarse` — **`[2026-08-05]` iron uses 5 runlist entries over 5 fused kernels** (encoder; 7 over 6 decoder). The "12" is both variants summed (`qkv_proj`, `mha_out_proj`,
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

`[2026-08-05]` **That implementation already exists.** Phase B built
`DispatchVector` in `programming_examples/llms/shared/infra/dispatch.py`, with the per-field
semantics written down there; `KernelCache.run_sequence` returns one, and `builders/block.py`
already emits four into the block's results artifact. So this work item is *wiring four modes into
an existing implementation* — and the rule above becomes a prohibition rather than a design task:
no mode gets its own counting.

The block's measured totals, as a calibration point for the gate below — four sequences summed:

```
host submissions  4     runlist entries  131     air launches   12
herd launches   146     sync boundaries  402     bytes  ~203 MB
```

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
   and an `atol` no greater than `1e-1`. Route them through `opcheck.py`'s `dispatch` seam, which
   D2 added for operators that are several ELFs rather than one module: such an operator's verdict
   is the **conjunction** of its end-to-end and per-boundary comparisons, and it is what the
   driver's objective check reads.
9. **`[2026-08-05]` Split `opcheck_specs.py` before adding to it.** It is **1043 lines** against
   the ~800 cap that [00](00-context-and-goals.md), [02](02-porting-conventions.md),
   [06a](06a-phase-c1-gate-and-small-operators.md) and [13](13-verification-and-acceptance.md) all
   gate on — a live violation no document recorded until now. D1 predicted it and named the seam:
   per-operator `_prepare_*` functions in one module, the `SPECS` catalogue in another, the same
   mechanism-versus-catalogue split `opcheck.py`/`opcheck_specs.py` and
   `registry_sweep.py`/`sweep_families.py` already draw. Phase E adds four modes' specs to that
   file, so it gets worse before it gets better. `sweep/registry_sweep.py` is also over, at 866.

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
a number and itself.

`[2026-08-05]` **Decided, and implemented in `agents/scripts/port-loop/phase_e_checks.py` before
any mode is built.** The criterion is **ordinal over driver-summed totals**, never an absolute
threshold — a threshold here would be measuring `build_addnorm_module`'s L1 capacity rather than
the taxonomy. The driver sums each mode's recorded vectors itself and prints the full four-by-six
table whether it passes or fails.

Four clauses gate, each following from the definition of the boundary rather than a guess at an
implementation:

1. **Distinctness.** No two modes' six-field totals are equal. If two modes share a vector, the
   vector is not measuring the boundary.
2. **`offload` is the host-mediated extreme:** its `host_submissions` strictly exceeds every other
   mode's, and `runlist_entries == host_submissions` — it aggregates nothing.
3. **`runlist` is finer than `coarse`:** `runlist.runlist_entries > coarse.runlist_entries`.
4. **`fused` removes intermediate host sync**, which is what MLIR-level fusion *is*:
   `fused.sync_boundaries < coarse.sync_boundaries`.

Two further predictions are **recorded with a verdict but do not halt**:
`fused.runlist_entries < coarse.runlist_entries`, and `fused.air_launches >= coarse.air_launches`.
Both assume a particular fused decomposition — a faithful whole-layer stitch may still row-block
its normalization, and `air_launches` is counted once per distinct ELF while `herd_launches`
accumulates per step ([03](03-measurement-model.md), and `dispatch.py:122-153`), which makes the
second weaker than it looks. If either is false, that is a finding for the mode's README.

This is also the natural shape for the driver's objective check: the four modes' recorded vectors
either differ in the predicted directions or they do not, and that is checkable from the artifacts
without trusting anything the session writes.

`[2026-08-05]` **And the vectors themselves now have a negative control.** `results/` is gitignored,
so a fabricated `dispatch_vectors` block is invisible to `guard_fingerprint`, `guard_check_tamper`
and every Codex diff; freshness was the only barrier, and no phase before E noticed. The driver
already re-runs each operator under `--fault-inject input` and requires failure — Phase E
additionally requires that run's summed totals to **equal** the clean run's. Injection perturbs one
input element after the reference exists and does not touch the dispatch path, so on an honest run
they are identical; D2's block clean and fault artifacts both total
4 / 131 / 12 / 146 / 402 / 202,902,528. A session cannot produce those six numbers without
dispatching.

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
