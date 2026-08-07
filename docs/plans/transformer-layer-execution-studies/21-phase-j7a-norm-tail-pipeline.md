# 21 — Phase J7a: the norm-tail pipeline


The first piece of iron's dataflow form built on this port, and the smallest. Three herds in one
segment joined by L1→L1 channels, replacing `fused`'s decomposed norm tail — which stages bf16
through **L3** between every stage and pays for it in precision.

## Why this one first

- **It has a measured target rather than a subjective one.** `fused` measures `mean_rel_L1`
  1.806e-2 against the block's 1.688e-2, and its own README names the cause: the tail decomposes
  `addnorm` into `elementwise_add` → `layer_norm` → `elementwise_mul` and rounds to bf16 through L3
  between each launch. Keeping those intermediates resident is the win, and it is checkable.
- **It is iron's `AIEAddAndNorm`**, a two-worker pipeline — the same shape, derived rather than
  placed by hand.
- **It is builder-only** and blocked on nothing.
- **It is the smallest place to prove the L1→L1 channel path** before the harder `mha_out_proj`
  pipeline (J7c) depends on it.

## The design, and the one constraint that decides it

```
herd A   stage_add     x|residual packed, ONE strided L3 fetch   -> Channel AtoB (L1->L1)
herd B   stage_norm    AtoB                                      -> Channel BtoC (L1->L1)
herd C   stage_scale   BtoC + gamma (one L3 fetch)               -> L3 out
```

**A column has two shim MM2S channels, and the budget is per column across the whole segment.**
Three 8-wide herds stack one tile of each into every column, so the column's L3 demand is the sum
over all three stages. Exceed two and AIR packet-multiplexes onto one queue — which until H9 lands
is the path that **silently miscompiles** past one trip on more than one column.

Measured at 4096×768, `herd_x=8`, 64 trips per tile:

| arrangement | packet-typed channels | places? |
|---|---|---|
| x, residual, gamma each streamed | **3 — packet path entered** | yes |
| **x\|residual packed**, gamma second | **0 — defect unreachable** | yes |

So **pack x and residual into one L3 buffer** and fetch both in one strided DMA. This is not a
workaround: `builders/addnorm.py`'s docstring proposes exactly it, and it is why J1's L2-staged
weight failed — 8 columns × 2 channels = 16 shim MM2S, already full before a third stream, and
staging through L2 cannot conjure a 17th port.

The declared `AtoB`/`BtoC` edges are **not** packet-typed and cost nothing from that budget. Only
L3-facing edges are budgeted.

## How the packing actually works — measured, including its one open risk

`[2026-08-07 00:55]` Probing the mechanism rather than assuming it, because the obvious form does
not work.

**Pack as PLANES, not wide.** Two candidates both collapse x and residual to one shim channel:

| packing | L3 | one DMA | L1 tile | existing kernels usable? |
|---|---|---|---|---|
| `wide` | `[rows, 2*cols]` | 2-D | `[rpc, 2*cols]` | **no** — x and residual interleaved *within* each row |
| **`planes`** | `[2, rows, cols]` | 3-D, `strides=[rows*cols, cols, 1]` | `[2, rpc, cols]` | **yes** — each plane is a contiguous `[rpc, cols]` tile |

Host side is then just `np.stack([x, residual])`.

**The subview into plane 1 hits H7's wall, one level down.** Taking `memref.subview %pk[1, 0, 0]`
to get the residual tile produces `memref<8x768xbf16, strided<[768, 1], offset: 6144>, 2>` — and
that will not cast to the identity `memref<8x768xbf16, 2>` a kernel signature normally declares.
Same offset-subview problem as H7, inside a herd on an L1 buffer rather than at a launch argument.

**The way through is to declare the strided types on the callee.** Give the external function's
operands `strided<[cols, 1], offset: 0>` and `strided<[cols, 1], offset: rpc*cols>` instead of the
identity layout. Measured: the module then lowers cleanly — one input channel, one output channel,
**zero packet-typed channels**, both subviews preserved.

**The risk this leaves, and it is the real one.** Whether the AIE lowering hands the C kernel a base
pointer that *includes* the subview's offset is **not** established by any compile-time check. If it
passes the plane-0 base for both operands, the residual tile silently reads x — plausible numbers,
wrong answer. This is premise item 3 and it is the first thing to measure on hardware, with a
deliberately asymmetric x and residual so the failure cannot hide.

**If the offset is not honoured**, the fallback is a packed-layout variant of the add kernel that
takes one `[2, rpc, cols]` operand and does the plane arithmetic itself. That is a small C change,
but it is a change — budget for it rather than discovering it mid-phase.

## What to build

- **`builders/norm_tail.py`** — the three-herd pipeline, modelled on `programming_examples/
  bottleneck/bottleneck.py` (heterogeneous named herds in one segment, `Channel` +
  `ChannelPut`/`ChannelGet` between stages) and `channel_examples/worker_to_self` (explicit
  `MemorySpace.L1`). The stage bodies are the existing kernels: `elementwise_add`'s,
  `layer_norm.py`'s `layer_norm_rows`, and `elementwise_mul`'s. `builders/layer_norm.py` is the
  template for a stage body — 8-column herd, `rows_per_tile` trips, one `CallOp` per trip.
- **Declare no placement and no buffer depth.** `air-place-herds` places; ping-pong labelling
  chooses depth. If you find yourself writing a tile coordinate, stop.
- **The packing helper.** The caller must produce the `[rows, 2*cols]` packed buffer. Keep that in
  the builder's own device-inputs helper, beside the reference, the way `mha_out_proj.py` keeps
  `mha_out_proj_device_inputs` beside `mha_out_proj_arg_layout` — so a caller cannot get the
  interleave right today and wrong next time.
- **An `opcheck` arm and a lit test**, as every operator here has.

## Gate

```
flock -x -w 1800 /tmp/mlir-air-npu.lock  ninja -C build-xrt check-programming-examples-transformer-layer
```

Allowlist `^programming_examples/transformer_layer/`.

Objective check, driver-owned, three clauses:

1. **Full-output `np.isclose` against the FP32 oracle at the registry tolerance, zero mismatches**,
   with the fault-injection negative control — the standard every operator here meets.
2. **`mean_rel_L1` at or below the block's 1.688e-2.** This is the phase's actual claim. The
   decomposed tail measures 1.806e-2; if the pipelined form does not beat it, the intermediates are
   not staying resident and the phase has not done its job even if the numbers pass.
3. **No packet-typed channel in the lowered IR.** Structural, and cheap: `air-opt` through
   `air-dma-to-channel`, count `"npu_dma_packet"`, require zero. Without it a later edit can add a
   third L3 stream and silently re-enter the miscompiling path — the numbers would still pass at
   one trip.

## What this phase must not do

- **Do not widen a tolerance.** The layer sits at the hard `1e-1` ceiling.
- **Do not hand-place a herd or hand-set a buffer depth.** The whole point is that the compiler
  derives them; a placement attribute in this builder falsifies the result.
- **Do not touch `mlir/`.** If the pipeline exposes a compiler defect, report the minimal shape in
  `work_not_completed` — that is how H9 came to exist.
- **Do not add a third L3-facing stream per column.** If a stage seems to need one, say so rather
  than accepting the packet path; until H9 lands it is silently wrong past one trip.

## Why J7a and not a J1 re-run, after H9

`[2026-08-07]` H9 landed the fusion fix and its session measured the next wall immediately:
multi-trip at `herd_x=8` now hits **shim 16-BD exhaustion at six trips** — column 0 carries
weight + x + residual, i.e. three packet tasks per trip, and 6 x 3 = 18 > 16. J1's target is
**64** trips. So J1 moves from *compiles-silently-wrong* to *refuses-loudly*, which is a real
improvement and still not a working J1.

**J7a is unaffected, and for a structural reason.** BD exhaustion counts *packet* tasks, and the
packed three-herd form has **no packet-typed channels at all** (measured: 0). Its DMAs are ordinary
BDs, and the existing streamed builders already run 64 trips at `herd_x=8` on that path today. The
same packing that keeps J7a off the miscompiling path also keeps it under the BD ceiling.

That is the second time this column budget has decided an outcome, and it is worth stating as the
rule it has become: **two or fewer L3-facing streams per column, everything else on L1->L1
channels.**

## Premise status

Measured before this spec was written:

1. **No packet path with packing** — confirmed (0 packet-typed channels at the real shape). ✅
2. **Places at `herd_x=8` across three herds**, 24 tiles of NPU2's 32 — confirmed via
   `air-place-herds`. ✅
3. **Numerically exact on hardware** — **not measured.** Compile-time checks cannot establish it,
   and this is where a design that passes both can still be wrong. It is the first thing to run.

Reproduce 1 and 2 with `probe_j7a_pipeline.py --packed`.
