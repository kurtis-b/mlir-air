# 23 — What the 2026-08-06/07 run established: standing rules and open items


Four phases ran overnight. Two compiler defects were found and fixed, one was found and left with
its shape recorded, and the first piece of iron's dataflow form landed. This document holds what
came out of it that is **not** already a phase spec: rules that should govern later work, and open
items nobody has claimed.

## The rule that explains more than anything else found

**A column has two shim MM2S channels, and the budget is per COLUMN across the whole segment.**
Exceed it and AIR packet-multiplexes onto one queue — and until H9 the packet queue silently
misdelivered every trip after the first on more than one column.

Measured through `air-dma-to-channel`:

| builder | L3→L1 streams | packet-typed channels |
|---|---|---|
| `addnorm` (x, residual, weight) | 3 | **3 — multiplexed** |
| `elementwise_add` (a, b) | 2 | 0 |
| `layer_norm` (in) | 1 | 0 |

Reproduce with `python3 agents/probes/probe_packet_streams.py`; the probes and what each one
establishes are indexed in [`agents/probes/README.md`](../../../agents/probes/README.md).

This one fact explains a chain of things that looked unrelated:

- why **`fused`'s decomposed tail already ran 64 trips on 8 columns correctly** while the fused
  `addnorm` could not — each decomposed launch stays within two streams;
- why `addnorm` needed its one-trip guard at all;
- why J1's L2-staged-weight attempt failed with *"no ShimNOCTile has sufficient DMA capacity"* —
  8 columns × 2 = 16, already full before the third stream;
- why **J7a works**: packing x and residual into one strided fetch puts each column at two.

**The design rule for everything downstream (J7b, J7c, J8): keep every column's L3-facing streams
at two or fewer; put the rest on L1→L1 channels, and pack co-indexed L3 operands into one fetch.**
A stage wanting a third L3 input per column is back on the packet path.

Note the budget is per column *across stacked herds*: three 8-wide herds put one tile of each into
every column, so their demands add. Getting this wrong per-herd instead of per-column is the first
mistake this run made.

## Match a probe's altitude to its claim

`air-opt` with a hand-built pass list answers *"does this pass fire"*. It does **not** answer
*"does this compile"*. The two diverge wherever a later pass rewrites what was measured.

This cost a spec claim: a strided-callee construction that lowered cleanly through `air-opt` never
compiles under `aircc`, because `air-to-aie` normalizes external callee signatures to identity
layout afterwards. J7a's session found it by compiling for real and routed around it.

**Use `aircc` — or `XRTBackend.compile(debug_ir=True)` — whenever the claim concerns anything
downstream of `air-to-aie`.** `air-opt` remains right for pass-level questions, which it answered
correctly several times in the same run. J7b's accumulator-ring claim was re-checked at `aircc`
altitude for this reason and holds (`pass_006`, 4 → 2 data-movement ops in the K loop).

## Silence is the wrong default for `air-fuse-packet-put-loops`

H1s settled "skip and warn, never refuse to compile" for `air-label-scf-for-to-ping-pong`, and that
was right **because declining leaves a correct single-buffered loop** — only the optimization is
lost.

`air-fuse-packet-put-loops` has the opposite property: declining leaves the per-channel put loops
whole-channel-after-whole-channel against a consumer ring built for per-iteration interleave, which
past one trip is **silently wrong data**. The untransformed program is the broken one.

The pass currently contains **zero** diagnostics on any decline path. Proposal, unclaimed:

> When it leaves two or more same-bounds packet put loops unfused **and the trip count exceeds
> one**, warn — naming the loop, the channels and the trip count.

The trip count is free (`isCandidate` already requires static bounds and stores them as the
grouping key). Not unconditional: at one trip the orders coincide and the unfused form is correct,
so an unconditional diagnostic would fire on most shipped designs. Not an error either, because the
pass cannot establish that the group's channels share a queue before placement —
`aie.shim_dma_allocation` does not exist until `air-to-aie`.

## Open items nobody has claimed

**1. `layer_norm`'s two-pass variance has no throughput measurement.** J7a's round-3 fix changed
`layer_norm.cc` from one-pass bf16 to two-pass f32 statistics — a large accuracy win
(`mean_rel_L1` ≈ 2.0e-3 → 8.1e-5 at 512×512, `atol_required` 0.0) — but it reads each row twice and
nothing measured the cost. `gate-h.sh` has a throughput leg; the transformer-layer suite does not.
Two execution modes dispatch this kernel and Phase F will report latency built on it. Measure
before/after on one shape and record it beside the accuracy figure.

**2. Large-mean activations are untested against all three norm operators.** The regime that
exposed the one-pass defect — a row whose mean is large next to its spread — is now covered for
`norm_tail` by the `128x768_offset` spec row. `layer_norm` and the fused `addnorm` (which
deliberately keeps one-pass, to preserve the block's 1.688e-2 baseline) have no equivalent. One
shared shape asserted across all three would pin the boundary once instead of each phase
rediscovering it.

**3. `fused` and `runlist` provenance figures are stale.** Both dispatch `build_layer_norm_module`
and will measure lower than their recorded 1.806e-2 / 1.755e-2 after the kernel fix. The J7a
session deliberately did not write numbers it had not measured. Refresh from a gate run.

**4. J1 is blocked on shim BD exhaustion**, not on correctness any more: `herd_x=8` multi-trip
refuses at six trips (column 0 carries weight + x + residual, three packet tasks per trip, 18 > 16)
against a 64-trip target. The candidate fix is loop-shaped packet BD programs on the shim rather
than one `aiex.dma_configure_task` per iteration. **Not on the goal path** — J7a reaches the same
dispatch collapse without the packet queue.

**5. `norm_tail_structure.py` checks at `air-dma-to-channel` altitude.** Sound for what it claims
(packet typing is decided there), but it cannot prove final routing or live L1-backed endpoints.
Strengthening it to assert on late IR would close the gap its own review recorded.

## Struck from the plan

**H4 — `air.disable_ping_pong` works.** Measured: set on a loop that is otherwise labeled, `unroll`
and `hoist_alloc` both go to zero and the buffers are not rotated, and the attribute survives every
pass to the labeler. The original "setting it changed nothing" was taken on a shape whose callee is
unannotated and which is therefore never labeled at all — it disabled something that was not
happening.
