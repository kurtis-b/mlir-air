# 22 — Phase J7b: the accumulator ring, derived

Partial sums that never leave the chip, with the **compiler** forming the ring rather than the
builder declaring it. This is the piece the goal names most directly.

## What exists, and what has never been dispatched

Three accumulate-into-C kernels are ported, compiled and exported, and `grep matmul_with_acc` across
every builder returns nothing. Meanwhile `fused-cast` stages partials in a full-size f32 scratch in
**L3** — `qkv_f32`, `ffn_up_f32`, `ffn_out_f32` are whole `[seq, ffn]` launch arguments listed in
`block.py`'s `host_writes` — where iron's `of_o_acc_in`/`of_o_acc_out` ring never leaves the memtile.

## The two conditions, measured — and at which altitude

**Read this first.** The table below was produced with `air-opt` and a hand-built pass list. That
answers *"what does this pass do"*, not *"what does the toolchain do"* — and the difference bit
J7a: a workaround I measured as lowering cleanly through `air-opt` never compiles under `aircc`,
because `air-to-aie` normalizes external callee signatures afterwards.

**`[2026-08-07 02:50]` Re-checked at the right altitude, and it holds.** The in-loop-alloc in-place
shape was compiled through `XRTBackend(debug_ir=True)` — the real `aircc` pipeline — and
`pass_006_after_air-hoist-dma-in-accum-pattern.mlir` shows the same **4 → 2**: both accumulator
DMAs lifted out of the K loop, leaving only the A and B fetches. (The compile fails at link, having
no kernel object; irrelevant, since the hoist pass runs sixth and its dump is written regardless.)

So the design below rests on an `aircc`-level measurement, not a pass-list one. Reproduce with
`probe_accum_aircc.py`. Keep the gate clause reading aircc's dump anyway — what is confirmed is
that the ring *forms*, not that a future builder will keep it.

## The two conditions, measured

`air-hoist-dma-in-accum-pattern` runs unconditionally, second in the pipeline, and forms the ring
by itself — under two conditions that doc 16 did not state and that are easy to get wrong. Measured
with `air-opt --air-dependency,--air-hoist-dma-in-accum-pattern`, counting data-movement ops left
inside the K loop (A and B fetches must stay; the accumulator's two should lift):

| accumulator kernel | L1 buffers allocated | before → after | ring? |
|---|---|---|---|
| in-place — one `C`, read-add-write | outside the loop (herd scope) | 4 → 4 | **no** |
| in-place — one `C`, read-add-write | **inside the loop** | **4 → 2** | **YES** |
| two-buffer — `pAcc` in, `C` out | outside the loop | 4 → 4 | no |
| two-buffer — `pAcc` in, `C` out | inside the loop | 4 → 4 | no |

**Condition 1 — use the in-place kernel.** `areSymmetricDmaOps` requires the incoming DMA's
destination and the outgoing DMA's source to be the *same* memref.
`ffn_matmul_bf16_bf16_up_proj(A, B, C)` reads C, adds, writes C — one memref, matches.
`ffn_matmul_with_acc_bf16_bf16_down_proj(A, B, pAcc, C)` uses two, both `__restrict` so they may
not alias, and **never matches at any alloc site**. That is the kernel doc 16 told J7 to call.

**Condition 2 — allocate the accumulator inside the K loop.** Counter-intuitive: you write
"allocate per iteration" to get "resident across iterations". `isIncomingDmaOp` requires the DMA to
depend on both the loop's first iter-arg **and** an `air.execute` holding a `memref.alloc`;
`isOutgoingDmaOp` requires its token's users to include both an `air.wait_all` **and** an
`air.execute` holding a `memref.dealloc`. Allocate at herd scope — the natural way — and neither
predicate holds. The pass hoists the alloc and the mirrored DMA pair together.

## Scope: the FFN down-projection ONLY

One GEMM, not two. The o-projection is the same construction and is a follow-on phase — landing one
accumulator ring that provably forms is worth more than starting two.

## What to build

**`builders/ffn_accum.py`, exporting `build_ffn_accum_module(...)`.** The names are fixed because
the driver's objective check imports them; a different name fails the phase for the wrong reason.
Register the operator as `ffn_accum` in `opcheck_specs.py` so `phase_c_operator_check` finds it,
with a fault-injected twin like every other operator here.

The FFN down-projection as a K-loop over the in-place accumulator: zero C once before the loop
(`ffn_zero_bf16_*` — the contract requires C zeroed or holding a valid partial sum), then per K
step fetch C, call, store C. Let the pass collapse it.

**Do not hand-build the ring.** If it does not collapse, that is a finding about the pass's
matcher and it is worth more than a workaround.

Carry J7a's column rule: **two or fewer L3-facing streams per column**. A GEMM stage fetching A, B
and C per column is already three — check the lowered IR for packet-typed channels before assuming
it is safe, and pack or restructure if it is not.

## Gate

The transformer-layer suite, allowlist `^programming_examples/transformer_layer/`, plus a
driver-owned objective check with **three** clauses:

1. Full-output `np.isclose` at the registry's GEMM tolerance, zero mismatches, fault-injection
   control.
2. **`grep matmul_with_acc`-equivalent: the in-place accumulating entry point is actually
   dispatched.** It never has been.
3. **Structural: the C DMAs are no longer inside the K loop.** Read aircc's `--debug-ir` dump for
   `air-hoist-dma-in-accum-pattern` and count data-movement ops in the loop body, exactly as
   `probe_accum_hoist.py` does.

**Clause 3 is the phase.** The numbers are identical whether the ring formed or not — a DDR
round-trip per K step is invisible to `np.isclose`. Two of the four cells in the table above would
ship as working code and pass every numerical gate.

## What this phase must not do

- Do not use the two-buffer `..._with_acc_...` entry point for the automatic route; it cannot
  match. Build it only as the hand-written comparison point, if at all.
- Do not declare a memory space for the accumulator. The point is that the compiler decides.
- Do not touch `mlir/`.

Reproduce the table with `probe_accum_hoist.py --shape {inplace,twobuf,inloop,inloop_twobuf}`.
