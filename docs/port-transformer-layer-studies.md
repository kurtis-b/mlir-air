# Port ledger — `exper/transformer-layer-execution-studies` onto `main`

The research branch is frozen at tag `pre-port-20260829` (`4a4f06a0`). Everything it holds is
ported here slice by slice, refactor-first, one concern per PR, ≤ 500 added lines per PR
(`agents/WORKFLOW.md`), or is deliberately left on the tag. The classification behind the numbers
is the port map (`agents/.state/4a/`, operator-approved 2026-08-29): **include** ports as-is,
**refactor** is re-derived onto what `main` now provides (air.api, `llms/shared`), **exclude**
stays on the tag.

Decisions that shaped the include set (operator, 2026-08-29): no `transformer_layer` example on
`main` (the execution-mode study continues as full end-to-end LLM inference); int4 covers
GGUF Q4_0 (ported) and Q4_K_M (new work, tracked separately); plan docs stay on the tag unless a
living path cites one; multi-launch uses the ELF path only; the Qwen3-0.6B QKV stage ports at 2 launches only, A/B against main's 8-launch form, no 4-launch stage (Q7, 2026-08-30); the GEMM kernel-registry rows port as
authored data under the cap.

## Scoreboard (branch-added lines; updated in the PR that moves a row)

| Cluster | Include | Refactor | Excluded | Landed | Remaining | PRs |
|---|---:|---:|---:|---:|---:|---|
| B `agents/` | 1,017 | 0 | 3,194 (+1,397 already on main) | 839 | 178 | #6 devq core (B1); B2 new-job + selftest (this PR, 432); B3 audit script rides with E's GEMM feature |
| F compiler | 1,942 | 6,099 | 2,831 | 2,281 | 5,760 | H10 (#7, 290); shared-L1 put guard (#9, 155); shrink-memref extent (#10, 440); H3 attribute verifier (#11, 123); split-l2 short offsets (#12, 438); split-l2 repeated feed (#13, 373 after its review fixes); split-l2 far-side pairing (this PR, 462 after its review fix) |
| E `llms/` + kernels | 12,972 | 7,233 | 7,741 | 12,822 | 7,383 | qwen3_1_7b o_ffn rewire + sidecar bind, net -12 (this PR, 338 added); qwen o_ffn mixed-method rewire (#51, 302 added after its review fix); registry qwen3_0_6b family, SERIES COMPLETE (#50, 248 carried after its review round; R5c-13); registry baseline_1024 o_proj rows, family complete (#49, 337 carried; R5c-12); registry baseline_1024 ffn_down rows + short-M test (#48, 296 carried; R5c-11); registry baseline_1024 ffn_up rows (#47, 475 carried after its review fix; R5c-10); registry baseline_1024 skeleton + qkv (#46, 397 carried after its review fix; R5c-9); registry baseline_512 o_proj rows, family complete (#45, 428 carried after its review fix; R5c-8); registry baseline_512 ffn_up rows (#44, 475 carried after its review fix; R5c-7); registry baseline_512 qkv rows (#43, 470 carried; R5c-6); registry baseline_512 skeleton + ffn_down (#42, 302 carried; R5c-5); registry baseline_768 o_proj rows, family complete (#41, 430 carried after its review fixes; R5c-4); registry baseline_768 ffn_down rows (#40, 430 carried after its review fixes; R5c-3); registry baseline_768 ffn_up rows (#39, 423 carried; R5c-2); registry herd + restructure + qkv rows (#38, 637 carried after its review fixes; R5c-1); GEMM object-link audit + remaining renames (#37, 234 after its review fixes; R5b); per-(tile_m,tile_n) GEMM object minting (#36, 429; R5a); r=64 int4 GEMV row strip (#35, 113 after its review fixes; R4); W4 three-arm verify gate + default flip (#34, 418 after its review fixes; R3c); W4 decode driver switch + artifact guard (#33, 336 after its review fixes; R3b); W4 decode pack + host test (#32, 482 after its review fixes; R3a); Q15 causal-window lit (#31, 40); R1 census fail-open fixes across example dirs (#30, 287 after its review fixes); SmolLM2 int4 verify adapter + docs (#29, 406; the Q4_0 feature complete); prompt_len regression test + guide (this PR, main-side follow-up of #21, not branch lines); SmolLM2 int4 decode/inference/Makefile (#27, 464; plan PR 8); SmolLM2-1.7B GGUF q4_0 loader (#26, 432; plan PR 7); promoted q4_1→q4_0 route (#25, 409; plan PR 6); q4_0 GEMV harness + device lit (#24, 370; plan PR 5); q4_0 codec + repack + self-test (#23, 342; plan PR 4); GGUF container reader (#22, 356; plan PR 3); prompt_len on the Llama drivers (#21, 26; plan PR 2); int4_gs plumbing (#20, 147; int4 Q4_0 plan PR 1); Qwen3-0.6B decode QKV at 2 launches (#19, 257); 2-launch QKV ELF builder (#17, 479); qkv-heads kernel + layout (#16, 395); rms/qkv host-ABI seam refactor (#15, main-side structure-only, net +3 — not branch lines); verify_runner host tests (#14, 412); next: NOT decode_qkv4 (closed above, Q7) and NOT the LM-head / idle-row family — that one is BLOCKED on an F-cluster split-L2 crash, reproducer and analysis in the section above; take its `tileChannelOpByFactor` prerequisite first. After that, in the split plan's dependency order, the LM-head / idle-row family (ranked first by the split plan on **tag-era** measurements — `2e14f533` idle-row fill (devq 679/674), `93ef7040` + `f0262b18` Qwen3-0.6B LM head 19→10→3 launches (devq 476/471, 688/691), `1e234f18` Llama m_input 8 (devq 563/564). **None re-measured on main**, and this ledger makes no performance claim of its own; the ranking is the plan's, cited so it can be audited. It also carries a re-derivation risk: the tag's `herd_rows` was written on the raw-bindings `matvec.py` while main's is air.api with none, and main has since grown `use_lock_race_condition_fix_v2` alongside the v1 flag the marked-herd path uses), then the qwen3 / llama32_1b_int4 / smollm2 int4 model rows, int4_awq q4_0, and the registry rows (data, ≈11 PRs) |
| D `transformer_layer/` | 0 | 0 | 69,330 | — | 0 | excluded (Q1) |
| C plan docs | 0 | 0 | 12,805 | — | 0 | excluded (Q3) |
| other | 0 | 29 | 41 | 0 | 29 | rides with B |
| **total** | **15,931** | **13,361** | **95,942** | **15,942** | **13,350** | 35 PRs include-only; 62 at full refactor size |

Loop-stop condition: Remaining = 0 for the include set; refactor rows close when their
re-derivation lands or is recorded as not needed.

## Closed with no PR — verified 2026-09-05 against `origin/main` `9c33271d`

The remaining-E split plan (`agents/.state/4b/remaining-e-split-plan.md`, written 2026-08-30 when
main was `830176cf`) found rows that need no port because main already has them in re-derived
form, an operator decision excluded them, or the branch deliberately dropped them. Those claims
were **re-checked against today's main** before the numbers moved, since main has advanced well
past the plan's base:

| row | lines | verified how |
|---|---:|---|
| `llms/verify/{test_verify_runner.py, run_verify_host_tests.lit, verify_runner.py}` | 310 | main is AHEAD, not behind: **7 test functions vs the tag's 5**, including `test_a_stored_flag_does_not_count_as_read`. Porting the tag's copy would remove tests. |
| `shared/infra/decode_qkv4.py` + `test_decode_qkv4.py` | 427 | `decode_qkv2.py` and its test are on main (#19); `decode_qkv4.py` is absent from main, which is what **Q7 decided** ("2 launches only, no 4-launch stage"). |
| `shared/builders/rms_qkv_qknorm_rope_multi.py` residue | 767 | tag-vs-main is **+688/−44**, and every tag-only definition is accounted for: `build_rms_qkv_qknorm_rope_gemv4_module` (Q7), `_build_qkv_heads_gemv` (main has it as `matrix_vector_multiplication/bf16/matvec_heads.py`), `_build_qkv_heads_gemv_wholehead` (dropped: **devq 552**, 0.588 ms either way — the tag records it at `shared/builders/rms_qkv_qknorm_rope_multi.py:1232`), and the `qkv_heads_*` / `qkv2_prep_weight` helpers (landed in `shared/infra/qkv2_layout.py`) — **except `qkv_heads_row_map`, which #17 dropped as unused and which is absent from main entirely**; it is deliberately not ported, not an oversight. |
| `shared/builders/rms_gemv_rope_multi.py` residue | 1 | the plan sized this row at 32; today the tag-vs-main diff is **+1/−1**, one comment line. The smaller number is the one recorded. |

**1,505 lines closed.** E's Remaining moves 8,888 → 7,383 and the total 14,855 → 13,350.

One row from the plan is **not** closed and not counted: `matvec_int4_packed_add.py` (+3). The
file lives at `matrix_vector_multiplication/int4_awq/` on **both** main and the tag, and the +3 is
a `BoolAttr` import plus two `l1_part_op.attributes["air.shrinkage"] = BoolAttr.get(False)`
opt-out lines. Those are not branch-added work: main **removed** them in `f51b9385`
("[air-opt] Stop air-shrink-memref-sizes-by-access retyping subviews it did not shrink", #1909),
the compiler fix that made the opt-out unnecessary. So porting them would re-add a workaround
main has already retired — moot, and recorded with that history so a later audit can check it.

The plan's unnumbered residue rows (smollm2_1_7b_int4 deltas, int4_awq study tooling,
`channel_examples` churn) are left in Remaining until each is measured the same way. The plan
estimates ~5,900 lines close in total against its own branch-added accounting; only what has been
re-verified against current main is moved here.

## Next slice: the LM-head / idle-row family — risk measured 2026-09-05 against `460aadcc`

The split plan calls this "the biggest re-derivation risk in this whole plan". Measured rather
than inherited, the risk is **smaller and differently shaped** than recorded:

| question | answer |
|---|---|
| Does main's `matvec.py` still have `herd_rows`? | **No.** Tag: 26 references, raw bindings. Main: **0**, fully air.api (#1849). This is the real work. |
| Does the marker need a compiler change? | **No.** `air.lock_race_fix_required` appears in **neither** compiler — not the tag's `mlir/`, not main's. It is a pure Python convention: `matvec.py` stamps the herd at `herd_rows > 1`, `dispatch.py` reads it, and `KernelCache.compile_and_cache` supplies `use_lock_race_condition_fix` "for that mark and for no other reason". No F-cluster dependency. |
| Is `use_lock_race_condition_fix_v2` in the way? | **No, and it does not supersede v1.** They are different mechanisms: v1 inserts extra dummy DMA BDs; v2 daisy-chains locks for shared-L2 fan-in/fan-out buffers (`Conversion/Passes.td:208-222`), needs `air.no_split`, and has its own opt-out `air.no_chain_lock`. They are mutually exclusive — `xrt.py:311` raises if both are set — but this path needs only v1, which main already exposes. |
| What is missing on main? | The driver-side half: main's `cache.py` supplies no `use_lock_race_condition_fix`, and main has no reader for the mark. |

So the slice *looked* like: re-derive `herd_rows` onto air.api's `matvec.py`, port the
mark-and-supply convention, and leave v2 alone.

**It is blocked, and the blocker is in the compiler — measured 2026-09-05, devq 928.** The
plumbing was written and the default path verified byte-identical, but the path it enables does
not compile on main:

```
make -f .../matrix_vector_multiplication/bf16/Makefile run \
     M=2048 K=8192 TILE_M=2 M_INPUT=1 HERD_M=4 HERD_ROWS=2
->  aircc: mlir/lib/IR/MLIRContext.cpp:1251: AffineMap::get(...):
    Assertion `willBeValidAffineMap(...)' failed.
    AIRSplitL2MemrefForBufferConstraintPass::runOnOperation()
      -> xilinx::tileChannelOpByFactor(...)
        -> mlir::AffineMap::replace(...)
```

devq 928, Turbo: the two configs the npu2 lit runs today **PASS unchanged** at the default, and
the 2-row config aborts. Note precisely what that does and does not establish: the `herd_rows2`
leg exits **at compile time, before any device execution**, so it proves the default is unaffected
and the compiler blocker is real — and **nothing about the multi-row plumbing itself, which
remains UNVALIDATED.** Its row indexing, its locking and its numerics have never run; when the
compiler side is fixed they must each be verified, not assumed.

**Consequence for the port's order, which the plan did not state: this E-cluster family has an
F-cluster prerequisite** — and it is a *specific* one, not "somewhere in F's 5,760 lines":

- the tag carries `971bab2a` "fix(air-split-l2-memref): size the split offset map from the map it
  composes", plus a 305-line regression test `air_split_l2_memref_multi_symbol_offset.mlir`;
- 4a classified it SUPERSEDED by `48225ba5` (#1934) `mapForExpr`, and **main does have that fix** —
  `AIRMiscPasses.cpp:1685`, *inside* `tileChannelOpByFactor` (1554), which is exactly the crash
  site. So this is **not an unported fix**; it is a case main's fix does not cover;
- and **the tag's regression test for that case is absent from main** (`git ls-files` finds it on
  the tag, not on main), so nothing pins the multi-symbol-offset shape either way.

**Tested 2026-09-05 with the test's COMPLETE RUN line — the tag's test FAILS on main, so the
prerequisite stands.** Two runs, because the first measured only half the gate:

| what was run | result |
|---|---|
| `air-opt … --air-split-l2-memref=…` alone | **exit 0** — main does *not* assert on this input |
| the test's actual RUN line, `air-opt … \| FileCheck %s` | **exit 1**: `multi_symbol_offset.mlir:198: error: CHECK: expected string not found` |

So `48225ba5` (#1934) changed the *failure mode* without preserving the *behaviour the tag pinned*:
main stopped asserting on the multi-symbol offset, and now emits a different channel put than the
test requires — line 198 wants `air.channel.put …[%c0, %[[Q0]], %c0, …]` and main produces an extra
`affine.apply #map3()[%arg3, %arg5]` feeding a different operand list. **That is a real, observable
delta, not a supersession**, so 4a's SUPERSEDED classification for `971bab2a` is at best partial.

Method worth recording with the result: **exit 0 from the producer is not a passing test when the
RUN line pipes into FileCheck.** Reading the producer's exit code and skipping every output
assertion is the same "half a gate" error this ledger has already paid for.

The crash is therefore a **different** shape from the one the tag fixed, and finding it is the
work. What is established:

- **The reproducer is host-only — no device, no devq**, which is much cheaper than how it was
  first found: on `feat/matvec-herd-rows`,
  `python3 matvec.py --compile-mode compile-and-xclbin --m 2048 --k 8192 --tile-m 2 --m-input 1
  --herd-m 4 --herd-rows 2` aborts with the same assertion.
- Running `--air-split-l2-memref` **alone** on the builder's own output does *not* crash (exit 0),
  so the offending map is produced by the aircc pipeline's earlier passes, not by the emitted IR
  as written.

**Answered 2026-09-05: the expectations move, because main's behaviour is deliberately different —
not wrong.** The test failed at CHECK 198 only in the **third offset slot**: it pins `%c0` there,
main emits the base's own `affine.apply`. That is intentional and documented in the pass itself —
`getOriginalApplyOperands` propagates a non-zero base on the split dim rather than zeroing it
"just because the access happens to be contiguous". Everything the test was written to pin — one
put per split, each carrying its own `Q_i` on the split dimension, in split order — holds on main.

So the row closes as a **ported regression test, no compiler change**: the lit is on main with that
one slot relaxed and the divergence documented at the CHECK. Mutation-checked so the relaxation did
not hollow it out — swapping `Q0`/`Q1`, giving split 1 the wrong `Q`, and using a wrong split index
each turn it red. Suite: **538 passed / 552, 0 failures** (the recorded 537/551 baseline plus this
test).

**Two things this did NOT establish, stated so they are not assumed:**

1. **It does not explain the `HERD_ROWS=2` crash.** That is a different shape — this test never
   asserted on main. The host-only reproducer above remains open and unexplained.
2. **`971bab2a`'s operand-ordering fix is now TESTED and ELIMINATED** (2026-09-05). Main is still
   missing that guarantee — Its `air::ExecuteOp` branch
   builds `originalApplyOperands` from `getUsedValuesDefinedAbove`, an unordered set, while the
   replacement map reuses the apply's expression verbatim — so symbol *i* need not bind to operand
   *i*. `mapForExpr` (#1934) removed the assert that used to hide it — but applying that
   fix and running the minimal reproducer below shows it **still aborts**. So it is not the fix for
   this shape, and it is reverted. The guarantee may still be worth restoring on its own merits;
   it is no longer a candidate explanation for this crash.

### The `HERD_ROWS=2` crash: minimal reproducer, and what it is not

**Control first — this is current main, not a stale install.** The crash was first seen through
`install-main/bin/aircc`, dated **Aug 30**. Replaying the *exact* argv (captured with `--verbose`)
against `build-xrt/bin/aircc`, which contains everything on main, reproduces the identical
assertion. Only two commits have touched `mlir/` since Aug 30 and both are #66's behaviour-
preserving refactor. Method note: an earlier replay with *guessed* flags produced a different error
(`'air.dma_memcpy_nd' op failed to get buffer`) — capture the argv, do not reconstruct it.

**Minimal reproducer — host-only, one second, and committed** so it can actually be run:

```sh
air-opt mlir/test/Transform/AIRMiscPasses/Inputs/split_l2_herd_rows2_pre_split.mlir \
  --air-split-l2-memref="max-launch-channels-mm2s=16 max-launch-channels-s2mm=16 tiles-per-l2-tile=4"
#  -> Assertion `willBeValidAffineMap(...)' failed.
#     AIRSplitL2MemrefForBufferConstraintPass -> tileChannelOpByFactor
```

The input is **143 lines** and lives under `Inputs/`, which `mlir/test/lit.cfg.py:93` excludes from
the testsuite — it has to, since running the pass on it aborts `air-opt`. Its header records where
it came from: aircc's own 24-pass prefix (printed by `aircc -v`) applied to the matvec GEMV example
built with a 2-row herd on `feat/matvec-herd-rows`, and the pass options are the ones aircc uses at
that stage. Nothing here needs the NPU, so this is a compiler-debugging loop of seconds — and any
candidate fix can be checked against it by anyone, which is how `971bab2a` was eliminated above.

**What it is not**: the same shape as the `multi_symbol_offset` lit (which passes on main, ported
in #82), and not fixed by `971bab2a`.

### Root cause, diagnosed by instrumentation 2026-09-05 — TWO asserts, not one

Printing the maps as `tileChannelOpByFactor` builds them (temporary `llvm::errs()` in
`mapForExpr` and the compose lambda, reverted) gives the sequence:

```
[DBG compose] original_map=()[s0, s1] -> (s0 * 2 + s1)   originalExpr=s0     <- then abort
```

**Assert 1 — an invalid map.** `composeAffineExprWithOffsetAndAffineMap` substitutes the split
offset with

```cpp
original_map.replace(getAffineSymbolExpr(0, ctx), const, /*dims=*/0, /*syms=*/1)
```

Those counts are hardcoded. With the **two**-symbol map above, substituting `s0` leaves `s1` —
symbol position 1 — inside a map declared to hold one symbol, so `AffineMap::get` asserts on
`willBeValidAffineMap`. A multi-symbol offset is exactly what a multi-level loop nest produces,
which is why a 2-row herd triggers it and a 1-row herd never did.

**Assert 2 — operands out of step with the map.** Preserving the map's own counts
(`original_map.getNumDims(), original_map.getNumSymbols()`) clears assert 1 and the pass runs
**32 map constructions further** — then aborts in `AffineMap::partialConstantFold` on
`getNumInputs(...)`: the map now correctly declares two symbols, while the operand list handed to
the rebuilt `affine.apply` no longer matches it, because one symbol became a constant.

**So the fix is not "keep the counts".** Substituting into the map and rebuilding the apply have to
stay consistent — substitute, then canonicalise the map *and* its operands together (MLIR's
`canonicalizeMapAndOperands` is the obvious tool), dropping the now-unused symbol and its operand
in one step. That needs the operand vector threaded into the compose lambda, which today returns a
map alone; it is a real change, not a one-liner, and it is the next slice.

**Attempted 2026-09-05, and it narrowed again rather than fixing:**

| attempt | result |
|---|---|
| keep the map's own counts in `replace` | clears assert 1; pass runs **32** map constructions further, then assert 2 |
| + `canonicalizeMapAndOperands` before building the apply | **its own precondition asserts** — it requires `map.getNumInputs() == operands.size()`, which does not hold at every call |
| guard that call, then print both counts at the create site | **RETRACTED — see the correction below.** The first 8 of 16 constructions are `numInputs=2, operands=1`; the 1/1 examples quoted here are the last 8 only. The check that produced this row errored out and was read as a pass. |

**That last row was wrong and is retracted** (corrected in the section below): the rebuilt
`affine.apply` **is** created with a mismatched operand list in 8 of 16 cases. The "downstream"
conclusion drawn from it does not hold.

**Where assert 2 actually comes from (2026-09-05, backtrace + three probes):**

The frames name it — `AffineApplyOp::fold` → `AffineMap::constantFold` → `partialConstantFold`.
So this is **MLIR folding an `affine.apply` that already exists in the IR** with map inputs ≠
operand count. It is not a bad `create` call; it is a malformed op reaching the folder. Three
probes then bound where it can come from:

| probe | result |
|---|---|
| is the committed input itself malformed? | **no** — it parses, `--verify-roundtrip`s and `--canonicalize`s clean (exit 0) |
| does the pass's own `AffineApplyOp::create` build one? | **YES** — the first **8** of 16 constructions print `numInputs=2, operands=1`. (An earlier entry claimed all 16 matched; that came from a filter that errored out and a sorted `head` showing only the 1/1 tail. Corrected here.) |
| do the `Util.cpp` offset-producer sites (1496, 1527) build one? | **not by inspection** — `composedMap` is built with `originalMap.getNumDims(), getNumSymbols()` and fed `affine_apply.getOperands()`, consistent by construction |

**So it IS built at the create site, and the earlier "downstream" conclusion is retracted.** The
raw capture, in order:

```
2 inputs, 1 operand   ()[s0, s1] -> (s0 + s1)          <- first 8, all malformed
2 inputs, 1 operand   ()[s0, s1] -> (s0 * 16 + s1 + 12)
1 input,  1 operand   ()[s0] -> (s0)                   <- last 8, consistent
```

**The mechanism this points at**: `composeAffineExprWithOffsetAndAffineMap` returns
`originalExpr + original_map.getResult(0)`. Those two expressions come from **different symbol
spaces** — `originalExpr`'s symbols index `originalApplyOperands`, while the split-info map's
symbols index its own operands — and adding them conflates `s0` of one with `s0` of the other while
the operand list stays as the first one's. That is why the composed map wants 2 inputs and only 1
operand is supplied. A fix has to rebase the second expression's symbols above the first's and
concatenate the operand lists, not just add the expressions.

Note the ordering: these malformed creates are only *reachable* once assert 1 is bypassed by the
counts fix, which is why assert 2 appeared to be a separate downstream problem.

### ROOT CAUSE (2026-09-05): the split-info entry stores a map without its operands

Following the dangling symbol back to its producer (`AIRMiscPasses.cpp:2856`):

```cpp
AffineMap applyMap;
auto apply = getAffineMapOnMemrefSplitDim(ci, *offsetDimOpt);
if (apply)
  applyMap = apply.getAffineMap();          // <- the operands are dropped here
infoEntryTy newEntry = {*offsetDimOpt, applyMap, splitDimOffset, ...};
```

`infoEntryTy` is `<split_dim, split_affine_map, split_offset, split_size, split_stride>` — it
carries **a map and no operands**. That is sound only while the stored map has exactly **one**
symbol: composition substitutes `s0` with the split offset, the result is a constant, and nothing
dangles. With a **two**-symbol map — which is what a multi-level loop nest produces, i.e. a herd
with more than one row — substituting `s0` leaves `s1` referring to an operand that was never
carried alongside it. `composeAffineExprWithOffsetAndAffineMap` then adds that residue to a
*different* apply's expression and keeps only that apply's operand list, producing exactly the
observed `()[s0, s1] -> (s0 + s1)` with one operand.

So both asserts are the same defect seen at two depths: a data structure that is only valid for
single-symbol maps, used with a multi-symbol one.

**Fix options, in order of honesty:**

1. **Carry the operands.** Store the producing apply's operands in `infoEntryTy` and use them when
   composing, rebasing the stored map's symbols above the original expression's. Correct, and it
   changes the tuple type and every user — a real slice, not a patch.
2. **Re-find the producer at composition time** and take its operands then, leaving the tuple
   alone. Smaller, but it re-does a lookup the producer already did and can diverge from it.
3. **Refuse the multi-symbol case** with a clear diagnostic instead of asserting. Does not unblock
   the 2-row herd, but turns a compiler abort into an explainable failure — worth doing regardless
   of which of (1) or (2) lands, because an assert is never the right answer to unsupported input.

Only (3) is safe to ship without a numerical gate; (1) and (2) change generated code and need the
`make run` before/after that this ledger's other rows use.

**(3) is done (2026-09-05).** `tileChannelOpByFactor` now refuses a split map with more than one
symbol and emits a diagnostic naming the op, the map and the reason; the caller already turned
failure into `signalPassFailure()`. On the committed reproducer the compiler goes from **abort
(exit 134)** to **a clean error (exit 1)**:

```
error: 'air.channel.get' op air-split-l2-memref cannot split this access: its offset map
       ()[s0, s1] -> (s0 * 2 + s1) has 2 symbols, and only one is supported ...
```

Pinned by `air_split_l2_memref_multi_symbol_refused.mlir`, and mutation-checked in two directions:
removing the guard fails that test, while **over**-refusing (`> 0` instead of `> 1`) still passes it
and breaks **3 other tests** in the suite — the new test catches the crash returning, the existing
suite catches over-refusal. Suite **539 passed / 553, 0 failures** (baseline 538/552 + this test).

**The 2-row herd is still blocked** — this changes an abort into an explanation, nothing more.
Options (1) and (2) remain the actual fix.

**Option (1) attempted 2026-09-05, and it uncovered a second, independent defect.** The structural
half alone — widen `infoEntryTy` to `<split_dim, split_affine_map, split_affine_map_operands,
split_offset, split_size, split_stride>`, populate the operands at the producer, read them
**nowhere** — should be behaviour-neutral. It is not:

```
FAIL: Transform/AIRMiscPasses/air_split_l2_memref.mlir
  #3 __memcpy_avx512_unaligned_erms
  #4 llvm::SmallVectorImpl<mlir::Value>::operator=(...)
  #5 xilinx::AIRSplitL2MemrefForBufferConstraintPass::runOnOperation()
```

A **segfault**, on an existing test, from adding a field nothing reads. Suite 538/553 with the
change, back to **539/553 with it reverted**, so it is unambiguously the change.

**The cause is NOT established, and a first attempt at explaining it was measured and refuted.**
The tempting story — "the tuple was all-POD, so a dangling copy survived by accident until a
heap-allocating member made it fatal" — rests on the old tuple being trivially copyable. It is not:

```cpp
static_assert(std::is_trivially_copyable<infoEntryTy>::value, ...);
//  -> error: static assertion failed   (on the tuple as it stands today)
```

So that mechanism is wrong. `SmallVector<Value>` may also sit in inline storage, and the trace
stops at `runOnOperation()` with no line info, so the copy site is unidentified. **What is
established is only the A/B**: adding a field nothing reads crashes an existing single-symbol test,
and reverting restores 539/553. The cause is **unknown**, and settling it needs a
line-symbolized or ASan build — the insertion into `opToSplitInfoMap` while iterating
(`AIRMiscPasses.cpp:3315-3329`) is a *candidate*, not a finding.

The consequence for the port holds regardless of mechanism: **option (1) cannot proceed until that
crash is understood**, because it requires exactly the change that triggers it. Reverted; no code
proposed.

Both asserts are reproducible in seconds against the committed input above, so a candidate can be
checked before it is believed — that is how `971bab2a`, "keep the counts", and "canonicalize at the
create site" were each eliminated. No compiler change is proposed from any of them; the tree is
clean.
Any plan that schedules the LM-head family as an E-side slice hits this abort on its first device
run.

The plumbing is preserved, unmerged, on `feat/matvec-herd-rows` (`0b7a27a8`) with its evidence in
the commit message; it is deliberately **not** proposed for main, because a CLI knob that aborts
the compiler is worse than no knob. The measured wins
this unlocks are the plan's, cited above with their devq ids, and **none has been re-measured on
main** — the port must land its own before/after, not carry those numbers forward.

## Slice order

**Amended 2026-09-05**: the split-L2 `tileChannelOpByFactor` prerequisite (see the section above)
comes before the LM-head / idle-row family — that E row aborts aircc today, so the order below is
not "compiler slices, then all E work": this particular E feature sits *behind* an F row that the
list treats as already covered.

Shared infra (B: devq) → compiler rows that are self-contained (H10 non-constant BD offset refusal,
`92b05de9` shared-L1 lock placement, shrink extent, split-L2 B/C/D, H3 verifier) → the `?` rows
settled by lit runs against `main`'s `air-opt` and one devq job → E model-side work (qwen3,
llama32_1b int4, 4-launch QKV, registry rows) → E refactors onto air.api / `llms/shared`.
