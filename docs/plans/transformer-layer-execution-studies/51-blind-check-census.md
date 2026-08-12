# 51 — The filed-but-not-fixed census of blind checks

`[2026-08-12]` Items **17** and **19** each closed with a list of the same defect class found
nearby and left explicitly *"Filed, not fixed"* / *"Reported, not fixed"*. This document is that
list, worked.

**The first finding is about the list itself.** By the time this worktree fast-forwarded to
`39a08a8b`, commit **`0d2ae8d5`** ("test: five checks that could not fail, made able to fail") had
already landed all five filed items. So the work here is not the fixes — it is the **independent
re-demonstration** of every one of them, plus the half nobody had touched (the dead-flag triage).
That re-demonstration was worth doing on its own terms: it found one claim in `0d2ae8d5`'s own
commit message that does not hold, and it corrected an assertion in the tree whose docstring
promised a red it cannot produce.

Auditing an audit is the right instinct in this codebase. The dominant defect class is checks that
could not fail; a *fix* for a check that could not fail is itself a check, and it can have the same
disease.

---

## The table

| Check | What it could not detect | The input that proves it | State after |
|---|---|---|---|
| `study/test_ladder_report.py` | `ladder_report.load` / `main` — the loader it exists to test. It hand-built the post-`load` row shape, so every check held whether or not `load` produced that shape | `load`'s `_ok` rule relaxed from `run_status == "passed"` to `!= "failed"` — a **skipped** rung counted as a measurement | **FIXED** (`0d2ae8d5`), re-verified: pre-fix **11/11 green**, post-fix red at `test_load_does_not_count_a_skipped_rung_as_passed` |
| `study/test_run_ladder.py:31` | `_spec()`, a hand-written stand-in for a real `SPECS` row, agreeing with the catalogue only until the catalogue moved | `opcheck_specs` `coarse` row `ffn_dim 3072 → 4096`, breaking the `ffn_dim == 4·emb_dim` relation the fixture bakes in | **FIXED**, re-verified: pre-fix **8/8 green**, post-fix red at `test_the_stand_in_row_still_matches_the_catalogue` |
| `study/test_profiles.py:143` | `expected_files()` compared against a **typed** list of the four CSV names rather than against what `run_ladder.walk` writes | `walk` renamed its output `<mode>.csv → <mode>_results.csv` | **FIXED**, re-verified: pre-fix **15/15 green**, post-fix red at `test_every_profile_expects_exactly_what_a_walk_of_it_writes` |
| `study/test_component_groups.py:38` | A typed `{mode: path}` map naming `pattern/coarse/cells.py` while `coarse.py` sat beside it unread — undetectable, since **both** open zero host buckets | A `time_cpu("smuggled_bucket")` added to `pattern/coarse/coarse.py`, the file the typed map did not name | **FIXED**, re-verified: pre-fix **20/20 green**, post-fix red at `test_the_host_bucket_derivation_can_tell_the_modes_apart` |
| `builders/test_block_cache.py:65` | The gate's own `SPECS` row, transcribed | Block gate row `4096x768 → 2048x768` | **FIXED, WITH A CORRECTION** — see §2. The derivation is right; the *pin added alongside it* cannot go red, and its docstring said it could |
| `pattern/test_blocked_attention.py` | Anything: **no negative control at all**. Worst case is the two "independent" implementations folded together, which is invisible to every agreement check because it makes them agree *better* | `builders/mha_attention.py`'s oracle `chunked_attention_reference` made to delegate to `blocked_attention` | **FIXED**, re-verified: pre-fix **5/5 green** on the collapse, post-fix red at `test_the_two_implementations_are_not_the_same_arithmetic` |
| Every host arm's `make` target | The lit's pinned count. `make` ran the script **bare**, and the pinned count is the only thing that catches a test that stops being *defined* | One test function deleted from `test_blocked_attention.py` | **FIXED** via `lit_pin.py`, re-verified: bare script reports **`9/9 passed`, exit 0**; `make blocked-attention-tests` now **exit 2**, `lit_pin: FAILED` |
| `--seed`, `llms/verify/verify_runner.py` | That a *seeded* verification was not seeded — parsed, never read | The flag re-introduced as `p.add_argument("--seed", type=int, default=0)` | **FIXED** (removed, not wired), guard re-verified: names `['--seed (args.seed)']` and fails |
| `--arch`, `vector_tanh.py` | That asking for `aie2` silently built `aie2p` | `--arch aie2` vs `--arch aie2p`, `--print-module-only` | **FIXED HERE** — §3 |
| `-v/--verbose`, `attn.py` | That `-v` did nothing; the only call site pinned `verbose=False` as a literal | AST audit; the literal at the call site is itself the proof | **FIXED HERE** — §3 |

Suite: **357/357 in 19 modules, before and after.** No count moved, so no lit literal moved.

---

## 1. Method, and one trap that nearly faked the whole thing

Every row above is established the same way: take the check as it stood at **`0d2ae8d5^`**, inject
the defect into the **production** file, and run the old check and the new check against the *same*
injected tree.

The harness (`inject.sh`) purges `__pycache__` before **every** interpreter invocation, and that is
not hygiene theatre. The first run of the `ladder_report` demonstration left the suite reading
**356/357 with a pristine working tree** — `git diff` empty, the assertion still failing. The cause:
the injected and pristine sources were the *same byte length* (`== "passed"` and `!= "failed"` are
both 11 characters) and the edit-and-restore happened inside one second. CPython validates a `.pyc`
on `(source_mtime_seconds, source_size)`, so both were unchanged and the **stale injected bytecode
was reused**.

That trap can fake either direction — a "fixed, now green" claim on injected bytecode, or a
"catches it, red" on code that was never injected. Any before/after demonstration in this repository
that edits a same-length string and re-runs within a second is suspect unless it purges. Recorded
here because the next person doing this will hit it.

## 2. `test_block_cache` — the one claim that does not hold

`0d2ae8d5`'s commit message says of all four transcriptions:

> Each demonstrated: move the production value and the derived check goes red where the
> transcription stayed silently green.

For three of the four that reproduces exactly (rows 2–4 above). **For `test_block_cache.py` it does
not.** Measured — block gate `SPECS` row moved `4096x768 → 2048x768`:

```
PRE-FIX  (SHAPE hand-transcribed 4096) : exit=0   block cache tests: 9/9 passed
POST-FIX (SHAPE derived from SPECS)    : exit=0   block cache tests: 10/10 passed
HEAD SHAPE = {'seq_len': 2048, 'emb_dim': 768, 'ffn_dim': 3072, 'num_heads': 12, 'head_dim': 64}
```

No red. `SHAPE` silently **followed** the production value to 2048. The pin added alongside the
derivation, `test_the_shape_under_test_is_the_gate_s_own_specs_row`, asserts
`SHAPE == rows[0]["shape"]` — and `SHAPE` is produced by `_gate_shape()` from *that same row*. It is
`dict(x) == x`, true by construction. That is the tautology risk in its other direction: a check
that computes its expectation with the thing under test.

**This is not a defect to fix, and that is the point worth writing down.** Derivation exists so the
checks track the gate instead of a number transcribed once; demanding a red here would mean putting
the transcription back. The honest description is that the drift is **absorbed**, not detected — and
the module's other checks are shape-agnostic properties of the caching decision, so absorbing it is
correct. What the pin still catches is structural: the block row disappearing or being duplicated,
and a row that no longer carries the fields `block_config` needs.

What *was* wrong was the docstring, which read "This is the assertion that turns that drift red."
A false claim about what a check detects is how the next reader decides not to add a real one. It
has been corrected in place to state what it does and does not detect, with the measurement above,
and to point at `study/test_run_ladder.py`'s catalogue pin as the drift-detecting check of this
family — that one **can** go red, precisely because its fixture is deliberately *not* derived.

**Which risk each of the four traded:**

| Module | Traded | Now |
|---|---|---|
| `test_run_ladder.py` | transcription → *nothing* | Fixture stays hand-written; the pin derives the catalogue **by AST** and asserts the fixture's *format and relations* against it. Not tautological — the two sides come from different places |
| `test_profiles.py` | transcription → *nothing* | Expectation compared against the output of a **stubbed `walk`**, i.e. against behaviour, not against `expected_files()` itself |
| `test_component_groups.py` | transcription → *nothing* | Reads the whole pattern **package**; the derivation is a text scan, independent of the taxonomy it checks |
| `test_block_cache.py` | transcription → **tautology** | Accepted deliberately. Derivation is correct here; the accompanying pin is structural only, and now says so |

## 3. The dead flags — the count did not reproduce

Item 19 recorded *"0 dead flags across 115 study and agent scripts"*, **9 in the wider tree**, two of
dispatch shape fixed, and **"7 remaining"**. After `0d2ae8d5` removed `--seed`, six should remain.

**Two remain.** Both an attribute-load audit and a stricter namespace-scoped variant agree, and the
figure reproduces in this worktree and in the shared checkout:

```
programming_examples/flash_attention/dataflow_based/attn.py:709            -v      (args.verbose)
programming_examples/primitives/vector_examples/vector_tanh/vector_tanh.py:136  --arch  (args.arch)
TOTAL: 2 dead flags in 2 files
```

The gap from 9 is scope, not fixes: the shared checkout carries a vendored **`llvm/`** subtree (201
dead flags on its own) and a **`sandbox/`** venv of site-packages (302). Rooted at the repository's
own sources the answer is 2; rooted one level up it is whatever third-party code happens to be on
disk, which is not a property of this project and moves when someone re-creates the venv. **A count
published without the root it was taken at is not reproducible** — the "claim without an artifact"
rule, in its counting form. Item 19's "7 remaining" should be read as retracted, superseded by 2.

Both were triaged as *asking for something changes nothing*, and both are fixed.

**`vector_tanh.py --arch` — item 19's defect verbatim, one step worse.** It declared
`choices=["aie2", "aie2p"]`, so argparse **validated** the request against a list of two and then
nothing read it. Every sibling primitive (`vector_exp`, `vector_mul`, `vector_reciprocal`,
`vector_rsqrt`) threads `arch` into `build_module` to select a different lowering; `vector_tanh`'s
`build_module` has no `arch` parameter at all. Measured, host-only:

```
BEFORE   --arch aie2   -> exit 0, 34-line module, md5 44bd7e0628c4de646d3839f1bf3f51e7
         --arch aie2p  -> exit 0, 34-line module, md5 44bd7e0628c4de646d3839f1bf3f51e7
         IDENTICAL -- asking for the other target silently got you this one

AFTER    --arch aie2   -> exit 2, "argument --arch: invalid choice: 'aie2' (choose from 'aie2p')"
         --arch aie2p  -> exit 0, md5 44bd7e0628c4de646d3839f1bf3f51e7  (unchanged)
```

Fixed by **refusing** rather than wiring: this example lowers to the AIE2P hardware tanh intrinsic
(`math.tanh → aievec.tanh → xllvm.intr.aie2p.tanh`) and has no `aie2` path, so wiring would have
meant inventing an untested lowering. The flag is kept rather than deleted because the sibling
Makefiles drive it through `AIE_TARGET`, and that knob should fail loudly rather than vanish — the
Makefile still reads `AIE_TARGET ?= aie2p` and `make AIE_TARGET=aie2` now stops instead of quietly
building aie2p. A guard reads the value after parsing, so the flag cannot go dead again and the next
person to widen `choices` without giving `build_module` an arch path gets told so by name.

**`attn.py -v/--verbose`** was parsed and never read while its only call site passed
`verbose=False` as a **literal** to `XRTRunner`. Wired to `args.verbose`; default behaviour is
unchanged, since `store_true` yields `False` when the flag is absent.

Audit after both: **0 dead flags** across the repository's own sources.

## 4. The `make`/lit gap — what was chosen, and the one arm left out

`lit_pin.py` **reads the pins out of the sibling `.lit`** and asserts them, so the number lives in
one place and `make` and `lit` cannot disagree. That is the right call among the three options: it
is one mechanism applied to every arm rather than a thorough fix on three, it keeps the pin where it
already is instead of copying it into the Makefile (which would be the transcription defect this
census is about), and it does not require `FileCheck` — which is in neither `build-xrt/bin` nor
`install-xrt/bin` and is reached by the lit suites only through an absolute path in
`lit.site.cfg.py`, so requiring it would break targets documented "No NPU required".

Verified across all six rewired targets (seven invocations): `block-cache-tests`,
`blocked-attention-tests`, `reference-tests`, `sweep-families-tests`, `registry-writer-tests`,
`seam-tests` (two, `POOL` and `DISPATCH`) — all exit 0 with a `[lit-pin]` line. No `make` recipe in
the example still runs a test script bare.

**Left out, deliberately: `make registry-resolution`.** It is a `make`/lit pair where the make
target pins nothing, but it is not the same defect and does not take the same fix:

- It is **not blind to a resolution failure** — `registry_sweep.py` returns 1 itself. It is blind
  only to the shape set *shrinking*: at 12 shapes it would print `PASS: all 12 ... resolve` and exit
  0 while the lit's `CHECK-COUNT-36` goes red.
- `lit_pin` **refuses** `CHECK-COUNT`, `CHECK-NOT` and `{{regex}}` rather than approximating them —
  a checker that mis-modelled a directive would pass where FileCheck fails, which is this same
  defect one layer down. That lit uses all three, so routing it through `lit_pin` would break a
  working target with a refusal.
- The target is parameterised by `FAMILY` while the lit pins `baseline_768`, so there is no single
  literal the two could share.
- It is a device-gated arm (`run_npu2_...`), outside item 17's "host arm" census.

Recorded rather than fixed. If it is ever worth closing, the move is to teach `lit_pin` `COUNT` and
`NOT` deliberately — not to approximate them.

## 5. What this leaves

- Nothing on items 17's and 19's lists is outstanding.
- One assertion in the tree (`test_block_cache`) now describes its own reach honestly instead of
  overstating it.
- Item 19's "7 remaining dead flags" is **retracted**: 2, both fixed, audit rooted at the repository.
- `make registry-resolution` is a known, characterised, unfixed instance — blind to a shrinking
  shape set only.
