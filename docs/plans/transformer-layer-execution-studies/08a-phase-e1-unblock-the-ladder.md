# 08a — Phase E1: Unblock the sequence ladder

Everything built so far runs at `seq = 4096` and nowhere else, because two GEMMs of the same method
at different `tile_n` cannot coexist. This sub-phase makes them coexist, proves it at a second point
on the ladder, and splits the two modules that are over the size cap before four more modes' specs
land in them.

**No execution strategy is built here.** E2 through E5 each need more than one sequence length and
one of them cannot be built at all until this lands, which is why it is first and separate.

## The collision, in two places, from one root cause

`llms/shared/builders/gemm_builder.py` mints both the GEMM's MLIR symbol suffix and its object file
name from the **method alone**. `tile_n` arrives later and separately, as a tile parameter. They
never meet.

The table is at `gemm_builder.py:21-26` and it is the whole of it:

```
fused-cast   sym_suffix "_m64"   obj "mm_m64.o"   tile_m 64
drain        sym_suffix "_m32"   obj "mm_m32.o"   tile_m 32
```

`gemm_method_spec(method)` (`gemm_builder.py:61`) is the only selector; `_spec_with_tiles`
(`gemm_builder.py:45`) merges the registry's `tile_n` in afterwards.

| Where it bites | Symptom |
|---|---|
| `stitch_elf` (`llms/shared/infra/stitching.py:318`) | Two same-method GEMMs at different `tile_n` declare `f32_to_bf16_mn_<suffix>` twice with **different memref types** — the operand types are functions of `tile_n` (`matrix_multiplication/bf16_in_bf16_out/run.py:238-244`). `_extract_private_funcs` collects both texts into one `set()` and `Module.parse` fails with `redefinition of symbol named ...` (`stitching.py:408-412`). |
| `compile_gemm_mm` (`llms/shared/infra/external_kernels.py:133`) | The object is named from the method while `tile_n` is baked in as `-DDIM_N`, so two such GEMMs **write the same file** and aiecc links whichever was written last. It does not fail. D2 measured what it does instead: the FFN's up-projection got the o-projection's 96-wide micro-kernel and returned **exactly zero for 32 of every 128 output columns**, which the GeLU passed through and the down-projection's 3072-deep reduction smeared over the whole FFN output. |

`builders/block.py:373-429` works the second one around by interleaving — each artifact's external
objects are compiled immediately before its own ELF — and says so at length. That workaround is
correct and it is also a trap for the next caller, because nothing announces it.

### What the fix is

Mint **both** names per `(method, tile_n)`. The natural spelling keeps the existing shape:

```
fused-cast, tile_n 96    ->  sym_suffix "_m64n96"    obj "mm_m64n96.o"
drain,      tile_n 128   ->  sym_suffix "_m32n128"   obj "mm_m32n128.o"
```

Three things to know before you start:

- **`gemm_method_spec` has no external callers.** Nineteen files import from `gemm_builder`, but
  every one of them goes through `gemm_registry_config` or `_build_gemm_module`. The single file
  that imports `gemm_method_spec` by name, `llms/qwen25_0_5b/qwen25_0_5b_prefill.py:61`, never calls
  it — its own comment at :65 explains that it synthesizes a `direct` spec instead, because
  `gemm_method_spec` only knows the two high-precision external methods. **The signature is yours to
  change.** Confirm this yourself before relying on it; it is the kind of claim that ages.
- **`tile_n` takes four values across the registry** — `{32, 64, 96, 128}` — so this mints up to
  eight `mm_*.o` objects rather than two. That is more compiles, each smaller. It is not a problem;
  it is worth knowing before you see eight objects appear.
- **`sweep/sweep_families.py:107-111` duplicates the method→`tile_m` table** and cross-references
  `gemm_builder.py:21-26` in a comment at :45. If the two drift, the sweep plans configurations the
  builders cannot build. Keep them in step, and say in the comment which is authoritative.

## This is a shared-infrastructure change, and the gate carries the rule

`gemm_builder.py` is not in `programming_examples/transformer_layer/`. Ten shipped LLM deployments
resolve their GEMM configurations through it, and
[13 §The cross-deployment regression rule](13-verification-and-acceptance.md#the-cross-deployment-regression-rule)
is explicit: after any shared-infrastructure change, re-run `make verify` on every sibling model,
serialized under `flock`. Your gate does that itself (`gate-e1.sh`), so you do not have to — but it
means a naming change that breaks one shipped model fails this sub-phase, and the failure will
arrive hours after you finish.

So: keep the change minimal and behaviour-preserving. Every shape that resolved before must resolve
to the same tiles, the same method and the same micro-kernel; only the *names* move. If you find
yourself changing which configuration a shape gets, stop — that is a different change and it is not
this one.

Phases C and D were both forbidden from touching this file. You are not. That is the only reason
this sub-phase exists.

## Work items

1. **`(method, tile_n)`-aware `sym_suffix` and object name** in `llms/shared/builders/gemm_builder.py`,
   with `sweep/sweep_families.py`'s duplicate table kept in step.
2. **Remove `builders/block.py`'s interleaving workaround.** It exists only for the object-name
   collision. Once the objects have distinct names, `compile_block_artifacts` can build every
   object and then every ELF. Keep the comment that records *why* it was there and what it cost —
   the zeroed 32-of-128 columns are the most instructive thing in that file — but say it is
   historical.
3. **A second point on the sequence ladder for `ffn`.** Add an `opcheck` spec at a `baseline_768`
   sequence length other than 4096, with a `CHECK` line in `run_npu2_ffn_peano.lit`. **`seq = 64` is
   the recommended point**: it is the cheapest on hardware, its two registry rows
   (`64x768x3072` and `64x3072x768`) both resolve to `drain`, and two `drain` GEMMs at `tile_n 128`
   and `96` in one module is precisely the collision this sub-phase removes. Any non-4096 point
   satisfies the gate; pick the cheap one unless you have a reason.
4. **Split `opcheck_specs.py`.** It is **1043 lines** against the ~800-line cap that
   [00](00-context-and-goals.md), [02](02-porting-conventions.md),
   [06a](06a-phase-c1-gate-and-small-operators.md) and [13](13-verification-and-acceptance.md) all
   gate on — a live violation, and E2–E5 add four more modes' specs to it. D1 named the seam and it
   is the same mechanism-versus-catalogue split the file's own neighbours already draw
   (`opcheck.py`/`opcheck_specs.py`, `registry_sweep.py`/`sweep_families.py`): the per-operator
   `_prepare_*` functions in one module, the `SPECS` catalogue in another. `opcheck.py` imports
   `SPECS` from one place (`opcheck.py:124`); keep that import working.
5. **Split `sweep/registry_sweep.py`**, at 866 lines, the same way.

Items 4 and 5 are mechanical and they are here rather than later on purpose: the file gets worse
before it gets better, and doing it while nothing else is in flight is the cheap moment.

## What this sub-phase must not do

- **Do not build an execution strategy.** No `pattern/offload/`, `runlist/`, `coarse/` or `fused/`.
  Those are E2–E5 and each has its own document.
- **Do not write registry rows.** `kernel_registry/details/*.json` is outside your allowlist and the
  tamper check halts the run for it. Every shape you need is already registered — C4 measured all 36
  `baseline_768` points.
- **Do not change which configuration a shape resolves to.** See above.

## Gate

```
flock -x -w 1800 /tmp/mlir-air-npu.lock  agents/scripts/port-loop/gate-e1.sh
```

Two legs. The transformer-layer lit suite on real hardware, then `make verify` in each of the ten
shipped `programming_examples/llms/<model>/` directories. Both must pass.

The driver then checks, independently of anything you write:

- **The names actually separate.** It resolves two same-method, different-`tile_n` GEMMs through
  `gemm_registry_config` — the FFN's up-projection (`4096x768x3072`, `drain`, `tile_n 128`) and the
  o-projection (`4096x768x768`, `drain`, `tile_n 96`) — and requires `sym_suffix` **and** `obj` to
  differ. Both are `drain` today and both names are identical today; that is the collision.
- **The ladder moved.** `ffn` must carry a fresh, declared, contract-satisfying result at a
  `baseline_768` shape whose `seq_len` is not 4096, read from the `shape` dict rather than the
  `shape_key` string. Before this sub-phase `build_ffn_module` cannot build at any other ladder point
  at all, so this is not something a laxer test can produce.
- **That point gets its own fault injection.** The driver re-runs `opcheck.py --operator ffn
  --shape-key <the new key> --fault-inject input` and requires it to **fail**. The generic
  per-operator control injects only an operator's *first* declared shape, which for `ffn` is a Phase
  C row — so without this clause the one new point here would be the only one never injected. That is
  D1's recorded lesson repeated deliberately.
- **Nothing regressed.** The whole D1 `baseline_768` coverage clause and the D2 `block` verdict are
  re-derived from their artifacts. Changing GEMM naming is exactly the change that could quietly
  break them.

## Risks

- **The ten-model leg is the expensive one.** C4's gate spent hours in it. It runs after your
  session ends, so budget nothing for it yourself, but a regression found there costs a whole gate
  cycle to re-run.
- **`direct` is not in `gemm_method_spec`.** It raises `ValueError` on anything but `fused-cast` and
  `drain` (`gemm_builder.py:101`), while `sweep_families.py`'s `METHODS` knows all three. That
  asymmetry is deliberate and pre-existing — do not "fix" it here.
- **The removed interleaving is a silent-failure path.** If the naming fix is subtly wrong and you
  have already removed the workaround, the symptom is not a build error; it is zeros in part of an
  output tensor. The block's ten-stage per-boundary comparison is what catches it, and it is in your
  gate. Do item 1 and item 2 in that order, and run the block between them.
