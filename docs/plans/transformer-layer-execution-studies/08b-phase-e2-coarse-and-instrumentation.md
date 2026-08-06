# 08b — Phase E2: `coarse`, and the instrumentation the other three modes reuse

`coarse` is the coarse-runlist point of the taxonomy — few fused kernels over one runlist, iron's
`hybrid`. **It is already built.** `builders/block.py` is a fused-operator sequence over
`KernelCache.run_sequence`, it passes on hardware, and it already records a dispatch vector per
sequence.

So this sub-phase is not "build `coarse`". It is: give it a strategy directory, and settle the
**artifact contract** that E3, E4 and E5 will each be measured against. That contract is the
expensive part, because three later sub-phases and the whole distinguishability gate read it.

## Do not write a second block

`builders/block.py` stays where it is. `pattern/coarse/` **wraps** it — imports `block_config`,
`run_block`, `describe_block`, `BLOCK_BOUNDARIES` — and adds the mode layer on top. The reasons are
concrete: `block.py` is enrolled in `run_npu2_block_peano.lit`, in `opcheck.py --operator block`, and
in the D1/D2 coverage clauses that E1 re-runs. Moving it churns gate files to no benefit, and the
harness would stop being able to prove D2 has not regressed.

Per [convention rule 7](02-porting-conventions.md), the directory and the operator are `coarse`; only
the CSV value is `hybrid`.

Per [convention rule 8](02-porting-conventions.md), do not port iron's per-strategy `reference.py`
re-export shims. Import `pattern/reference.py`, the FP32 golden model D2 produced. It is the
correctness anchor for all four modes and `pattern/__init__.py` already says so.

## The layout, and one rule that is not optional

```
programming_examples/transformer_layer/pattern/
├── reference.py       # the shared FP32 oracle          (exists, D2)
├── test_reference.py  # its composition check           (exists, D2)
└── coarse/
    ├── __init__.py
    ├── coarse.py      # the mode wrapper over builders/block.py
    └── README.md      # what boundary this mode isolates, and what it costs
```

**Each mode gets its own `KernelCache` directory, and it is not a style choice.**
`builders/block_cache.py` keys a cached ELF by a fingerprint over the configuration, the emitted
MLIR and every kernel source — which is sound — but the *cache directory itself* is chosen by name
(`BLOCK_CACHE_DIR = "block_cache"`, `opcheck_specs.py:550`). Two modes pointed at one directory can
trade ELFs whose fingerprints happen to agree, and the result is numerically valid output attributed
to the wrong execution boundary. That failure would not show up in any equivalence check; it would
show up as a dispatch vector that quietly describes another mode.

Give `coarse` a directory of its own, add it to `transformer_layer/.gitignore` **and** to the
`clean` target, in the same commit. [15](15-environment-notes.md) predicted exactly this leak before
D2 and D2 leaked a 6.3 MB `block_cache/` into the source tree anyway.

## The artifact contract

Each mode is an `opcheck.py` operator, dispatched through the `dispatch` seam D2 added
(`opcheck.py:330`), whose verdict is the conjunction of the end-to-end comparison and the
per-boundary `stages`. That is already how `block` works — `_prepare_block` (`opcheck_specs.py:553`)
is the only spec returning `dispatch` instead of `module`, and it is your model.

For `coarse` specifically, and for every mode after it:

| Field | Requirement |
|---|---|
| `operator` | the mode name — `coarse` here; `offload`, `runlist`, `fused` later |
| `shape` | `seq_len 4096, emb_dim 768, ffn_dim 3072, num_heads 12, head_dim 64` — the forced gate configuration, exactly as D2's block |
| `n_elements` | `4096 * 768 = 3145728`, the whole layer output. A comparison over a slice does not pass |
| `stages` | at least 8 entries with **distinct** names, each `n_mismatch == 0` and no smaller than one 4096×768 boundary tensor |
| `rtol` / `atol` | `1.6e-2` exactly, and at most `1e-1`. The driver re-derives both |
| `ref_dtype` | `"float32"` |
| `execution_mode` | `"hybrid"` for this mode — the CSV value, per rule 7 |
| `dispatch_vectors` | a non-empty list of `DispatchVector.as_row()` dicts, one per `run_sequence` call |

### `dispatch_vectors`, precisely

`DispatchVector` lives in `llms/shared/infra/dispatch.py:120` and its `as_row()` (`:172`) is the
serializer. **Use it. Do not write your own counting** — the whole comparison is meaningless if two
modes disagree about what a submission is, and `dispatch.py:122-153` already writes down what each
field means, including the deliberate asymmetry that `air_launches` is counted once per distinct ELF
while `herd_launches` accumulates per step.

Two things about the six keys that will bite if you assume otherwise:

- **`runlist_entries_per_submission` is a derived mean, not a count** (`dispatch.py:166`). The driver
  reconstructs the total as `Σ round(entries_per_submission × host_submissions)` over the recorded
  rows. For the block's four rows — `(1, 2.0), (1, 64.0), (1, 1.0), (1, 64.0)` — that is 131, which
  is the number [08](08-phase-e-execution-strategies.md) quotes. A naive sum of the means would give
  131 too here, only because every submission count is 1; it will not once a mode submits more than
  once per sequence.
- **The product must be an exact integer.** The driver rejects a vector whose
  `entries_per_submission × host_submissions` is not integral, because that is the shape a fabricated
  number takes.

### The fault-injected run carries the vectors too

The driver proves the numbers came from hardware rather than from a text editor by comparing them
against the run **it** initiates: it re-runs the mode with `--fault-inject input`, requires that run
to fail, and then requires the fault artifact's six driver-summed totals to **equal** the clean run's.
Fault injection perturbs one input element after the reference exists; it does not change the
dispatch path, so the vectors are identical. They already are for `block` today — clean and fault
artifacts both total 4 / 131 / 12 / 146 / 402 / 202,902,528.

This costs you nothing as long as the mode's `dispatch` callable emits `dispatch_vectors`
unconditionally, on the injected path as well as the clean one. `_prepare_block` already does. Do not
add a "skip instrumentation when injecting" shortcut.

## What `coarse` measures, and why it is a result rather than a disappointment

D2's four recorded vectors, in order — qkv+mha, norm 1, ffn, norm 2:

```
host_submissions  runlist_entries  air_launches  herd_launches  sync_boundaries      bytes
       1                2               6             10              9          80,216,064
       1               64               1             64            193          18,875,904
       1                1               4              8              7          84,934,656
       1               64               1             64            193          18,875,904
```

**The two 64-entry rows are the normalization points, and they are 64 dispatches each, not one.**
`build_addnorm_module` requires `rows == herd_x * rows_per_call`, which at `cols = 768` caps a call
at 64 of the layer's 4096 rows (`builders/block.py::norm_rows` derives it — do not carry a row count
across widths). So 128 of `coarse`'s 131 entries, 98%, are one operator's row blocking, and its
dispatch numbers are dominated by `addnorm` rather than by the GEMMs.

There are four submissions rather than one because a dispatch argument is a whole BO, so the
sequence cannot be expressed as a single runlist. Both facts belong in `coarse/README.md`. They are
the calibration point the distinguishability gate is defined against, and iron's count is not -- which is **5 entries over 5 kernels** for the encoder, not the "12" this
plan long quoted (that figure sums both workload variants).

## Work items

1. `pattern/coarse/` — a wrapper over `builders/block.py`, its own `README.md`, its own cache
   directory in `.gitignore` and `clean`.
2. A `coarse` operator spec in the `SPECS` catalogue (E1 split it out of `opcheck_specs.py`; add to
   the catalogue module, not back into the mechanism module), using the `dispatch` seam.
3. `run_npu2_coarse_peano.lit` — **both recipes in one file**, the clean check and the
   `--fault-inject input --expect-failure` twin, following `run_npu2_block_peano.lit`. That file is
   one file for two recipes deliberately: they share a working directory and therefore the ELF
   cache. A new `.lit` anywhere under `transformer_layer/` joins the suite automatically; enrolment
   is path-based and there is no `CMakeLists.txt` in the example.
4. The `execution_mode` mapping in one place, so `coarse` → `hybrid` is not spelled out per call
   site.

## Gate

```
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  ninja -C build-xrt check-programming-examples-transformer-layer
```

Every test in the suite passes on real hardware, including the new one.

The driver then, independently: re-derives `coarse`'s verdict from `n_mismatch` / `ref_dtype` /
`rtol` / `atol` rather than reading `passed`; requires exactly one fresh `coarse` result at the
forced configuration with full-layer `n_elements` and ≥8 distinctly-named clean stages; validates the
`dispatch_vectors` contract above; re-runs `coarse` under `--fault-inject input` and requires it to
**fail**; and requires the fault run's summed vector totals to equal the clean run's.

## Risks

- **Two full-layer runs now live in the suite** — `block` and `coarse` — and each lit test starts
  with `make clean`, so the second does not inherit the first's ELF cache. That is real minutes on
  every gate. It is the price of keeping D2's gate provable and it is worth it; if it becomes the
  dominant cost, say so in `work_not_completed` rather than deleting either test.
- **The wrapper must not quietly become a fork.** If you find yourself copying logic out of
  `block.py` into `coarse.py`, stop: the reason this mode is cheap is that it is the same code.
