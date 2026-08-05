# `coarse` — few fused kernels, one runlist per sequence

The coarse-runlist point of the Phase E execution-strategy taxonomy — iron's
`hybrid`; per porting convention 7 the directory and the operator are `coarse`
and only the CSV `execution_mode` value keeps the old name, mapped in one place
(`pattern/__init__.py::EXECUTION_MODE_CSV`).

The mode **is** `builders/block.py`: the D2 layer, five operator launches over
four separately compiled ELFs dispatched through `KernelCache.run_sequence`.
`coarse.py` wraps it — its own ELF cache directory (`coarse_cache/`), its own
operator name in the `SPECS` catalogue, its `execution_mode` value — and adds
no device logic. That is deliberate: the reason this mode is cheap is that it
is the same code the block gate already proves correct, and the reason it is a
wrapper rather than a re-export is that each mode's artifact must be a separate
claim the driver can hold to the mode contract independently.

## What this mode isolates

The boundary between *fused-operator granularity* and *dispatch aggregation*.
Each of the layer's four sequences is one host submission whose runlist holds
every kernel invocation in that sequence; within a sequence, intermediates stay
on the device. What `coarse` does **not** do is cross a sequence boundary
without the host: a dispatch argument is a whole BO (`run.set_arg` takes a
buffer, never a buffer plus an offset), so the row-blocked normalization points
cannot share a runlist with the tensors they consume, and the layer cannot be
one submission.

## What it measures, and why the numbers look the way they do

Four recorded `DispatchVector` rows, one per sequence — qkv+mha, norm 1, ffn,
norm 2:

```
host_submissions  runlist_entries  air_launches  herd_launches  sync_boundaries      bytes
       1                2               6             10              9          80,216,064
       1               64               1             64            193          18,875,904
       1                1               4              8              7          84,934,656
       1               64               1             64            193          18,875,904
```

Driver-summed totals: 4 submissions, 131 runlist entries, 12 air launches,
146 herd launches, 402 sync boundaries, 202,902,528 bytes. The fault-injected
twin totals identically — injection perturbs one input element after the
reference exists and never touches the dispatch path.

Two facts about that table are the calibration the distinguishability gate is
defined against, and neither matches iron's "12 entries" figure:

- **The two 64-entry rows are the normalization points, and they are 64
  dispatches each, not one.** `build_addnorm_module` requires one kernel call
  per tile, and L1 caps a call at 64 of the layer's 4096 rows at `emb_dim` 768
  (`builders/block.py::norm_rows` derives it — the cap moves with the width).
  So 128 of the 131 entries — 98% — are one operator's row blocking, and
  `coarse`'s dispatch numbers are dominated by `addnorm`, not by the GEMMs.
- **Four submissions rather than one** because a dispatch argument is a whole
  BO, so the sequence cannot be expressed as a single runlist (see above).

E4's `runlist` mode owns the ordinal claim *more entries than coarse*, and 131
is the number it is measured against — which is why `coarse` had to be
measured before any decomposition claim could mean anything.

## What it costs

A second full-layer run in the lit suite (`run_npu2_coarse_peano.lit` beside
`run_npu2_block_peano.lit`). Each lit test starts with `make clean` in its own
working directory, so `coarse` compiles its four ELFs into its own
`coarse_cache/` rather than inheriting `block`'s — real minutes on every gate.
That is the price of two things worth paying for: D2's gate stays provable
against its own artifact set, and the two modes can never trade ELFs. The
cache directory is chosen by NAME (`KernelCache(cache_dir=...)`), and two modes
pointed at one directory can exchange ELFs whose fingerprints happen to agree —
numerically valid output attributed to the wrong execution boundary, a failure
no equivalence check would surface. `coarse_cache/` is gitignored and in
`make clean`, in the same commit that created it, because the driver's negative
control runs `opcheck.py` from the source directory and the cache lands there —
exactly the leak D2's `block_cache/` had.
