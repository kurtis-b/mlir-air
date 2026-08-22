# 32 — The first cost-decomposed four-mode ladder, and what it caught

> **`[2026-08-22]` Every latency in this document is under the OLD clock** (the per-boundary
> comparison inside the timed region, ~24–27 ms per forward at 512; [54 §5](54-first-full-profile-and-decoder-families.md)).
> Its orderings stand. The absolute numbers to cite are the re-walk under the forward-only clock
> with rule S1 once per plan — `results/rewalk-doc32-w3`, devq 508, same `--warmup 2 --samples 5`,
> same mode order, Turbo: at 512 `offload` **75.0** / `runlist` **43.2** / `coarse` **12.7** /
> `fused` **9.7** ms; at 1024 **102.9** / **68.3** / **21.2** / **15.4** (`results/rewalk-doc32-w2`, the same ladder under the old
> clock, devq 353: 101.7 / 73.3 / 43.1 / 39.1 and 153.2 / 130.0 / 78.4 / 77.6). `fused` < `coarse` < `runlist` <
> `offload` 8/8, as before; the `fused`/`coarse` margin at 1024 that had "nearly closed" is 27 %
> under the new clock (15.4 vs 21.2) — the comparison had been hiding it.

`[2026-08-10]` The first ladder walked with schema v2's cost columns: four modes × 512/1024,
`--warmup 2 --samples 5`, one process per rung, **two walks** (results/ladder-v2-w1, -w2,
gitignored; 16/16 rungs passed). Three findings, in decreasing order of solidity — and the third
is a **machine-state anomaly the new instruments localized on their first day**, which is why
this document leads with what may NOT be concluded from the walk.

## What may not be concluded: any cross-day latency comparison

**`runlist` and `offload` read ~15–20× above their recorded latencies at the same shapes**
(runlist ~1.9 s, offload ~2.4 s at 1024, against [29](25-mode-rebuilds-and-results.md)'s 164–183 ms for
offload at 1024 on 2026-08-09), while `coarse` and `fused` match their records (~48/~90–105 ms).

**It is the machine, not the day's code.** A worktree at `3b9f811a` — before every one of today's
changes — with today's warm caches symlinked in reproduces the inflation: runlist 1968 ms,
offload 2408 ms at 1024. The same bisect run reproduced doc 27's runlist bytes **byte-for-byte**
(55,246,848), so the old code ran exactly as recorded in everything the environment does not touch.

**The decomposition attributes it.** The inflation tracks `context_loads` exactly: the two
zero-load modes are unaffected, and unattributed host time per load —
`(avg_latency − device_ms − sync_ms − host_cpu_ms) / context_loads` — reads **~78–80 ms per
`hw_context` load** from `runlist` (24 loads) and `offload` (30 loads) *independently*, both
walks, both lengths. Yesterday's recorded offload minimums (78–82 ms total at 512, 30 loads
inside) bound the same cost at **≤ 2.6 ms/load on 2026-08-09**. So the machine's context-creation
cost rose ~30× overnight. Cause unknown; `xrt-smi examine` shows nothing anomalous from
userspace. **`[2026-08-11]` A clean `amdxdna` reload does NOT clear it** — the same rung read
2514 ms avg (~80.9 ms/load residual) immediately after — and nothing on disk changed since May
(XRT userspace, `/lib/firmware/amdnpu`, kernel all pre-date the anomaly), so it is runtime
platform state. The remaining diagnostic is a **full reboot** followed by one `offload` rung —
**do not publish any latency comparison against pre-2026-08-10 numbers until that is done**, and
treat even this walk's runlist/offload-vs-rest gaps as conditional on the anomaly
([27](27-common-ladder-result.md) called runlist/offload latency-indistinguishable at healthy
load costs; today's clean ~25% separation is the per-load cost times a 24-vs-30 difference, not
necessarily the modes).

**`[2026-08-11]` RESOLVED: it was the NPU power mode.** The full reboot did NOT clear it (verdict
rung 2717 ms avg, 82.5 ms/load residual), and a userspace experiment then refuted runtime PM as
the mechanism: pinning the device active with a held `hw_context` from a second process (device
`runtime_status: active` throughout) left the rung at 82.2 ms/load — the cost is in context
creation itself, not device wake. What settled it was the boot history: the machine **rebooted
itself at 01:09 on 2026-08-10 — the exact onset** — and a reboot resets the non-persistent
`xrt-smi` power mode to `Default`. The healthy window (08-04 → 08-09) sat inside one
uninterrupted boot (08-03 09:50 → 08-10 01:08) that carried the **Turbo** set for the C4
registry sweep — `registry_sweep.py`'s `require_turbo()` is the paper trail; the registry's
numbers are Turbo-conditional by design and C4 ran under that guard on 08-04. Every "it survives
X" observation follows: an `amdxdna` reload and a reboot BOTH reset pmode, and nothing on disk
changes because pmode is runtime state. (One correction in passing: the `[2026-08-11]` "kernel
unchanged" claim above was wrong — the 08-10 overnight boot ran `6.14.0-1020-oem`, not the
healthy window's `7.0.0-28-generic` — but the kernel is not the discriminator: the post-reboot
slow rung ran back on `7.0.0-28-generic`.)

**Confirmation, same rung, minutes apart on 2026-08-11:** after
`sudo xrt-smi configure --device 0000:64:00.1 --pmode turbo` the rung reads **156.2 ms avg**
(min 146.1; device 20.6 / sync 13.8 / host_cpu 11.6 ms), residual **3.7 ms/load** — back inside
doc [29](25-mode-rebuilds-and-results.md)'s 164–183 ms band. At `Default`, the same rung minutes earlier:
2717/2547 ms avg, ~82 ms/load, device_ms ~38. So `Default` costs ~22× on context creation and
~1.8× on device compute at this shape.

**Consequences.** (1) **`pmode` is a measurement condition.** Every latency in this document's
2026-08-10 walks was measured at `Default` (they ran after the 01:09 reboot); every pre-08-10
record is Turbo-conditional. The byte and count columns are pmode-independent. (2) Turbo must be
re-set after every reboot or `amdxdna` reload, and verified (`xrt-smi examine -r platform`)
before any latency run; `run_mode.py` now enforces this itself — the same `require_turbo()`
guard the registry sweep has, checked before anything is prepared (exit 2, no row), so a
`Default`-pmode measurement can no longer run silently. (3) That `fused`/`coarse` at `Default` still matched their records is
consistent with their totals being host-dominated at these shapes (device_ms 10–13 of ~50–105 ms
total) — which is exactly why the anomaly presented as context-load-specific.

`fused` and `coarse` latencies ARE comparable to their records, and match them.

## What is solid: the byte totals, walk-identical, and a changed warm ordering

| mode | 512 bytes | 1024 bytes | context_loads | device/sync/host @1024 (ms, w1) |
|---|---|---|---|---|
| `runlist` | **20,447,232** | **40,894,464** | 24 | 45.1 / 5.7 / 0.0 |
| `fused` | 21,233,664 | 42,467,328 | 0 | 10.3 / 1.2 / 0.0 |
| `coarse` | 22,020,096 | 44,040,192 | 0 | 13.4 / 2.0 / 0.0 |
| `offload` | 44,040,192 | 99,090,432 | 30 | 40.3 / 23.6 / 24.0 |

Every byte total is identical between the walks, and `fused`/`coarse`/`offload` reproduce
[27](27-common-ladder-result.md)'s figures exactly. **`runlist` does not, by design**: it is
14,352,384 bytes below doc 27's 55,246,848 at 1024 — the static-weight set, to the byte — because
the targeted pool eviction (queue item 4, landed this morning) stopped destroying its
static-weight pools per head. The pre-today bisect run read doc 27's figure exactly, so the delta
is the fix and nothing else.

**Consequence: the warm steady-state DRAM ordering is now `runlist < fused < coarse < offload`**
at both lengths — doc 27's `fused < coarse < runlist` no longer holds warm. This does not
contradict the taxonomy's mechanism claims (the intermediate-traffic story is unchanged; what
moved is re-upload traffic that was never intermediate), but every sentence citing doc 27's
ordering should now cite it as the ordering *under the wholesale-eviction implementation*.
`runlist` under `fused` decomposes cleanly: its front moves 2×[S,E] fewer bytes than the block
front ([30](25-mode-rebuilds-and-results.md)'s solved system, at this length ~3.1 MB) against its tail's
small broadcast surcharge.

## The reconfiguration columns, on their first walk

`offload`-ELF **30**, `runlist` **24** (12 heads × 2 attention artifacts), `coarse` and `fused`
**0** — per dispatch, identical across walks and lengths. `runlist`'s middle regime is now a
recorded number rather than an inference, and `offload`'s definitional contradiction (the
maximum reconfiguration cost in the mode defined to minimize it, on its default path) is now a
column any results reader can see.

## Two wiring defects the walk caught (both fixed same-day)

- **`offload`'s extra reported the cumulative counter, not the per-dispatch delta** — measured as
  `context_loads 210` (= 30 × 7 harness iterations) at warmup 2 / samples 5, against 30 from a
  warmup 1 / samples 2 run of the same code. `run_mode` records the LAST timed dispatch's extra,
  so a cumulative value multiplies with the iteration count. Fixed (`4ced893b`) and re-validated
  at the exact configuration that read 210.
- The walks completed in ~91 s each, which initially read as "did not dispatch" — it is real
  (warm caches, light shapes, 7 dispatches/rung), but it is also the reason the anomalous
  latencies got noticed at all: the numbers were checked against the recorded distributions
  before being believed, per the README's standing rule.

## `[2026-08-11]` The post-flip walk: the first unconditional four-mode comparison

> **`[2026-08-18]` Read this section's ORDERINGS as current and its LATENCIES as superseded.** A
> re-walk under this walk's exact conditions (devq 353; warmup 2 / samples 5, same mode order, two
> walks, Turbo verified in-job; roots at `results/rewalk-doc32-w{1,2}`) reproduces
> `fused` < `coarse` **8/8** on avgs and minimums — but every absolute number here moved **down
> 13–26%** (all 8 (mode, length) pairs, same direction), consistent with the three 2026-08-12
> compiler changes, which postdate these trees. The @1024 `fused`/`coarse` margin has nearly closed
> (0.76 ms avg / 0.04 ms min in the re-walk's w2), so treat that pair as tied at 1024. Dispatch
> vectors and `sync` counts are unchanged. Also measured on the way: `fused`@1024 at **warmup 1** is
> unstable (101.5 then 80.0 across two walks) and stabilizes to 0.4% at warmup 2 — a cross-mode
> comparison at warmup 1 compares warmup tails, not modes, which is what made devq 352's inversion
> look real.

The re-walk item 3's flip owed, run the day the flip landed: four modes at 512 and 1024,
`--warmup 2 --samples 5`, walked twice under one devq measure job each, **Turbo verified
in-job before either walk** (trap 0), with `offload` on its NEW default — the shared xclbin
(`npu_unique_xclbin_count 1` over 30 dispatches in every offload row, which is the CSV's own
attestation that the shared path ran). Artifacts:
`results/postflip-ladder-w1/`, `-w2/` (gitignored), devq job log 224.

**Latency orders `fused` < `coarse` < `runlist` < `offload` — on averages AND minimums, both
walks, both lengths, 16/16 orderings.** For the first time all four modes fully separate:
doc 27's `runlist`/`offload` indistinguishability is gone, removed by the flip (`offload`'s
ELF-era intra-walk spread of 120–316% collapsed to ~3–16% shared; one walk-1 sample at 1024
reached 238 ms against a 163.6 min, so its band is not yet tight — but no statistic in any
walk brings the pair closer than 23 ms).

| avg ms (w1 / w2) | 512 | 1024 |
|---|---|---|
| `fused` | 45.73 / 45.23 | 100.84 / 99.48 |
| `coarse` | 53.66 / 48.25 | 105.02 / 111.95 |
| `runlist` | 77.82 / 77.51 | 150.96 / 152.94 |
| `offload` | 109.49 / 99.72 | 179.88 / 175.48 |

Minimums preserve the same order (512: 44.4 / 47.2 / 76.8 / 102.6 on walk 1; 1024: 98.6 /
103.5 / 149.9 / 163.6; walk 2 agrees).

**Bytes are walk-identical and reproduce this doc's warm ordering** — `runlist` < `fused` <
`coarse` < `offload`: 20,447,232 / 21,233,664 / 22,020,096 / 44,040,192 at 512 and
40,894,464 / 42,467,328 / 44,040,192 / 99,090,432 at 1024.

**The reconfiguration column shows the flip doing its work**: `offload`'s timed-region
`context_loads` is now **0** (the standing context is created once at setup and outlives every
dispatch — against 30 *per dispatch* on the ELF path in this doc's first walk), `runlist` keeps
its measured 24, `coarse`/`fused` 0. `offload`'s host half remains visible as `host_cpu_ms`
4.2 / 12.7 (512 / 1024), which no other mode has — the two taxonomy costs and the mode's host
residue are now all separately legible in one row.

**One confound, declared**: this walk is not single-variable against doc 27's or this doc's
earlier walks — it also carries the two-pass f32 `addnorm` (item 7's kernel change, merged the
same day) in the three modes that dispatch `addnorm` on the device; `offload` norms on the host
and does not. Within-walk cross-mode comparison is unaffected (that asymmetry is real mode
behavior); against pre-flip records, both the packaging and the kernel changed.
