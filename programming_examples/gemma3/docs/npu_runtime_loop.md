# Gemma3 1B Text NPU Runtime Loop

This runbook is the agent-facing loop for Gemma3 1B text NPU work. It is
decision-complete by design: pick one production blocker, implement one
runtime-owned boundary, collect real hardware evidence, update blocker state,
and stop.

Scope is Gemma3 1B text with prompt length 1024 and decode context 1024.
Probe, HF, synthetic, compile-only, and self-test paths are useful diagnostics,
but they do not clear production NPU blockers.

## Current Baseline

The production-shaped runtime boundary is `gemma3.npu.inference_runtime`.

Current entrypoints:

- `prepare_runtime()`: discovers artifacts, prepares model shape state, builds
  preflight, weight, BO, buffer, argument-binding, and launch-order plans, and
  creates the optional `Gemma3KernelCache`.
- `run_npu_prefill()`: enters `gemma3.npu.prefill_runner`, which now checks
  the Gemma-owned runtime cache for per-layer production prefill artifacts
  (`gemma3_prefill_kv_L*`) and reports a concrete artifact blocker when it
  cannot launch. Explicit JSON evidence is accepted only when passed by path
  for validation or self-test fixtures. HF, synthetic, and probe cache evidence
  cannot satisfy the production path.
- `generate()`: requires production NPU prefill K/V before decode can launch as
  a production handoff.
- `gemma3.npu.runtime_cache`: owns cached XRT artifacts, persistent BO sets,
  static/intermediate write policy, launch timing, readback accounting, and
  cache statistics.
- `gemma3.npu.prefill_runner`: owns the production prefill K/V executor
  boundary, result contract, and all-layer validation helper used by the
  blocker ledger.

Accepted milestones are NPU-only. Diagnostics can guide implementation, but a
diagnostic result must stay labeled diagnostic until it satisfies the evidence
contract below.

## Llama-Derived Runtime Map

Use `programming_examples/llama32_1b` as the runtime-organization reference,
not as imported code and not as accepted Gemma3 evidence. The Gemma3 loop maps
onto the same control-plane shape:

```text
compile/cache artifacts -> prepare_runtime -> run_npu_prefill -> generate -> profile/verify
```

| Llama runtime pattern | Gemma3 runtime boundary | Contract implication |
| --- | --- | --- |
| `KernelCache` artifact manifest | `Gemma3KernelCache` and `gemma3_npu_kernel_manifest.json` | Production prefill cannot pass until the cache manifest names `gemma3_prefill_kv_L0` through `gemma3_prefill_kv_L25` and each binary exists. |
| Per-layer `bo_key` reuse | Gemma3 runtime cache BO sets keyed by layer/runtime artifact | Static weights and mutable K/V buffers must not alias across layers accidentally. |
| `static_input_indices` | Gemma3 static projection/norm BO inputs | Static BO writes happen during setup/preload and are skipped inside timed windows after first use. |
| `intermediate_indices` | Gemma3 virtual/intermediate buffers | Intermediate outputs are not treated as accepted readback evidence unless the contract names them. |
| K/V handoff from prefill to decode | `run_npu_prefill()` result consumed by `generate()` | Decode cannot clear with HF, synthetic, repeated-current-token, or host-replaced K/V. |
| Profile/verify split | `gemma3.evidence.npu_runtime_contracts` plus paper comparison | Contract validation decides blocker state; paper comparison only compares accepted result cells. |

The exact runtime-contract validator for all pre-paper gates is:

```bash
PYTHONPATH=programming_examples/gemma3:sandbox/lib/python3.12/site-packages \
python3 -m gemma3.evidence.npu_runtime_contracts \
  --model-variant gemma3-1b \
  --prompt-len 1024 \
  --decode-context 1024 \
  --runtime-cache-dir programming_examples/gemma3/build_peano/runtime_cache/gemma3-1b \
  --prefill-result programming_examples/gemma3/results/gemma3_1b_npu_prefill_runtime.json \
  --decode-result programming_examples/gemma3/results/gemma3_1b_npu_runtime_decode_loop.json
```

Add `--paper-result <paper-cell.json>` only when validating a TTFT/TPS/power
cell. Add `--allow-blocked` only for loop inspection; it must not be used to
clear a blocker.

## Contract Gate Table

| Gate | Required runtime boundary | Accepted evidence JSON | Exact validator command | Blocker cleared | Next blocker |
| --- | --- | --- | --- | --- | --- |
| Prefill artifacts | `Gemma3KernelCache.load_manifest()` plus `run_npu_prefill()` | Runtime cache manifest and `results/gemma3_1b_npu_prefill_runtime.json` | `PYTHONPATH=programming_examples/gemma3:sandbox/lib/python3.12/site-packages python3 -m gemma3.evidence.npu_runtime_contracts --model-variant gemma3-1b --prompt-len 1024 --decode-context 1024 --runtime-cache-dir programming_examples/gemma3/build_peano/runtime_cache/gemma3-1b --prefill-result programming_examples/gemma3/results/gemma3_1b_npu_prefill_runtime.json --decode-result programming_examples/gemma3/results/gemma3_1b_npu_runtime_decode_loop.json` | `production-prefill-runtime-artifacts-not-cached` | `production-prefill-runtime-arguments-not-bound` or production K/V readiness |
| Production K/V | `run_npu_prefill()` production executor | `results/gemma3_1b_npu_prefill_runtime.json` | `PYTHONPATH=programming_examples/gemma3:sandbox/lib/python3.12/site-packages python3 -m gemma3.evidence.npu_runtime_contracts --model-variant gemma3-1b --prompt-len 1024 --decode-context 1024 --runtime-cache-dir programming_examples/gemma3/build_peano/runtime_cache/gemma3-1b --prefill-result programming_examples/gemma3/results/gemma3_1b_npu_prefill_runtime.json --decode-result programming_examples/gemma3/results/gemma3_1b_npu_runtime_decode_loop.json` | `npu-prefill-kv-cache-not-wired`, then `prefill-1k-npu-not-wired` when timed | Decode handoff |
| Decode handoff | `generate()` consumes the same NPU-produced K/V descriptor | `results/gemma3_1b_npu_runtime_decode_loop.json` | `PYTHONPATH=programming_examples/gemma3:sandbox/lib/python3.12/site-packages python3 -m gemma3.evidence.npu_runtime_contracts --model-variant gemma3-1b --prompt-len 1024 --decode-context 1024 --runtime-cache-dir programming_examples/gemma3/build_peano/runtime_cache/gemma3-1b --prefill-result programming_examples/gemma3/results/gemma3_1b_npu_prefill_runtime.json --decode-result programming_examples/gemma3/results/gemma3_1b_npu_runtime_decode_loop.json` | `generate-prefill-kv-cache-blocked` and `prefill-produced-kv-cache-not-wired` | Attention reduction |
| Attention reduction | Decode attention reduction is NPU-owned or proven unnecessary | Decode runtime JSON with `attention_reduction_mode=npu` or `not-required` | `PYTHONPATH=programming_examples/gemma3:sandbox/lib/python3.12/site-packages python3 -m gemma3.evidence.npu_runtime_contracts --model-variant gemma3-1b --prompt-len 1024 --decode-context 1024 --runtime-cache-dir programming_examples/gemma3/build_peano/runtime_cache/gemma3-1b --prefill-result programming_examples/gemma3/results/gemma3_1b_npu_prefill_runtime.json --decode-result programming_examples/gemma3/results/gemma3_1b_npu_runtime_decode_loop.json` | `npu-attention-reduction-not-wired` | Static BO route |
| Static BO route | FusedDQP decode projections use manifest-backed contiguous static BOs | Decode runtime JSON with `static_projection_argument_mode=manifest-contiguous-static-bo` and nonzero BO sets | `PYTHONPATH=programming_examples/gemma3:sandbox/lib/python3.12/site-packages python3 -m gemma3.evidence.npu_runtime_contracts --model-variant gemma3-1b --prompt-len 1024 --decode-context 1024 --runtime-cache-dir programming_examples/gemma3/build_peano/runtime_cache/gemma3-1b --prefill-result programming_examples/gemma3/results/gemma3_1b_npu_prefill_runtime.json --decode-result programming_examples/gemma3/results/gemma3_1b_npu_runtime_decode_loop.json` | `production-contiguous-static-weight-bo-not-used-by-fused-dqp-route` | Logits/sampling |
| Logits/sampling | Final norm/logits/sampling are NPU-owned or timed/accounted host work | Decode runtime JSON with `logits_sampling_mode`, `sampling_policy`, and ownership records | `PYTHONPATH=programming_examples/gemma3:sandbox/lib/python3.12/site-packages python3 -m gemma3.evidence.npu_runtime_contracts --model-variant gemma3-1b --prompt-len 1024 --decode-context 1024 --runtime-cache-dir programming_examples/gemma3/build_peano/runtime_cache/gemma3-1b --prefill-result programming_examples/gemma3/results/gemma3_1b_npu_prefill_runtime.json --decode-result programming_examples/gemma3/results/gemma3_1b_npu_runtime_decode_loop.json` | `logits-sampling-not-wired` or `logits-sampling-host-diagnostic-only` | Paper cell |
| Paper cell | Blocker-free TTFT/TPS/power result | Paper result JSON passed with `--paper-result` | `PYTHONPATH=programming_examples/gemma3:sandbox/lib/python3.12/site-packages python3 -m gemma3.evidence.npu_runtime_contracts --model-variant gemma3-1b --prompt-len 1024 --decode-context 1024 --runtime-cache-dir programming_examples/gemma3/build_peano/runtime_cache/gemma3-1b --prefill-result programming_examples/gemma3/results/gemma3_1b_npu_prefill_runtime.json --decode-result programming_examples/gemma3/results/gemma3_1b_npu_runtime_decode_loop.json --paper-result <paper-cell.json>` | Paper-comparable 1B/1k NPU cell | Broaden only after the 1B/1k cell is stable |


## Iteration Loop

1. Inspect current state.
2. Read the blocker ledger.
3. Choose exactly one blocker in the required order.
4. Implement one production runtime boundary.
5. Run focused self-test/lit diagnostics.
6. Run the exact hardware path for the selected blocker.
7. Validate the production evidence contract.
8. Update evidence docs and blocker state.
9. Stop.

Do not broaden to 4B, vision, longer contexts, new benchmark cells, or unrelated
refactors during the same iteration.

## Required Blocker Order

Resolve blockers in this order:

1. Production prefill K/V cache.
2. Decode handoff from NPU-produced prefill K/V.
3. NPU-owned attention reduction.
4. Production contiguous static BO route.
5. Final norm, logits, and sampling policy.
6. 1B/1k paper-cell timing and power.

The ordering is intentional. For example, decode timing over HF or synthetic
K/V is still diagnostic even if later decode stages are NPU-owned.


Current first unresolved runtime sub-blocker: `production-prefill-runtime-artifacts-not-cached`.
`run_npu_prefill()` currently finds no cached `gemma3_prefill_kv_L*` production
artifacts, records zero prefill launches, and keeps `prefill-1k-npu-not-wired`
and `npu-prefill-kv-cache-not-wired` active.

## Milestones

| Milestone | Accepted evidence | Blocker effect |
| --- | --- | --- |
| Layer-0 production K/V parity | Layer 0 records `source=production-npu-prefill-kv-cache`, `owner=npu`, `status=PREFILL_KV_CACHE_READY`, nonzero NPU launches, no layer blockers, and parity against the reference K/V rows. | Narrows implementation risk only; no paper blocker clears. |
| All-26-layer production K/V parity | All 26 text layers satisfy the production K/V contract for model `gemma3-1b`, prompt 1024, decode context 1024. | Clears `npu-prefill-kv-cache-not-wired` only. |
| Full timed 1B/1k prefill | `run_npu_prefill` produces the all-layer production K/V cache and reports a timed 1024-token NPU prefill result with nonzero launch accounting. | Clears `prefill-1k-npu-not-wired`. |
| NPU-only decode handoff | `generate` consumes the NPU-produced cache without HF, synthetic, or host-replaced K/V and records the handoff in operation ownership. | Enables decode paper-cell work; does not by itself clear attention/logits blockers. |
| Final 1B/1k paper cell | NPU-only TTFT, TPS, and power records match the paper cell dimensions, prompt/decode counts, timed-window policy, and blocker-free result contract. | Paper-comparable NPU 1B/1k cell is ready. |

## Evidence Contract

`gemma3.evidence.npu_runtime_contracts` is the normative pass/fail interface.
It emits one stable line per contract and returns nonzero when any requested
contract is blocked unless `--allow-blocked` is passed.

Production prefill artifacts must satisfy all of these fields before
`production-prefill-runtime-artifacts-not-cached` can clear:

- Runtime cache manifest is `gemma3_npu_kernel_manifest.json` under the selected
  `--runtime-cache-dir`.
- The manifest contains `gemma3_prefill_kv_L0` through
  `gemma3_prefill_kv_L25`.
- Every referenced `output_binary` exists; every referenced `insts` file exists
  when present.
- `run_npu_prefill()` no longer reports
  `production-prefill-runtime-artifacts-not-cached`.

Production prefill K/V evidence must satisfy all of these fields before it can
clear `npu-prefill-kv-cache-not-wired`:

- Top-level or runtime-cache payload has `model_variant=gemma3-1b`.
- `prompt_token_count=1024` and `decode_context=1024`.
- `layer_count=26`.
- `status=PREFILL_KV_CACHE_READY`.
- `source=production-npu-prefill-kv-cache`.
- `owner=npu`.
- `prefill_kernel_launch_count` or summed layer `kernel_launch_count` is
  greater than zero.
- Every layer has `status=PREFILL_KV_CACHE_READY`,
  `source=production-npu-prefill-kv-cache`, `owner=npu`, no blockers, and K/V
  reference correlation at least `0.99`.

Decode handoff evidence must show that `generate()` consumes the same
NPU-produced K/V descriptor. HF K/V, synthetic K/V, repeated-current-token K/V,
single-current-token diagnostic K/V, and host-replaced K/V never clear this
gate.

Attention reduction evidence must state `attention_reduction_mode=npu` or
`attention_reduction_mode=not-required`. Host reduction, missing mode metadata,
or `npu-attention-reduction-not-wired` keeps the gate blocked.

Static BO route evidence must state
`static_projection_argument_mode=manifest-contiguous-static-bo`, must report a
nonzero static projection BO-set count, and must not carry
`production-contiguous-static-weight-bo-not-used-by-fused-dqp-route`.

Logits/sampling evidence must state `logits_sampling_mode` and
`sampling_policy`. The work must be NPU-owned, or it must be explicitly timed
and accounted as host work in `operation_ownership`; diagnostic host logits do
not clear the gate.

Paper-cell evidence must include prompt/decode shape, warmup and timed
iterations, timed-window policy, aligned power snapshot, launch counts, host
fallback records, blocker-free status, and the TTFT/TPS value.

Compile-only success, self-tests, lit tests, standalone kernel probes,
HF-produced K/V, synthetic K/V, and host-replaced K/V are provisional
diagnostics. They must not clear production prefill, production decode handoff,
or paper-cell readiness.

## Clear Conditions

`npu-prefill-kv-cache-not-wired` clears only when all 26 production K/V layers
exist for Gemma3 1B, prompt 1024, decode context 1024, and the evidence contract
passes.

`prefill-1k-npu-not-wired` clears only when the full timed 1024-token NPU
prefill contract runs through `run_npu_prefill` with nonzero launch accounting
and no production prefill blockers.

`npu-attention-reduction-not-wired` clears only when tiled attention reduction
is NPU-owned, or when the production attention route proves no cross-tile
host-side reduction is required for the paper-cell shape.

The 1B/1k paper cell is ready only when NPU-only TTFT, TPS, and power evidence
exists with exact model, prompt length, decode-token count, layer count,
timed-window policy, and blocker-free result records.

## Commands

Run commands from the repository root. Source existing environment scripts
instead of reconfiguring or rebuilding from scratch.

### 1. Environment Check

```bash
git status --short --branch
bash agents/scripts/doctor.sh env
which python3
python3 --version
which aircc
which mlir-opt
which lit
```

If NPU/XRT tools are missing from `PATH`, recover the shell state before
running hardware commands:

```bash
source /home/cj/iron/ironenv/bin/activate
source /opt/xilinx/xrt/setup.sh
```

### 2. Inspect Blocker Ledger

```bash
PYTHONPATH=programming_examples/gemma3:sandbox/lib/python3.12/site-packages \
python3 -m gemma3.evidence.reproduction_blockers
```

### 3. Prepare Runtime For Gemma3 1B/1024

```bash
PYTHONPATH=programming_examples/gemma3:sandbox/lib/python3.12/site-packages \
python3 -m gemma3.npu.inference_runtime \
  --prepare-runtime \
  --model-variant gemma3-1b \
  --prompt-len 1024 \
  --decode-context 1024 \
  --quantized-weights required \
  --json \
  --result-json /tmp/gemma3_1b_npu_prepare_runtime.json
```

### 4. Run Production Prefill Boundary

Use this command when the implementation should produce production prefill
K/V through the runtime executor. Do not pass `--prefill-evidence-json` unless
you are deliberately validating an already-recorded production fixture:

```bash
PYTHONPATH=programming_examples/gemma3:sandbox/lib/python3.12/site-packages \
python3 -m gemma3.npu.inference_runtime \
  --run-npu-prefill \
  --model-variant gemma3-1b \
  --prompt-len 1024 \
  --decode-context 1024 \
  --quantized-weights required \
  --json \
  --result-json programming_examples/gemma3/results/gemma3_1b_npu_prefill_runtime.json
```

### 5. Validate Runtime Contracts

This command is the normative pass/fail check for all current Gemma3 1B/1k NPU
runtime gates. It exits nonzero while the current blocked state remains blocked:

```bash
PYTHONPATH=programming_examples/gemma3:sandbox/lib/python3.12/site-packages \
python3 -m gemma3.evidence.npu_runtime_contracts \
  --model-variant gemma3-1b \
  --prompt-len 1024 \
  --decode-context 1024 \
  --runtime-cache-dir programming_examples/gemma3/build_peano/runtime_cache/gemma3-1b \
  --prefill-result programming_examples/gemma3/results/gemma3_1b_npu_prefill_runtime.json \
  --decode-result programming_examples/gemma3/results/gemma3_1b_npu_runtime_decode_loop.json
```

For a status-only loop over current blocked artifacts, use the Make target,
which passes `--allow-blocked` intentionally:

```bash
make -C programming_examples/gemma3 model-validate
```

### 6. Run Decode Handoff Boundary

Use this after production prefill K/V evidence passes:

```bash
PYTHONPATH=programming_examples/gemma3:sandbox/lib/python3.12/site-packages \
python3 -m gemma3.npu.inference_runtime \
  --generate \
  --model-variant gemma3-1b \
  --prompt-len 1024 \
  --decode-context 1024 \
  --decode-tokens 1 \
  --quantized-weights required \
  --json \
  --result-json programming_examples/gemma3/results/gemma3_1b_npu_runtime_decode_loop.json
```

`--no-run-hardware` is a diagnostic setup check only. It must not be used as
accepted hardware evidence.

### 7. Focused Diagnostics

```bash
PYTHONPATH=programming_examples/gemma3:sandbox/lib/python3.12/site-packages \
python3 -m gemma3.npu.inference_runtime --self-test
```

```bash
sandbox/bin/lit -v --filter=model_loop_npu_runtime_shell build-xrt/programming_examples
```

Diagnostics can catch contract regressions, but they do not clear production
blockers without the hardware evidence command for the selected blocker.

### 8. Paper Comparison Validation

```bash
PYTHONPATH=programming_examples/gemma3:sandbox/lib/python3.12/site-packages \
python3 -m gemma3.evidence.paper_compare --validate
```

## Blocker Ledger Names Covered

The runbook covers the blocker names emitted by
`gemma3.evidence.reproduction_blockers`: `missing-safetensors`,
`missing-config-json`, `missing-tokenizer`, `missing-processor`,
`missing-python-safetensors`, `missing-python-tokenizer-package`,
`environment-not-paper-comparable`, `unmeasured-nonlinear-host-fallbacks`,
`npu-model-execution-not-implemented`, `production-prefill-runtime-artifacts-not-cached`,
`production-prefill-runtime-arguments-not-bound`, `prefill-1k-npu-not-wired`,
`prefill-produced-kv-cache-not-wired`, `npu-prefill-kv-cache-not-wired`,
`generate-prefill-kv-cache-blocked`, `npu-attention-reduction-not-wired`,
`production-contiguous-static-weight-bo-not-used-by-fused-dqp-route`,
`logits-sampling-not-wired`, `logits-sampling-host-diagnostic-only`,
`vision-path-contract-missing`, and `vision-npu-path-not-validated`.

## Templates

### Blocker Selection Record

```text
date:
branch:
dirty status:
selected blocker:
required-order position:
why this blocker is first unresolved:
files expected to change:
evidence command:
expected result JSON:
non-goals:
```

### Implementation Iteration Note

```text
runtime boundary implemented:
entrypoint:
owned by npu:
remaining host fallbacks:
diagnostic commands run:
hardware command run:
result JSON:
blocker ledger before:
blocker ledger after:
```

### Production Evidence Checklist

```text
model_variant=gemma3-1b:
prompt_token_count=1024:
decode_context=1024:
layer_count=26:
status=PREFILL_KV_CACHE_READY:
source=production-npu-prefill-kv-cache:
owner=npu:
nonzero prefill kernel launches:
all layers status=PREFILL_KV_CACHE_READY:
all layers source=production-npu-prefill-kv-cache:
all layers owner=npu:
layer blockers empty:
HF/synthetic/probe cache absent:
```

### Final Verification Summary

```text
selected blocker:
production evidence accepted:
result files updated:
docs updated:
commands passed:
commands failed or not run:
blockers cleared:
blockers remaining:
next required blocker:
```
