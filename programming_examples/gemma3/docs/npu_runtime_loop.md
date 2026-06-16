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

Production prefill K/V evidence must satisfy all of these fields before it can
clear `npu-prefill-kv-cache-not-wired`:

- Top-level or runtime-cache payload has `model_variant=gemma3-1b`.
- `prompt_token_count=1024`.
- `decode_context=1024`.
- `layer_count=26`.
- `status=PREFILL_KV_CACHE_READY`.
- `source=production-npu-prefill-kv-cache`.
- `owner=npu`.
- `prefill_kernel_launch_count` or summed layer `kernel_launch_count` is
  greater than zero.
- Every layer has `status=PREFILL_KV_CACHE_READY`.
- Every layer has `source=production-npu-prefill-kv-cache`.
- Every layer has `owner=npu`.
- No layer has blockers.

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

### 5. Validate Production Prefill Evidence

This command is the normative pass/fail check for clearing
`npu-prefill-kv-cache-not-wired`:

```bash
PYTHONPATH=programming_examples/gemma3:sandbox/lib/python3.12/site-packages \
python3 -c 'from pathlib import Path; from gemma3.npu.prefill_runner import has_all_layer_production_prefill_evidence; ok = has_all_layer_production_prefill_evidence("gemma3-1b", prompt_len=1024, decode_context=1024, layers=26, path=Path("programming_examples/gemma3/results/gemma3_1b_production_prefill_kv_cache.json")); raise SystemExit(0 if ok else 1)'
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
