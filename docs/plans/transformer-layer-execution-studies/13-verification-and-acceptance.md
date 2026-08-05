# 13 — Verification and Acceptance

Every gate in the plan, in one place.

## Environment

Hardware runs need:

```bash
source utils/env_setup.sh <install> <mlir-aie> <llvm-aie> <llvm>
source <xrt>/setup.sh
sudo xrt-smi configure --pmode turbo
xrt-smi examine -r all          # confirm "Power Mode : Turbo"
```

and every NPU command wrapped in the repository-wide lock:

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock <command>
```

`KernelCache` serializes internally on `/tmp/npu.lock` — deliberately a different inode, to
avoid flock self-deadlock. Do not unify them.

Per AGENTS.md, prefer incremental rebuilds: a targeted `ninja` in the existing build directory,
then the narrowest useful test.

## Gate table

| Level | Command | Gates on |
|---|---|---|
| Conventions | `black --check .`, clang-format / clang-tidy, plus the [02](02-porting-conventions.md) checklist at review | No iron-shaped code lands: no `AIE*` operator classes, no `op.py`/`design.py` pairs, no `REUSE.toml`, no module materially over ~800 lines |
| Kernels compile | `ninja check-programming-examples-transformer-layer` | Phase A — compile-only lit, no NPU needed, PR-gate-safe |
| Runlist spike | Hardware test with the real separately-compiled artifacts | Phase B — the taxonomy's load-bearing assumption |
| Operator numerics | `transformer_layer/opcheck.py --operator <op>`, one `run_npu2_<op>_peano.lit` per operator | Phase C — full-output `np.isclose` at registry `rtol`/`atol` vs an FP32 reference, zero mismatches |
| Check discriminates | `opcheck.py --operator <op> --fault-inject input`, which must **fail** | Phase C — a vacuous check passes under injection; the driver fails the phase for it |
| Registry coverage | Rows present in `supported_kernels.md` + `details/<Kernel>_bf16.md`, and `gemm_config()` resolving | Phase C — every case-matrix shape registered or provably dynamic |
| Operators at the block's width | `opcheck.py` at the `baseline_768` widths, one `run_npu2_<op>_peano.lit` each | Phase D1 — every operator right at the width the block runs, including the pre-add `addnorm` |
| Single block | `opcheck.py --operator block`, `run_npu2_block_peano.lit` | Phase D2 — launch maps, layouts, external linking, BO reuse. Full `seq × hidden` output, zero mismatches, plus a clean per-boundary intermediate at each of ten stages |
| The golden model's composition | `make reference-tests`, `run_reference_tests.lit` | Phase D2 — host-only, pins erf vs tanh GeLU, post-add vs pre-add residual, and QKV column order, which a numerical comparison would survive |
| Strategy equivalence | `pytest programming_examples/transformer_layer/pattern/` | Phase E — all four modes vs the shared FP32 reference |
| Strategy distinguishability | Dispatch vector per mode | Phase E — the vectors separate the modes as predicted |
| Harness plumbing | `unattended_reboot smoke-test` | Phase F — plot/regeneration path only, measures nothing |
| End-to-end setup | `unattended_reboot execution-smoke-test` | Phase F — **≥1 row with `run_status=passed` per measurement CSV** |
| Full suite | `unattended_reboot start --suite-profile <profile>` | Phase G — complete manifest, counts derived from the profile |
| Cross-run sanity | `compare_results_roots <old> <new>` | Median/p90 drift within per-mode tolerance |
| Sliding window | `make verify` with window-crossing prompts | Goal 1 |
| Quantized path | `make verify` under a gate exercising the quantized path | Goal 2 |
| LLM regression | `make verify` in each `llms/<model>/` | Shared-infrastructure changes did not break the ten shipped models |

## Two gates that deserve emphasis

**`execution-smoke-test` must check rows, not files.** A broken environment still writes
complete, well-formed CSVs full of failed rows. iron shipped a smoke test that checked only that
expected files existed and were non-empty, and it reported 21/21 passed on a machine where every
measurement had failed. The gate must require at least one `run_status=passed` row per
measurement CSV and report the first `failure_message` verbatim.

**Goal 1's verify must cross the window boundary.** The standard gate is a top-5 token-set
inclusion check over 32 decoded tokens. Short prompts never reach the window edge, so a pass
would prove nothing about windowing. Long-prompt fixtures are part of the gate, not an extra.

## The cross-deployment regression rule

Phase B modifies `programming_examples/llms/shared/infra/cache.py`. Phase C may touch
`shared/builders/`. Phase A may extend `matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc`.
Goal 2 hoists builders into `shared/`.

`[2026-08-05]` **Phase E almost certainly joins them.** Phases C and D were both forbidden from
touching `llms/shared/builders/gemm_builder.py`, and both hit the same consequence: its symbol and
object names are minted from the GEMM method alone, ignoring `tile_n`, so two same-method GEMMs at
different `tile_n` collide — at the symbol level in `stitch_elf`, and at the object level in
`compile_gemm_mm`, which is the same bug reached twice. It confines the whole study to
`seq = 4096`. Phase E needs the sequence ladder, so it is the phase that has to make the change,
and its gate has to carry this rule.

All of these touch shared infrastructure that the ten shipped LLM deployments depend on. The rule
from `deploy-new-llm` applies: **after any shared-infrastructure change, re-run `make verify` on
every sibling model**, serialized under `flock`.

```bash
for m in llama32_1b llama32_1b_int4 llama32_3b smollm2_1_7b \
         qwen25_0_5b qwen25_1_5b qwen25_3b qwen3_0_6b qwen3_1_7b qwen3_4b; do
  (cd programming_examples/llms/$m && flock -x -w 1800 /tmp/mlir-air-npu.lock make verify) \
    || echo "REGRESSION: $m"
done
```

This is the most expensive check in the plan and the one most likely to be skipped. It is also
the one that catches the failures that matter most.

## Correctness standards in this repository

Worth restating, because they differ from what is intuitive:

- **Kernel numerics gate on element-wise `np.isclose`** at the registry's `rtol`/`atol` against
  an FP32 reference — explicitly **not** cosine similarity. The `kernel_registry` README is
  direct about this. Cosine hides per-element errors that matter.
- **Per-layer cosine (`make diagnosis`) is informational only.** It never fails a run. It is a
  localization lens for finding *where* a regression happened, not a gate.
- **Model correctness gates on top-k token-set inclusion** (k=5, first divergence over 32
  tokens) against a Hugging Face bf16 reference, mirroring vLLM's `check_logprobs_close`.
- **Reductions accumulate in FP32** even when inputs and outputs are bf16.

## Pre-existing issues worth fixing en route

None of these are caused by this port, but several phases touch the same files.

- **Stale skill paths.** All 15 `SKILL.md` files (in both `.claude/skills/` and `.codex/skills/`)
  reference `programming_examples/llms/llama_kernel_builder/`, renamed to `llms/shared/` in
  commit `2f20c2fa`. `deploy-new-llm` Step 3 `test -d`'s that path and always reports MISSING.
  Goal 1 edits these files anyway.
- **`verify/README.md` is stale** — titled for Llama-3.2-1B and documenting a
  `runners/npu_runner.py` that no longer exists; per-model `verify_adapter.py` replaced it.
- **`llama32_1b_int4/README.md` is stale** — claims int4 decode is a follow-up PR when it landed
  in `aa73c0d7`, and points at `../llama_kernel_builder/`. Goal 2 fixes this as its first step.
- **`docs/ai_skills.md` is an explicit placeholder** promising to document the LLM-deployment
  skills. Once this port lands, it is a natural home for the methodology.
