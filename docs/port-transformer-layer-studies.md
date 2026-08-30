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
living path cites one; multi-launch uses the ELF path only; the GEMM kernel-registry rows port as
authored data under the cap.

## Scoreboard (branch-added lines; updated in the PR that moves a row)

| Cluster | Include | Refactor | Excluded | Landed | Remaining | PRs |
|---|---:|---:|---:|---:|---:|---|
| B `agents/` | 1,017 | 0 | 3,194 (+1,397 already on main) | 839 | 178 | #6 devq core (B1); B2 new-job + selftest (this PR, 432); B3 audit script rides with E's GEMM feature |
| F compiler | 1,942 | 6,099 | 2,831 | 290 | 7,751 | H10 non-constant BD offset refusal (this PR, 290); prepared: shared-L1 put guard, shrink extent, H3 verifier, split-l2 B |
| E `llms/` + kernels | 14,477 | 7,233 | 6,236 | 377 | 21,333 | qkv-heads kernel + layout (this PR, 377); prepared: verify_runner host tests (304), qwen3 QKV seam refactor; next: the 2-launch ELF builder, then the driver switch (E2 plan) |
| D `transformer_layer/` | 0 | 0 | 69,330 | — | 0 | excluded (Q1) |
| C plan docs | 0 | 0 | 12,805 | — | 0 | excluded (Q3) |
| other | 0 | 29 | 41 | 0 | 29 | rides with B |
| **total** | **17,436** | **13,361** | **94,437** | **1,506** | **29,291** | 35 PRs include-only; 62 at full refactor size |

Loop-stop condition: Remaining = 0 for the include set; refactor rows close when their
re-derivation lands or is recorded as not needed.

## Slice order

Shared infra (B: devq) → compiler rows that are self-contained (H10 non-constant BD offset refusal,
`92b05de9` shared-L1 lock placement, shrink extent, split-L2 B/C/D, H3 verifier) → the `?` rows
settled by lit runs against `main`'s `air-opt` and one devq job → E model-side work (qwen3,
llama32_1b int4, 4-launch QKV, registry rows) → E refactors onto air.api / `llms/shared`.
