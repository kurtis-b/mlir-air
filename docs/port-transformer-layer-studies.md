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
| E `llms/` + kernels | 14,477 | 7,233 | 6,236 | 9,524 | 12,186 | registry baseline_512 qkv rows (this PR, 468 carried; R5c-6); registry baseline_512 skeleton + ffn_down (#42, 302 carried; R5c-5); registry baseline_768 o_proj rows, family complete (#41, 430 carried after its review fixes; R5c-4); registry baseline_768 ffn_down rows (#40, 430 carried after its review fixes; R5c-3); registry baseline_768 ffn_up rows (#39, 423 carried; R5c-2); registry herd + restructure + qkv rows (#38, 637 carried after its review fixes; R5c-1); GEMM object-link audit + remaining renames (#37, 234 after its review fixes; R5b); per-(tile_m,tile_n) GEMM object minting (#36, 429; R5a); r=64 int4 GEMV row strip (#35, 113 after its review fixes; R4); W4 three-arm verify gate + default flip (#34, 418 after its review fixes; R3c); W4 decode driver switch + artifact guard (#33, 336 after its review fixes; R3b); W4 decode pack + host test (#32, 482 after its review fixes; R3a); Q15 causal-window lit (#31, 40); R1 census fail-open fixes across example dirs (#30, 287 after its review fixes); SmolLM2 int4 verify adapter + docs (#29, 406; the Q4_0 feature complete); prompt_len regression test + guide (this PR, main-side follow-up of #21, not branch lines); SmolLM2 int4 decode/inference/Makefile (#27, 464; plan PR 8); SmolLM2-1.7B GGUF q4_0 loader (#26, 432; plan PR 7); promoted q4_1→q4_0 route (#25, 409; plan PR 6); q4_0 GEMV harness + device lit (#24, 370; plan PR 5); q4_0 codec + repack + self-test (#23, 342; plan PR 4); GGUF container reader (#22, 356; plan PR 3); prompt_len on the Llama drivers (#21, 26; plan PR 2); int4_gs plumbing (#20, 147; int4 Q4_0 plan PR 1); Qwen3-0.6B decode QKV at 2 launches (#19, 257); 2-launch QKV ELF builder (#17, 479); qkv-heads kernel + layout (#16, 395); rms/qkv host-ABI seam refactor (#15, main-side structure-only, net +3 — not branch lines); verify_runner host tests (#14, 412); next: shared decode_qkv4 + test, then the qwen3 / llama32_1b_int4 / smollm2 int4 model rows, int4_awq q4_0, registry rows (data, ≈11 PRs) |
| D `transformer_layer/` | 0 | 0 | 69,330 | — | 0 | excluded (Q1) |
| C plan docs | 0 | 0 | 12,805 | — | 0 | excluded (Q3) |
| other | 0 | 29 | 41 | 0 | 29 | rides with B |
| **total** | **17,436** | **13,361** | **94,437** | **12,644** | **18,153** | 35 PRs include-only; 62 at full refactor size |

Loop-stop condition: Remaining = 0 for the include set; refactor rows close when their
re-derivation lands or is recorded as not needed.

## Slice order

Shared infra (B: devq) → compiler rows that are self-contained (H10 non-constant BD offset refusal,
`92b05de9` shared-L1 lock placement, shrink extent, split-L2 B/C/D, H3 verifier) → the `?` rows
settled by lit runs against `main`'s `air-opt` and one devq job → E model-side work (qwen3,
llama32_1b int4, 4-launch QKV, registry rows) → E refactors onto air.api / `llms/shared`.
