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
| E `llms/` + kernels | 12,972 | 7,233 | 7,741 | 12,822 | 7,383 | qwen3_1_7b o_ffn rewire + sidecar bind, net -12 (this PR, 338 added); qwen o_ffn mixed-method rewire (#51, 302 added after its review fix); registry qwen3_0_6b family, SERIES COMPLETE (#50, 248 carried after its review round; R5c-13); registry baseline_1024 o_proj rows, family complete (#49, 337 carried; R5c-12); registry baseline_1024 ffn_down rows + short-M test (#48, 296 carried; R5c-11); registry baseline_1024 ffn_up rows (#47, 475 carried after its review fix; R5c-10); registry baseline_1024 skeleton + qkv (#46, 397 carried after its review fix; R5c-9); registry baseline_512 o_proj rows, family complete (#45, 428 carried after its review fix; R5c-8); registry baseline_512 ffn_up rows (#44, 475 carried after its review fix; R5c-7); registry baseline_512 qkv rows (#43, 470 carried; R5c-6); registry baseline_512 skeleton + ffn_down (#42, 302 carried; R5c-5); registry baseline_768 o_proj rows, family complete (#41, 430 carried after its review fixes; R5c-4); registry baseline_768 ffn_down rows (#40, 430 carried after its review fixes; R5c-3); registry baseline_768 ffn_up rows (#39, 423 carried; R5c-2); registry herd + restructure + qkv rows (#38, 637 carried after its review fixes; R5c-1); GEMM object-link audit + remaining renames (#37, 234 after its review fixes; R5b); per-(tile_m,tile_n) GEMM object minting (#36, 429; R5a); r=64 int4 GEMV row strip (#35, 113 after its review fixes; R4); W4 three-arm verify gate + default flip (#34, 418 after its review fixes; R3c); W4 decode driver switch + artifact guard (#33, 336 after its review fixes; R3b); W4 decode pack + host test (#32, 482 after its review fixes; R3a); Q15 causal-window lit (#31, 40); R1 census fail-open fixes across example dirs (#30, 287 after its review fixes); SmolLM2 int4 verify adapter + docs (#29, 406; the Q4_0 feature complete); prompt_len regression test + guide (this PR, main-side follow-up of #21, not branch lines); SmolLM2 int4 decode/inference/Makefile (#27, 464; plan PR 8); SmolLM2-1.7B GGUF q4_0 loader (#26, 432; plan PR 7); promoted q4_1→q4_0 route (#25, 409; plan PR 6); q4_0 GEMV harness + device lit (#24, 370; plan PR 5); q4_0 codec + repack + self-test (#23, 342; plan PR 4); GGUF container reader (#22, 356; plan PR 3); prompt_len on the Llama drivers (#21, 26; plan PR 2); int4_gs plumbing (#20, 147; int4 Q4_0 plan PR 1); Qwen3-0.6B decode QKV at 2 launches (#19, 257); 2-launch QKV ELF builder (#17, 479); qkv-heads kernel + layout (#16, 395); rms/qkv host-ABI seam refactor (#15, main-side structure-only, net +3 — not branch lines); verify_runner host tests (#14, 412); next: NOT decode_qkv4 — that row is closed above (Q7: 2 launches only). The next features, in the split plan's dependency order, are the LM-head / idle-row family (ranked first by the split plan on **tag-era** measurements — `2e14f533` idle-row fill (devq 679/674), `93ef7040` + `f0262b18` Qwen3-0.6B LM head 19→10→3 launches (devq 476/471, 688/691), `1e234f18` Llama m_input 8 (devq 563/564). **None re-measured on main**, and this ledger makes no performance claim of its own; the ranking is the plan's, cited so it can be audited. It also carries a re-derivation risk: the tag's `herd_rows` was written on the raw-bindings `matvec.py` while main's is air.api with none, and main has since grown `use_lock_race_condition_fix_v2` alongside the v1 flag the marked-herd path uses), then the qwen3 / llama32_1b_int4 / smollm2 int4 model rows, int4_awq q4_0, and the registry rows (data, ≈11 PRs) |
| D `transformer_layer/` | 0 | 0 | 69,330 | — | 0 | excluded (Q1) |
| C plan docs | 0 | 0 | 12,805 | — | 0 | excluded (Q3) |
| other | 0 | 29 | 41 | 0 | 29 | rides with B |
| **total** | **15,931** | **13,361** | **95,942** | **15,942** | **13,350** | 35 PRs include-only; 62 at full refactor size |

Loop-stop condition: Remaining = 0 for the include set; refactor rows close when their
re-derivation lands or is recorded as not needed.

## Closed with no PR — verified 2026-09-05 against `origin/main` `9c33271d`

The remaining-E split plan (`agents/.state/4b/remaining-e-split-plan.md`, written 2026-08-30 when
main was `830176cf`) found rows that need no port because main already has them in re-derived
form, an operator decision excluded them, or the branch deliberately dropped them. Those claims
were **re-checked against today's main** before the numbers moved, since main has advanced well
past the plan's base:

| row | lines | verified how |
|---|---:|---|
| `llms/verify/{test_verify_runner.py, run_verify_host_tests.lit, verify_runner.py}` | 310 | main is AHEAD, not behind: **7 test functions vs the tag's 5**, including `test_a_stored_flag_does_not_count_as_read`. Porting the tag's copy would remove tests. |
| `shared/infra/decode_qkv4.py` + `test_decode_qkv4.py` | 427 | `decode_qkv2.py` and its test are on main (#19); `decode_qkv4.py` is absent from main, which is what **Q7 decided** ("2 launches only, no 4-launch stage"). |
| `shared/builders/rms_qkv_qknorm_rope_multi.py` residue | 767 | tag-vs-main is **+688/−44**, and every tag-only definition is accounted for: `build_rms_qkv_qknorm_rope_gemv4_module` (Q7), `_build_qkv_heads_gemv` (main has it as `matrix_vector_multiplication/bf16/matvec_heads.py`), `_build_qkv_heads_gemv_wholehead` (dropped: **devq 552**, 0.588 ms either way — the tag records it at `shared/builders/rms_qkv_qknorm_rope_multi.py:1232`), and the `qkv_heads_*` / `qkv2_prep_weight` helpers (landed in `shared/infra/qkv2_layout.py`) — **except `qkv_heads_row_map`, which #17 dropped as unused and which is absent from main entirely**; it is deliberately not ported, not an oversight. |
| `shared/builders/rms_gemv_rope_multi.py` residue | 1 | the plan sized this row at 32; today the tag-vs-main diff is **+1/−1**, one comment line. The smaller number is the one recorded. |

**1,505 lines closed.** E's Remaining moves 8,888 → 7,383 and the total 14,855 → 13,350.

One row from the plan is **not** closed and not counted: `matvec_int4_packed_add.py` (+3). The
file lives at `matrix_vector_multiplication/int4_awq/` on **both** main and the tag, and the +3 is
a `BoolAttr` import plus two `l1_part_op.attributes["air.shrinkage"] = BoolAttr.get(False)`
opt-out lines. Those are not branch-added work: main **removed** them in `f51b9385`
("[air-opt] Stop air-shrink-memref-sizes-by-access retyping subviews it did not shrink", #1909),
the compiler fix that made the opt-out unnecessary. So porting them would re-add a workaround
main has already retired — moot, and recorded with that history so a later audit can check it.

The plan's unnumbered residue rows (smollm2_1_7b_int4 deltas, int4_awq study tooling,
`channel_examples` churn) are left in Remaining until each is measured the same way. The plan
estimates ~5,900 lines close in total against its own branch-added accounting; only what has been
re-verified against current main is moved here.

## Slice order

Shared infra (B: devq) → compiler rows that are self-contained (H10 non-constant BD offset refusal,
`92b05de9` shared-L1 lock placement, shrink extent, split-L2 B/C/D, H3 verifier) → the `?` rows
settled by lit runs against `main`'s `air-opt` and one devq job → E model-side work (qwen3,
llama32_1b int4, 4-launch QKV, registry rows) → E refactors onto air.api / `llms/shared`.
