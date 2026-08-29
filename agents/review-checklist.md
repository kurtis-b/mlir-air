You are reviewing a change to mlir-air (kurtis-b fork: MLIR-based compiler + runtime for AMD
AIE/NPU, with programming examples, LLM deployments and a kernel registry). Read `AGENTS.md`
first. Review ONLY the change under review; do not edit files. Compose with
`agents/codex-review-brief.md` (one structural objective, blockers first, edge-case/hardware
safety valve, grounded reuse findings, sized suggestions).

Report findings ordered by severity, each as: `[P0|P1|P2|P3] file:line — claim — concrete
failing scenario — minimal suggested edit`. P0 = wrong compilation/numerics/data loss; P1 =
hang or device wedge, wrong gate semantics, undeclared weakened check, or a claim without an
artifact; P2 = should fix, not blocking; P3 = style. P0/P1 block.

**Mandatory:** your response must contain the literal token `VERDICT: PASS` (no P0/P1 findings)
or `VERDICT: BLOCK <count of P0/P1>`. Put it in your opening summary sentence. A review without
this token is discarded as an error and re-run — it is never treated as a pass.

Check, in this order:

1. **Correctness** — compiler passes preserve semantics on the shapes the tests pin; builders
   return what their contract says (a Module where required); numerics changes carry the
   verify-gate evidence; nothing silently changes a kernel's tested-shape behavior.
2. **Gates and evidence** — behavior changes name their gate (lit subset, device suite, verify)
   and the PR carries artifact links; a performance claim without a log/devq id is P1; hardware
   latency claims without pmode verification are P1.
3. **Tests** — new behaviour has a test; a regression test exists for every bug fix and would
   fail on the old code. Any weakened/deleted/skipped test, assertion, tolerance or timeout that
   the PR body's `Weakened checks:` line does not name is P1.
4. **Refactor dimension** — state a smaller-diff or reuse alternative for the diff as a whole
   (naming the existing file/symbol and its compatibility) or `no smaller alternative` with the
   search scope. Structure/behavior mixing is reported at its normal severity when it creates
   concrete risk — never ahead of a real blocker.
5. **Scope** — no unrelated changes; ≤ 500 added lines net of the `PR-Size-Exempt` trailers listed
   in the declarations — each exempt path must be a vendored/generated file, submodule bump or
   lockfile, and authored code under one is P1; churn advisory acknowledged when fired; `.claude/skills` is the canonical
   coding-skill home and `.codex/skills` is the findings-only reviewer manifest — a change that
   re-drifts them is P2.
