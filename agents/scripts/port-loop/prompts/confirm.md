Confirm one fix. This is a narrow review with a single question, not a general code review.

## Why you are being asked

The review loop is review → fix, repeated. That means the **last fix to run is never reviewed by
anything** — the round that requested it has already finished, and there is no round after it. You
are that missing round, and you are looking at exactly that one diff.

This has bitten before: Phase C4's round-3 review raised two blocking findings, both were fixed,
and the only thing that ever checked those fixes was a human reading them afterwards.

## What to review

Exactly this diff, in the repository at /home/cj/mlir-air:

```
git diff __FIX_BASE_SHA__..HEAD
```

Start by running that, and `git log --oneline __FIX_BASE_SHA__..HEAD`. Read the changed files in
full where the diff alone is not enough to judge them. **Nothing outside this diff is in scope** —
not the rest of the phase, not pre-existing code the fix happens to sit next to.

Phase: __PHASE_ID__ — __PHASE_NAME__
Specification: `__PHASE_DOC__`

These are the round __ROUND__ findings the diff was written to address:

```json
__BLOCKING_JSON__
```

## The question

For each finding above:

1. **Is it actually fixed?** Not "was something changed nearby" — does the diff resolve the defect
   the finding describes? A fix that addresses a symptom while the described failure mode survives
   is not fixed.
2. **Was it fixed in the code, or in the check?** Fixing the test instead of the code is the
   failure this whole harness exists to catch. If the diff narrowed an assertion, loosened a
   tolerance, removed a case, or made a gate accept what it previously rejected, that belongs in
   `weakened_gates` and it halts the run.
3. **Did the fix break something else?** A late fix is written under time pressure by a session
   with no reviewer after it. Look for the ordinary consequences: a changed signature with a stale
   caller, an invariant that held before and does not now, a shared-infrastructure behaviour the
   ten LLM deployments under `programming_examples/llms/` depend on.

If a finding was *not* fixed and the diff says why — a session is instructed to push back on a
finding it believes is wrong rather than make a change it does not believe in — judge the argument.
A well-reasoned refusal is a correct outcome, not a blocking finding.

## Calibration — read this before you write anything

A blocking finding here **halts the run before the gate**, and a human has to come back to it. Be
deliberate about that.

Report as `blocking` only:

- a finding above that is not actually resolved, or
- a new defect introduced by this diff, with a specific failing input or condition, or
- a gate this diff made weaker than it was at `__FIX_BASE_SHA__`.

Everything else is `non_blocking`, including: style, naming, structure, module size, missing
documentation, test coverage you would have liked, opportunities to simplify, and anything about
code this diff did not touch. Those are worth recording and they are not worth stopping for.

If the fixes are sound, say so and return an empty `blocking` list. That is the expected outcome.

## Output

Return the structured verdict. `verdict` is `pass` only when `blocking` and `weakened_gates` are
both empty. Cite concrete file paths. A finding you cannot tie to a specific file and a specific
failure mode belongs in `non_blocking`.
