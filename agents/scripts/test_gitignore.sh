#!/usr/bin/env bash
# Hermetic probes for the root .gitignore's anchoring and negation guarantees
# (R1): the out-of-tree toolchain stanzas ignore only ROOT paths, and the
# bootstrap negation block keeps the workflow's own files tracked. Run from
# anywhere inside the repo.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
pass=0; fail=0
ign()   { if git check-ignore -q -- "$1"; then pass=$((pass+1)); else echo "FAIL ignored-expected: $1"; fail=$((fail+1)); fi; }
noign() { if git check-ignore -q -- "$1"; then echo "FAIL unignored-expected: $1"; fail=$((fail+1)); else pass=$((pass+1)); fi; }
# root-anchored toolchain/artifact stanzas
ign  llvm/x
ign  my_install/y
ign  foo.o
ign  air.mlir
ign  coarse_cache/z
ign  results/r/x
ign  results_unattended_1/x
# anchoring: the same names below the root stay visible
noign programming_examples/foo/llvm/x
noign programming_examples/foo/my_install/y
noign programming_examples/foo/bar.o
noign programming_examples/foo/air.mlir
# bootstrap negation block: the workflow's own files stay tracked
noign CLAUDE.md
noign .claude/settings.json
noign .claude/hooks/guard.sh
noign .claude/skills/README.md 2>/dev/null || true
noign agents/scripts/pr.sh
echo "gitignore probes: $pass passed, $fail failed"
exit "$((fail > 0))"
