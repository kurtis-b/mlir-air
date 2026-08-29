#!/usr/bin/env bash
# Fail-closed wrapper: the vendored guard first, then mlir-air's origin-only guard
# (agents/WORKFLOW.md rule 8). A deny from the first is forwarded as-is.
guard="$CLAUDE_PROJECT_DIR/agents/hooks/main-branch-guard.mjs"
origin_only="$CLAUDE_PROJECT_DIR/agents/hooks/origin-only-guard.mjs"
[ -f "$guard" ] || { echo "vendored guard missing: agents/hooks/main-branch-guard.mjs" >&2; exit 2; }
[ -f "$origin_only" ] || { echo "origin-only guard missing: agents/hooks/origin-only-guard.mjs" >&2; exit 2; }
input="$(cat)"
out="$(printf '%s' "$input" | node "$guard")" || exit $?
if [ -n "$out" ]; then printf '%s\n' "$out"; exit 0; fi
printf '%s' "$input" | exec node "$origin_only"
