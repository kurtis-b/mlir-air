#!/usr/bin/env bash
# Fail-closed wrapper for the vendored guard.
guard="$CLAUDE_PROJECT_DIR/agents/hooks/main-branch-guard.mjs"
[ -f "$guard" ] || { echo "vendored guard missing: agents/hooks/main-branch-guard.mjs" >&2; exit 2; }
exec node "$guard"
