#!/usr/bin/env bash
# Fail-closed wrapper for the canonical guard in the agent-standards submodule.
guard="$CLAUDE_PROJECT_DIR/agent-standards/hooks/main-branch-guard.mjs"
[ -f "$guard" ] || { echo "agent-standards submodule missing: run git submodule update --init agent-standards" >&2; exit 2; }
exec node "$guard"
