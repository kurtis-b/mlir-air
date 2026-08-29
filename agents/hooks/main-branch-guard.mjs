#!/usr/bin/env node
// Canonical main-branch guard — the agent-side enforcement layer of the shared
// agent workflow (../WORKFLOW.md). Vendored into consumer repos as the
// `agent-standards/` submodule and run as a Claude Code PreToolUse(Bash) hook
// via each repo's `.claude/hooks/guard.sh` wrapper.
//
// Policy (binds agents only — the human is exempt by using a terminal):
//   G1  no commit-creating git command while on main (belt: G2 makes this unreachable)
//   G2  never check out main (`git switch|checkout main|-`); branch from origin/main
//   G3  never `git merge` except `git merge origin/main` to sync a conflicted PR
//       branch (integration is human-merged PRs); never `git reset --hard`
//   G4  never push to main, force-push, bulk-push, or delete remote branches
//   G5  never `gh pr merge`/`gh pr review` or merge/review REST endpoints — human-only
//   G6  never touch the main ref (`git branch … main`, `git update-ref refs/heads/main`)
//
// A guardrail against habitual mistakes, not an adversarial sandbox; the GitHub
// ruleset on main is the server-side layer that binds everyone.
// Errors degrade to permissionDecision "ask", never fail-open.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { spawnSync } from "node:child_process";

function emit(decision, reason) {
  process.stdout.write(
    JSON.stringify({
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: decision,
        permissionDecisionReason: reason,
      },
    }) + "\n",
  );
  process.exit(0);
}

const deny = (reason) =>
  emit(
    "deny",
    `${reason}\nShared workflow rule (agent-standards/WORKFLOW.md): work on a branch cut from ` +
      "origin/main and let the human merge. If this command is truly intended, the human runs it " +
      "themselves in a terminal.",
  );

const BRANCH_RECIPE = "git fetch origin && git switch -c <type>/<slug> origin/main";

function readInput() {
  const raw = fs.readFileSync(0, "utf8").trim();
  return raw ? JSON.parse(raw) : {};
}

// Value-taking options that may sit between `git` and its subcommand.
const GIT_VALUE_OPTS = new Set(["-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"]);
// Value-taking options of `git push` (separate-token form).
const PUSH_VALUE_OPTS = new Set(["-o", "--push-option", "--receive-pack", "--exec"]);
// Value-taking flags of `gh` that may precede the subcommand words.
const GH_VALUE_OPTS = new Set(["-R", "--repo", "--hostname"]);
// Value-taking options of `git merge` (separate-token form).
const MERGE_VALUE_OPTS = new Set(["-m", "--message", "-F", "--file", "-s", "--strategy", "-X", "--strategy-option", "--cleanup", "--into-name"]);
// git switch/checkout options whose value names the branch being created.
const BRANCH_CREATING_OPTS = new Set(["-c", "-C", "-b", "-B", "--create", "--force-create", "--orphan"]);
const BRANCH_CREATING_ATTACHED = /^(?:--create|--force-create|--orphan)=(.+)$/;
const COMMIT_CREATING = new Set(["commit", "rebase", "cherry-pick", "revert", "am"]);
const PUSH_ALL_FLAGS = new Set(["--all", "--branches", "--mirror"]);
const GH_API_HUMAN_ONLY =
  /pulls\/[^\s/]+\/(merge|reviews)|merge-upstream|\b(mergePullRequest|enablePullRequestAutoMerge|addPullRequestReview|submitPullRequestReview)\b/;

const isMainRef = (ref) => {
  const r = ref.replace(/^\+/, "");
  return r === "main" || r === "refs/heads/main";
};

function currentBranch(cwd) {
  const r = spawnSync("git", ["rev-parse", "--abbrev-ref", "HEAD"], { cwd, encoding: "utf8" });
  return r.status === 0 ? r.stdout.trim() : null;
}

function repoToplevel(cwd) {
  const r = spawnSync("git", ["rev-parse", "--show-toplevel"], { cwd, encoding: "utf8" });
  return r.status === 0 ? r.stdout.trim() : null;
}

// The guard polices only the consumer checkout itself; a git call inside a
// different repo (e.g. a vendored submodule) is out of scope.
function inProjectRepo(cwd) {
  const projectDir = process.env.CLAUDE_PROJECT_DIR;
  if (!projectDir) return true; // can't scope-check -> stay conservative
  const top = repoToplevel(cwd);
  if (!top) return false; // not a git repo at all
  try {
    return fs.realpathSync(top) === fs.realpathSync(projectDir);
  } catch {
    return true;
  }
}

// Prose inside quotes (commit messages, PR bodies) must not trip the token scan,
// but option values (paths, branch names, refspecs) must survive it. Quoted
// spans become numbered placeholders; positions that consume a value restore
// the original text via `restore`.
function collapseQuotes(command) {
  const originals = [];
  const scannable = command.replace(/"([^"]*)"|'([^']*)'/g, (_, dq, sq) => {
    originals.push(dq ?? sq);
    return `__QUOTED_${originals.length - 1}__`;
  });
  const restore = (token) =>
    token.replace(/__QUOTED_(\d+)__/g, (_, n) => originals[Number(n)] ?? "");
  return { scannable, restore };
}

// Best-effort expansion of a path-like token: placeholders back, then $VAR /
// ${VAR} from this hook's own environment, then ~. Returns null when the token
// still contains shell constructs we cannot evaluate — callers fall back to the
// directory they already had.
function resolveToken(token, restore) {
  let s = restore(token);
  s = s.replace(/\$\{(\w+)\}|\$(\w+)/g, (m, a, b) => process.env[a || b] ?? m);
  if (s.startsWith("~")) s = path.join(os.homedir(), s.slice(1));
  if (/[$`]/.test(s)) return null;
  return s;
}

// Deny reason for one `git push` invocation, given its argv after `push`.
function checkPush(args, branchNow, restore) {
  const bare = [];
  let repoFlagSeen = false;
  for (let i = 0; i < args.length; i++) {
    const t = args[i];
    if (t.startsWith("-")) {
      if (t === "--force" || t === "--force-if-includes" || t.startsWith("--force-with-lease")) {
        return "Force push is forbidden, always — append commits instead.";
      }
      // Short flags may be fused (-uf, -fq): inspect each character.
      if (/^-[a-zA-Z]+$/.test(t)) {
        if (t.includes("f")) return "Force push is forbidden, always — append commits instead.";
        if (t.includes("d")) return "Deleting remote branches is forbidden for agents.";
      }
      if (PUSH_ALL_FLAGS.has(t)) return `Bulk push (${t}) is forbidden: it would include main.`;
      if (t === "--delete") return "Deleting remote branches is forbidden for agents.";
      if (t === "--repo" || t.startsWith("--repo=")) {
        repoFlagSeen = true;
        if (t === "--repo") i += 1;
        continue;
      }
      if (PUSH_VALUE_OPTS.has(t)) i += 1;
      continue;
    }
    bare.push(restore(t));
  }
  // Normally bare[0] is the remote; with --repo the remote is already supplied,
  // so every bare token is a potential refspec.
  const refspecs = repoFlagSeen ? bare : bare.slice(1);
  for (const spec of refspecs) {
    if (spec.replace(/^\+/, "").startsWith(":")) {
      return `Empty-source refspec (\`${spec}\`) deletes a remote branch — forbidden for agents.`;
    }
    const dst = spec.includes(":") ? spec.split(":").pop() : spec;
    if (isMainRef(dst)) return `git push targets main (refspec \`${spec}\`).`;
    if (dst.includes("*")) return `Wildcard refspec (\`${spec}\`) can update main.`;
    if (/[$`]/.test(dst)) {
      return `Push destination \`${spec}\` cannot be resolved by the guard — use a literal branch name.`;
    }
  }
  if (refspecs.length === 0 || refspecs.some((s) => s.replace(/^\+/, "") === "HEAD")) {
    if (branchNow() === "main") return "Pushing from a checkout of main updates main.";
  }
  return null;
}

// The one permitted merge: syncing the current feature branch from origin/main
// when its PR conflicts. Everything else is human-only integration.
function checkMerge(args, branchNow, restore) {
  const bare = [];
  for (let i = 0; i < args.length; i++) {
    const t = args[i];
    if (t.startsWith("-") && t !== "-") {
      if (t === "--continue" || t === "--abort" || t === "--quit") return null;
      if (MERGE_VALUE_OPTS.has(t)) i += 1;
      continue;
    }
    bare.push(restore(t));
  }
  if (branchNow() === "main") return "No merges on main.";
  if (bare.length === 1 && bare[0] === "origin/main") return null;
  return "`git merge` is forbidden for agents except `git merge origin/main` to sync a conflicted PR branch — integration happens via human-merged PRs.";
}

// Deny reason for `git switch`/`git checkout`: main may never be checked out,
// re-created, or reached via `-`. Pathspec forms (`--`, `.`) are not switches.
function checkSwitchTarget(args, restore) {
  for (let i = 0; i < args.length; i++) {
    const t = args[i];
    if (t === "--") return null;
    if (t.startsWith("-") && t !== "-") {
      const attached = t.match(BRANCH_CREATING_ATTACHED);
      const created = attached
        ? restore(attached[1])
        : BRANCH_CREATING_OPTS.has(t) && args[i + 1]
          ? restore(args[i + 1])
          : null;
      if (created !== null) return isMainRef(created) ? "Re-creating main is forbidden." : null;
      if (GIT_VALUE_OPTS.has(t)) i += 1;
      continue;
    }
    if (t === ".") continue; // pathspec, not a branch
    if (t === "-") return "`switch -` may land on main — name the branch explicitly.";
    if (isMainRef(restore(t))) {
      return `Checking out main is forbidden for agents (later commands could write to it). Branch instead: ${BRANCH_RECIPE}`;
    }
    return null; // first bare target is a non-main branch
  }
  return null;
}

// Deny reason for one `gh` invocation: flag-position-independent match of the
// human-only subcommands, plus merge/review REST endpoints.
function checkGh(args, segmentText) {
  const words = [];
  for (let i = 0; i < args.length && words.length < 2; i++) {
    const t = args[i];
    if (t.startsWith("-")) {
      if (GH_VALUE_OPTS.has(t)) i += 1;
      continue;
    }
    words.push(t);
  }
  if (words[0] === "pr" && words[1] === "merge") {
    return "`gh pr merge` is human-only: the merge click is the recorded human approval.";
  }
  if (words[0] === "pr" && words[1] === "review") {
    return "`gh pr review` is human-only (any form). Use `gh pr comment` for advisory notes.";
  }
  if (words[0] === "api" && GH_API_HUMAN_ONLY.test(segmentText)) {
    return "Merge/review API endpoints are human-only.";
  }
  return null;
}

function check(command, baseCwd) {
  const { scannable, restore } = collapseQuotes(command);
  let dir = baseCwd;

  for (const segment of scannable.split(/[;&|\n]+/)) {
    const cdMatch = segment.trim().match(/^cd\s+(\S+)$/);
    if (cdMatch) {
      const target = resolveToken(cdMatch[1], restore);
      if (target) dir = path.resolve(dir, target);
      continue;
    }

    const tokens = segment.trim().split(/\s+/);
    for (let i = 0; i < tokens.length; i++) {
      const tok = tokens[i];

      if (tok === "gh" || tok.endsWith("/gh")) {
        const why = checkGh(tokens.slice(i + 1), restore(segment));
        if (why) deny(why);
        continue;
      }

      if (tok !== "git" && !tok.endsWith("/git")) continue;
      let gitDir = dir;
      let j = i + 1;
      while (j < tokens.length && tokens[j].startsWith("-")) {
        if (tokens[j] === "-C" && tokens[j + 1]) {
          const target = resolveToken(tokens[j + 1], restore);
          if (target) gitDir = path.resolve(dir, target);
        }
        j += GIT_VALUE_OPTS.has(tokens[j]) ? 2 : 1;
      }
      const sub = tokens[j];
      if (!sub || !inProjectRepo(gitDir)) continue;
      const args = tokens.slice(j + 1);
      const restoredArgs = args.map(restore);

      if (sub === "switch" || sub === "checkout") {
        const why = checkSwitchTarget(args, restore);
        if (why) deny(why);
      } else if (sub === "merge") {
        const why = checkMerge(args, () => currentBranch(gitDir), restore);
        if (why) deny(why);
      } else if (sub === "reset") {
        if (restoredArgs.includes("--hard")) {
          deny("`git reset --hard` is forbidden — use `git restore <path>` for targeted rollback.");
        }
        if (currentBranch(gitDir) === "main") deny("No history edits on main.");
      } else if (sub === "branch") {
        if (restoredArgs.some((a) => isMainRef(a))) deny("Modifying the main branch ref is forbidden.");
      } else if (sub === "update-ref") {
        if (restoredArgs.some((a) => a === "refs/heads/main")) {
          deny("Editing refs/heads/main directly is forbidden.");
        }
      } else if (COMMIT_CREATING.has(sub)) {
        if (currentBranch(gitDir) === "main") {
          deny(`\`git ${sub}\` while checked out on main. Branch first: ${BRANCH_RECIPE}`);
        }
      } else if (sub === "push") {
        const why = checkPush(args, () => currentBranch(gitDir), restore);
        if (why) deny(`${why} main only moves via a PR merged by the human.`);
      }
    }
  }
}

try {
  const input = readInput();
  if (input.tool_name && input.tool_name !== "Bash") process.exit(0);
  const command = input?.tool_input?.command ?? "";
  if (command) {
    const cwd = input.cwd || process.env.CLAUDE_PROJECT_DIR || process.cwd();
    check(command, cwd);
  }
  process.exit(0);
} catch (err) {
  emit(
    "ask",
    `main-branch-guard hook error (${err?.message || err}) — verify this command does not write to main or merge/review a PR before approving.`,
  );
}
