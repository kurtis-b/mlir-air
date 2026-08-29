#!/usr/bin/env node
// Origin-only guard — mlir-air's repo-specific PreToolUse(Bash) layer, run by
// .claude/hooks/guard.sh after the vendored main-branch-guard.mjs (kept verbatim).
// Enforces agents/WORKFLOW.md §Git workflow rule 8: `origin` is the only push
// target; `upstream` (Xilinx/mlir-air) is pull-only. Denies:
//   U1  `git push` to `upstream` or to any URL naming Xilinx/ (incl. --repo forms)
//   U2  `git remote add|set-url|rename` naming `upstream` or Xilinx/, and
//       `git config remote.upstream.*` writes — re-arming the neutered push URL
//   U3  `gh` writes addressed to Xilinx/… via -R/--repo (pr/issue/release
//       create|edit|close|comment|ready|reopen|delete|lock|pin|transfer|upload),
//       `gh repo set-default Xilinx/…`, and `gh api` requests naming Xilinx/ with a
//       write method (-X/--method other than GET) or a body (-f/-F/--field/
//       --raw-field/--input)
// Reads of upstream stay allowed (git fetch upstream, gh pr view -R Xilinx/…,
// gh api GET): syncing upstream into main is the operator's own action.
// Sibling layers: upstream push URL = no_push, `gh repo set-default kurtis-b/mlir-air`,
// permissions.deny in .claude/settings.json. Errors degrade to "ask", never fail-open.

import fs from "node:fs";
import process from "node:process";

const UPSTREAM = /(^upstream$)|(^https?:\/\/[^/]*github\.com\/Xilinx\/)|(github\.com[:/]Xilinx\/)/i;
const NAMES_XILINX = /\bXilinx\//i;
const GIT_VALUE_OPTS = new Set(["-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"]);
const GH_VALUE_OPTS = new Set(["-R", "--repo", "--hostname"]);
const GH_WRITE_VERBS = new Set(["create", "edit", "close", "comment", "ready", "reopen", "delete", "lock", "pin", "transfer", "upload", "merge", "review"]);
const GH_API_BODY = new Set(["-f", "-F", "--field", "--raw-field", "--input"]);

function emit(decision, reason) {
  process.stdout.write(JSON.stringify({ hookSpecificOutput: { hookEventName: "PreToolUse", permissionDecision: decision, permissionDecisionReason: reason } }) + "\n");
  process.exit(0);
}
const deny = (why) => emit("deny", `${why}\nOrigin-only rule (agents/WORKFLOW.md §Git workflow 8): upstream Xilinx/mlir-air is pull-only — push to origin (kurtis-b/mlir-air) and open PRs there. Upstream syncs are the operator's own action.`);

function collapseQuotes(command) {
  const originals = [];
  const scannable = command.replace(/"([^"]*)"|'([^']*)'/g, (_, dq, sq) => { originals.push(dq ?? sq); return `__QUOTED_${originals.length - 1}__`; });
  return { scannable, restore: (t) => t.replace(/__QUOTED_(\d+)__/g, (_, n) => originals[Number(n)] ?? "") };
}

// U1: the remote is the --repo value or the first bare token after `push`.
function checkPush(args) {
  for (let i = 0; i < args.length; i++) {
    const t = args[i];
    if (t.startsWith("--repo=")) return UPSTREAM.test(t.slice(7)) ? t.slice(7) : null;
    if (t === "--repo") return args[i + 1] && UPSTREAM.test(args[i + 1]) ? args[i + 1] : null;
    if (t.startsWith("-")) { if (["-o", "--push-option", "--receive-pack", "--exec"].includes(t)) i += 1; continue; }
    return UPSTREAM.test(t) ? t : null;
  }
  return null;
}

// U3: gh writes addressed to Xilinx/… ; `gh api` writes naming Xilinx/ anywhere.
function checkGh(args) {
  const words = [];
  let repo = null, method = "GET", body = false;
  for (let i = 0; i < args.length; i++) {
    const t = args[i];
    if (t.startsWith("-")) {
      const eq = t.indexOf("=");
      const flag = eq > 0 ? t.slice(0, eq) : t;
      const val = eq > 0 ? t.slice(eq + 1) : args[i + 1];
      if (flag === "-R" || flag === "--repo") repo = val ?? repo;
      if (flag === "-X" || flag === "--method") method = (val ?? "").toUpperCase();
      if (GH_API_BODY.has(flag)) body = true;
      if (eq < 0 && (GH_VALUE_OPTS.has(flag) || flag === "-X" || flag === "--method" || GH_API_BODY.has(flag))) i += 1;
      continue;
    }
    words.push(t);
  }
  const [w0, w1] = words;
  if (w0 === "repo" && w1 === "set-default" && words.slice(2).some((w) => NAMES_XILINX.test(w))) {
    return "`gh repo set-default` must stay on kurtis-b/mlir-air; pointing gh at Xilinx/ makes every bare `gh pr` command target upstream.";
  }
  if (repo && NAMES_XILINX.test(repo) && ["pr", "issue", "release"].includes(w0) && GH_WRITE_VERBS.has(w1)) {
    return `\`gh ${w0} ${w1}\` addressed to ${repo} would write to upstream.`;
  }
  if (w0 === "api" && args.some((a) => NAMES_XILINX.test(a)) && (method !== "GET" || body)) {
    return "`gh api` write (non-GET method or request body) naming Xilinx/ would write to upstream.";
  }
  return null;
}

function check(command) {
  const { scannable, restore } = collapseQuotes(command);
  for (const segment of scannable.split(/[;&|\n]+/)) {
    const tokens = segment.trim().split(/\s+/);
    for (let i = 0; i < tokens.length; i++) {
      const tok = tokens[i];
      if (tok === "gh" || tok.endsWith("/gh")) {
        const why = checkGh(tokens.slice(i + 1).map(restore));
        if (why) deny(why);
        continue;
      }
      if (tok !== "git" && !tok.endsWith("/git")) continue;
      let j = i + 1;
      while (j < tokens.length && tokens[j].startsWith("-")) j += GIT_VALUE_OPTS.has(tokens[j]) ? 2 : 1;
      const sub = tokens[j];
      const args = tokens.slice(j + 1).map(restore);
      if (sub === "push") {
        const remote = checkPush(args);
        if (remote) deny(`git push to \`${remote}\` targets upstream.`);
      } else if (sub === "remote" && ["add", "set-url", "rename"].includes(args[0]) && args.some((a) => UPSTREAM.test(a))) {
        deny(`\`git remote ${args[0]}\` naming upstream/Xilinx would re-arm a push path to upstream (its push URL is deliberately \`no_push\`).`);
      } else if (sub === "config" && !args.some((a) => a.startsWith("--get") || a === "-l" || a === "--list") && args.some((a) => /^remote\.upstream\./.test(a))) {
        deny("Editing `remote.upstream.*` config would re-arm a push path to upstream.");
      }
    }
  }
}

try {
  const raw = fs.readFileSync(0, "utf8").trim();
  const input = raw ? JSON.parse(raw) : {};
  if (input.tool_name && input.tool_name !== "Bash") process.exit(0);
  const command = input?.tool_input?.command ?? "";
  if (command) check(command);
  process.exit(0);
} catch (err) {
  emit("ask", `origin-only-guard hook error (${err?.message || err}) — verify this command does not push to or write upstream Xilinx/mlir-air before approving.`);
}
