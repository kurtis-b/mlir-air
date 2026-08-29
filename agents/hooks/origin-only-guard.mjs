#!/usr/bin/env node
// Origin-only guard — mlir-air's repo-specific PreToolUse(Bash) layer, run by
// .claude/hooks/guard.sh after the vendored main-branch-guard.mjs (kept verbatim).
// Enforces agents/WORKFLOW.md §Git workflow rule 8: `origin` is the only push
// target; `upstream` (Xilinx/mlir-air) is pull-only. Denies:
//   U1  `git push` to `upstream` or to any URL naming Xilinx/ (incl. --repo forms),
//       and any forced refspec (`+ref`) — the vendored guard strips the `+` and the
//       settings allow-list auto-approves `git push origin …`
//   U2  re-arming a push path: `git remote add|set-url|rename` naming upstream or
//       Xilinx/, `git config` (or one-shot `git -c`) setting any remote's url/pushurl
//       or a url.*.insteadOf to Xilinx/, and any write to `remote.upstream.*`
//   U3  `gh` addressed to Xilinx/ — via -R/--repo (separate, `=` or attached form)
//       or via a GH_REPO environment naming Xilinx/ — unless the subcommand is
//       read-only (view/list/status/checks/diff/download/search, or `api` with GET
//       and no body); `gh repo set-default` may only view or name a non-Xilinx repo
//       (no `--unset`, no interactive no-argument form)
// Reads of upstream stay allowed (git fetch upstream, gh pr view -R Xilinx/…,
// gh api GET): syncing upstream into main is the operator's own action.
// Sibling layers: upstream push URL = no_push, `gh repo set-default kurtis-b/mlir-air`,
// permissions.deny in .claude/settings.json. Errors degrade to "ask", never fail-open.

import fs from "node:fs";
import process from "node:process";

const UPSTREAM = /(^upstream$)|(^https?:\/\/[^/]*github\.com\/Xilinx\/)|(github\.com[:/]Xilinx\/)/i;
const NAMES_XILINX = /\bXilinx\//i;
const GIT_VALUE_OPTS = new Set(["-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"]);
const GH_VALUE_OPTS = new Set(["-R", "--repo", "--hostname", "-X", "--method", "-f", "-F", "--field", "--raw-field", "--input", "-H", "--header", "-p", "--preview", "-t", "--template", "-q", "--jq"]);
const GH_API_BODY = new Set(["-f", "-F", "--field", "--raw-field", "--input"]);
const GH_READ_VERBS = new Set(["view", "list", "status", "checks", "diff", "download", "search", "browse"]);
const ENV_REPO = process.env.GH_REPO || "";

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

// `-RX`, `-XPOST`, `-fk=v` → ["-R","X"], … ; long options keep their `=` form.
function splitAttachedShort(args) {
  const out = [];
  for (const t of args) {
    const m = t.match(/^-([RXfFHptq])(.+)$/);
    if (m && !t.startsWith("--")) out.push(`-${m[1]}`, m[2]); else out.push(t);
  }
  return out;
}

// A git config write that would point any remote (or an insteadOf rewrite) at Xilinx/, or any
// write to remote.upstream.* — as `git config KEY VALUE`, `git config KEY=VALUE`-style -c, or `--unset KEY`.
function configWriteReason(keyValue) {
  const [key, ...rest] = keyValue.split("="); const value = rest.join("=");
  if (/^remote\.upstream\./i.test(key)) return "Editing `remote.upstream.*` config would re-arm a push path to upstream.";
  if (/^remote\.[^.]+\.(url|pushurl)$/i.test(key) && NAMES_XILINX.test(value)) return `Pointing \`${key}\` at Xilinx/ would make an allowed push write upstream.`;
  if (/^url\..*\.(push)?insteadof$/i.test(key) && (NAMES_XILINX.test(key) || NAMES_XILINX.test(value))) return "A url.*.insteadOf rewrite naming Xilinx/ would redirect pushes upstream.";
  return null;
}

// U1: the remote is the --repo value or the first bare token after `push`; any `+refspec` is a force.
function checkPush(args) {
  let remote = null;
  for (let i = 0; i < args.length; i++) {
    const t = args[i];
    if (t.startsWith("--repo=")) { remote ??= t.slice(7); continue; }
    if (t === "--repo") { remote ??= args[i + 1] ?? ""; i += 1; continue; }
    if (t.startsWith("-")) { if (["-o", "--push-option", "--receive-pack", "--exec"].includes(t)) i += 1; continue; }
    if (t.startsWith("+")) return `Forced refspec \`${t}\` overwrites the remote branch — append commits instead.`;
    remote ??= t;
  }
  if (remote !== null && UPSTREAM.test(remote)) return `git push to \`${remote}\` targets upstream.`;
  return null;
}

// U3: gh addressed to Xilinx/ (flag or environment) must be read-only.
function checkGh(rawArgs) {
  const args = splitAttachedShort(rawArgs);
  const words = [];
  let repo = null, method = "GET", body = false, unset = false, view = false;
  for (let i = 0; i < args.length; i++) {
    const t = args[i];
    if (t.startsWith("-")) {
      const eq = t.indexOf("=");
      const flag = eq > 0 ? t.slice(0, eq) : t;
      const val = eq > 0 ? t.slice(eq + 1) : args[i + 1];
      if (flag === "-R" || flag === "--repo") repo = val ?? repo;
      if (flag === "-X" || flag === "--method") method = (val ?? "").toUpperCase();
      if (GH_API_BODY.has(flag)) body = true;
      if (flag === "--unset") unset = true;
      if (flag === "--view") view = true;
      if (eq < 0 && GH_VALUE_OPTS.has(flag)) i += 1;
      continue;
    }
    words.push(t);
  }
  const [w0, w1] = words;
  if (w0 === "repo" && w1 === "set-default") {
    if (view) return null;
    if (unset) return "`gh repo set-default --unset` drops the safe origin default; gh would then prefer the upstream remote for every bare `gh pr` command.";
    const target = words[2];
    if (!target) return "`gh repo set-default` with no repository is interactive; name kurtis-b/mlir-air explicitly.";
    if (NAMES_XILINX.test(target)) return "`gh repo set-default` must stay on kurtis-b/mlir-air; pointing gh at Xilinx/ makes every bare `gh pr` command target upstream.";
    return null;
  }
  const target = repo ?? (ENV_REPO || null);
  const targetsXilinx = (target !== null && NAMES_XILINX.test(target)) || (w0 === "api" && args.some((a) => NAMES_XILINX.test(a)));
  if (!targetsXilinx) return null;
  if (w0 === "api") {
    if (method === "GET" && !body) return null;
    return "`gh api` write (non-GET method or request body) naming Xilinx/ would write to upstream.";
  }
  if (w0 === "search" || (w1 !== undefined && GH_READ_VERBS.has(w1))) return null;
  const how = repo !== null ? `-R ${repo}` : `GH_REPO=${target}`;
  return `\`gh ${w0}${w1 ? " " + w1 : ""}\` addressed to upstream (${how}) is not a read-only subcommand.`;
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
      while (j < tokens.length && tokens[j].startsWith("-")) {
        // one-shot config: `-c key=value` or `-ckey=value`
        const attached = tokens[j].match(/^-c(.+)$/);
        const kv = attached ? restore(attached[1]) : tokens[j] === "-c" ? restore(tokens[j + 1] ?? "") : null;
        if (kv !== null) { const why = configWriteReason(kv); if (why) deny(why); }
        j += GIT_VALUE_OPTS.has(tokens[j]) ? 2 : 1;
      }
      const sub = tokens[j];
      const args = tokens.slice(j + 1).map(restore);
      if (sub === "push") {
        const why = checkPush(args);
        if (why) deny(why);
      } else if (sub === "remote" && ["add", "set-url", "rename"].includes(args[0]) && args.some((a) => UPSTREAM.test(a))) {
        deny(`\`git remote ${args[0]}\` naming upstream/Xilinx would re-arm a push path to upstream (its push URL is deliberately \`no_push\`).`);
      } else if (sub === "config") {
        const bare = args.filter((a) => !a.startsWith("-"));
        const reading = args.some((a) => a.startsWith("--get") || a === "-l" || a === "--list");
        if (!reading && bare.length > 0) {
          const why = configWriteReason(bare.length > 1 ? `${bare[0]}=${bare.slice(1).join(" ")}` : bare[0]);
          if (why) deny(why);
        }
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
