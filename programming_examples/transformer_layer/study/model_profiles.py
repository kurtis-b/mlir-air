# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Named MODEL walks for `run_model.py` (doc 56 section 3.1, H1a) -- the sibling of
`profiles.py` at whole-model scope.

CONTRACT
    A `ModelProfile` is a list of `ModelRung`s over the adapter's models
    (`llms/shared/model_adapter.MODELS`): each rung is one phase of one model
    at one compiled prefill M and one context, and produces exactly one schema
    v3 model row. `bind(compiled)` applies the applicability rule -- a prefill
    rung whose artifact set is not compiled is a SKIP with the reason, not a
    failure and not a silent omission -- so `expected_rows` says, before the
    walk, how many rows will measure and how many will skip. The manifest's
    row-count clauses and `resume.plan` read exactly that.

    The rung's resume identity is `(mode, seq, *extra)` where `extra` is
    `resume.MODEL_KEY_FIELDS` in order: decode rungs share seq_len 1 and differ
    in context, so the pair alone cannot key them.

THE TWO CURVES ARE LABELLED, NOT INFERRED (doc 56 section 3.4). A prefill rung
here has prompt length == compiled M and no chunking: that is the
KERNEL-SCALING curve (tok/s against the kernel's M), and every such row's
`study_case_label` says `kernel-scaling`. It is NOT an ubatch curve -- the
ubatch curve holds the prompt fixed and varies the chunk, and is H1b's.

THE DECODE RUNGS END at the named context: `context_end_tokens` is the ctx,
the prompt is `ctx - n_tokens` tokens and the n_tokens decode steps fill the
context up to it. The production verify gate runs the SAME prompt as every
measurement -- the decode rung's `ctx - 32` tokens, the prefill rung's full
M tokens -- with `M + 32` of KV room, so the row's plan hash, its prompt and
the gate's are one workload.

FOOTGUNS
    - `model-smoke`'s M=512 and M=1024 prefill rungs for Qwen3-0.6B need the
      six registry rows doc 56 section 5 names and a compile
      (`run_model.py compile`); until then they are derived skips and the
      profile is complete WITHOUT them -- `expected_rows` counts them as
      skipped. Measuring them later is a resume: a skip is re-derived every
      session and becomes a walk when the artifact set appears.
    - `decode_n_tokens` is also the gate's decode length; keep it at the
      gate's 32 unless both move.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.dirname(_HERE)
_PE = os.path.dirname(_EXAMPLE)
for _p in (_PE, os.path.join(_PE, "llms"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import schema  # noqa: E402

#: The study's execution mode for the drivers' per-ELF `load_and_run` path: one
#: submission per ELF, split at every host op -- doc 03's `hybrid`, code name
#: `coarse` (convention 7).
DRIVER_MODE = "coarse"

#: Prefill M the shipped drivers compile for (`build_session` hard-codes it).
SHIPPED_PREFILL_M = 2048

KERNEL_SCALING = "kernel-scaling"
DECODE_CONTEXT = "decode-context"
#: `[2026-08-25]` doc 56 H1b: the ubatch curve -- the LOGICAL prompt fixed,
#: the physical chunk varied. A rung with this label runs the drivers'
#: incremental (chunked) prefill: M is the compiled ubatch, context_end the
#: logical prompt (context_end > M means > 1 chunk), and the adapter policy
#: is "chunked", not "whole". Distinct from KERNEL_SCALING by label AND by
#: mechanism, never inferred (doc 56 section 3.4).
UBATCH = "ubatch"


@dataclass(frozen=True)
class ModelRung:
    model_id: str
    phase: str  # prefill | decode
    M: int  # the compiled prefill artifact set this rung runs on
    context_end: int  # prefill: prompt length (== M); decode: the context it ends at
    n_tokens: int = 0  # decode tokens sampled; 0 for prefill
    precision_plan: str = "bf16"
    curve: str = KERNEL_SCALING
    skip_reason: str | None = None

    # -- resume identity (resume.rung_key(mode, seq, extra) == resume.row_key(row))
    @property
    def mode(self) -> str:
        return DRIVER_MODE

    @property
    def seq(self) -> int:
        return self.M if self.phase == "prefill" else 1

    @property
    def ubatch_tokens(self) -> int:
        return self.seq

    @property
    def extra(self) -> tuple:
        return ("model", self.model_id, self.phase, str(self.ubatch_tokens), str(self.context_end), self.precision_plan)

    @property
    def context_start(self) -> int:
        return 0 if self.phase == "prefill" else self.context_end - self.n_tokens

    @property
    def case_id(self) -> str:
        return f"{self.model_id}/{self.phase}/M{self.M}/ctx{self.context_end}/{self.precision_plan}"

    @property
    def label(self) -> str:
        if self.curve == UBATCH:
            n_chunks = self.context_end // self.M
            return (f"{self.curve} prefill: prompt {self.context_end} tokens in {n_chunks} x ubatch {self.M} chunk(s), "
                    f"incremental KV, no padding ({self.model_id}, {self.precision_plan})")
        if self.phase == "prefill":
            return f"{self.curve} prefill: prompt {self.context_end} tokens = M {self.M}, no chunking ({self.model_id}, {self.precision_plan})"
        return f"{self.curve} decode: {self.n_tokens} tokens ending at context {self.context_end} on the M {self.M} artifact set ({self.model_id}, {self.precision_plan})"

    @property
    def csv_name(self) -> str:
        return f"model_{self.model_id}.csv"

    @property
    def prompt_tokens(self) -> int:
        """Tokens the measurement's prompt holds."""
        return self.context_end if self.phase == "prefill" else self.context_end - self.n_tokens

    @property
    def gate_prompt_tokens(self) -> int:
        """Tokens the gate's prompt holds: EXACTLY the measurement's prompt
        (H1a review finding 3 -- the gate must run the timed workload, a full
        M-token prompt for a prefill rung, not a padded shorter one). The
        gate's 32 generation slots come from `gate_max_seq`, never from
        shortening the prompt."""
        return self.prompt_tokens

    @property
    def gate_max_seq(self) -> int:
        """KV / RoPE capacity the gate runs with (LLMS_VERIFY_MAX_SEQ): the
        prompt the gate prefills plus the 32 generation steps. For a prefill
        rung the prompt is context_end (== M kernel-scaling; > M on an ubatch
        rung, whose gate runs the CHUNKED path over the logical prompt); for
        a decode rung the gate prefills at the compiled M as before."""
        return (self.context_end if self.phase == "prefill" else self.M) + GATE_N_TOKENS


#: The production gate decodes 32 tokens after each prompt (verify_runner.GATE_N_TOKENS).
GATE_N_TOKENS = 32


@dataclass(frozen=True)
class ModelProfile:
    name: str
    description: str
    models: tuple
    prefill_Ms: dict  # model_id -> tuple of M
    decode_ctxs: tuple  # contexts, on the SHIPPED_PREFILL_M artifact set
    decode_n_tokens: int = GATE_N_TOKENS
    prefill_samples: int = 3
    prefill_warmup: int = 1
    decode_warmup: int = 1
    precision_plan: str = "bf16"
    #: `[2026-08-25]` doc 56 H1b ubatch points: (model_id, logical_tokens,
    #: ubatch) triples. Each becomes ONE prefill rung with curve=UBATCH on the
    #: M=ubatch artifact set (whose cache must hold the rectangular FA ELFs
    #: for every chunk context > ubatch): context_end = the logical prompt,
    #: chunked through the driver's incremental path. logical % ubatch must be
    #: 0 (the scheduler has no padding path).
    ubatch_points: tuple = ()
    #: (model_id, M) -> {"prefill_cache": dir, "decode_cache": dir}; set by `bind`.
    compiled: dict = field(default_factory=dict)
    skip_notes: dict = field(default_factory=dict)

    def bind(self, compiled: dict, skip_notes: dict | None = None) -> "ModelProfile":
        """The profile with its applicability rule applied to what is compiled."""
        return replace(self, compiled=dict(compiled), skip_notes=dict(skip_notes or {}))

    def _skip(self, model_id: str, M: int, phase: str) -> str | None:
        if (model_id, M) in self.compiled:
            return None
        why = self.skip_notes.get((model_id, M), "")
        return (
            f"no compiled prefill artifact set for {model_id} at M={M}"
            + (f": {why}" if why else "")
            + (" -- compile it with `run_model.py compile` (its own cwd, a build-class devq job)" if M != SHIPPED_PREFILL_M else " -- the shipped build_peano caches are missing")
        )

    def rungs(self) -> tuple:
        out = []
        for model_id in self.models:
            for M in self.prefill_Ms.get(model_id, ()):
                out.append(ModelRung(model_id, "prefill", M, M, 0, self.precision_plan, KERNEL_SCALING, self._skip(model_id, M, "prefill")))
            for mid, logical, ubatch in self.ubatch_points:
                if mid != model_id:
                    continue
                if logical % ubatch:
                    raise ValueError(f"ubatch point ({mid}, {logical}, {ubatch}): logical prompt is not a whole number of chunks")
                out.append(ModelRung(model_id, "prefill", ubatch, logical, 0, self.precision_plan, UBATCH, self._skip(model_id, ubatch, "prefill")))
            for ctx in self.decode_ctxs:
                M = SHIPPED_PREFILL_M
                out.append(ModelRung(model_id, "decode", M, ctx, self.decode_n_tokens, self.precision_plan, DECODE_CONTEXT, self._skip(model_id, M, "decode")))
        return tuple(out)

    def artifact_sets(self) -> list:
        """Distinct (model_id, M) the walk dispatches on, in rung order, measurable only."""
        seen = []
        for r in self.rungs():
            if r.skip_reason is None and (r.model_id, r.M) not in seen:
                seen.append((r.model_id, r.M))
        return seen

    def expected_files(self) -> list:
        return [f"model_{m}.csv" for m in self.models]

    def expected_rows(self) -> dict:
        out = {}
        for r in self.rungs():
            counts = out.setdefault(r.csv_name, {"rows": 0, "measured": 0, "skipped": 0})
            counts["rows"] += 1
            counts["skipped" if r.skip_reason else "measured"] += 1
        return out

    def summary(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "models": list(self.models),
            "prefill_Ms": {k: list(v) for k, v in self.prefill_Ms.items()},
            "decode_ctxs": list(self.decode_ctxs),
            "ubatch_points": [list(t) for t in self.ubatch_points],
            "decode_n_tokens": self.decode_n_tokens,
            "prefill_samples": self.prefill_samples,
            "prefill_warmup": self.prefill_warmup,
            "decode_warmup": self.decode_warmup,
            "precision_plan": self.precision_plan,
            "rungs": [{"case_id": r.case_id, "skip_reason": r.skip_reason} for r in self.rungs()],
            "expected_rows": self.expected_rows(),
        }


PROFILES = {
    "ubatch-curve": ModelProfile(
        name="ubatch-curve",
        description=(
            "doc 56 H1b (section 5, the first real milestone): the two-point Qwen3-0.6B bf16 "
            "ubatch curve -- the SAME 1024-token prompt at ubatch 512 (2 chunks, incremental "
            "KV, rectangular FA at (512,1024)) and ubatch 1024 (1 chunk) -- chunked prefill "
            "through the driver's incremental path, verify gate per artifact set on the same "
            "prompt through the same chunked path"
        ),
        models=("qwen3_0_6b",),
        prefill_Ms={"qwen3_0_6b": ()},
        decode_ctxs=(),
        ubatch_points=(("qwen3_0_6b", 1024, 512), ("qwen3_0_6b", 1024, 1024)),
    ),
    "model-smoke": ModelProfile(
        name="model-smoke",
        description=(
            "doc 56 H1a: both models, decode at ctx 512/1024/2048 on the shipped M=2048 "
            "artifact set, and the Qwen3-0.6B kernel-scaling prefill curve at M 512/1024/2048 "
            "(512 and 1024 skip until their artifact sets are compiled)"
        ),
        models=("qwen3_0_6b", "llama32_1b"),
        prefill_Ms={"qwen3_0_6b": (512, 1024, 2048), "llama32_1b": (2048,)},
        decode_ctxs=(512, 1024, 2048),
    ),
}


def profile(name: str) -> ModelProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown model profile {name!r}; known: {sorted(PROFILES)}") from None


def discover_compiled(models, compiled_root: str | Path | None, llms_dir: str | Path | None = None) -> tuple:
    """(compiled, notes): every (model_id, M) with a manifest on disk.

    M == SHIPPED_PREFILL_M is the model's own `build_peano` caches; any other M
    is `<compiled_root>/<model_id>/M<M>/prefill_kernel_cache` with the shipped
    decode cache. A directory without `manifest.json` is not compiled.
    """
    llms = Path(llms_dir) if llms_dir else Path(_PE) / "llms"
    compiled, notes = {}, {}
    for model_id in models:
        build = llms / model_id / "build_peano"
        decode = build / "decode_kernel_cache"
        prefill = build / "prefill_kernel_cache"
        if (prefill / "manifest.json").is_file() and (decode / "manifest.json").is_file():
            compiled[(model_id, SHIPPED_PREFILL_M)] = {"prefill_cache": str(prefill), "decode_cache": str(decode)}
        else:
            notes[(model_id, SHIPPED_PREFILL_M)] = f"{build} holds no prefill+decode manifests"
        if compiled_root:
            for d in sorted(Path(compiled_root).glob(f"{model_id}/M*/prefill_kernel_cache")):
                try:
                    M = int(d.parent.name[1:])
                except ValueError:
                    continue
                if (d / "manifest.json").is_file() and (decode / "manifest.json").is_file():
                    compiled[(model_id, M)] = {"prefill_cache": str(d), "decode_cache": str(decode)}
                else:
                    note = d / "compile.json"
                    failed = None
                    if note.is_file():
                        try:
                            failed = json.loads(note.read_text(encoding="utf-8")).get("failed")
                        except ValueError:
                            failed = None
                    notes[(model_id, M)] = (
                        f"compile failed: {failed}" if failed
                        else f"{d} has no manifest.json (compile did not finish)"
                    )
    return compiled, notes


def main(argv=None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", nargs="?", default=None)
    ap.add_argument("--compiled-root", default=None)
    args = ap.parse_args(argv)
    if args.name is None:
        for name in sorted(PROFILES):
            print(f"{name}: {PROFILES[name].description}")
        return 0
    prof = profile(args.name)
    compiled, notes = discover_compiled(prof.models, args.compiled_root)
    print(json.dumps(prof.bind(compiled, notes).summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
