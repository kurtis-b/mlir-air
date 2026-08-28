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
    #: `[2026-08-26]` doc 56 H4 (queue item 20): (model_id, M, precision_plan)
    #: triples. Each becomes ONE kernel-scaling prefill rung at that M under
    #: THAT plan, so one profile -- one walk, one session, one prompt -- can
    #: carry a precision A/B whose two arms are two artifact sets. Distinct
    #: from `prefill_Ms`, which takes the profile's own `precision_plan`.
    prefill_points: tuple = ()
    #: `[2026-08-26]` doc 56 H2b (queue item 24): the DECODE mirror of
    #: `prefill_points` -- (model_id, context_end, precision_plan) triples, one
    #: decode rung each on the shipped prefill M under THAT plan. Same reason
    #: as its prefill sibling: when a precision becomes a DEFAULT, its A/B
    #: against the precision it replaced has to be one walk in one session on
    #: one prompt, or the comparison is two sessions' drift as much as the
    #: precision. Distinct from `decode_ctxs`, which takes the profile's own
    #: `precision_plan`.
    decode_points: tuple = ()
    #: (model_id, M, precision_plan) -> {"prefill_cache": dir, "decode_cache": dir};
    #: set by `bind`. The precision plan is part of the ARTIFACT SET key because
    #: a plan selects ELFs (H2b the decode set, H4 the prefill set) -- two rungs
    #: at one M under two plans are two sets and two worker processes.
    compiled: dict = field(default_factory=dict)
    skip_notes: dict = field(default_factory=dict)

    def bind(self, compiled: dict, skip_notes: dict | None = None) -> "ModelProfile":
        """The profile with its applicability rule applied to what is compiled."""
        return replace(self, compiled=dict(compiled), skip_notes=dict(skip_notes or {}))

    def _skip(self, model_id: str, M: int, plan: str) -> str | None:
        if (model_id, M, plan) in self.compiled:
            return None
        why = self.skip_notes.get((model_id, M, plan), "")
        return (
            f"no compiled artifact set for {model_id} at M={M} under {plan}"
            + (f": {why}" if why else "")
            + (" -- compile it with `run_model.py compile` (its own cwd, a build-class devq job)" if M != SHIPPED_PREFILL_M else " -- the shipped build_peano caches are missing")
        )

    def precision_plans_used(self) -> tuple:
        """Every precision plan any rung of this profile runs under, in rung
        order -- what `discover_compiled` must look for."""
        seen = [self.precision_plan]
        for _mid, _M, plan in self.prefill_points:
            if plan not in seen:
                seen.append(plan)
        for _mid, _ctx, plan in self.decode_points:
            if plan not in seen:
                seen.append(plan)
        return tuple(seen)

    def rungs(self) -> tuple:
        out = []
        for model_id in self.models:
            for M in self.prefill_Ms.get(model_id, ()):
                out.append(ModelRung(model_id, "prefill", M, M, 0, self.precision_plan, KERNEL_SCALING, self._skip(model_id, M, self.precision_plan)))
            for mid, M, plan in self.prefill_points:
                if mid != model_id:
                    continue
                out.append(ModelRung(model_id, "prefill", M, M, 0, plan, KERNEL_SCALING, self._skip(model_id, M, plan)))
            for mid, logical, ubatch in self.ubatch_points:
                if mid != model_id:
                    continue
                if logical % ubatch:
                    raise ValueError(f"ubatch point ({mid}, {logical}, {ubatch}): logical prompt is not a whole number of chunks")
                out.append(ModelRung(model_id, "prefill", ubatch, logical, 0, self.precision_plan, UBATCH, self._skip(model_id, ubatch, self.precision_plan)))
            for ctx in self.decode_ctxs:
                M = SHIPPED_PREFILL_M
                out.append(ModelRung(model_id, "decode", M, ctx, self.decode_n_tokens, self.precision_plan, DECODE_CONTEXT, self._skip(model_id, M, self.precision_plan)))
            for mid, ctx, plan in self.decode_points:
                if mid != model_id:
                    continue
                M = SHIPPED_PREFILL_M
                out.append(ModelRung(model_id, "decode", M, ctx, self.decode_n_tokens, plan, DECODE_CONTEXT, self._skip(model_id, M, plan)))
        return tuple(out)

    def artifact_sets(self) -> list:
        """Distinct (model_id, M, precision_plan) the walk dispatches on, in rung
        order, measurable only."""
        seen = []
        for r in self.rungs():
            key = (r.model_id, r.M, r.precision_plan)
            if r.skip_reason is None and key not in seen:
                seen.append(key)
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
            "prefill_points": [list(t) for t in self.prefill_points],
            "decode_points": [list(t) for t in self.decode_points],
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
    "w4-decode": ModelProfile(
        name="w4-decode",
        description=(
            "doc 56 H2a (queue item 17): the EXISTING llama32_1b_int4 decode -- bf16 NPU "
            "prefill on dequantized AWQ weights + int4 NPU decode, the shipped build_peano "
            "caches -- under the study's decomposition: decode at ctx 512/1024/2048 on the "
            "shipped M=2048 artifact set, precision_plan_id=w4_decode, quant_* populated "
            "from the packing code, the prediction written before the walk"
        ),
        models=("llama32_1b_int4",),
        prefill_Ms={"llama32_1b_int4": ()},
        decode_ctxs=(512, 1024, 2048),
        precision_plan="w4_decode",
    ),
    "bfp16-prefill": ModelProfile(
        name="bfp16-prefill",
        description=(
            "doc 56 H4 (queue item 20): the FIRST performance number for the bfp16 prefill "
            "path. Llama-3.2-1B-AWQ prefill at the shipped ubatch M=2048 measured twice in "
            "one walk, one session, on the SAME 2048-token prompt -- the baseline arm "
            "(precision_plan_id=w4_decode, the bf16 prefill stitchers on dequantized AWQ "
            "weights, the shipped build_peano set) against the bfp16 arm "
            "(precision_plan_id=w_bfp16_prefill, rms_gemms_rope_bfp16 / o_ffn_bfp16 over "
            "bfp16ebs8 weight BOs, 248 launches against 328). quant_* from the bfp16 "
            "packer's own contract; the prediction written before the walk "
            "(results/item20-h4-20260826/PREDICTION.md)"
        ),
        models=("llama32_1b_int4",),
        prefill_Ms={"llama32_1b_int4": ()},
        decode_ctxs=(),
        prefill_points=(("llama32_1b_int4", SHIPPED_PREFILL_M, "w4_decode"),
                        ("llama32_1b_int4", SHIPPED_PREFILL_M, "w_bfp16_prefill")),
        precision_plan="w4_decode",
    ),
    "w4-decode-qwen": ModelProfile(
        name="w4-decode-qwen",
        description=(
            "doc 56 H2b (queue item 18): Qwen3-0.6B decode under precision_plan_id=w4_decode "
            "-- the flag-selected int4 O+FFN cascade (o_gemv_ffn_int4, RTN gs=128 packed "
            "weights, QKV and LM head bf16) at ctx 512/1024/2048 on the shipped M=2048 "
            "prefill set + the w4_decode decode set from `run_model.py compile-decode`; "
            "quant_* populated from qwen3_0_6b.w4_decode_pack, the prediction written "
            "before the walk (results/item18-h2b-20260826/PREDICTION.md)"
        ),
        models=("qwen3_0_6b",),
        prefill_Ms={"qwen3_0_6b": ()},
        decode_ctxs=(512, 1024, 2048),
        precision_plan="w4_decode",
    ),
    "w4-default-qwen": ModelProfile(
        name="w4-default-qwen",
        description=(
            "doc 56 H2b (queue item 24): the standing Qwen3-0.6B decode numbers RE-TAKEN "
            "after `QWEN3_W4_DECODE` became the default -- both precisions in ONE walk, one "
            "session, one prompt per context, so the A/B is the precision and not two "
            "sessions' drift. Six decode rungs: ctx 512/1024/2048 under precision_plan_id="
            "w4_decode (the new default: the int4 O+FFN cascade on "
            "<compiled_root>/qwen3_0_6b/w4_decode/decode_kernel_cache) and the same three "
            "under bf16 (the shipped build_peano decode set), all on the shipped M=2048 "
            "prefill set. Supersedes item 18's `w4-decode-qwen` walks as the standing "
            "number; that profile stays as it was, so its walks stay reproducible"
        ),
        models=("qwen3_0_6b",),
        prefill_Ms={"qwen3_0_6b": ()},
        decode_ctxs=(512, 1024, 2048),
        precision_plan="w4_decode",
        decode_points=(("qwen3_0_6b", 512, "bf16"),
                       ("qwen3_0_6b", 1024, "bf16"),
                       ("qwen3_0_6b", 2048, "bf16")),
    ),
    "h5-cells": ModelProfile(
        name="h5-cells",
        description=(
            "doc 56 H5 (queue item 21): the PLANNER-SELECTED cells plus two declared "
            "controls. Not a Cartesian matrix -- 15 rungs against a 288-cell axis "
            "product. The selection rule is applied per cell in "
            "results/item21-h5-20260827/rederive_selection.py and its output is the "
            "authority; this description states which of these rungs it selects and "
            "which are here for another reason, because an earlier version of this "
            "text asserted reasons the rule did not actually give. "
            "SELECTED: the four Qwen prefill rungs by S1 -- the LM head went 10 "
            "air.launches to 3 at f0262b18, moving the plan to 479/479/619/955 "
            "against the 486/486/626/962 the newest passing rows recorded; the three "
            "Qwen w4_decode rungs by S1b, comparing today's w4 artifact set against "
            "the recorded one FOR THAT PRECISION (an earlier scan compared it against "
            "the bf16 set and manufactured the difference); and the four llama32_1b "
            "rungs by S1b plus S1c. "
            "NOT SELECTED, and here as the CONTROL ARM of the precision A/B: the "
            "three Qwen bf16 decode rungs. The rule REJECTS them -- plan, bytes and "
            "timing contract all stand -- but item 24 established that an A/B whose "
            "two arms are two sessions measures session drift as much as precision, "
            "so the control belongs in the same walk. "
            "CARRIED AS A DERIVED SKIP: the Qwen w_bfp16_prefill rung, refused twice "
            "over (the planner has no qk-norm bfp16 prefill stitcher, and no artifact "
            "set exists), so the walk demonstrates that its skips are derived. "
            "S1c, added after the review: a standing row whose root carries no "
            "`timing` block predates item 19's contract, so its device_ms is not the "
            "quantity measured today and the cell cannot be rejected on bytes."
        ),
        models=("qwen3_0_6b", "llama32_1b"),
        prefill_Ms={"qwen3_0_6b": (512, 1024, 2048), "llama32_1b": (2048,)},
        decode_ctxs=(512, 1024, 2048),
        precision_plan="bf16",
        ubatch_points=(("qwen3_0_6b", 1024, 512),),
        prefill_points=(("qwen3_0_6b", SHIPPED_PREFILL_M, "w_bfp16_prefill"),),
        decode_points=(("qwen3_0_6b", 512, "w4_decode"),
                       ("qwen3_0_6b", 1024, "w4_decode"),
                       ("qwen3_0_6b", 2048, "w4_decode")),
    ),
    "h5-cold": ModelProfile(
        name="h5-cold",
        description=(
            "doc 56 H5 (queue item 21), the COLD/WARM control. The planner cannot select on this "
            "axis -- `plan()` has no cold/warm term, its cost model is steady state -- so this is "
            "a control, not a planner-selected cell. It is `h5-cells`'s ctx-2048 bf16 decode rung "
            "with `decode_warmup = 0`, so the FIRST timed token is the first dispatch of every "
            "decode ELF in the process and `samples_s[0]` against the median of the rest is the "
            "cold/warm ratio. Its own profile rather than a knob on `h5-cells` because the row "
            "key does not carry the warm-up count: sharing a profile would collide, and folding "
            "the cold token into the standing 32-token mean would move the standing number by the "
            "act of measuring it."
        ),
        models=("qwen3_0_6b",),
        prefill_Ms={"qwen3_0_6b": ()},
        decode_ctxs=(2048,),
        decode_warmup=0,
        precision_plan="bf16",
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


def discover_compiled(models, compiled_root: str | Path | None, llms_dir: str | Path | None = None,
                      precision_plan: str = "bf16", precision_plans=None) -> tuple:
    """(compiled, notes), both keyed by **(model_id, M, precision_plan)**.

    A precision plan selects an ARTIFACT SET, so it is part of the key. Which
    set it selects is the plan's own property (`model_adapter.plan_phase`):

    - the model's SHIPPED plan (its binding's first) is the `build_peano`
      caches, prefill and decode both; other prefill `M` live at
      `<compiled_root>/<model_id>/M<M>/prefill_kernel_cache` with the shipped
      decode cache -- unchanged from H1a;
    - a `decode`-phase plan (`w4_decode`, doc 56 H2b) keeps that prefill set
      and takes `<compiled_root>/<model_id>/<plan>/decode_kernel_cache`;
    - a `prefill`-phase plan (`w_bfp16_prefill`, doc 56 H4) is the mirror: it
      takes `<compiled_root>/<model_id>/<plan>/M<M>/prefill_kernel_cache` and
      keeps the SHIPPED decode set, because `bfp16ebs8` names the prefill
      GEMM's weight operand and leaves the decode GEMV alone.

    A missing set is a derived skip NAMING what to run, never a silent
    fall-through to the shipped bytes.

    `precision_plans` `[2026-08-26]` (queue item 20): look for ALL of these
    plans, so one profile can carry a precision A/B. `precision_plan` remains
    for callers with a single plan.
    """
    from shared.model_adapter import MODELS as _BINDINGS, plan_phase

    plans = tuple(precision_plans) if precision_plans else (precision_plan,)
    llms = Path(llms_dir) if llms_dir else Path(_PE) / "llms"
    compiled, notes = {}, {}
    for model_id in models:
        build = llms / model_id / "build_peano"
        shipped_prefill = build / "prefill_kernel_cache"
        shipped_decode = build / "decode_kernel_cache"
        try:
            shipped_plan = _BINDINGS[model_id].precision_plans[0]
        except Exception:
            shipped_plan = "bf16"
        for plan in plans:
            phase = plan_phase(plan)
            is_shipped = plan == shipped_plan
            # -- decode set
            if is_shipped or phase == "prefill":
                decode, decode_note = shipped_decode, None
            else:
                decode = (Path(compiled_root) / model_id / plan / "decode_kernel_cache") if compiled_root else None
                decode_note = (
                    f"no {plan} decode artifact set"
                    + (f" at {decode}" if decode else " (no --compiled-root)")
                    + f" -- compile it with `run_model.py compile-decode --model {model_id} "
                      f"--precision-plan {plan}` (its own cwd, a build-class devq job)"
                )
            decode_ok = decode is not None and (decode / "manifest.json").is_file()
            # -- prefill sets, per M
            if phase == "prefill" and not is_shipped:
                roots = []
                if compiled_root:
                    roots = sorted(Path(compiled_root).glob(f"{model_id}/{plan}/M*/prefill_kernel_cache"))
                if not roots:
                    notes[(model_id, SHIPPED_PREFILL_M, plan)] = (
                        f"no {plan} prefill artifact set under "
                        + (f"{Path(compiled_root) / model_id / plan}" if compiled_root else "(no --compiled-root)")
                        + f" -- it replaces the PREFILL ELFs, so it must be compiled and placed at "
                          f"<compiled-root>/{model_id}/{plan}/M<M>/prefill_kernel_cache")
                for d in roots:
                    try:
                        M = int(d.parent.name[1:])
                    except ValueError:
                        continue
                    if (d / "manifest.json").is_file() and decode_ok:
                        compiled[(model_id, M, plan)] = {"prefill_cache": str(d), "decode_cache": str(decode)}
                    else:
                        notes[(model_id, M, plan)] = _compile_note(d, decode, decode_ok, decode_note)
                continue
            if (shipped_prefill / "manifest.json").is_file() and decode_ok:
                compiled[(model_id, SHIPPED_PREFILL_M, plan)] = {"prefill_cache": str(shipped_prefill), "decode_cache": str(decode)}
            else:
                notes[(model_id, SHIPPED_PREFILL_M, plan)] = (
                    decode_note if ((shipped_prefill / "manifest.json").is_file() and not decode_ok and decode_note)
                    else f"{build} holds no prefill+decode manifests")
            if compiled_root:
                for d in sorted(Path(compiled_root).glob(f"{model_id}/M*/prefill_kernel_cache")):
                    try:
                        M = int(d.parent.name[1:])
                    except ValueError:
                        continue
                    if (d / "manifest.json").is_file() and decode_ok:
                        compiled[(model_id, M, plan)] = {"prefill_cache": str(d), "decode_cache": str(decode)}
                    else:
                        notes[(model_id, M, plan)] = _compile_note(d, decode, decode_ok, decode_note)
    return compiled, notes


def _compile_note(d: Path, decode, decode_ok: bool, decode_note) -> str:
    """Why a prefill cache directory is not a usable artifact set."""
    note = d / "compile.json"
    failed = None
    if note.is_file():
        try:
            failed = json.loads(note.read_text(encoding="utf-8")).get("failed")
        except ValueError:
            failed = None
    if failed:
        return f"compile failed: {failed}"
    if (d / "manifest.json").is_file() and not decode_ok and decode_note:
        return decode_note
    return f"{d} has no manifest.json (compile did not finish)"


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
    compiled, notes = discover_compiled(prof.models, args.compiled_root, precision_plans=prof.precision_plans_used())
    print(json.dumps(prof.bind(compiled, notes).summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
