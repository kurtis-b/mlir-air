# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The model adapter seam (doc 56 section 3.1, H1a): one narrow interface over the
Qwen3-0.6B and Llama-3.2-1B production drivers for the model runner
(`transformer_layer/study/run_model.py`).

    prepare(model, precision_plan, compiled_shapes)   -> ModelSession
    prefill(token_ids, ubatch_policy, state)          -> PhaseResult
    decode(state, n_tokens)                            -> PhaseResult
    dispatch_vector(scope)                             -> the seven-key record
    verify_against_hf(...)                             -> the production gate's verdict

CONTRACT
    Built ON the drivers' own `Session` / `prepare_runtime` / `run_npu_prefill` /
    `run_npu_decode_step` -- the functions `make run`, `make profile` and the
    verify adapters already share -- never a fork of them. The two drivers have
    the same shape (llama32_1b_inference.py is the template qwen3_0_6b_inference.py
    mirrors), so the per-model difference is a `ModelBinding` row: module names,
    the planner's `ModelSpec`, the HF id, the verify adapter's dotted path.

    THE CLOCK is the forward pass only (operator rule 2026-08-22): a phase's
    `elapsed_s` runs from the first host op of the forward (the embedding
    lookup) to the instant its logits are CPU-readable. Tokenization, EOS
    padding, the HF gate and any comparison sit outside it. `tokens_per_second`
    counts VALID tokens only -- the padded tail rows a prefill dispatches are
    excluded from the numerator (doc 56 section 3.4).

    THE DISPATCH VECTOR is read off the drivers' `Profiler` (every
    `KernelCache.load_and_run` records one breakdown; every `time_cpu` block one
    host op), snapshotted at the phase boundary. Its `air_launches` and
    `herd_launches` are the launches EXECUTED in the scope -- each submission
    contributes its artifact's compile-time `LaunchCounts`, the counts the cache
    manifest persists -- which is the boundary count doc 57 prices at ~107 us,
    not the per-module figure the layer schema's `air_launches_per_elf` carries.
    `model_dispatch_vector_from_manifest` derives the SAME record statically
    from a cache manifest and a `Plan`, which is what the host tests compare
    the measured one against and what H0's "the plan reproduces the shipped
    sequence" gate becomes at model scope.

    `import model_adapter` needs no `air`, no XRT and no weights: every driver
    import is deferred into `prepare`, so the planner half (`plan_for`,
    `model_dispatch_vector_from_manifest`, `MODELS`) is host-testable.

FOOTGUNS
    - The drivers compile for ONE prefill M (`build_session` hard-codes 2048);
      `compiled_shapes` names the cache directories to bind, and `prepare`
      refuses rather than compiles when one is missing -- compilation is a
      separate, device-lock-holding step (`compile_prefill`), never a side
      effect of a measurement.
    - `precision_plan` other than `bf16` is refused here with the reason: these
      two drivers implement only the bf16 path; `w4_decode` is the int4 sibling
      driver (H2a), not a flag on these.
    - One `ModelSession` per process. `prepare_runtime` mutates the weights
      object in place with idempotency guards; the drivers say "do not call
      twice", and this module keeps that promise by refusing a second `prepare`.
    - `verify_against_hf` runs the production verify subprocess (`make verify`'s
      command line), which COMPILES the artifact set it verifies into the cache
      root it is pointed at. Point it at the root the rows ran on, and run it
      outside the clock.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LLMS = _HERE.parent
_PE = _LLMS.parent
for _p in (str(_PE), str(_LLMS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.plan import LLAMA32_1B, QWEN3_0_6B, NPU2_CAPS, Workload, decoder_graph, plan as _plan  # noqa: E402

#: The seven keys, in the schema's order (`study/schema.py` MODEL_DISPATCH_VECTOR_KEYS).
DISPATCH_VECTOR_KEYS = (
    "scope",
    "host_submissions",
    "runlist_entries",
    "air_launches",
    "herd_launches",
    "sync_boundaries",
    "bytes_transferred",
)

#: Precision plans these drivers implement. `w4_decode` lives in the int4 sibling
#: drivers (llama32_1b_int4); `w_bfp16_prefill` and `a8` are H4 / later.
SUPPORTED_PRECISION_PLANS = ("bf16",)

PREFILL_CACHE = "prefill_kernel_cache"
DECODE_CACHE = "decode_kernel_cache"


@dataclass(frozen=True)
class ModelBinding:
    """What differs between the two drivers, and nothing else."""

    model_id: str  # llms/ directory and the schema's model_id
    package: str  # driver module prefix: <package>_inference etc.
    spec: object  # shared.plan ModelSpec
    hf_id: str  # the checkpoint the driver loads for its default variant
    model_variant: str  # the driver's --model choice
    verify_adapter: str  # verify_runner --runner dotted path
    prefill_prompt_len_kwarg: bool  # run_npu_prefill accepts prompt_len=

    @property
    def directory(self) -> Path:
        return _LLMS / self.model_id


MODELS: dict[str, ModelBinding] = {
    "qwen3_0_6b": ModelBinding(
        "qwen3_0_6b", "qwen3_0_6b", QWEN3_0_6B, "Qwen/Qwen3-0.6B", "instruct",
        "qwen3_0_6b.verify_adapter", prefill_prompt_len_kwarg=False,
    ),
    "llama32_1b": ModelBinding(
        "llama32_1b", "llama32_1b", LLAMA32_1B, "meta-llama/Llama-3.2-1B-Instruct", "instruct",
        "llama32_1b.verify_adapter", prefill_prompt_len_kwarg=True,
    ),
}


# ---------------------------------------------------------------------------
# The planner half: pure, host-testable.
# ---------------------------------------------------------------------------


def plan_for(model_id: str, phase: str, M: int, kv_len: int, ctx: int = 2048, precision_plan: str = "bf16"):
    """The `Plan` whose sha keys a row: `plan(decoder_graph(spec), Workload(...))`."""
    spec = MODELS[model_id].spec
    return _plan(decoder_graph(spec), Workload(phase, M, kv_len, ctx, precision_plan), NPU2_CAPS)


def read_cache_manifest(cache_dir: str | Path) -> dict:
    """`KernelCache`'s manifest.json: name -> {output_binary, kernel, insts, launches}."""
    path = Path(cache_dir) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def launch_counts_of(manifests: dict[str, dict]) -> dict[str, dict]:
    """kernel name -> {air_launches, herd_launches} over one or more manifests."""
    out: dict[str, dict] = {}
    for manifest in manifests.values():
        for name, info in manifest.items():
            if info.get("launches"):
                out[name] = dict(info["launches"])
    return out


def model_dispatch_vector_from_manifest(plan_, launch_counts: dict[str, dict], scope: str) -> dict:
    """The record a phase WOULD produce, from its plan's ELF sequence and the
    cache manifest's compile-time launch counts: one submission per device
    stage instance, its artifact's launches executed once per submission.

    Sync boundaries and bytes are runtime facts (which BOs were static on that
    call) and are reported as 0 here; the measured vector fills them. A stage
    whose ELF is not in the manifest raises: the plan names an artifact the
    cache does not hold, which is exactly the mismatch this exists to surface.
    """
    if plan_.spec_name not in MODELS:
        raise KeyError(f"plan is for {plan_.spec_name!r}, which this adapter does not bind")
    spec_layers = MODELS[plan_.spec_name].spec.n_layers
    submissions = air = herd = 0
    for stage in plan_.stages:
        if stage.where != "device":
            continue
        counts = launch_counts.get(stage.name)
        if counts is None:
            raise KeyError(
                f"plan stage {stage.name!r} has no artifact in the cache manifest "
                f"(manifest holds {sorted(launch_counts)})"
            )
        repeats = spec_layers if stage.repeated else 1
        submissions += repeats
        air += repeats * int(counts["air_launches"])
        herd += repeats * int(counts["herd_launches"])
    return {
        "scope": scope,
        "host_submissions": submissions,
        "runlist_entries": submissions,
        "air_launches": air,
        "herd_launches": herd,
        "sync_boundaries": 0,
        "bytes_transferred": 0,
    }


def plan_launches_match_manifest(plan_, launch_counts: dict[str, dict]) -> list[str]:
    """Every device stage's planned launch count against the manifest's. Empty
    means the plan reproduces the shipped artifacts (H0's gate, per artifact)."""
    problems = []
    for stage in plan_.stages:
        if stage.where != "device":
            continue
        counts = launch_counts.get(stage.name)
        if counts is None:
            problems.append(f"{stage.name}: planned, not in the cache manifest")
        elif int(counts["air_launches"]) != stage.launches:
            problems.append(
                f"{stage.name}: plan says {stage.launches} air.launch, "
                f"manifest says {counts['air_launches']}"
            )
    return problems


def weights_source(hf_id: str) -> str:
    """`<hf_id>@<snapshot commit>` from the local HF cache; `@unknown` when the
    revision cannot be resolved offline. Never a guess."""
    try:
        from huggingface_hub import snapshot_download

        local = Path(snapshot_download(hf_id, local_files_only=True))
        sha = local.name
        if len(sha) == 40 and all(c in "0123456789abcdef" for c in sha):
            return f"{hf_id}@{sha}"
        return f"{hf_id}@{local}"
    except Exception:
        return f"{hf_id}@unknown"


# ---------------------------------------------------------------------------
# The device half.
# ---------------------------------------------------------------------------


@dataclass
class ModelSession:
    """The driver `Session` plus what the adapter added: which caches, at which M."""

    binding: ModelBinding
    driver: object  # the <package>_inference module
    session: object  # the driver's Session
    prefill_M: int
    prefill_cache_dir: Path
    decode_cache_dir: Path
    precision_plan: str
    n_tokens_reserved: int
    weights_source: str
    prepare_s: float


@dataclass
class DecodeState:
    """KV state after a prefill: what `decode` continues from."""

    k_cache: object
    v_cache: object
    current_pos: int  # next position to write
    next_token: int
    prompt_len: int
    max_seq: int


@dataclass
class PhaseResult:
    phase: str
    elapsed_s: float  # the forward-pass clock, summed over samples
    samples_s: list  # per-sample forward-pass seconds
    logical_tokens: int
    measured_tokens: int
    context_start: int
    context_end: int
    dispatch: dict  # the seven-key record for the phase (or per token)
    decomposition: dict  # device_ms / sync_ms / host_cpu_ms / host_ops over the phase
    tokens: list = field(default_factory=list)
    state: DecodeState | None = None

    @property
    def tokens_per_second(self) -> float:
        return self.measured_tokens / self.elapsed_s if self.elapsed_s > 0 else 0.0


class _ProfilerMark:
    """Snapshot of the per-name record counts in one or more Profilers."""

    def __init__(self, profilers):
        self.profilers = list(profilers)
        self.kernels = [{k: len(v) for k, v in p.kernel_breakdowns.items()} for p in self.profilers]
        self.cpu = [{k: len(v) for k, v in p.cpu_times.items()} for p in self.profilers]


def _delta(mark: _ProfilerMark, launch_counts: dict[str, dict], scope: str) -> tuple[dict, dict]:
    """(dispatch vector, decomposition) accumulated since `mark`."""
    subs = air = herd = syncs = nbytes = 0
    device_ms = sync_ms = host_ms = 0.0
    host_ops = 0
    elfs = set()
    for p, k0, c0 in zip(mark.profilers, mark.kernels, mark.cpu):
        for name, records in p.kernel_breakdowns.items():
            new = records[k0.get(name, 0):]
            if not new:
                continue
            elfs.add(name)
            counts = launch_counts.get(name)
            if counts is None:
                raise KeyError(f"{name!r} was dispatched but the cache manifest has no launch counts for it")
            subs += len(new)
            air += len(new) * int(counts["air_launches"])
            herd += len(new) * int(counts["herd_launches"])
            for r in new:
                syncs += int(r["n_written"]) + int(r["n_readback"])
                nbytes += int(r["bytes_written"]) + int(r.get("bytes_readback", 0))
                device_ms += float(r["kernel_ms"])
                sync_ms += float(r["write_ms"]) + float(r["read_ms"])
        for name, times in p.cpu_times.items():
            new = times[c0.get(name, 0):]
            host_ops += len(new)
            host_ms += 1000.0 * sum(new)
    vector = {
        "scope": scope,
        "host_submissions": subs,
        "runlist_entries": subs,  # load_and_run: one xrt.run per submission
        "air_launches": air,
        "herd_launches": herd,
        "sync_boundaries": syncs,
        "bytes_transferred": nbytes,
    }
    decomposition = {"device_ms": device_ms, "sync_ms": sync_ms, "host_cpu_ms": host_ms, "host_ops": host_ops, "distinct_elfs": len(elfs)}
    return vector, decomposition


class ModelAdapter:
    """One adapter per model id; `prepare` once per process."""

    def __init__(self, model_id: str):
        if model_id not in MODELS:
            raise KeyError(f"unknown model {model_id!r}; the adapter binds {sorted(MODELS)}")
        self.binding = MODELS[model_id]
        self.ms: ModelSession | None = None
        self._scratch_layout = None
        self._launch_counts: dict[str, dict] = {}
        self._last: dict[str, dict] = {}  # scope -> last dispatch vector

    # -- import the driver ---------------------------------------------------

    def _import_driver(self):
        model_dir = str(self.binding.directory)
        for p in (str(_LLMS), model_dir):
            if p in sys.path:
                sys.path.remove(p)
            sys.path.insert(0, p)
        import importlib

        return importlib.import_module(f"{self.binding.package}_inference")

    # -- prepare --------------------------------------------------------------

    def prepare(self, model: str, precision_plan: str, compiled_shapes: dict, *, n_tokens: int = 64, verbose: bool = False) -> ModelSession:
        """Bind the compiled artifacts, load weights and tokenizer, run the
        driver's `prepare_runtime`. `compiled_shapes` is
        `{"prefill_M": int, "prefill_cache": dir, "decode_cache": dir}`.

        `n_tokens` reserves KV / RoPE room past the prefill M, exactly as the
        driver's `build_session` reserves `args.n_tokens`.
        """
        if model != self.binding.model_id:
            raise ValueError(f"adapter is bound to {self.binding.model_id!r}, asked to prepare {model!r}")
        if precision_plan not in SUPPORTED_PRECISION_PLANS:
            raise ValueError(
                f"precision_plan {precision_plan!r} is not implemented by the {model} bf16 driver "
                f"(it implements {list(SUPPORTED_PRECISION_PLANS)}; w4_decode is the int4 sibling driver, "
                "doc 56 H2a) -- a derived skip, not a failure"
            )
        if self.ms is not None:
            raise RuntimeError("prepare() twice in one process: the drivers' prepare_runtime is one-shot")
        M = int(compiled_shapes["prefill_M"])
        prefill_dir = Path(compiled_shapes["prefill_cache"])
        decode_dir = Path(compiled_shapes["decode_cache"])
        for label, d in (("prefill", prefill_dir), ("decode", decode_dir)):
            if not (d / "manifest.json").is_file():
                raise FileNotFoundError(
                    f"no compiled {label} artifact set at {d} (no manifest.json); compile it first -- "
                    "prepare() never compiles"
                )

        t0 = time.perf_counter()
        driver = self._import_driver()
        from shared.infra.cache import KernelCache, Profiler

        prefill_cache = KernelCache(str(prefill_dir), verbose=verbose, profiler=Profiler(enabled=True))
        decode_cache = KernelCache(str(decode_dir), verbose=verbose, profiler=Profiler(enabled=True))
        for cache, d in ((prefill_cache, prefill_dir), (decode_cache, decode_dir)):
            # A cache compiled from `cd build_peano` (the Makefiles' cwd) records
            # RELATIVE binary paths; `load_manifest` resolves them against the
            # cwd and the backend loads them by that path later. Anchor them to
            # the cache's parent so the worker can run from its own directory.
            old_cwd = os.getcwd()
            os.chdir(d.parent)
            try:
                ok = cache.load_manifest()
            finally:
                os.chdir(old_cwd)
            if not ok:
                raise RuntimeError(f"the cache manifest under {d} names a binary that is missing")
            from air.backend.xrt import XRTCompileArtifact

            for name, art in list(cache.artifacts.items()):
                binary, insts = Path(art.output_binary), art.insts
                if not binary.is_absolute():
                    binary = (d.parent / binary).resolve()
                if insts and not Path(insts).is_absolute():
                    insts = str((d.parent / insts).resolve())
                cache.artifacts[name] = XRTCompileArtifact(str(binary), art.kernel, insts)
        self._launch_counts = launch_counts_of({"prefill": read_cache_manifest(prefill_dir), "decode": read_cache_manifest(decode_dir)})

        import importlib

        weights_mod = importlib.import_module(f"{self.binding.package}_weights")
        config = weights_mod.LlamaConfig()
        self._scratch_layout = self._restore_scratch_layout(config, M)
        hf_id = self.binding.hf_id
        weights = weights_mod.load_weights(hf_id, config=config)
        from ml_dtypes import bfloat16
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        rope_lut = weights_mod.generate_rope_lut(config=config, seq_len=M + n_tokens).astype(bfloat16)
        driver.prepare_runtime(prefill_cache, decode_cache, weights, config, M, rope_lut)
        session = driver.Session(
            config=config, seq_len=M, weights=weights, tokenizer=tokenizer,
            prefill_cache=prefill_cache, decode_cache=decode_cache,
            rope_lut_bf16=rope_lut, model_variant=self.binding.model_variant,
        )
        self.ms = ModelSession(
            binding=self.binding, driver=driver, session=session, prefill_M=M,
            prefill_cache_dir=prefill_dir, decode_cache_dir=decode_dir,
            precision_plan=precision_plan, n_tokens_reserved=n_tokens,
            weights_source=weights_source(hf_id), prepare_s=time.perf_counter() - t0,
        )
        return self.ms

    def _restore_scratch_layout(self, config, M):
        """What `compile_all_kernels` leaves in the prefill module and a
        run-only process lacks: the QKV ELF's f32-scratch arg layout
        (`_FUSED_SCRATCH_FOR`), derived from the registry exactly as the
        builder derives it (`alloc_gemm_scratch` over the Q/K/V specs at M,
        base arg 17). Without it the block runner passes 17 args to an ELF that
        declares 18 when Q is `fused-cast` (M=2048) -- the drivers' own
        `--run-only` path does exactly that today, and this adapter refuses to
        measure a path the gate does not run. Models whose prefill module has
        no such global (llama32_1b) need nothing.
        """
        import importlib

        prefill_mod = importlib.import_module(f"{self.binding.package}_prefill")
        if not hasattr(prefill_mod, "_FUSED_SCRATCH_FOR"):
            return None
        from shared.builders.gemm_builder import gemm_registry_config
        from shared.infra.stitching import alloc_gemm_scratch

        q_dim, kv_dim = config.n_heads * config.head_dim, config.n_kv_heads * config.head_dim
        specs = [
            (gemm_registry_config(M, config.emb_dim, q_dim, "bf16", "high"), M, q_dim),
            (gemm_registry_config(M, config.emb_dim, kv_dim, "bf16", "high"), M, kv_dim),
            (gemm_registry_config(M, config.emb_dim, kv_dim, "bf16", "high"), M, kv_dim),
        ]
        _args, scratch_for = alloc_gemm_scratch(specs, base_arg_count=17)
        prefill_mod._FUSED_SCRATCH_FOR = scratch_for
        return list(scratch_for)

    # -- tokens ---------------------------------------------------------------

    def tokenize(self, text: str) -> list:
        """The gate's tokenization (`verify_runner._tokenize`: plain `encode`, no
        chat template), so a study prompt and the gate's prompt are one token
        sequence."""
        return list(self.ms.session.tokenizer.encode(text))

    def pad(self, token_ids: list) -> list:
        """EOS-pad to the compiled M, as the drivers' `run_once` does."""
        s = self.ms.session
        ids = list(token_ids)
        if len(ids) > s.seq_len:
            raise ValueError(f"prompt of {len(ids)} tokens exceeds the compiled M {s.seq_len}; no chunking in H1a")
        return ids + [s.tokenizer.eos_token_id] * (s.seq_len - len(ids))

    # -- the phases -----------------------------------------------------------

    def _profilers(self):
        s = self.ms.session
        return (s.prefill_cache.profiler, s.decode_cache.profiler)

    def prefill(self, token_ids: list, ubatch_policy: str, state=None, *, samples: int = 1, warmup: int = 0) -> PhaseResult:
        """Prefill `token_ids` (valid tokens; padded here to M). `ubatch_policy`
        must be `whole` -- H1a has no chunking -- and `state` must be None (a
        fresh context). Runs `warmup` untimed forwards, then `samples` timed
        ones; the last one's KV state is returned for `decode`.
        """
        if ubatch_policy != "whole":
            raise ValueError(f"ubatch_policy {ubatch_policy!r}: H1a prefill is whole-prompt only (doc 56 section 3.4); chunking is H1b")
        if state is not None:
            raise ValueError("prefill() starts a fresh context; decode() continues one")
        s = self.ms.session
        drv = self.ms.driver
        valid = list(token_ids)
        prompt_len = len(valid)
        padded = self.pad(valid)
        max_seq = s.seq_len + self.ms.n_tokens_reserved
        kwargs = dict(tokenizer=s.tokenizer, cpu_attn=False, profile=False, quiet=True)
        if self.binding.prefill_prompt_len_kwarg:
            kwargs["prompt_len"] = prompt_len

        def forward():
            t0 = time.perf_counter()
            out = drv.run_npu_prefill(padded, s.weights, s.config, s.prefill_cache, s.decode_cache, s.rope_lut_bf16, max_seq, **kwargs)
            return time.perf_counter() - t0, out

        for _ in range(warmup):
            forward()
        mark = _ProfilerMark(self._profilers())
        times = []
        out = None
        for _ in range(samples):
            dt, out = forward()
            times.append(dt)
        vector, decomposition = _delta(mark, self._launch_counts, "prefill")
        # the vector is per forward: divide the accumulated counts by the sample count
        vector = {k: (v if k == "scope" else v // samples) for k, v in vector.items()}
        decomposition = {k: (v if k == "distinct_elfs" else v / samples) for k, v in decomposition.items()}
        decomposition["host_ops"] = int(round(decomposition["host_ops"]))
        self._last["prefill"] = vector
        prefill_token, logits_row, k_cache, v_cache, got_len = out
        if int(got_len) != prompt_len:
            raise RuntimeError(f"driver counted prompt_len {got_len}, adapter passed {prompt_len}: the prompt holds an EOS id")
        st = DecodeState(k_cache=k_cache, v_cache=v_cache, current_pos=prompt_len, next_token=int(prefill_token), prompt_len=prompt_len, max_seq=max_seq)
        return PhaseResult(
            phase="prefill", elapsed_s=sum(times), samples_s=times, logical_tokens=prompt_len,
            measured_tokens=prompt_len * samples, context_start=0, context_end=prompt_len,
            dispatch=vector, decomposition=decomposition, tokens=[int(prefill_token)], state=st,
        )

    def decode(self, state: DecodeState, n_tokens: int, *, warmup: int = 0) -> PhaseResult:
        """Decode `n_tokens` greedily from `state` (each token one sample). The
        first `warmup` tokens are generated untimed and not counted; they still
        advance the context, which `context_start` reports.
        """
        s = self.ms.session
        drv = self.ms.driver
        from ml_dtypes import bfloat16

        def step(tok, pos):
            t0 = time.perf_counter()
            x = s.weights.embed_table[tok].astype(bfloat16)
            next_tok, _logits = drv.run_npu_decode_step(x, s.weights, s.config, s.decode_cache, s.rope_lut_bf16, state.k_cache, state.v_cache, pos)
            return time.perf_counter() - t0, int(next_tok)

        tok, pos = state.next_token, state.current_pos
        if pos + warmup + n_tokens > state.max_seq:
            raise ValueError(f"decode of {warmup + n_tokens} tokens from position {pos} exceeds the reserved max_seq {state.max_seq}")
        for _ in range(warmup):
            _, tok = step(tok, pos)
            pos += 1
        start = pos
        mark = _ProfilerMark(self._profilers())
        times, tokens = [], []
        for _ in range(n_tokens):
            dt, tok = step(tok, pos)
            pos += 1
            times.append(dt)
            tokens.append(tok)
        vector, decomposition = _delta(mark, self._launch_counts, "decode")
        per_token = {k: (v if k == "scope" else v // n_tokens) for k, v in vector.items()}
        per_token["scope"] = "decode_token"
        decomposition = {k: (v if k == "distinct_elfs" else v / n_tokens) for k, v in decomposition.items()}
        decomposition["host_ops"] = int(round(decomposition["host_ops"]))
        self._last["decode"] = vector
        self._last["decode_token"] = per_token
        state.next_token, state.current_pos = tok, pos
        return PhaseResult(
            phase="decode", elapsed_s=sum(times), samples_s=times, logical_tokens=n_tokens,
            measured_tokens=n_tokens, context_start=start, context_end=pos,
            dispatch=per_token, decomposition=decomposition, tokens=tokens, state=state,
        )

    def dispatch_vector(self, scope: str) -> dict:
        """The last measured record for `scope` (prefill | decode | decode_token),
        strictly the seven keys."""
        if scope not in self._last:
            raise KeyError(f"no {scope!r} phase has been measured yet (have {sorted(self._last)})")
        return {k: self._last[scope][k] for k in DISPATCH_VECTOR_KEYS}

    # -- the gate ---------------------------------------------------------------

    def verify_against_hf(self, prompts_file: str | Path, report_dir: str | Path, *, cache_root: str | Path, max_seq: int, cwd: str | Path, timeout_s: int = 3600, extra_env: dict | None = None) -> dict:
        """Run the production verify gate (`make verify`'s command line:
        `verify_runner.py --runner <adapter> --prompts topk_token`) over
        `prompts_file`, against the artifact set under `cache_root` at
        `max_seq`. Returns `{passed, returncode, report_json, per_prompt, log}`.

        Outside the clock by construction: a subprocess, after the rows.
        """
        runner = _LLMS / "verify" / "verify_runner.py"
        cmd = [
            sys.executable, str(runner), f"--runner={self.binding.verify_adapter}",
            "--prompts", "topk_token", "--model", self.binding.model_variant,
            "--prompts-file", str(prompts_file), "--report-dir", str(report_dir), "--no-strict",
        ]
        env = dict(os.environ)
        env["LLMS_VERIFY_CACHE_ROOT"] = str(cache_root)
        env["LLMS_VERIFY_MAX_SEQ"] = str(int(max_seq))
        env.update(extra_env or {})
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        Path(cwd).mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout_s)
        log = (proc.stdout or "") + (proc.stderr or "")
        reports = sorted(Path(report_dir).glob("verify_topk_token_*.json"), key=lambda p: p.stat().st_mtime)
        per_prompt, report_json = [], None
        if reports:
            report_json = reports[-1]
            payload = json.loads(report_json.read_text(encoding="utf-8"))
            for rec in payload.get("topk_checks") or []:
                per_prompt.append({"prompt_idx": rec["prompt_idx"], "status": rec["status"], "fail_reason": rec.get("fail_reason"), "divergence_step": rec.get("divergence_step")})
        passed = proc.returncode == 0 and "[verify] PASS" in log and bool(per_prompt) and all(r["status"] == "OK" for r in per_prompt)
        return {
            "passed": passed, "returncode": proc.returncode, "report_json": str(report_json) if report_json else None,
            "per_prompt": per_prompt, "log": log, "command": cmd, "wall_s": time.perf_counter() - t0,
            "cache_root": str(cache_root), "max_seq": int(max_seq),
        }


# ---------------------------------------------------------------------------
# Compilation: a separate step, its own cwd, never inside a measurement.
# ---------------------------------------------------------------------------


COMPILE_NOTE = "compile.json"


def compile_prefill(model_id: str, M: int, cache_dir: str | Path, *, cwd: str | Path, verbose: bool = False, o_ffn_gemm_method: str | None = None) -> dict:
    """Compile the model's prefill artifact set at seq_len M into `cache_dir`,
    running in `cwd` (XRTBackend writes `air_project/` into the cwd -- two
    compiles from one directory clobber each other). Returns the manifest.

    `o_ffn_gemm_method` forces the O+FFN cascade's GEMM method (Qwen3-0.6B
    driver only; see `build_o_ffn_qwen_module`). It is written into
    `<cache_dir>/compile.json` as `artifact_deviation`, which the worker copies
    onto every row measured on the set: a forced method is not the plan."""
    adapter = ModelAdapter(model_id)
    driver_dir = adapter.binding.directory
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    old = os.getcwd()
    os.chdir(cwd)
    try:
        drv = adapter._import_driver()
        import importlib

        prefill_mod = importlib.import_module(f"{adapter.binding.package}_prefill")
        weights_mod = importlib.import_module(f"{adapter.binding.package}_weights")
        from shared.infra.cache import KernelCache, Profiler

        cache = KernelCache(str(Path(cache_dir).resolve()), verbose=verbose, profiler=Profiler(enabled=False))
        kwargs = {"o_ffn_gemm_method": o_ffn_gemm_method} if o_ffn_gemm_method else {}
        t0 = time.perf_counter()
        prefill_mod.compile_all_kernels(cache, weights_mod.LlamaConfig(), int(M), verbose=verbose, cpu_attn=False, **kwargs)
        wall = time.perf_counter() - t0
    finally:
        os.chdir(old)
    manifest = read_cache_manifest(cache_dir)
    note = {"model_id": model_id, "M": int(M), "wall_s": wall, "cwd": str(cwd), "driver_dir": str(driver_dir), "driver": drv.__name__,
            "artifact_deviation": (
                {"o_ffn_gemm_method": o_ffn_gemm_method,
                 "why": "the o_ffn cascade builder binds 4-arg fused-cast GEMMs only; the registry's best at this M is another method"}
                if o_ffn_gemm_method else None)}
    (Path(cache_dir) / COMPILE_NOTE).write_text(json.dumps(note, indent=1), encoding="utf-8")
    manifest["_compile"] = note
    return manifest


def artifact_key(model_id: str, M: int, precision_plan: str, cache_dirs: list) -> str:
    """sha256 over the model, M, precision and the cache manifests' binaries
    (their names, sizes and mtimes): names one compiled artifact SET. The plan
    hash names the planned sequence; this names the bytes that ran it."""
    h = hashlib.sha256(f"{model_id}|{M}|{precision_plan}".encode())
    for d in cache_dirs:
        manifest = read_cache_manifest(d)
        for name in sorted(manifest):
            info = manifest[name]
            binary = Path(info["output_binary"])
            if not binary.is_absolute():
                binary = Path(d).parent / binary
            st = binary.stat() if binary.exists() else None
            h.update(f"{name}|{binary.name}|{st.st_size if st else -1}|{int(st.st_mtime) if st else -1}".encode())
    return h.hexdigest()
