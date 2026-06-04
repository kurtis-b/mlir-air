#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real Gemma3 execution smoke and CPU benchmark helpers.

This module is deliberately CPU/HF oriented. It proves that local real Gemma3
artifacts can execute without importing AIR and can produce baseline timing
records without claiming NPU paper parity.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from gemma3_artifacts import MODEL_SPECS, Gemma3ArtifactError, load_real_model_artifacts
from gemma3_power import begin_power_window, finish_power_window


class Gemma3ExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Gemma3RealExecutionReport:
    model_variant: str
    backend: str
    status: str
    weights_dir: str | None
    tokenizer_path: str | None
    model_class: str
    tokenizer_class: str
    processor_class: str | None
    prompt_token_count: int
    logits_shape: tuple[int, ...]
    pixel_values_shape: tuple[int, ...] | None
    logits_checksum: float
    load_seconds: float
    forward_seconds: float
    notes: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format(self) -> str:
        shape = "x".join(str(dim) for dim in self.logits_shape) if self.logits_shape else "none"
        return (
            f"real_execution model={self.model_variant} backend={self.backend} "
            f"status={self.status} model_class={self.model_class} "
            f"tokens={self.prompt_token_count} logits_shape={shape} "
            f"checksum={self.logits_checksum:.6f} load_s={self.load_seconds:.3f} "
            f"forward_s={self.forward_seconds:.3f}"
        )


@dataclass(frozen=True)
class Gemma3RealBenchmarkReport:
    model_variant: str
    backend: str
    status: str
    metric: str
    weights_dir: str | None
    prompt_token_count: int
    decode_tokens: int
    warmup_iters: int
    timed_iters: int
    local_value: float
    unit: str
    load_seconds: float
    mean_seconds: float
    logits_shape: tuple[int, ...]
    power_snapshot: dict[str, Any] | None = None
    notes: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format(self) -> str:
        shape = "x".join(str(dim) for dim in self.logits_shape) if self.logits_shape else "none"
        return (
            f"real_benchmark model={self.model_variant} backend={self.backend} "
            f"status={self.status} metric={self.metric} tokens={self.prompt_token_count} "
            f"decode_tokens={self.decode_tokens} local={self.local_value:.6f} "
            f"unit={self.unit} mean_s={self.mean_seconds:.6f} logits_shape={shape}"
        )


def _trim_inputs(inputs: Any, max_prompt_tokens: int) -> Any:
    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    input_ids = inputs["input_ids"]
    if input_ids.shape[-1] > max_prompt_tokens:
        inputs["input_ids"] = input_ids[..., :max_prompt_tokens]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][..., :max_prompt_tokens]
    return inputs


def _exact_text_inputs(tokenizer: Any, torch_module: Any, prompt: str, target_tokens: int) -> dict[str, Any]:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    tokenized = tokenizer(
        prompt.strip() or "Gemma3 paper reproduction benchmark.",
        add_special_tokens=True,
        return_attention_mask=False,
    )
    ids = list(tokenized["input_ids"])
    if ids and isinstance(ids[0], list):
        ids = list(ids[0])
    if not ids:
        raise Gemma3ExecutionError("tokenizer produced no prompt tokens")
    repeat_ids = ids[1:] if len(ids) > 1 else ids
    while len(ids) < target_tokens:
        ids.extend(repeat_ids[: target_tokens - len(ids)])
    input_ids = torch_module.tensor([ids[:target_tokens]], dtype=torch_module.long)
    attention_mask = torch_module.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _load_cpu_model_inputs(
    *,
    model_variant: str,
    weights_dir: Path | None,
    prompt: str,
    max_prompt_tokens: int,
    vision_smoke: bool,
) -> tuple[Any, Any, Any, Any, Any, float]:
    try:
        import torch
    except Exception as exc:
        raise Gemma3ExecutionError("python:torch is required") from exc

    try:
        inventory = load_real_model_artifacts(
            model_variant,
            weights_dir=weights_dir,
            strict=True,
        )
    except Gemma3ArtifactError as exc:
        raise Gemma3ExecutionError(str(exc)) from exc
    if inventory.weights_dir is None:
        raise Gemma3ExecutionError("resolved weights_dir is missing")

    include_image = vision_smoke or model_variant == "gemma3-4b-vision"
    load_start = perf_counter()
    processor = None
    if model_variant == "gemma3-1b":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise Gemma3ExecutionError("python:transformers is required") from exc
        tokenizer = AutoTokenizer.from_pretrained(inventory.weights_dir)
        model = AutoModelForCausalLM.from_pretrained(
            inventory.weights_dir,
            dtype=torch.bfloat16,
        )
        inputs = _exact_text_inputs(tokenizer, torch, prompt, max_prompt_tokens)
    else:
        try:
            from PIL import Image
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as exc:
            raise Gemma3ExecutionError("python:transformers and python:Pillow are required") from exc
        processor = AutoProcessor.from_pretrained(inventory.weights_dir)
        model = AutoModelForImageTextToText.from_pretrained(
            inventory.weights_dir,
            dtype=torch.bfloat16,
        )
        tokenizer = processor.tokenizer
        if include_image:
            image = Image.new("RGB", (224, 224), color=(32, 64, 96))
            text = prompt if "<start_of_image>" in prompt else "<start_of_image> " + prompt
            inputs = processor(text=text, images=image, return_tensors="pt")
        else:
            inputs = _exact_text_inputs(tokenizer, torch, prompt, max_prompt_tokens)
    model.eval()
    return inventory, model, tokenizer, processor, inputs, perf_counter() - load_start


def run_cpu_smoke(
    *,
    model_variant: str,
    weights_dir: Path | None = None,
    prompt: str = "Gemma3 paper reproduction smoke.",
    max_prompt_tokens: int = 16,
    vision_smoke: bool = False,
) -> Gemma3RealExecutionReport:
    try:
        import torch
    except Exception as exc:
        raise Gemma3ExecutionError("python:torch is required") from exc

    inventory, model, tokenizer, processor, inputs, load_seconds = _load_cpu_model_inputs(
        model_variant=model_variant,
        weights_dir=weights_dir,
        prompt=prompt,
        max_prompt_tokens=max_prompt_tokens,
        vision_smoke=vision_smoke,
    )

    prompt_token_count = int(inputs["input_ids"].shape[-1])
    pixel_values = inputs.get("pixel_values") if hasattr(inputs, "get") else None
    pixel_values_shape = (
        tuple(int(dim) for dim in pixel_values.shape) if pixel_values is not None else None
    )
    forward_start = perf_counter()
    with torch.no_grad():
        output = model(**inputs)
    forward_seconds = perf_counter() - forward_start
    logits = output.logits.detach().float()
    checksum = float(logits[0, -1, : min(64, logits.shape[-1])].sum())

    return Gemma3RealExecutionReport(
        model_variant=model_variant,
        backend="cpu-hf",
        status="PASS",
        weights_dir=inventory.weights_dir,
        tokenizer_path=inventory.tokenizer_path,
        model_class=type(model).__name__,
        tokenizer_class=type(tokenizer).__name__,
        processor_class=type(processor).__name__ if processor is not None else None,
        prompt_token_count=prompt_token_count,
        logits_shape=tuple(int(dim) for dim in logits.shape),
        pixel_values_shape=pixel_values_shape,
        logits_checksum=checksum,
        load_seconds=load_seconds,
        forward_seconds=forward_seconds,
        notes=("not a paper timing; CPU smoke only",),
    )


def run_cpu_benchmark(
    *,
    model_variant: str,
    weights_dir: Path | None = None,
    prompt: str = "Gemma3 paper reproduction benchmark.",
    max_prompt_tokens: int = 16,
    metric: str = "prefill_ttft_seconds",
    decode_tokens: int = 1,
    warmup_iters: int = 0,
    timed_iters: int = 1,
    vision_smoke: bool = False,
    power_sample: bool = False,
    run_id: str | None = None,
) -> Gemma3RealBenchmarkReport:
    if metric not in ("prefill_ttft_seconds", "decode_tps", "vision_ttft_seconds"):
        raise ValueError("metric must be prefill_ttft_seconds, decode_tps, or vision_ttft_seconds")
    if timed_iters <= 0 or warmup_iters < 0:
        raise ValueError("timed_iters must be positive and warmup_iters must be non-negative")
    if metric == "decode_tps" and decode_tokens <= 0:
        raise ValueError("decode_tps requires positive decode_tokens")
    try:
        import torch
    except Exception as exc:
        raise Gemma3ExecutionError("python:torch is required") from exc

    inventory, model, _tokenizer, _processor, inputs, load_seconds = _load_cpu_model_inputs(
        model_variant=model_variant,
        weights_dir=weights_dir,
        prompt=prompt,
        max_prompt_tokens=max_prompt_tokens,
        vision_smoke=vision_smoke or metric == "vision_ttft_seconds",
    )
    prompt_token_count = int(inputs["input_ids"].shape[-1])

    def run_prefill() -> Any:
        return model(**inputs)

    def prepare_decode_state() -> tuple[Any, Any, Any]:
        prefill_output = model(**inputs, use_cache=True)
        past_key_values = prefill_output.past_key_values
        next_token = prefill_output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.clone()
        return past_key_values, next_token, attention_mask

    def run_decode_tokens(state: tuple[Any, Any, Any]) -> Any:
        past_key_values, next_token, attention_mask = state
        output = None
        for _ in range(decode_tokens):
            decode_inputs: dict[str, Any] = {
                "input_ids": next_token,
                "past_key_values": past_key_values,
                "use_cache": True,
            }
            if attention_mask is not None:
                extension = torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, extension], dim=-1)
                decode_inputs["attention_mask"] = attention_mask
            output = model(**decode_inputs)
            past_key_values = output.past_key_values
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return output

    with torch.no_grad():
        if metric == "decode_tps":
            for _ in range(warmup_iters):
                run_decode_tokens(prepare_decode_state())
            decode_states = [prepare_decode_state() for _ in range(timed_iters)]
            elapsed = 0.0
            output = None
            power_window = begin_power_window(sample=power_sample, run_id=run_id, target_backend="cpu")
            for state in decode_states:
                start = perf_counter()
                output = run_decode_tokens(state)
                elapsed += perf_counter() - start
        else:
            for _ in range(warmup_iters):
                run_prefill()
            elapsed = 0.0
            output = None
            power_window = begin_power_window(sample=power_sample, run_id=run_id, target_backend="cpu")
            for _ in range(timed_iters):
                start = perf_counter()
                output = run_prefill()
                elapsed += perf_counter() - start
    power_snapshot = (
        finish_power_window(power_window, elapsed_seconds=elapsed).to_json_dict()
        if power_sample
        else None
    )
    mean_seconds = elapsed / float(timed_iters)

    if metric == "decode_tps":
        local_value = float(decode_tokens) / mean_seconds if mean_seconds else 0.0
        unit = "tokens_per_second"
        logits_shape: tuple[int, ...] = ()
    else:
        local_value = mean_seconds
        unit = "seconds"
        logits_shape = tuple(int(dim) for dim in output.logits.shape)

    return Gemma3RealBenchmarkReport(
        model_variant=model_variant,
        backend="cpu-hf",
        status="PASS",
        metric=metric,
        weights_dir=inventory.weights_dir,
        prompt_token_count=prompt_token_count,
        decode_tokens=decode_tokens if metric == "decode_tps" else 0,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        local_value=local_value,
        unit=unit,
        load_seconds=load_seconds,
        mean_seconds=mean_seconds,
        logits_shape=logits_shape,
        power_snapshot=power_snapshot,
        notes=(
            "real CPU/HF measurement for the requested sequence length",
            "decode_tps excludes prefill by constructing the KV cache before the timed decode loop"
            if metric == "decode_tps"
            else "prefill_ttft measures one full prompt forward pass",
        ),
    )


def _self_test() -> None:
    report = Gemma3RealExecutionReport(
        model_variant="gemma3-1b",
        backend="cpu-hf",
        status="PASS",
        weights_dir="/tmp/gemma",
        tokenizer_path="/tmp/gemma/tokenizer.json",
        model_class="Gemma3ForCausalLM",
        tokenizer_class="GemmaTokenizer",
        processor_class=None,
        prompt_token_count=4,
        logits_shape=(1, 4, 8),
        pixel_values_shape=None,
        logits_checksum=1.25,
        load_seconds=0.1,
        forward_seconds=0.2,
        notes=("fixture",),
    )
    if report.to_json_dict()["status"] != "PASS":
        raise AssertionError(report)
    benchmark = Gemma3RealBenchmarkReport(
        model_variant="gemma3-1b",
        backend="cpu-hf",
        status="PASS",
        metric="prefill_ttft_seconds",
        weights_dir="/tmp/gemma",
        prompt_token_count=4,
        decode_tokens=0,
        warmup_iters=0,
        timed_iters=1,
        local_value=0.2,
        unit="seconds",
        load_seconds=0.1,
        mean_seconds=0.2,
        logits_shape=(1, 4, 8),
        notes=("fixture",),
    )
    print(report.format())
    print(benchmark.format())
    print("GEMMA3_REAL_EXECUTION_SELF_TEST: PASS")


def _write_json(path: Path, report: Gemma3RealExecutionReport | Gemma3RealBenchmarkReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 real execution smoke and CPU benchmark helper")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--cpu-benchmark", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default="gemma3-1b")
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--prompt", default="Gemma3 paper reproduction smoke.")
    parser.add_argument("--max-prompt-tokens", type=int, default=16)
    parser.add_argument("--metric", choices=["prefill_ttft_seconds", "decode_tps", "vision_ttft_seconds"], default="prefill_ttft_seconds")
    parser.add_argument("--decode-tokens", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--timed-iters", type=int, default=1)
    parser.add_argument("--vision-smoke", action="store_true")
    parser.add_argument("--power-sample", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    if args.cpu_smoke:
        try:
            report = run_cpu_smoke(
                model_variant=args.model_variant,
                weights_dir=args.weights_dir,
                prompt=args.prompt,
                max_prompt_tokens=args.max_prompt_tokens,
                vision_smoke=args.vision_smoke,
            )
        except Gemma3ExecutionError as exc:
            print(f"GEMMA3_REAL_EXECUTION_BLOCKED: {exc}")
            return 2
        print(report.format())
        if args.json:
            _write_json(args.json, report)
            print(f"GEMMA3_REAL_EXECUTION_JSON: {args.json}")
        return 0
    if args.cpu_benchmark:
        try:
            report = run_cpu_benchmark(
                model_variant=args.model_variant,
                weights_dir=args.weights_dir,
                prompt=args.prompt,
                max_prompt_tokens=args.max_prompt_tokens,
                metric=args.metric,
                decode_tokens=args.decode_tokens,
                warmup_iters=args.warmup_iters,
                timed_iters=args.timed_iters,
                vision_smoke=args.vision_smoke,
                power_sample=args.power_sample,
                run_id=args.run_id,
            )
        except (Gemma3ExecutionError, ValueError) as exc:
            print(f"GEMMA3_REAL_BENCHMARK_BLOCKED: {exc}")
            return 2
        print(report.format())
        if args.json:
            _write_json(args.json, report)
            print(f"GEMMA3_REAL_BENCHMARK_JSON: {args.json}")
        return 0
    parser.error("one of --self-test, --cpu-smoke, or --cpu-benchmark is required")


if __name__ == "__main__":
    raise SystemExit(main())
