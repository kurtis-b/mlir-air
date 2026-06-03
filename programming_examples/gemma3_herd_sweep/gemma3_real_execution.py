#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real Gemma3 execution smoke helpers.

This module is deliberately CPU/HF oriented. It proves that local real Gemma3
artifacts can execute without importing AIR or claiming NPU paper parity.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from gemma3_artifacts import MODEL_SPECS, Gemma3ArtifactError, load_real_model_artifacts


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


def _trim_inputs(inputs: Any, max_prompt_tokens: int) -> Any:
    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    input_ids = inputs["input_ids"]
    if input_ids.shape[-1] > max_prompt_tokens:
        inputs["input_ids"] = input_ids[..., :max_prompt_tokens]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][..., :max_prompt_tokens]
    return inputs


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
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = _trim_inputs(inputs, max_prompt_tokens)
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
            inputs = processor(text=prompt, return_tensors="pt")
            inputs = _trim_inputs(inputs, max_prompt_tokens)
    model.eval()
    load_seconds = perf_counter() - load_start

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
    print(report.format())
    print("GEMMA3_REAL_EXECUTION_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 real execution smoke helper")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default="gemma3-1b")
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--prompt", default="Gemma3 paper reproduction smoke.")
    parser.add_argument("--max-prompt-tokens", type=int, default=16)
    parser.add_argument("--vision-smoke", action="store_true")
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
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(
                json.dumps(report.to_json_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"GEMMA3_REAL_EXECUTION_JSON: {args.json}")
        return 0
    parser.error("one of --self-test or --cpu-smoke is required")


if __name__ == "__main__":
    raise SystemExit(main())
