#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 model-loop buffer object planning.

The plan is shape-only: it estimates the L3/XRT BO contracts needed by a future
model runner without allocating BOs, compiling kernels, or claiming execution.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from gemma3_artifacts import MODEL_SPECS, model_spec
from gemma3_npu_preflight import Gemma3NPUPreflightPlan, build_preflight_plan
from gemma3_norm_weight_plan import Gemma3NormWeightPlan, build_norm_weight_plan
from gemma3_weight_plan import Gemma3StaticWeightPlan, build_weight_plan


BF16_BYTES = 2
KV_STRATEGIES = ("benchmark-cell", "monolithic")


@dataclass(frozen=True)
class Gemma3BORecord:
    name: str
    scope: str
    dtype: str
    shape: tuple[int, ...]
    bytes: int
    static: bool
    notes: str

    def format(self) -> str:
        shape = "x".join(str(dim) for dim in self.shape) if self.shape else "scalar"
        static = "static" if self.static else "dynamic"
        return (
            f"bo name={self.name} scope={self.scope} dtype={self.dtype} "
            f"shape={shape} bytes={self.bytes} class={static} notes={self.notes}"
        )


@dataclass(frozen=True)
class Gemma3BOPlan:
    model_variant: str
    status: str
    layers: int
    prompt_len: int
    decode_context: int
    kv_strategy: str
    kv_record_count: int
    max_bo_bytes: int
    total_bytes: int
    dynamic_bytes: int
    static_bytes: int
    records: tuple[Gemma3BORecord, ...]
    blockers: tuple[str, ...]

    def format(self, *, include_records: bool = False) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        lines = [
            f"bo_plan model={self.model_variant} status={self.status} "
            f"layers={self.layers} prompt_len={self.prompt_len} "
            f"decode_context={self.decode_context} kv_strategy={self.kv_strategy} "
            f"kv_records={self.kv_record_count} max_bo_bytes={self.max_bo_bytes} "
            f"total_bytes={self.total_bytes} "
            f"dynamic_bytes={self.dynamic_bytes} static_bytes={self.static_bytes} "
            f"blockers={blockers}"
        ]
        if include_records:
            lines.extend(record.format() for record in self.records)
        return "\n".join(lines)

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["records"] = [asdict(record) for record in self.records]
        return data


def _bf16_bytes(shape: tuple[int, ...]) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count * BF16_BYTES


def _record(name: str, scope: str, shape: tuple[int, ...], *, static: bool = False, notes: str = "") -> Gemma3BORecord:
    return Gemma3BORecord(
        name=name,
        scope=scope,
        dtype="bf16",
        shape=shape,
        bytes=_bf16_bytes(shape),
        static=static,
        notes=notes,
    )


def _attention_kind(layer_index: int) -> str:
    return "global_full" if layer_index % 6 == 5 else "local_swa"


def _local_cache_tokens(decode_context: int, sliding_window: int | None) -> int:
    if sliding_window is None or sliding_window <= 0:
        return decode_context
    return min(decode_context, int(sliding_window))


def _kv_cache_records(
    *,
    layers: int,
    decode_context: int,
    kv_heads: int,
    head_dim: int,
    sliding_window: int | None,
    kv_strategy: str,
) -> list[Gemma3BORecord]:
    if kv_strategy not in KV_STRATEGIES:
        choices = ", ".join(KV_STRATEGIES)
        raise ValueError(f"kv_strategy must be one of: {choices}")
    if kv_strategy == "monolithic":
        return [
            _record("kv_cache_k", "model", (layers, decode_context, kv_heads, head_dim), notes="all-layer K cache"),
            _record("kv_cache_v", "model", (layers, decode_context, kv_heads, head_dim), notes="all-layer V cache"),
        ]

    records: list[Gemma3BORecord] = []
    for layer_index in range(layers):
        attention_kind = _attention_kind(layer_index)
        cache_tokens = decode_context if attention_kind == "global_full" else _local_cache_tokens(decode_context, sliding_window)
        if attention_kind == "global_full":
            notes = "global attention layer K/V cache; full decode context"
        else:
            notes = f"local SWA layer K/V cache; window={cache_tokens}"
        records.extend(
            [
                _record(f"kv_cache_k_L{layer_index}", "layer", (cache_tokens, kv_heads, head_dim), notes=f"{notes}; K"),
                _record(f"kv_cache_v_L{layer_index}", "layer", (cache_tokens, kv_heads, head_dim), notes=f"{notes}; V"),
            ]
        )
    return records


def build_bo_plan_from_preflight(
    preflight: Gemma3NPUPreflightPlan,
    weight_plan: Gemma3StaticWeightPlan,
    norm_weight_plan: Gemma3NormWeightPlan | None = None,
    *,
    prompt_len: int,
    decode_context: int,
    kv_strategy: str = "benchmark-cell",
) -> Gemma3BOPlan:
    if prompt_len <= 0 or decode_context <= 0:
        raise ValueError("prompt_len and decode_context must be positive")
    if preflight.layers is None or preflight.hidden_size is None:
        raise ValueError("preflight plan must include layers and hidden size")
    if preflight.intermediate_size is None or preflight.head_dim is None:
        raise ValueError("preflight plan must include intermediate size and head dim")
    if preflight.num_attention_heads is None or preflight.num_key_value_heads is None:
        raise ValueError("preflight plan must include attention head counts")

    layers = int(preflight.layers)
    hidden = int(preflight.hidden_size)
    intermediate = int(preflight.intermediate_size)
    heads = int(preflight.num_attention_heads)
    kv_heads = int(preflight.num_key_value_heads)
    head_dim = int(preflight.head_dim)
    kv_records = _kv_cache_records(
        layers=layers,
        decode_context=decode_context,
        kv_heads=kv_heads,
        head_dim=head_dim,
        sliding_window=preflight.sliding_window,
        kv_strategy=kv_strategy,
    )

    records = [
        Gemma3BORecord(
            name="static_projection_weights",
            scope="model",
            dtype="q4nx",
            shape=(weight_plan.static_bo_bytes,),
            bytes=weight_plan.static_bo_bytes,
            static=True,
            notes="Q4NX packed weights plus scale/min metadata",
        ),
    ]
    if norm_weight_plan is not None and norm_weight_plan.static_bo_bytes > 0:
        records.append(
            _record(
                "static_norm_weights",
                "model",
                (norm_weight_plan.static_bo_bytes // BF16_BYTES,),
                static=True,
                notes="BF16 RMSNorm/QK-Norm vector weights",
            )
        )
    records.extend([
        _record("token_embeddings", "model", (prompt_len, hidden), notes="host-prepared embedding input"),
        _record("layer_input", "layer", (prompt_len, hidden), notes="ping buffer"),
        _record("layer_output", "layer", (prompt_len, hidden), notes="pong buffer"),
        _record("decode_token_state", "decode", (1, hidden), notes="single-token decode state"),
        _record("prefill_q", "layer", (prompt_len, heads, head_dim), notes="attention query"),
        _record("prefill_k", "layer", (prompt_len, kv_heads, head_dim), notes="attention key append"),
        _record("prefill_v", "layer", (prompt_len, kv_heads, head_dim), notes="attention value append"),
        _record("prefill_attention_out", "layer", (prompt_len, hidden), notes="FlowQKV output"),
        _record("mlp_gate", "layer", (prompt_len, intermediate), notes="MLP gate projection"),
        _record("mlp_up", "layer", (prompt_len, intermediate), notes="MLP up projection"),
        _record("mlp_down", "layer", (prompt_len, hidden), notes="MLP down projection output"),
        _record("decode_q", "decode", (1, heads, head_dim), notes="decode query"),
        _record("decode_k", "decode", (1, kv_heads, head_dim), notes="decode key append"),
        _record("decode_v", "decode", (1, kv_heads, head_dim), notes="decode value append"),
        _record("decode_attention_out", "decode", (1, hidden), notes="FlowKV output"),
    ])
    records.extend(kv_records)
    static_bytes = sum(record.bytes for record in records if record.static)
    dynamic_bytes = sum(record.bytes for record in records if not record.static)
    return Gemma3BOPlan(
        model_variant=preflight.model_variant,
        status="READY_FOR_BO_ALLOCATION",
        layers=layers,
        prompt_len=prompt_len,
        decode_context=decode_context,
        kv_strategy=kv_strategy,
        kv_record_count=len(kv_records),
        max_bo_bytes=max(record.bytes for record in records),
        total_bytes=static_bytes + dynamic_bytes,
        dynamic_bytes=dynamic_bytes,
        static_bytes=static_bytes,
        records=tuple(records),
        blockers=("xrt-bo-allocation-not-wired",),
    )


def build_bo_plan(
    model_variant: str,
    *,
    weights_dir: Path | None = None,
    prompt_len: int | None = None,
    decode_context: int | None = None,
    kv_strategy: str = "benchmark-cell",
) -> Gemma3BOPlan:
    spec = model_spec(model_variant)
    prompt_len = prompt_len or spec.prefill_lengths[0]
    decode_context = decode_context or spec.max_decode_context
    preflight = build_preflight_plan(model_variant, weights_dir=weights_dir)
    weight_plan = build_weight_plan(model_variant, weights_dir=weights_dir)
    norm_weight_plan = build_norm_weight_plan(model_variant, weights_dir=weights_dir)
    return build_bo_plan_from_preflight(
        preflight,
        weight_plan,
        norm_weight_plan,
        prompt_len=prompt_len,
        decode_context=decode_context,
        kv_strategy=kv_strategy,
    )


def _self_test() -> None:
    from gemma3_npu_preflight import ProjectionPlan

    preflight = Gemma3NPUPreflightPlan(
        model_variant="gemma3-1b",
        status="READY_FOR_NPU_WIRING",
        blocker="npu-model-execution-not-implemented",
        layers=2,
        hidden_size=1152,
        intermediate_size=6912,
        head_dim=256,
        num_attention_heads=4,
        num_key_value_heads=1,
        sliding_window=512,
        attention_pattern="5-local-1-global",
        projections=(ProjectionPlan("q_proj", (1024, 1152), (1024, 1280), 32, 5, True, 0.1, 0.01),),
    )
    weight_plan = Gemma3StaticWeightPlan(
        model_variant="gemma3-1b",
        status="READY_FOR_STATIC_BO_PRELOAD",
        layers=2,
        tensor_count=14,
        static_bo_bytes=36003840,
        families=(),
        records=(),
        blockers=(),
    )
    norm_weight_plan = Gemma3NormWeightPlan(
        model_variant="gemma3-1b",
        status="READY_FOR_NORM_WEIGHT_PRELOAD",
        layers=2,
        tensor_count=12,
        static_bo_bytes=20480,
        families=(),
        records=(),
        blockers=(),
    )
    plan = build_bo_plan_from_preflight(preflight, weight_plan, norm_weight_plan, prompt_len=1024, decode_context=2048)
    if plan.status != "READY_FOR_BO_ALLOCATION" or plan.static_bytes != 36024320:
        raise AssertionError(plan)
    if plan.kv_strategy != "benchmark-cell" or plan.kv_record_count != 4:
        raise AssertionError(plan)
    if not any(record.name == "kv_cache_k_L0" and record.shape == (512, 1, 256) for record in plan.records):
        raise AssertionError(plan.records)
    if not any(record.name == "kv_cache_k_L1" and record.shape == (512, 1, 256) for record in plan.records):
        raise AssertionError(plan.records)
    print(plan.format(include_records=True))
    print("GEMMA3_BO_PLAN_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 model-loop BO planning")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default="gemma3-1b")
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--prompt-len", type=int)
    parser.add_argument("--decode-context", type=int)
    parser.add_argument("--kv-strategy", choices=KV_STRATEGIES, default="benchmark-cell")
    parser.add_argument("--include-records", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    plan = build_bo_plan(
        args.model_variant,
        weights_dir=args.weights_dir,
        prompt_len=args.prompt_len,
        decode_context=args.decode_context,
        kv_strategy=args.kv_strategy,
    )
    if args.json:
        print(json.dumps(plan.to_json_dict(), indent=2, sort_keys=True))
    else:
        print(plan.format(include_records=args.include_records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
