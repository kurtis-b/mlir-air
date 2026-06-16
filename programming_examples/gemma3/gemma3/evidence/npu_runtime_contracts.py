#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 NPU runtime evidence contract validator.

This validator is intentionally stricter than the diagnostic recognizers. It
only accepts production-owned runtime evidence for the Gemma3 1B/1k NPU loop and
keeps HF, synthetic, repeated-current-token, and host-replaced K/V paths blocked.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


CONTRACT_VERSION = "gemma3-npu-runtime-contract-v1"
PREFILL_READY = "PREFILL_KV_CACHE_READY"
PREFILL_PRODUCTION_SOURCE = "production-npu-prefill-kv-cache"
PREFILL_ARTIFACT_BLOCKER = "production-prefill-runtime-artifacts-not-cached"
PREFILL_1K_BLOCKER = "prefill-1k-npu-not-wired"
PREFILL_PRODUCED_BLOCKER = "prefill-produced-kv-cache-not-wired"
NPU_PREFILL_BLOCKER = "npu-prefill-kv-cache-not-wired"
GENERATE_PREFILL_BLOCKER = "generate-prefill-kv-cache-blocked"
ATTENTION_REDUCTION_BLOCKER = "npu-attention-reduction-not-wired"
STATIC_BO_BLOCKER = "production-contiguous-static-weight-bo-not-used-by-fused-dqp-route"
LOGITS_BLOCKER = "logits-sampling-not-wired"
LOGITS_HOST_DIAGNOSTIC_BLOCKER = "logits-sampling-host-diagnostic-only"
MANIFEST_FILE = "gemma3_npu_kernel_manifest.json"
PREFILL_ARTIFACT_TEMPLATE = "gemma3_prefill_kv_L{layer_index}"
EXPECTED_LAYERS = {"gemma3-1b": 26}
MIN_KV_REFERENCE_CORRELATION = 0.99
DIAGNOSTIC_KV_CONTRACTS = {
    "host-hf-prefill-kv-cache",
    "synthetic-prefill-kv-cache",
    "single-current-token-kv",
    "repeated-current-token",
}


@dataclass(frozen=True)
class ContractResult:
    name: str
    blockers: tuple[str, ...]

    @property
    def status(self) -> str:
        return "BLOCKED" if self.blockers else "PASS"

    def format(self) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        return f"contract {self.name} status={self.status} blockers={blockers}"


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in items if item))


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _load_json_or_blocker(path: Path, blocker: str) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    try:
        return _read_json(path), ()
    except Exception as exc:
        return None, (blocker, f"json-load-failed:{path.name}:{type(exc).__name__}")


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return ()


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _float_value(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _blockers(data: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(item) for item in _as_sequence(data.get("blockers")))


def _cache_payload(data: Mapping[str, Any]) -> Mapping[str, Any]:
    cache = data.get("prefill_kv_cache")
    return cache if isinstance(cache, Mapping) else data


def _layer_payloads(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(layer for layer in _as_sequence(payload.get("layers")) if isinstance(layer, Mapping))


def _expected_layers(model_variant: str) -> int:
    return EXPECTED_LAYERS.get(model_variant, 0)


def _launch_count(data: Mapping[str, Any], payload: Mapping[str, Any]) -> int:
    count = _int_value(data.get("prefill_kernel_launch_count"))
    if count <= 0:
        count = _int_value(payload.get("kernel_launch_count"))
    if count <= 0:
        count = sum(_int_value(layer.get("kernel_launch_count")) for layer in _layer_payloads(payload))
    if count <= 0:
        count = _int_value(data.get("kernel_launch_count"))
    return count


def _kv_reference_correlation(layer: Mapping[str, Any]) -> float | None:
    for key in ("kv_reference_correlation", "reference_correlation"):
        value = _float_value(layer.get(key))
        if value is not None:
            return value
    key_corr = _float_value(layer.get("key_reference_correlation"))
    value_corr = _float_value(layer.get("value_reference_correlation"))
    if key_corr is not None and value_corr is not None:
        return min(key_corr, value_corr)
    return None


def _prefill_cache_contract_blockers(
    data: Mapping[str, Any],
    *,
    model_variant: str,
    prompt_len: int,
    decode_context: int,
    require_reference_correlation: bool,
) -> tuple[str, ...]:
    payload = _cache_payload(data)
    expected_layers = _expected_layers(model_variant)
    blockers: list[str] = []
    if data.get("model_variant") not in (None, model_variant):
        blockers.append("prefill-evidence-model-mismatch")
    if payload.get("model_variant") not in (None, model_variant):
        blockers.append("prefill-cache-model-mismatch")
    if _int_value(data.get("prompt_len", payload.get("prompt_token_count"))) != prompt_len:
        blockers.append("prefill-evidence-prompt-len-mismatch")
    if _int_value(payload.get("prompt_token_count", data.get("prompt_len"))) != prompt_len:
        blockers.append("prefill-cache-prompt-len-mismatch")
    if _int_value(data.get("decode_context", payload.get("decode_context"))) != decode_context:
        blockers.append("prefill-cache-decode-context-mismatch")
    if expected_layers and _int_value(payload.get("layer_count")) != expected_layers:
        blockers.append("prefill-cache-layer-count-mismatch")
    if payload.get("status") != PREFILL_READY:
        blockers.append(NPU_PREFILL_BLOCKER)
    if data.get("entrypoint") == "run_npu_prefill" and data.get("status") != PREFILL_READY:
        blockers.append(NPU_PREFILL_BLOCKER)
    if payload.get("source") != PREFILL_PRODUCTION_SOURCE:
        blockers.append(PREFILL_PRODUCED_BLOCKER)
    if payload.get("owner") != "npu":
        blockers.append(NPU_PREFILL_BLOCKER)
    if _blockers(payload):
        blockers.extend(_blockers(payload))
    launch_count = _launch_count(data, payload)
    if launch_count <= 0:
        blockers.append(PREFILL_1K_BLOCKER)

    layers = _layer_payloads(payload)
    if expected_layers and len(layers) != expected_layers:
        blockers.append("prefill-cache-layer-records-missing")
    layer_by_index = {
        _int_value(layer.get("layer_index"), -1): layer
        for layer in layers
        if "layer_index" in layer
    }
    for layer_index in range(expected_layers):
        layer = layer_by_index.get(layer_index)
        if layer is None:
            blockers.append(f"prefill-cache-layer-missing:L{layer_index}")
            continue
        if layer.get("status") != PREFILL_READY:
            blockers.append(NPU_PREFILL_BLOCKER)
        if layer.get("source") != PREFILL_PRODUCTION_SOURCE:
            blockers.append(PREFILL_PRODUCED_BLOCKER)
        if layer.get("owner") != "npu":
            blockers.append(NPU_PREFILL_BLOCKER)
        if _blockers(layer):
            blockers.extend(_blockers(layer))
        if _int_value(layer.get("prompt_token_count")) != prompt_len:
            blockers.append("prefill-cache-layer-prompt-len-mismatch")
        if require_reference_correlation:
            correlation = _kv_reference_correlation(layer)
            if correlation is None:
                blockers.append("prefill-kv-reference-correlation-missing")
            elif correlation < MIN_KV_REFERENCE_CORRELATION:
                blockers.append("prefill-kv-reference-correlation-below-threshold")
    return _dedupe(blockers)


def _manifest_artifacts(runtime_cache_dir: Path) -> tuple[Mapping[str, Any], ...] | None:
    manifest = runtime_cache_dir / MANIFEST_FILE
    if not manifest.exists():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"runtime cache manifest must be a list: {manifest}")
    return tuple(item for item in data if isinstance(item, Mapping))


def _path_exists(path_value: Any, manifest_dir: Path) -> bool:
    if not path_value:
        return False
    path = Path(str(path_value))
    if not path.is_absolute():
        path = manifest_dir / path
    return path.exists()


def _contract_prefill_artifacts(
    *,
    runtime_cache_dir: Path,
    prefill_result: Mapping[str, Any],
    model_variant: str,
) -> ContractResult:
    blockers: list[str] = []
    manifest_dir = runtime_cache_dir
    try:
        artifacts = _manifest_artifacts(runtime_cache_dir)
    except Exception:
        artifacts = None
        blockers.append(PREFILL_ARTIFACT_BLOCKER)
    if artifacts is None:
        blockers.append(PREFILL_ARTIFACT_BLOCKER)
    else:
        expected = {
            PREFILL_ARTIFACT_TEMPLATE.format(layer_index=layer_index)
            for layer_index in range(_expected_layers(model_variant))
        }
        by_name = {str(item.get("name")): item for item in artifacts}
        missing = expected - set(by_name)
        if missing:
            blockers.append(PREFILL_ARTIFACT_BLOCKER)
        else:
            for name in sorted(expected):
                item = by_name[name]
                if not _path_exists(item.get("output_binary"), manifest_dir):
                    blockers.append("prefill-runtime-artifact-binary-missing")
                insts = item.get("insts")
                if insts is not None and not _path_exists(insts, manifest_dir):
                    blockers.append("prefill-runtime-artifact-insts-missing")
    if PREFILL_ARTIFACT_BLOCKER in _blockers(prefill_result):
        blockers.append(PREFILL_ARTIFACT_BLOCKER)
    return ContractResult("production_prefill_artifacts", _dedupe(blockers))


def _contract_production_kv(
    *,
    prefill_result: Mapping[str, Any],
    model_variant: str,
    prompt_len: int,
    decode_context: int,
) -> ContractResult:
    blockers = _prefill_cache_contract_blockers(
        prefill_result,
        model_variant=model_variant,
        prompt_len=prompt_len,
        decode_context=decode_context,
        require_reference_correlation=True,
    )
    return ContractResult("production_kv", blockers)


def _operation_record(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for item in _as_sequence(data.get("operation_ownership")):
        if isinstance(item, Mapping) and item.get("name") == name:
            return item
    return {}


def _cache_signature(data: Mapping[str, Any]) -> tuple[Any, ...]:
    payload = _cache_payload(data)
    layers = _layer_payloads(payload)
    return (
        payload.get("model_variant"),
        payload.get("status"),
        payload.get("source"),
        payload.get("owner"),
        _int_value(payload.get("prompt_token_count")),
        _int_value(payload.get("decode_context")),
        _int_value(payload.get("layer_count")),
        tuple(
            (
                _int_value(layer.get("layer_index"), -1),
                layer.get("key_buffer"),
                layer.get("value_buffer"),
                tuple(_as_sequence(layer.get("key_shape"))),
                tuple(_as_sequence(layer.get("value_shape"))),
            )
            for layer in sorted(layers, key=lambda item: _int_value(item.get("layer_index"), -1))
        ),
    )


def _contract_decode_handoff(
    *,
    prefill_result: Mapping[str, Any],
    decode_result: Mapping[str, Any],
    model_variant: str,
    prompt_len: int,
    decode_context: int,
) -> ContractResult:
    blockers: list[str] = []
    blockers.extend(
        _prefill_cache_contract_blockers(
            decode_result,
            model_variant=model_variant,
            prompt_len=prompt_len,
            decode_context=decode_context,
            require_reference_correlation=False,
        )
    )
    for blocker in (GENERATE_PREFILL_BLOCKER, PREFILL_PRODUCED_BLOCKER, NPU_PREFILL_BLOCKER):
        if blocker in _blockers(decode_result):
            blockers.append(blocker)
    record = _operation_record(decode_result, "decode_prefill_kv_cache")
    if record.get("owner") != "npu":
        blockers.append(GENERATE_PREFILL_BLOCKER)
    status = str(record.get("status", ""))
    if any(label in status for label in ("diagnostic", "synthetic", "host", "single-token", "repeated-current-token")):
        blockers.append("decode-handoff-diagnostic-kv-cache")
    loop = _as_mapping(decode_result.get("npu_decode_loop"))
    cache_contract = str(loop.get("attention_cache_contract", ""))
    cache_mode = str(loop.get("attention_cache_mode", ""))
    if cache_contract in DIAGNOSTIC_KV_CONTRACTS or cache_mode in DIAGNOSTIC_KV_CONTRACTS:
        blockers.append("decode-handoff-diagnostic-kv-cache")
    if _cache_signature(prefill_result) != _cache_signature(decode_result):
        blockers.append("decode-prefill-kv-cache-descriptor-mismatch")
    return ContractResult("decode_handoff", _dedupe(blockers))


def _contract_attention_reduction(decode_result: Mapping[str, Any]) -> ContractResult:
    blockers: list[str] = []
    mode = decode_result.get("attention_reduction_mode")
    if mode not in ("npu", "not-required"):
        blockers.append(ATTENTION_REDUCTION_BLOCKER)
    if ATTENTION_REDUCTION_BLOCKER in _blockers(decode_result):
        blockers.append(ATTENTION_REDUCTION_BLOCKER)
    return ContractResult("attention_reduction", _dedupe(blockers))


def _contract_static_bo_route(decode_result: Mapping[str, Any]) -> ContractResult:
    blockers: list[str] = []
    loop = _as_mapping(decode_result.get("npu_decode_loop"))
    mode = decode_result.get("static_projection_argument_mode", loop.get("static_projection_argument_mode"))
    bo_sets = _int_value(decode_result.get("static_projection_bo_set_count", loop.get("static_projection_bo_set_count")))
    if mode != "manifest-contiguous-static-bo" or bo_sets <= 0:
        blockers.append(STATIC_BO_BLOCKER)
    loop_gaps = tuple(str(item) for item in _as_sequence(loop.get("remaining_paper_gaps")))
    if STATIC_BO_BLOCKER in _blockers(decode_result) or STATIC_BO_BLOCKER in loop_gaps:
        blockers.append(STATIC_BO_BLOCKER)
    return ContractResult("static_bo_route", _dedupe(blockers))


def _contract_logits_sampling(decode_result: Mapping[str, Any]) -> ContractResult:
    blockers: list[str] = []
    mode = decode_result.get("logits_sampling_mode")
    policy = decode_result.get("sampling_policy")
    record = _operation_record(decode_result, "logits_sampling")
    owner = record.get("owner")
    timed = bool(record.get("timed_window", False))
    status = str(record.get("status", ""))
    if not mode or mode in ("none", "not-wired", "missing", "host-diagnostic", "not-applicable"):
        blockers.append(LOGITS_BLOCKER if mode != "host-diagnostic" else LOGITS_HOST_DIAGNOSTIC_BLOCKER)
    elif not policy:
        blockers.append("sampling-policy-missing")
    elif mode == "npu":
        if owner != "npu":
            blockers.append(LOGITS_BLOCKER)
    elif str(mode).startswith("host"):
        accounted = timed or any(label in status for label in ("timed", "accounted", "measured"))
        if owner != "host-fallback" or not accounted:
            blockers.append(LOGITS_HOST_DIAGNOSTIC_BLOCKER)
    else:
        blockers.append(LOGITS_BLOCKER)
    for blocker in (LOGITS_BLOCKER, LOGITS_HOST_DIAGNOSTIC_BLOCKER):
        if blocker in _blockers(decode_result):
            blockers.append(blocker)
    return ContractResult("logits_sampling", _dedupe(blockers))


def _host_fallbacks_accounted(data: Mapping[str, Any]) -> bool:
    fallback_count = _int_value(data.get("host_fallback_count"))
    records = tuple(item for item in _as_sequence(data.get("host_fallbacks")) if isinstance(item, Mapping))
    if fallback_count == 0 and not records:
        return True
    for record in records:
        if not (
            bool(record.get("measured"))
            or bool(record.get("timed_window"))
            or str(record.get("status", "")) in {"measured", "measured-host-fallback", "timed", "accounted"}
        ):
            return False
    return bool(records)


def _contract_paper_cell(
    paper_result: Mapping[str, Any],
    *,
    model_variant: str,
    prompt_len: int,
    decode_context: int,
) -> ContractResult:
    blockers: list[str] = []
    if paper_result.get("model_variant") != model_variant:
        blockers.append("paper-cell-model-mismatch")
    if _int_value(paper_result.get("prompt_len", paper_result.get("sequence_length"))) != prompt_len:
        blockers.append("paper-cell-prompt-len-mismatch")
    if _int_value(paper_result.get("decode_context", prompt_len), prompt_len) != decode_context:
        blockers.append("paper-cell-decode-context-mismatch")
    if _int_value(paper_result.get("warmup_iters")) <= 0:
        blockers.append("paper-cell-warmup-iters-missing")
    if _int_value(paper_result.get("timed_iters")) <= 0:
        blockers.append("paper-cell-timed-iters-missing")
    if not paper_result.get("timed_window_policy") and not paper_result.get("timing_window"):
        blockers.append("paper-cell-timed-window-policy-missing")
    power = _as_mapping(paper_result.get("power_snapshot"))
    if not power or power.get("aligned_with_timed_window") is not True:
        blockers.append("paper-cell-power-window-not-aligned")
    if _int_value(paper_result.get("kernel_launch_count", paper_result.get("prefill_kernel_launch_count"))) <= 0:
        blockers.append("paper-cell-launch-count-missing")
    if not _host_fallbacks_accounted(paper_result):
        blockers.append("paper-cell-host-fallbacks-unaccounted")
    if _blockers(paper_result):
        blockers.extend(_blockers(paper_result))
    if paper_result.get("local_value") is None:
        blockers.append("paper-cell-local-value-missing")
    status = str(paper_result.get("status", ""))
    if "BLOCKED" in status or str(paper_result.get("correctness", "PASS")) != "PASS":
        blockers.append("paper-cell-status-blocked")
    return ContractResult("paper_cell", _dedupe(blockers))


def evaluate_contracts(
    *,
    model_variant: str,
    prompt_len: int,
    decode_context: int,
    runtime_cache_dir: Path,
    prefill_result_path: Path,
    decode_result_path: Path,
    paper_result_path: Path | None = None,
) -> tuple[ContractResult, ...]:
    prefill_result, prefill_load_blockers = _load_json_or_blocker(prefill_result_path, NPU_PREFILL_BLOCKER)
    decode_result, decode_load_blockers = _load_json_or_blocker(decode_result_path, GENERATE_PREFILL_BLOCKER)
    if prefill_result is None:
        prefill_result = {}
    if decode_result is None:
        decode_result = {}

    results = [
        _contract_prefill_artifacts(
            runtime_cache_dir=runtime_cache_dir,
            prefill_result=prefill_result,
            model_variant=model_variant,
        ),
        _contract_production_kv(
            prefill_result=prefill_result,
            model_variant=model_variant,
            prompt_len=prompt_len,
            decode_context=decode_context,
        ),
        _contract_decode_handoff(
            prefill_result=prefill_result,
            decode_result=decode_result,
            model_variant=model_variant,
            prompt_len=prompt_len,
            decode_context=decode_context,
        ),
        _contract_attention_reduction(decode_result),
        _contract_static_bo_route(decode_result),
        _contract_logits_sampling(decode_result),
    ]
    if prefill_load_blockers:
        results[0] = ContractResult(results[0].name, _dedupe((*results[0].blockers, *prefill_load_blockers)))
        results[1] = ContractResult(results[1].name, _dedupe((*results[1].blockers, *prefill_load_blockers)))
    if decode_load_blockers:
        for index in range(2, len(results)):
            results[index] = ContractResult(
                results[index].name,
                _dedupe((*results[index].blockers, *decode_load_blockers)),
            )
    if paper_result_path is not None:
        paper_result, paper_load_blockers = _load_json_or_blocker(paper_result_path, "paper-cell-json-missing")
        if paper_result is None:
            paper_result = {}
        paper_contract = _contract_paper_cell(
            paper_result,
            model_variant=model_variant,
            prompt_len=prompt_len,
            decode_context=decode_context,
        )
        if paper_load_blockers:
            paper_contract = ContractResult(
                paper_contract.name,
                _dedupe((*paper_contract.blockers, *paper_load_blockers)),
            )
        results.append(paper_contract)
    return tuple(results)


def _write_manifest(cache_dir: Path, *, complete: bool) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    layer_count = 26 if complete else 1
    for layer_index in range(layer_count):
        name = PREFILL_ARTIFACT_TEMPLATE.format(layer_index=layer_index)
        binary = cache_dir / f"{name}.elf"
        insts = cache_dir / f"{name}.insts.bin"
        binary.write_bytes(b"elf")
        insts.write_bytes(b"insts")
        artifacts.append(
            {
                "name": name,
                "output_binary": str(binary),
                "kernel": name,
                "insts": str(insts),
            }
        )
    (cache_dir / MANIFEST_FILE).write_text(json.dumps(artifacts), encoding="utf-8")


def _prefill_fixture(*, ready: bool) -> dict[str, Any]:
    layers = []
    for layer_index in range(26):
        layers.append(
            {
                "layer_index": layer_index,
                "attention_kind": "global_full" if layer_index % 6 == 5 else "local_swa",
                "key_buffer": f"kv_cache_k_L{layer_index}",
                "value_buffer": f"kv_cache_v_L{layer_index}",
                "key_shape": [1024, 1, 256],
                "value_shape": [1024, 1, 256],
                "prompt_token_count": 1024,
                "retained_token_count": 1024,
                "read_window_token_count": 1024,
                "status": PREFILL_READY if ready else "PREFILL_KV_CACHE_NOT_PRODUCED",
                "source": PREFILL_PRODUCTION_SOURCE if ready else "none",
                "owner": "npu" if ready else "missing",
                "blockers": [] if ready else [PREFILL_ARTIFACT_BLOCKER, NPU_PREFILL_BLOCKER],
                "kernel_launch_count": 1 if ready else 0,
                "kv_reference_correlation": 0.999 if ready else None,
            }
        )
    return {
        "runtime_contract_version": CONTRACT_VERSION,
        "entrypoint": "run_npu_prefill",
        "status": PREFILL_READY if ready else "PREFILL_KV_CACHE_BLOCKED",
        "model_variant": "gemma3-1b",
        "prompt_len": 1024,
        "decode_context": 1024,
        "prefill_kernel_launch_count": 26 if ready else 0,
        "prefill_host_fallback_count": 0,
        "blockers": [] if ready else [PREFILL_ARTIFACT_BLOCKER, PREFILL_1K_BLOCKER, PREFILL_PRODUCED_BLOCKER, NPU_PREFILL_BLOCKER],
        "prefill_kv_cache": {
            "model_variant": "gemma3-1b",
            "status": PREFILL_READY if ready else "PREFILL_KV_CACHE_NOT_PRODUCED",
            "source": PREFILL_PRODUCTION_SOURCE if ready else "none",
            "owner": "npu" if ready else "missing",
            "layer_count": 26,
            "prompt_token_count": 1024,
            "decode_context": 1024,
            "blockers": [] if ready else [PREFILL_ARTIFACT_BLOCKER, NPU_PREFILL_BLOCKER],
            "layers": layers,
        },
    }


def _decode_fixture(prefill: Mapping[str, Any], *, ready: bool) -> dict[str, Any]:
    result = dict(prefill)
    result.update(
        {
            "runtime_contract_version": CONTRACT_VERSION,
            "entrypoint": "generate",
            "status": "DECODE_RUNTIME_PASS" if ready else "BLOCKED",
            "decode_tokens": 1,
            "kernel_launch_count": 182 if ready else 0,
            "host_fallback_count": 0,
            "blockers": [] if ready else [GENERATE_PREFILL_BLOCKER, PREFILL_ARTIFACT_BLOCKER, NPU_PREFILL_BLOCKER, ATTENTION_REDUCTION_BLOCKER, STATIC_BO_BLOCKER, LOGITS_BLOCKER],
            "attention_reduction_mode": "npu" if ready else None,
            "logits_sampling_mode": "npu" if ready else "not-wired",
            "sampling_policy": "argmax" if ready else None,
            "operation_ownership": [
                {
                    "name": "decode_prefill_kv_cache",
                    "phase": "decode",
                    "owner": "npu" if ready else "missing",
                    "timed_window": False,
                    "status": "production-npu-prefill-kv-cache" if ready else "requires-production-npu-prefill-kv-cache",
                    "blockers": [] if ready else [NPU_PREFILL_BLOCKER],
                },
                {
                    "name": "logits_sampling",
                    "phase": "decode",
                    "owner": "npu" if ready else "missing",
                    "timed_window": True,
                    "status": "sampled" if ready else "not-wired",
                    "blockers": [] if ready else [LOGITS_BLOCKER],
                },
            ],
            "npu_decode_loop": {
                "status": "DECODE_LOOP_PASS" if ready else None,
                "attention_cache_contract": "production-npu-prefill-kv-cache" if ready else "single-current-token-kv",
                "static_projection_argument_mode": "manifest-contiguous-static-bo" if ready else "diagnostic",
                "static_projection_bo_set_count": 182 if ready else 0,
                "remaining_paper_gaps": [] if ready else [STATIC_BO_BLOCKER, LOGITS_BLOCKER],
            },
        }
    )
    return result


def _paper_fixture(*, ready: bool) -> dict[str, Any]:
    return {
        "model_variant": "gemma3-1b",
        "prompt_len": 1024,
        "decode_context": 1024,
        "decode_tokens": 1,
        "warmup_iters": 1 if ready else 0,
        "timed_iters": 3 if ready else 0,
        "timed_window_policy": "compile-load-bo-preload-prefill-excluded;decode-token-loop-timed" if ready else None,
        "power_snapshot": {"aligned_with_timed_window": ready},
        "kernel_launch_count": 182 if ready else 0,
        "host_fallback_count": 0,
        "blockers": [] if ready else [LOGITS_BLOCKER],
        "status": "PASS" if ready else "BLOCKED",
        "correctness": "PASS" if ready else "BLOCKED",
        "local_value": 41.1 if ready else None,
    }


def _self_test() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        pass_cache = root / "pass-cache"
        blocked_cache = root / "blocked-cache"
        _write_manifest(pass_cache, complete=True)
        _write_manifest(blocked_cache, complete=False)

        pass_prefill = root / "pass-prefill.json"
        pass_decode = root / "pass-decode.json"
        pass_paper = root / "pass-paper.json"
        blocked_prefill = root / "blocked-prefill.json"
        blocked_decode = root / "blocked-decode.json"
        blocked_paper = root / "blocked-paper.json"

        prefill_pass_data = _prefill_fixture(ready=True)
        prefill_blocked_data = _prefill_fixture(ready=False)
        pass_prefill.write_text(json.dumps(prefill_pass_data), encoding="utf-8")
        pass_decode.write_text(json.dumps(_decode_fixture(prefill_pass_data, ready=True)), encoding="utf-8")
        pass_paper.write_text(json.dumps(_paper_fixture(ready=True)), encoding="utf-8")
        blocked_prefill.write_text(json.dumps(prefill_blocked_data), encoding="utf-8")
        blocked_decode.write_text(json.dumps(_decode_fixture(prefill_blocked_data, ready=False)), encoding="utf-8")
        blocked_paper.write_text(json.dumps(_paper_fixture(ready=False)), encoding="utf-8")

        pass_results = evaluate_contracts(
            model_variant="gemma3-1b",
            prompt_len=1024,
            decode_context=1024,
            runtime_cache_dir=pass_cache,
            prefill_result_path=pass_prefill,
            decode_result_path=pass_decode,
            paper_result_path=pass_paper,
        )
        blocked_results = evaluate_contracts(
            model_variant="gemma3-1b",
            prompt_len=1024,
            decode_context=1024,
            runtime_cache_dir=blocked_cache,
            prefill_result_path=blocked_prefill,
            decode_result_path=blocked_decode,
            paper_result_path=blocked_paper,
        )
        if any(result.blockers for result in pass_results):
            raise AssertionError(pass_results)
        if not any(result.blockers for result in blocked_results):
            raise AssertionError(blocked_results)
        for result in pass_results:
            print(result.format())
        for result in blocked_results:
            print(result.format())
    print("GEMMA3_NPU_RUNTIME_CONTRACTS_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Gemma3 NPU runtime evidence contracts")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-variant", default="gemma3-1b")
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--decode-context", type=int, default=1024)
    parser.add_argument("--runtime-cache-dir", type=Path)
    parser.add_argument("--prefill-result", type=Path)
    parser.add_argument("--decode-result", type=Path)
    parser.add_argument("--paper-result", type=Path)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0

    missing = [
        name
        for name, value in (
            ("--runtime-cache-dir", args.runtime_cache_dir),
            ("--prefill-result", args.prefill_result),
            ("--decode-result", args.decode_result),
        )
        if value is None
    ]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))

    results = evaluate_contracts(
        model_variant=args.model_variant,
        prompt_len=args.prompt_len,
        decode_context=args.decode_context,
        runtime_cache_dir=args.runtime_cache_dir,
        prefill_result_path=args.prefill_result,
        decode_result_path=args.decode_result,
        paper_result_path=args.paper_result,
    )
    for result in results:
        print(result.format())
    if any(result.blockers for result in results) and not args.allow_blocked:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
