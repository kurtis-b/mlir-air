#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 production prefill K/V cache runner contract.

This module owns the Gemma3 prefill-KV runtime boundary. Probe/HF/synthetic
cache paths are intentionally not imported here. The default path is the
Gemma-owned runtime executor; explicit JSON evidence is accepted only when a
caller passes an evidence path for validation or self-test fixtures.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from gemma3.paths import RESULTS_DIR


PREFILL_KV_CACHE_READY_STATUS = "PREFILL_KV_CACHE_READY"
PREFILL_KV_CACHE_PARTIAL_STATUS = "PREFILL_KV_CACHE_PARTIAL"
PREFILL_KV_CACHE_NOT_PRODUCED_STATUS = "PREFILL_KV_CACHE_NOT_PRODUCED"
PREFILL_KV_CACHE_BLOCKED_STATUS = "PREFILL_KV_CACHE_BLOCKED"
PREFILL_KV_CACHE_PRODUCTION_SOURCE = "production-npu-prefill-kv-cache"
PREFILL_KV_CACHE_NOT_PRODUCED_SOURCE = "none"
PREFILL_1K_NPU_BLOCKER = "prefill-1k-npu-not-wired"
PREFILL_PRODUCED_KV_CACHE_BLOCKER = "prefill-produced-kv-cache-not-wired"
NPU_PREFILL_KV_CACHE_BLOCKER = "npu-prefill-kv-cache-not-wired"
PRODUCTION_PREFILL_ARTIFACTS_BLOCKER = "production-prefill-runtime-artifacts-not-cached"
PRODUCTION_PREFILL_ARGUMENTS_BLOCKER = "production-prefill-runtime-arguments-not-bound"
PRODUCTION_PREFILL_EVIDENCE_NAME = "gemma3_1b_production_prefill_kv_cache.json"
RUNTIME_PREFILL_RESULT_NAME = "gemma3_1b_npu_prefill_runtime.json"
PRODUCTION_PREFILL_ARTIFACT_TEMPLATE = "gemma3_prefill_kv_L{layer_index}"


@dataclass(frozen=True)
class Gemma3PrefillKVLayerProduction:
    layer_index: int
    attention_kind: str
    key_buffer: str
    value_buffer: str
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    prompt_token_count: int
    retained_token_count: int
    read_window_token_count: int
    status: str
    source: str
    owner: str
    blockers: tuple[str, ...]
    kernel_launch_count: int
    elapsed_seconds: float | None

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Gemma3ProductionPrefillResult:
    status: str
    source: str
    owner: str
    blockers: tuple[str, ...]
    layers: tuple[Gemma3PrefillKVLayerProduction, ...]
    elapsed_seconds: float | None
    kernel_launch_count: int
    host_fallback_count: int
    host_runtime_count: int
    evidence_path: str | None
    runtime_cache: dict[str, object] | None
    error: str | None = None

    @property
    def ready(self) -> bool:
        return (
            self.status == PREFILL_KV_CACHE_READY_STATUS
            and self.source == PREFILL_KV_CACHE_PRODUCTION_SOURCE
            and self.owner == "npu"
            and not any(layer.blockers for layer in self.layers)
        )

    @property
    def produced_layer_count(self) -> int:
        return sum(1 for layer in self.layers if layer.owner == "npu" and layer.status == PREFILL_KV_CACHE_READY_STATUS)

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["layers"] = [layer.to_json_dict() for layer in self.layers]
        return data


def default_evidence_path(model_variant: str) -> Path:
    if model_variant == "gemma3-1b":
        return RESULTS_DIR / PRODUCTION_PREFILL_EVIDENCE_NAME
    return RESULTS_DIR / f"{model_variant}_production_prefill_kv_cache.json"


def candidate_evidence_paths(model_variant: str, explicit: Path | None = None) -> tuple[Path, ...]:
    paths: list[Path] = []
    if explicit is not None:
        paths.append(explicit)
    paths.append(default_evidence_path(model_variant))
    if model_variant == "gemma3-1b":
        paths.append(RESULTS_DIR / RUNTIME_PREFILL_RESULT_NAME)
    return tuple(dict.fromkeys(paths))


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in items if item))


def _relevant_prefill_blockers(blockers: Iterable[str]) -> tuple[str, ...]:
    selected = [
        blocker
        for blocker in blockers
        if blocker
        in (
            PREFILL_1K_NPU_BLOCKER,
            PREFILL_PRODUCED_KV_CACHE_BLOCKER,
            NPU_PREFILL_KV_CACHE_BLOCKER,
            "prefill-runtime-launch-not-implemented",
            PRODUCTION_PREFILL_ARTIFACTS_BLOCKER,
            PRODUCTION_PREFILL_ARGUMENTS_BLOCKER,
        )
    ]
    return _dedupe(selected or (PREFILL_PRODUCED_KV_CACHE_BLOCKER, NPU_PREFILL_KV_CACHE_BLOCKER))


def _setup_cache_layers(session: Any) -> tuple[Any, ...]:
    cache = getattr(getattr(session, "setup", None), "prefill_kv_cache", None)
    return tuple(getattr(cache, "layers", ()) or ())


def _setup_layer_lookup(session: Any) -> dict[int, Any]:
    return {int(layer.layer_index): layer for layer in _setup_cache_layers(session)}


def _layer_from_setup(
    layer: Any,
    *,
    status: str,
    source: str,
    owner: str,
    blockers: Iterable[str],
    kernel_launch_count: int = 0,
    elapsed_seconds: float | None = None,
) -> Gemma3PrefillKVLayerProduction:
    return Gemma3PrefillKVLayerProduction(
        layer_index=int(layer.layer_index),
        attention_kind=str(layer.attention_kind),
        key_buffer=str(layer.key_buffer),
        value_buffer=str(layer.value_buffer),
        key_shape=tuple(int(dim) for dim in layer.key_shape),
        value_shape=tuple(int(dim) for dim in layer.value_shape),
        prompt_token_count=int(layer.prompt_token_count),
        retained_token_count=int(layer.retained_token_count),
        read_window_token_count=int(layer.read_window_token_count),
        status=status,
        source=source,
        owner=owner,
        blockers=_dedupe(blockers),
        kernel_launch_count=int(kernel_launch_count),
        elapsed_seconds=elapsed_seconds,
    )


def _cache_payload(data: Mapping[str, Any]) -> Mapping[str, Any]:
    cache = data.get("prefill_kv_cache")
    if isinstance(cache, Mapping):
        return cache
    return data


def _layer_payloads(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    layers = payload.get("layers", ())
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
        return ()
    return tuple(layer for layer in layers if isinstance(layer, Mapping))


def _int_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(int(dim) for dim in value)


def _layer_from_evidence(
    *,
    layer_payload: Mapping[str, Any],
    setup_layer: Any,
    prompt_len: int,
) -> Gemma3PrefillKVLayerProduction:
    blockers = tuple(str(item) for item in layer_payload.get("blockers", ()) or ())
    kernel_launch_count = int(layer_payload.get("kernel_launch_count", 0) or 0)
    if kernel_launch_count <= 0:
        kernel_launch_count = int(layer_payload.get("prefill_kernel_launch_count", 0) or 0)
    ready = (
        layer_payload.get("status") == PREFILL_KV_CACHE_READY_STATUS
        and layer_payload.get("source") == PREFILL_KV_CACHE_PRODUCTION_SOURCE
        and layer_payload.get("owner") == "npu"
        and not blockers
        and kernel_launch_count > 0
    )
    if not ready:
        return _layer_from_setup(
            setup_layer,
            status=PREFILL_KV_CACHE_NOT_PRODUCED_STATUS,
            source=PREFILL_KV_CACHE_NOT_PRODUCED_SOURCE,
            owner="missing",
            blockers=_relevant_prefill_blockers(blockers),
        )
    key_shape = _int_tuple(layer_payload.get("key_shape")) or tuple(int(dim) for dim in setup_layer.key_shape)
    value_shape = _int_tuple(layer_payload.get("value_shape")) or tuple(int(dim) for dim in setup_layer.value_shape)
    retained = int(layer_payload.get("retained_token_count", key_shape[0] if key_shape else prompt_len) or 0)
    read_window = int(layer_payload.get("read_window_token_count", retained) or 0)
    return Gemma3PrefillKVLayerProduction(
        layer_index=int(setup_layer.layer_index),
        attention_kind=str(layer_payload.get("attention_kind", setup_layer.attention_kind)),
        key_buffer=str(layer_payload.get("key_buffer", setup_layer.key_buffer)),
        value_buffer=str(layer_payload.get("value_buffer", setup_layer.value_buffer)),
        key_shape=key_shape,
        value_shape=value_shape,
        prompt_token_count=int(layer_payload.get("prompt_token_count", prompt_len) or prompt_len),
        retained_token_count=retained,
        read_window_token_count=read_window,
        status=PREFILL_KV_CACHE_READY_STATUS,
        source=PREFILL_KV_CACHE_PRODUCTION_SOURCE,
        owner="npu",
        blockers=(),
        kernel_launch_count=kernel_launch_count,
        elapsed_seconds=(
            None
            if layer_payload.get("elapsed_seconds") is None
            else float(layer_payload.get("elapsed_seconds"))
        ),
    )


def _result_from_evidence(
    *,
    session: Any,
    data: Mapping[str, Any],
    evidence_path: Path,
    runtime_cache_stats: dict[str, object] | None,
) -> Gemma3ProductionPrefillResult:
    setup = session.setup
    payload = _cache_payload(data)
    setup_layers = _setup_layer_lookup(session)
    layer_count = int(getattr(setup, "layers", 0) or len(setup_layers))
    prompt_len = int(setup.prompt_len)
    decode_context = int(setup.decode_context)
    invalid: list[str] = []
    if payload.get("model_variant") not in (None, setup.model_variant):
        invalid.append("prefill-evidence-model-mismatch")
    if int(payload.get("prompt_token_count", prompt_len) or 0) != prompt_len:
        invalid.append("prefill-evidence-prompt-len-mismatch")
    if int(payload.get("decode_context", decode_context) or 0) != decode_context:
        invalid.append("prefill-evidence-decode-context-mismatch")
    if int(payload.get("layer_count", layer_count) or 0) != layer_count:
        invalid.append("prefill-evidence-layer-count-mismatch")
    if payload.get("source") != PREFILL_KV_CACHE_PRODUCTION_SOURCE:
        invalid.append("prefill-evidence-not-production-source")
    if payload.get("owner") != "npu":
        invalid.append("prefill-evidence-not-npu-owned")

    payload_by_layer = {
        int(layer.get("layer_index", -1)): layer
        for layer in _layer_payloads(payload)
        if "layer_index" in layer
    }
    layers: list[Gemma3PrefillKVLayerProduction] = []
    for layer_index in range(layer_count):
        setup_layer = setup_layers.get(layer_index)
        if setup_layer is None:
            invalid.append(f"prefill-evidence-missing-setup-layer:{layer_index}")
            continue
        layer_payload = payload_by_layer.get(layer_index)
        if layer_payload is None:
            layers.append(
                _layer_from_setup(
                    setup_layer,
                    status=PREFILL_KV_CACHE_NOT_PRODUCED_STATUS,
                    source=PREFILL_KV_CACHE_NOT_PRODUCED_SOURCE,
                    owner="missing",
                    blockers=(NPU_PREFILL_KV_CACHE_BLOCKER,),
                )
            )
            continue
        layers.append(
            _layer_from_evidence(
                layer_payload=layer_payload,
                setup_layer=setup_layer,
                prompt_len=prompt_len,
            )
        )

    produced = sum(1 for layer in layers if layer.status == PREFILL_KV_CACHE_READY_STATUS and layer.owner == "npu")
    kernel_launch_count = int(data.get("prefill_kernel_launch_count", 0) or 0)
    if kernel_launch_count <= 0:
        kernel_launch_count = int(payload.get("kernel_launch_count", 0) or 0)
    if kernel_launch_count <= 0:
        kernel_launch_count = sum(layer.kernel_launch_count for layer in layers)
    if produced and kernel_launch_count <= 0:
        invalid.append("prefill-evidence-missing-launch-accounting")

    if invalid:
        blockers = _relevant_prefill_blockers([*invalid, *getattr(setup, "blockers", ())])
        status = PREFILL_KV_CACHE_BLOCKED_STATUS
        owner = "missing"
        source = PREFILL_KV_CACHE_NOT_PRODUCED_SOURCE
    elif produced == layer_count and layer_count > 0:
        blockers = _dedupe(
            blocker
            for blocker in getattr(setup, "blockers", ())
            if blocker == PREFILL_1K_NPU_BLOCKER
        )
        status = PREFILL_KV_CACHE_READY_STATUS
        owner = "npu"
        source = PREFILL_KV_CACHE_PRODUCTION_SOURCE
    elif produced:
        blockers = _dedupe((PREFILL_1K_NPU_BLOCKER, NPU_PREFILL_KV_CACHE_BLOCKER))
        status = PREFILL_KV_CACHE_PARTIAL_STATUS
        owner = "npu"
        source = PREFILL_KV_CACHE_PRODUCTION_SOURCE
    else:
        blockers = _relevant_prefill_blockers(getattr(setup, "blockers", ()))
        status = PREFILL_KV_CACHE_BLOCKED_STATUS
        owner = "missing"
        source = PREFILL_KV_CACHE_NOT_PRODUCED_SOURCE

    elapsed = data.get("elapsed_seconds", payload.get("elapsed_seconds"))
    return Gemma3ProductionPrefillResult(
        status=status,
        source=source,
        owner=owner,
        blockers=blockers,
        layers=tuple(layers),
        elapsed_seconds=None if elapsed is None else float(elapsed),
        kernel_launch_count=kernel_launch_count if status != PREFILL_KV_CACHE_BLOCKED_STATUS else 0,
        host_fallback_count=int(data.get("prefill_host_fallback_count", 0) or 0),
        host_runtime_count=int(data.get("host_runtime_count", 0) or 0),
        evidence_path=str(evidence_path),
        runtime_cache=runtime_cache_stats,
        error=",".join(invalid) if invalid else None,
    )


def _blocked_result(
    *,
    session: Any,
    blockers: Iterable[str],
    runtime_cache_stats: dict[str, object] | None,
    error: str | None = None,
) -> Gemma3ProductionPrefillResult:
    layers = tuple(
        _layer_from_setup(
            layer,
            status=PREFILL_KV_CACHE_NOT_PRODUCED_STATUS,
            source=PREFILL_KV_CACHE_NOT_PRODUCED_SOURCE,
            owner="missing",
            blockers=_relevant_prefill_blockers(blockers),
        )
        for layer in _setup_cache_layers(session)
    )
    return Gemma3ProductionPrefillResult(
        status=PREFILL_KV_CACHE_BLOCKED_STATUS,
        source=PREFILL_KV_CACHE_NOT_PRODUCED_SOURCE,
        owner="missing",
        blockers=_relevant_prefill_blockers(blockers),
        layers=layers,
        elapsed_seconds=None,
        kernel_launch_count=0,
        host_fallback_count=0,
        host_runtime_count=0,
        evidence_path=None,
        runtime_cache=runtime_cache_stats,
        error=error,
    )


def _runtime_cache_stats(runtime_cache: Any) -> dict[str, object] | None:
    if runtime_cache is None:
        return None
    stats = getattr(runtime_cache, "stats", None)
    if stats is None:
        return None
    return stats().to_json_dict()


def _runtime_cache_artifacts(runtime_cache: Any) -> Mapping[str, Any]:
    artifacts = getattr(runtime_cache, "artifacts", None)
    if not isinstance(artifacts, Mapping):
        return {}
    return artifacts


def _expected_prefill_artifact_names(session: Any) -> tuple[str, ...]:
    return tuple(
        PRODUCTION_PREFILL_ARTIFACT_TEMPLATE.format(layer_index=int(layer.layer_index))
        for layer in _setup_cache_layers(session)
    )


def _missing_artifact_error(missing: Sequence[str]) -> str:
    sample = ", ".join(missing[:4])
    if len(missing) > 4:
        sample = f"{sample}, ... (+{len(missing) - 4} more)"
    return f"missing production prefill runtime artifacts: {sample}"


def _run_runtime_prefill_executor(
    session: Any,
    token_ids: Sequence[int],
    *,
    runtime_cache: Any,
) -> Gemma3ProductionPrefillResult:
    """Run the production prefill executor, or report why it cannot launch."""
    if runtime_cache is None:
        return _blocked_result(
            session=session,
            blockers=(
                PRODUCTION_PREFILL_ARTIFACTS_BLOCKER,
                PREFILL_1K_NPU_BLOCKER,
                PREFILL_PRODUCED_KV_CACHE_BLOCKER,
                NPU_PREFILL_KV_CACHE_BLOCKER,
            ),
            runtime_cache_stats=None,
            error="production prefill runtime cache is not available",
        )
    expected_artifacts = _expected_prefill_artifact_names(session)
    if not expected_artifacts:
        return _blocked_result(
            session=session,
            blockers=(
                PRODUCTION_PREFILL_ARTIFACTS_BLOCKER,
                PREFILL_1K_NPU_BLOCKER,
                PREFILL_PRODUCED_KV_CACHE_BLOCKER,
                NPU_PREFILL_KV_CACHE_BLOCKER,
            ),
            runtime_cache_stats=_runtime_cache_stats(runtime_cache),
            error="production prefill K/V cache descriptor is unavailable",
        )
    artifacts = _runtime_cache_artifacts(runtime_cache)
    missing_artifacts = tuple(name for name in expected_artifacts if name not in artifacts)
    if missing_artifacts:
        return _blocked_result(
            session=session,
            blockers=(
                PRODUCTION_PREFILL_ARTIFACTS_BLOCKER,
                PREFILL_1K_NPU_BLOCKER,
                PREFILL_PRODUCED_KV_CACHE_BLOCKER,
                NPU_PREFILL_KV_CACHE_BLOCKER,
            ),
            runtime_cache_stats=_runtime_cache_stats(runtime_cache),
            error=_missing_artifact_error(missing_artifacts),
        )

    # Cached binaries alone are insufficient: the production executor also
    # needs a materialized argument plan for token embeddings, static BO slices,
    # and per-layer K/V output BOs before it can safely launch.
    return _blocked_result(
        session=session,
        blockers=(
            PRODUCTION_PREFILL_ARGUMENTS_BLOCKER,
            PREFILL_1K_NPU_BLOCKER,
            PREFILL_PRODUCED_KV_CACHE_BLOCKER,
            NPU_PREFILL_KV_CACHE_BLOCKER,
        ),
        runtime_cache_stats=_runtime_cache_stats(runtime_cache),
        error=(
            "production prefill artifacts are cached, but runtime argument "
            "materialization is not wired"
        ),
    )


def run_prefill_kv_cache(
    session: Any,
    token_ids: Sequence[int],
    *,
    runtime_cache: Any = None,
    evidence_path: Path | None = None,
) -> Gemma3ProductionPrefillResult:
    """Return production-prefill K/V cache status for a prepared session.

    Explicit evidence can validate an already-recorded production result. The
    default path does not scan result JSON, because the runtime entrypoint must
    launch or explicitly report the missing production executor boundary.
    """
    start = perf_counter()
    setup = session.setup
    runtime_cache_stats = _runtime_cache_stats(runtime_cache)
    if len(token_ids) != setup.prompt_len:
        raise ValueError(f"token_ids length {len(token_ids)} does not match prompt_len {setup.prompt_len}")
    if not setup.ready_for_entrypoints:
        return _blocked_result(
            session=session,
            blockers=setup.blockers or ("prefill-runtime-setup-blocked",),
            runtime_cache_stats=runtime_cache_stats,
        )

    if evidence_path is not None:
        try:
            data = json.loads(evidence_path.read_text(encoding="utf-8"))
            if not isinstance(data, Mapping):
                raise ValueError("prefill evidence root must be a JSON object")
            result = _result_from_evidence(
                session=session,
                data=data,
                evidence_path=evidence_path,
                runtime_cache_stats=runtime_cache_stats,
            )
            if result.elapsed_seconds is None:
                return Gemma3ProductionPrefillResult(
                    status=result.status,
                    source=result.source,
                    owner=result.owner,
                    blockers=result.blockers,
                    layers=result.layers,
                    elapsed_seconds=perf_counter() - start,
                    kernel_launch_count=result.kernel_launch_count,
                    host_fallback_count=result.host_fallback_count,
                    host_runtime_count=result.host_runtime_count,
                    evidence_path=result.evidence_path,
                    runtime_cache=result.runtime_cache,
                    error=result.error,
                )
            return result
        except Exception as exc:
            return _blocked_result(
                session=session,
                blockers=(
                    PREFILL_1K_NPU_BLOCKER,
                    PREFILL_PRODUCED_KV_CACHE_BLOCKER,
                    NPU_PREFILL_KV_CACHE_BLOCKER,
                    *getattr(setup, "blockers", ()),
                ),
                runtime_cache_stats=runtime_cache_stats,
                error=str(exc),
            )

    return _run_runtime_prefill_executor(
        session=session,
        token_ids=token_ids,
        runtime_cache=runtime_cache,
    )


def has_all_layer_production_prefill_evidence(
    model_variant: str,
    *,
    prompt_len: int,
    decode_context: int,
    layers: int,
    path: Path | None = None,
) -> bool:
    for candidate in candidate_evidence_paths(model_variant, explicit=path):
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, Mapping):
            continue
        payload = _cache_payload(data)
        if payload.get("model_variant") not in (None, model_variant):
            continue
        if payload.get("status") != PREFILL_KV_CACHE_READY_STATUS:
            continue
        if payload.get("source") != PREFILL_KV_CACHE_PRODUCTION_SOURCE:
            continue
        if payload.get("owner") != "npu":
            continue
        if int(payload.get("prompt_token_count", 0) or 0) != int(prompt_len):
            continue
        if int(payload.get("decode_context", 0) or 0) != int(decode_context):
            continue
        if int(payload.get("layer_count", 0) or 0) != int(layers):
            continue
        layer_payloads = _layer_payloads(payload)
        if len(layer_payloads) != int(layers):
            continue
        if any(
            layer.get("status") != PREFILL_KV_CACHE_READY_STATUS
            or layer.get("source") != PREFILL_KV_CACHE_PRODUCTION_SOURCE
            or layer.get("owner") != "npu"
            or layer.get("blockers")
            for layer in layer_payloads
        ):
            continue
        launch_count = int(data.get("prefill_kernel_launch_count", payload.get("kernel_launch_count", 0)) or 0)
        if launch_count <= 0:
            launch_count = sum(int(layer.get("kernel_launch_count", 0) or 0) for layer in layer_payloads)
        if launch_count <= 0:
            continue
        return True
    return False
