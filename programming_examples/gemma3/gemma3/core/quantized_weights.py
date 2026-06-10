#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 reusable Q4NX projection-weight cache.

The cache stores derived Q4NX projection payloads beside a resolved Gemma3
weights directory.  It is intentionally shared by CPU/iGPU benchmark guards and
NPU static-weight plumbing so paper-comparison paths do not silently fall back
to original BF16 projection weights.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np
from ml_dtypes import bfloat16

from gemma3.core.common import Q4NX_COLS, Q4NX_ROWS, pack_int4_low_first, unpack_int4_low_first
from gemma3.core.artifacts import MODEL_SPECS, Q4NX_PROJECTION_FAMILIES, load_real_model_artifacts
from gemma3.npu.weight_plan import (
    Gemma3ProjectionWeightRecord,
    Gemma3StaticWeightPlan,
    build_weight_plan,
    build_weight_plan_from_shapes,
)


Q4NX_MANIFEST_VERSION = 1
Q4NX_CACHE_DIRNAME = "q4nx"
Q4NX_MANIFEST_NAME = "q4nx_manifest.json"
Q4NX_TENSOR_DIRNAME = "tensors"
Q4NX_PAYLOAD_EXTENSION = ".q4nx"
Q4NX_BLOCK_BYTES = Q4NX_ROWS * Q4NX_COLS // 2 + Q4NX_COLS * 2 + Q4NX_COLS * 2
Q4NX_PAYLOAD_LAYOUT = "row-block-major-then-col-block-32x256"


@dataclass(frozen=True)
class Q4NXSourceFileRecord:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class Q4NXTensorRecord:
    source_tensor_key: str
    family: str
    layer_index: int
    shape: tuple[int, int]
    padded_shape: tuple[int, int]
    row_blocks: int
    col_blocks: int
    packed_weight_bytes: int
    scale_bytes: int
    min_bytes: int
    payload_bytes: int
    static_bo_offset: int
    relative_path: str
    payload_sha256: str
    source_safetensor: str | None

    @property
    def block_count(self) -> int:
        return int(self.row_blocks) * int(self.col_blocks)


@dataclass(frozen=True)
class Q4NXManifest:
    schema_version: int
    manifest_version: int
    model_variant: str
    weights_dir: str
    quantized_weights_dir: str
    status: str
    q4nx_contract: dict[str, object]
    source_safetensors: tuple[Q4NXSourceFileRecord, ...]
    tensor_count: int
    static_bo_bytes: int
    payload_bytes: int
    tensors: tuple[Q4NXTensorRecord, ...]

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["source_safetensors"] = [asdict(item) for item in self.source_safetensors]
        data["tensors"] = [asdict(item) for item in self.tensors]
        return data


@dataclass(frozen=True)
class Q4NXCacheValidation:
    valid: bool
    status: str
    reason: str
    manifest_path: str
    manifest_sha256: str | None
    tensor_count: int
    payload_bytes: int
    manifest: Q4NXManifest | None

    def format(self) -> str:
        sha = self.manifest_sha256[:16] if self.manifest_sha256 else "none"
        return (
            f"quantized_weights status={self.status} valid={self.valid} "
            f"manifest={self.manifest_path} tensors={self.tensor_count} "
            f"payload_bytes={self.payload_bytes} manifest_sha256={sha} "
            f"reason={self.reason}"
        )


def q4nx_contract_dict() -> dict[str, object]:
    return {
        "rows": Q4NX_ROWS,
        "cols": Q4NX_COLS,
        "block_bytes": Q4NX_BLOCK_BYTES,
        "payload_layout": Q4NX_PAYLOAD_LAYOUT,
        "packed_nibble_order": "low_nibble_first",
        "scale_dtype": "bfloat16",
        "min_offset_dtype": "bfloat16",
        "formula": "w[row, col] = scale[col] * q4[row, col] + min[col]",
        "projection_families": list(Q4NX_PROJECTION_FAMILIES),
    }


def default_quantized_weights_dir(weights_dir: Path) -> Path:
    return weights_dir.expanduser() / Q4NX_CACHE_DIRNAME


def manifest_path_for(quantized_weights_dir: Path) -> Path:
    return quantized_weights_dir / Q4NX_MANIFEST_NAME


def manifest_sha256(path: Path) -> str:
    return _file_sha256(path)


def _ceil_to(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _safe_tensor_filename(tensor_key: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", tensor_key).strip("._")
    if not stem:
        stem = "tensor"
    digest = hashlib.sha256(tensor_key.encode("utf-8")).hexdigest()[:16]
    return f"{stem}.{digest}{Q4NX_PAYLOAD_EXTENSION}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_file_records(paths: Iterable[str | Path]) -> tuple[Q4NXSourceFileRecord, ...]:
    records: list[Q4NXSourceFileRecord] = []
    for item in sorted(Path(path).expanduser() for path in paths):
        records.append(
            Q4NXSourceFileRecord(
                path=str(item),
                size_bytes=item.stat().st_size,
                sha256=_file_sha256(item),
            )
        )
    return tuple(records)


def _tensor_source_map(paths: Iterable[str | Path]) -> dict[str, str]:
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise RuntimeError("python:safetensors is required for Q4NX cache generation") from exc
    source: dict[str, str] = {}
    for filename in paths:
        with safe_open(str(filename), framework="np") as handle:
            for key in handle.keys():
                source.setdefault(key, str(filename))
    return source


def _load_matrix(paths: Iterable[str | Path], key: str) -> np.ndarray:
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise RuntimeError("python:safetensors is required for Q4NX cache generation") from exc
    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise RuntimeError("python:torch is required for BF16 safetensor loading") from exc
    for filename in paths:
        with safe_open(str(filename), framework="pt", device="cpu") as handle:
            if key in handle.keys():
                return handle.get_tensor(key).float().cpu().numpy()
    raise RuntimeError(f"tensor key not found in safetensors: {key}")


def serialize_q4nx_matrix(matrix: np.ndarray) -> bytes:
    """Serialize a matrix as row-block-major 32x256 Q4NX payload bytes."""
    source = np.asarray(matrix, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"Q4NX projection matrix must be rank-2, got shape={source.shape}")
    padded_rows = _ceil_to(source.shape[0], Q4NX_ROWS)
    padded_cols = _ceil_to(source.shape[1], Q4NX_COLS)
    padded = np.zeros((padded_rows, padded_cols), dtype=np.float32)
    padded[: source.shape[0], : source.shape[1]] = source

    chunks: list[bytes] = []
    for row_base in range(0, padded_rows, Q4NX_ROWS):
        for col_base in range(0, padded_cols, Q4NX_COLS):
            block = padded[
                row_base : row_base + Q4NX_ROWS,
                col_base : col_base + Q4NX_COLS,
            ]
            col_min = block.min(axis=0)
            col_max = block.max(axis=0)
            scale = (col_max - col_min) / 15.0
            quant_scale = np.where(scale == 0.0, 1.0, scale)
            q = np.rint((block - col_min[None, :]) / quant_scale[None, :])
            q = np.clip(q, 0, 15).astype(np.uint8)
            chunks.append(pack_int4_low_first(q).tobytes())
            chunks.append(quant_scale.astype(bfloat16).tobytes())
            chunks.append(col_min.astype(bfloat16).tobytes())
    return b"".join(chunks)


def _record_payload_path(record: Gemma3ProjectionWeightRecord) -> Path:
    return Path(Q4NX_TENSOR_DIRNAME) / _safe_tensor_filename(record.tensor_key)


def _manifest_tensor_record(
    record: Gemma3ProjectionWeightRecord,
    *,
    payload_bytes: int,
    payload_sha256: str,
    static_bo_offset: int,
    source_safetensor: str | None,
) -> Q4NXTensorRecord:
    return Q4NXTensorRecord(
        source_tensor_key=record.tensor_key,
        family=record.family,
        layer_index=record.layer_index,
        shape=record.shape,
        padded_shape=record.padded_shape,
        row_blocks=record.row_blocks,
        col_blocks=record.col_blocks,
        packed_weight_bytes=record.packed_weight_bytes,
        scale_bytes=record.scale_bytes,
        min_bytes=record.min_bytes,
        payload_bytes=payload_bytes,
        static_bo_offset=static_bo_offset,
        relative_path=str(_record_payload_path(record)),
        payload_sha256=payload_sha256,
        source_safetensor=source_safetensor,
    )


def _expected_payload_bytes(record: Gemma3ProjectionWeightRecord) -> int:
    return int(record.row_blocks) * int(record.col_blocks) * Q4NX_BLOCK_BYTES


def _manifest_from_json_dict(data: Mapping[str, Any]) -> Q4NXManifest:
    tensors = tuple(
        Q4NXTensorRecord(
            source_tensor_key=str(item["source_tensor_key"]),
            family=str(item["family"]),
            layer_index=int(item["layer_index"]),
            shape=tuple(int(dim) for dim in item["shape"]),
            padded_shape=tuple(int(dim) for dim in item["padded_shape"]),
            row_blocks=int(item["row_blocks"]),
            col_blocks=int(item["col_blocks"]),
            packed_weight_bytes=int(item["packed_weight_bytes"]),
            scale_bytes=int(item["scale_bytes"]),
            min_bytes=int(item["min_bytes"]),
            payload_bytes=int(item["payload_bytes"]),
            static_bo_offset=int(item["static_bo_offset"]),
            relative_path=str(item["relative_path"]),
            payload_sha256=str(item["payload_sha256"]),
            source_safetensor=item.get("source_safetensor"),
        )
        for item in data.get("tensors", ())
    )
    sources = tuple(
        Q4NXSourceFileRecord(
            path=str(item["path"]),
            size_bytes=int(item["size_bytes"]),
            sha256=str(item["sha256"]),
        )
        for item in data.get("source_safetensors", ())
    )
    return Q4NXManifest(
        schema_version=int(data["schema_version"]),
        manifest_version=int(data["manifest_version"]),
        model_variant=str(data["model_variant"]),
        weights_dir=str(data["weights_dir"]),
        quantized_weights_dir=str(data["quantized_weights_dir"]),
        status=str(data["status"]),
        q4nx_contract=dict(data["q4nx_contract"]),
        source_safetensors=sources,
        tensor_count=int(data["tensor_count"]),
        static_bo_bytes=int(data["static_bo_bytes"]),
        payload_bytes=int(data["payload_bytes"]),
        tensors=tensors,
    )


def load_q4nx_manifest(quantized_weights_dir: Path) -> Q4NXManifest:
    path = manifest_path_for(quantized_weights_dir)
    return _manifest_from_json_dict(json.loads(path.read_text(encoding="utf-8")))


def _plan_records_by_key(plan: Gemma3StaticWeightPlan) -> dict[str, Gemma3ProjectionWeightRecord]:
    return {record.tensor_key: record for record in plan.records}


def _manifest_records_by_key(manifest: Q4NXManifest) -> dict[str, Q4NXTensorRecord]:
    return {record.source_tensor_key: record for record in manifest.tensors}


def _validation_failure(
    *,
    manifest_path: Path,
    reason: str,
    status: str = "STALE",
    manifest: Q4NXManifest | None = None,
) -> Q4NXCacheValidation:
    manifest_sha = _file_sha256(manifest_path) if manifest_path.exists() else None
    return Q4NXCacheValidation(
        valid=False,
        status=status,
        reason=reason,
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha,
        tensor_count=manifest.tensor_count if manifest is not None else 0,
        payload_bytes=manifest.payload_bytes if manifest is not None else 0,
        manifest=manifest,
    )


def validate_q4nx_cache(
    model_variant: str,
    *,
    weights_dir: Path,
    quantized_weights_dir: Path | None = None,
    plan: Gemma3StaticWeightPlan | None = None,
    source_safetensors: Iterable[str | Path] | None = None,
) -> Q4NXCacheValidation:
    weights_dir = weights_dir.expanduser()
    quantized_weights_dir = quantized_weights_dir or default_quantized_weights_dir(weights_dir)
    manifest_path = manifest_path_for(quantized_weights_dir)
    if not manifest_path.exists():
        return _validation_failure(
            manifest_path=manifest_path,
            reason="missing-manifest",
            status="MISSING",
        )
    try:
        manifest = load_q4nx_manifest(quantized_weights_dir)
    except Exception as exc:
        return _validation_failure(
            manifest_path=manifest_path,
            reason=f"invalid-manifest:{exc}",
            status="INVALID",
        )
    if manifest.schema_version != 1 or manifest.manifest_version != Q4NX_MANIFEST_VERSION:
        return _validation_failure(
            manifest_path=manifest_path,
            reason="manifest-version-mismatch",
            manifest=manifest,
        )
    if manifest.model_variant != model_variant:
        return _validation_failure(
            manifest_path=manifest_path,
            reason="model-variant-mismatch",
            manifest=manifest,
        )
    if Path(manifest.weights_dir).expanduser() != weights_dir:
        return _validation_failure(
            manifest_path=manifest_path,
            reason="weights-dir-mismatch",
            manifest=manifest,
        )
    if manifest.q4nx_contract != q4nx_contract_dict():
        return _validation_failure(
            manifest_path=manifest_path,
            reason="q4nx-contract-mismatch",
            manifest=manifest,
        )

    source_safetensors = tuple(source_safetensors) if source_safetensors is not None else tuple(item.path for item in manifest.source_safetensors)
    try:
        current_sources = _source_file_records(source_safetensors)
    except Exception as exc:
        return _validation_failure(
            manifest_path=manifest_path,
            reason=f"source-digest-failed:{exc}",
            manifest=manifest,
        )
    if current_sources != manifest.source_safetensors:
        return _validation_failure(
            manifest_path=manifest_path,
            reason="source-safetensor-digest-mismatch",
            manifest=manifest,
        )

    plan = plan or build_weight_plan(model_variant, weights_dir=weights_dir)
    plan_records = _plan_records_by_key(plan)
    manifest_records = _manifest_records_by_key(manifest)
    if set(plan_records) != set(manifest_records):
        return _validation_failure(
            manifest_path=manifest_path,
            reason="tensor-key-set-mismatch",
            manifest=manifest,
        )
    if manifest.tensor_count != plan.tensor_count:
        return _validation_failure(
            manifest_path=manifest_path,
            reason="tensor-count-mismatch",
            manifest=manifest,
        )
    if manifest.static_bo_bytes != plan.static_bo_bytes:
        return _validation_failure(
            manifest_path=manifest_path,
            reason="static-bo-bytes-mismatch",
            manifest=manifest,
        )

    offset = 0
    payload_total = 0
    for key, plan_record in plan_records.items():
        manifest_record = manifest_records[key]
        expected = _manifest_tensor_record(
            plan_record,
            payload_bytes=manifest_record.payload_bytes,
            payload_sha256=manifest_record.payload_sha256,
            static_bo_offset=offset,
            source_safetensor=manifest_record.source_safetensor,
        )
        comparable_expected = asdict(expected)
        comparable_manifest = asdict(manifest_record)
        if comparable_expected != comparable_manifest:
            return _validation_failure(
                manifest_path=manifest_path,
                reason=f"tensor-record-mismatch:{key}",
                manifest=manifest,
            )
        if manifest_record.payload_bytes != _expected_payload_bytes(plan_record):
            return _validation_failure(
                manifest_path=manifest_path,
                reason=f"payload-size-contract-mismatch:{key}",
                manifest=manifest,
            )
        payload_path = quantized_weights_dir / manifest_record.relative_path
        if not payload_path.exists():
            return _validation_failure(
                manifest_path=manifest_path,
                reason=f"missing-payload:{key}",
                manifest=manifest,
            )
        if payload_path.stat().st_size != manifest_record.payload_bytes:
            return _validation_failure(
                manifest_path=manifest_path,
                reason=f"payload-byte-size-mismatch:{key}",
                manifest=manifest,
            )
        if _file_sha256(payload_path) != manifest_record.payload_sha256:
            return _validation_failure(
                manifest_path=manifest_path,
                reason=f"payload-sha256-mismatch:{key}",
                manifest=manifest,
            )
        offset += manifest_record.payload_bytes
        payload_total += manifest_record.payload_bytes
    if payload_total != manifest.payload_bytes:
        return _validation_failure(
            manifest_path=manifest_path,
            reason="payload-total-mismatch",
            manifest=manifest,
        )
    return Q4NXCacheValidation(
        valid=True,
        status="VALID",
        reason="cache-valid",
        manifest_path=str(manifest_path),
        manifest_sha256=_file_sha256(manifest_path),
        tensor_count=manifest.tensor_count,
        payload_bytes=manifest.payload_bytes,
        manifest=manifest,
    )


def _write_manifest_atomic(manifest: Q4NXManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
        prefix=path.name + ".",
        suffix=".tmp",
    ) as handle:
        tmp = Path(handle.name)
        json.dump(manifest.to_json_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def _write_payload_atomic(payload: bytes, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=str(path.parent),
        delete=False,
        prefix=path.name + ".",
        suffix=".tmp",
    ) as handle:
        tmp = Path(handle.name)
        handle.write(payload)
    tmp.replace(path)
    return digest


def build_q4nx_cache_from_matrices(
    *,
    model_variant: str,
    weights_dir: Path,
    quantized_weights_dir: Path,
    plan: Gemma3StaticWeightPlan,
    matrices: Mapping[str, np.ndarray],
    source_safetensors: Iterable[str | Path],
    tensor_source_map: Mapping[str, str] | None = None,
) -> Q4NXManifest:
    weights_dir = weights_dir.expanduser()
    quantized_weights_dir = quantized_weights_dir.expanduser()
    tensor_dir = quantized_weights_dir / Q4NX_TENSOR_DIRNAME
    tensor_dir.mkdir(parents=True, exist_ok=True)
    tensor_source_map = tensor_source_map or {}

    records: list[Q4NXTensorRecord] = []
    offset = 0
    payload_total = 0
    for record in plan.records:
        matrix = matrices[record.tensor_key]
        payload = serialize_q4nx_matrix(matrix)
        expected_bytes = _expected_payload_bytes(record)
        if len(payload) != expected_bytes:
            raise RuntimeError(
                f"Q4NX payload size mismatch for {record.tensor_key}: "
                f"got {len(payload)}, expected {expected_bytes}"
            )
        rel_path = _record_payload_path(record)
        digest = _write_payload_atomic(payload, quantized_weights_dir / rel_path)
        tensor_record = _manifest_tensor_record(
            record,
            payload_bytes=len(payload),
            payload_sha256=digest,
            static_bo_offset=offset,
            source_safetensor=tensor_source_map.get(record.tensor_key),
        )
        records.append(tensor_record)
        offset += len(payload)
        payload_total += len(payload)

    manifest = Q4NXManifest(
        schema_version=1,
        manifest_version=Q4NX_MANIFEST_VERSION,
        model_variant=model_variant,
        weights_dir=str(weights_dir),
        quantized_weights_dir=str(quantized_weights_dir),
        status="READY",
        q4nx_contract=q4nx_contract_dict(),
        source_safetensors=_source_file_records(source_safetensors),
        tensor_count=len(records),
        static_bo_bytes=plan.static_bo_bytes,
        payload_bytes=payload_total,
        tensors=tuple(records),
    )
    if manifest.payload_bytes != manifest.static_bo_bytes:
        raise RuntimeError(
            f"manifest payload/static bytes mismatch: "
            f"{manifest.payload_bytes} != {manifest.static_bo_bytes}"
        )
    _write_manifest_atomic(manifest, manifest_path_for(quantized_weights_dir))

    expected_names = {Path(record.relative_path).name for record in records}
    for stale in tensor_dir.glob(f"*{Q4NX_PAYLOAD_EXTENSION}"):
        if stale.name not in expected_names:
            stale.unlink()
    return manifest


def ensure_q4nx_cache(
    model_variant: str,
    *,
    weights_dir: Path | None = None,
    quantized_weights_dir: Path | None = None,
    force: bool = False,
) -> Q4NXManifest:
    inventory = load_real_model_artifacts(model_variant, weights_dir=weights_dir, strict=True)
    if inventory.weights_dir is None:
        raise RuntimeError("resolved weights_dir is missing")
    resolved_weights_dir = Path(inventory.weights_dir).expanduser()
    resolved_quantized_dir = quantized_weights_dir or default_quantized_weights_dir(resolved_weights_dir)
    plan = build_weight_plan(model_variant, weights_dir=resolved_weights_dir)
    if not force:
        validation = validate_q4nx_cache(
            model_variant,
            weights_dir=resolved_weights_dir,
            quantized_weights_dir=resolved_quantized_dir,
            plan=plan,
            source_safetensors=inventory.safetensors,
        )
        if validation.valid and validation.manifest is not None:
            return validation.manifest

    tensor_source = _tensor_source_map(inventory.safetensors)
    matrices = {
        record.tensor_key: _load_matrix(inventory.safetensors, record.tensor_key)
        for record in plan.records
    }
    return build_q4nx_cache_from_matrices(
        model_variant=model_variant,
        weights_dir=resolved_weights_dir,
        quantized_weights_dir=resolved_quantized_dir,
        plan=plan,
        matrices=matrices,
        source_safetensors=inventory.safetensors,
        tensor_source_map=tensor_source,
    )


def _tensor_record_for_key(manifest: Q4NXManifest, tensor_key: str) -> Q4NXTensorRecord:
    for record in manifest.tensors:
        if record.source_tensor_key == tensor_key:
            return record
    raise KeyError(f"tensor key not present in Q4NX manifest: {tensor_key}")


def load_q4nx_payload(quantized_weights_dir: Path, record: Q4NXTensorRecord) -> bytes:
    path = quantized_weights_dir / record.relative_path
    payload = path.read_bytes()
    if len(payload) != record.payload_bytes:
        raise RuntimeError(
            f"Q4NX payload byte-size mismatch for {record.source_tensor_key}: "
            f"got {len(payload)}, expected {record.payload_bytes}"
        )
    checksum = hashlib.sha256(payload).hexdigest()
    if checksum != record.payload_sha256:
        raise RuntimeError(
            f"Q4NX payload checksum mismatch for {record.source_tensor_key}: "
            f"got {checksum}, expected {record.payload_sha256}"
        )
    return payload


def load_q4nx_payload_for_tensor(
    manifest: Q4NXManifest,
    tensor_key: str,
) -> tuple[Q4NXTensorRecord, bytes]:
    record = _tensor_record_for_key(manifest, tensor_key)
    payload = load_q4nx_payload(Path(manifest.quantized_weights_dir), record)
    return record, payload


def decode_q4nx_payload(
    payload: bytes,
    record: Q4NXTensorRecord,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = int(record.row_blocks) * int(record.col_blocks) * Q4NX_BLOCK_BYTES
    if len(payload) != expected:
        raise ValueError(f"payload size mismatch: got {len(payload)}, expected {expected}")
    packed = np.empty(
        (record.row_blocks, record.col_blocks, Q4NX_ROWS * Q4NX_COLS // 2),
        dtype=np.int8,
    )
    scale = np.empty((record.row_blocks, record.col_blocks, Q4NX_COLS), dtype=bfloat16)
    min_offset = np.empty((record.row_blocks, record.col_blocks, Q4NX_COLS), dtype=bfloat16)
    cursor = 0
    packed_bytes = Q4NX_ROWS * Q4NX_COLS // 2
    param_bytes = Q4NX_COLS * np.dtype(bfloat16).itemsize
    for rb in range(record.row_blocks):
        for cb in range(record.col_blocks):
            packed[rb, cb] = np.frombuffer(
                payload[cursor : cursor + packed_bytes],
                dtype=np.int8,
            )
            cursor += packed_bytes
            scale[rb, cb] = np.frombuffer(
                payload[cursor : cursor + param_bytes],
                dtype=bfloat16,
            )
            cursor += param_bytes
            min_offset[rb, cb] = np.frombuffer(
                payload[cursor : cursor + param_bytes],
                dtype=bfloat16,
            )
            cursor += param_bytes
    return packed.copy(), scale.copy(), min_offset.copy()


def q4nx_dense_dequantize(payload: bytes, record: Q4NXTensorRecord) -> np.ndarray:
    packed, scale, min_offset = decode_q4nx_payload(payload, record)
    dense = np.empty(record.padded_shape, dtype=bfloat16)
    for rb in range(record.row_blocks):
        row_start = rb * Q4NX_ROWS
        for cb in range(record.col_blocks):
            col_start = cb * Q4NX_COLS
            q = unpack_int4_low_first(
                packed[rb, cb].view(np.uint8),
                Q4NX_ROWS * Q4NX_COLS,
            ).reshape(Q4NX_ROWS, Q4NX_COLS)
            block = q.astype(np.float32) * scale[rb, cb].astype(np.float32)[None, :]
            block += min_offset[rb, cb].astype(np.float32)[None, :]
            dense[
                row_start : row_start + Q4NX_ROWS,
                col_start : col_start + Q4NX_COLS,
            ] = block.astype(bfloat16)
    rows, cols = record.shape
    return dense[:rows, :cols]


def cpu_q4nx_projection(payload: bytes, record: Q4NXTensorRecord, activation: np.ndarray) -> np.ndarray:
    """Native packed-Q4NX CPU projection reference.

    This intentionally unpacks and dequantizes one 32x256 block inside the
    projection loops. It does not materialize the full dense projection weight.
    """
    vector = np.asarray(activation, dtype=np.float32).reshape(-1)
    out_dim, in_dim = record.shape
    if vector.size != in_dim:
        raise ValueError(f"activation size mismatch: got {vector.size}, expected {in_dim}")
    padded_activation = np.zeros(record.padded_shape[1], dtype=np.float32)
    padded_activation[:in_dim] = vector
    output = np.zeros(record.padded_shape[0], dtype=np.float32)

    cursor = 0
    packed_bytes = Q4NX_ROWS * Q4NX_COLS // 2
    param_bytes = Q4NX_COLS * np.dtype(bfloat16).itemsize
    for rb in range(record.row_blocks):
        row_start = rb * Q4NX_ROWS
        accum = np.zeros(Q4NX_ROWS, dtype=np.float32)
        for cb in range(record.col_blocks):
            col_start = cb * Q4NX_COLS
            packed = np.frombuffer(payload[cursor : cursor + packed_bytes], dtype=np.uint8)
            cursor += packed_bytes
            scale = np.frombuffer(payload[cursor : cursor + param_bytes], dtype=bfloat16).astype(np.float32)
            cursor += param_bytes
            min_offset = np.frombuffer(payload[cursor : cursor + param_bytes], dtype=bfloat16).astype(np.float32)
            cursor += param_bytes
            q = unpack_int4_low_first(packed, Q4NX_ROWS * Q4NX_COLS).reshape(Q4NX_ROWS, Q4NX_COLS)
            block = q.astype(np.float32) * scale[None, :]
            block += min_offset[None, :]
            accum += block @ padded_activation[col_start : col_start + Q4NX_COLS]
        output[row_start : row_start + Q4NX_ROWS] = accum
    return output[:out_dim].astype(bfloat16)


def _manifest_summary(manifest: Q4NXManifest) -> dict[str, object]:
    path = manifest_path_for(Path(manifest.quantized_weights_dir))
    return {
        "model_variant": manifest.model_variant,
        "weights_dir": manifest.weights_dir,
        "quantized_weights_dir": manifest.quantized_weights_dir,
        "q4nx_manifest": str(path),
        "q4nx_manifest_sha256": _file_sha256(path) if path.exists() else None,
        "tensor_count": manifest.tensor_count,
        "payload_bytes": manifest.payload_bytes,
        "static_bo_bytes": manifest.static_bo_bytes,
        "projection_weight_source": "q4nx",
    }


def format_manifest_summary(manifest: Q4NXManifest, *, elapsed_seconds: float | None = None) -> str:
    path = manifest_path_for(Path(manifest.quantized_weights_dir))
    sha = _file_sha256(path)[:16] if path.exists() else "none"
    elapsed = "" if elapsed_seconds is None else f" elapsed_s={elapsed_seconds:.3f}"
    return (
        f"quantized_weights model={manifest.model_variant} status={manifest.status} "
        f"cache={manifest.quantized_weights_dir} manifest={path} "
        f"tensors={manifest.tensor_count} payload_bytes={manifest.payload_bytes} "
        f"static_bo_bytes={manifest.static_bo_bytes} manifest_sha256={sha} "
        f"projection_weight_source=q4nx{elapsed}"
    )


def _synthetic_shapes() -> dict[str, tuple[int, int]]:
    return {
        "model.layers.0.self_attn.q_proj.weight": (64, 48),
        "model.layers.0.self_attn.k_proj.weight": (32, 48),
        "model.layers.0.self_attn.v_proj.weight": (32, 48),
        "model.layers.0.self_attn.o_proj.weight": (48, 64),
        "model.layers.0.mlp.gate_proj.weight": (96, 48),
        "model.layers.0.mlp.up_proj.weight": (96, 48),
        "model.layers.0.mlp.down_proj.weight": (48, 96),
    }


def _self_test() -> None:
    root = Path("/tmp/gemma3_q4nx_cache_self_test")
    if root.exists():
        shutil.rmtree(root)
    weights_dir = root / "weights"
    cache_dir = weights_dir / Q4NX_CACHE_DIRNAME
    weights_dir.mkdir(parents=True, exist_ok=True)
    source_file = weights_dir / "model.safetensors"
    source_file.write_bytes(b"synthetic source v1")
    plan = build_weight_plan_from_shapes("gemma3-1b", _synthetic_shapes())
    rng = np.random.default_rng(3)
    matrices = {
        record.tensor_key: rng.uniform(-0.75, 0.75, size=record.shape).astype(np.float32)
        for record in plan.records
    }
    tensor_source = {record.tensor_key: str(source_file) for record in plan.records}
    manifest = build_q4nx_cache_from_matrices(
        model_variant="gemma3-1b",
        weights_dir=weights_dir,
        quantized_weights_dir=cache_dir,
        plan=plan,
        matrices=matrices,
        source_safetensors=(source_file,),
        tensor_source_map=tensor_source,
    )
    validation = validate_q4nx_cache(
        "gemma3-1b",
        weights_dir=weights_dir,
        quantized_weights_dir=cache_dir,
        plan=plan,
        source_safetensors=(source_file,),
    )
    if not validation.valid or validation.manifest is None:
        raise AssertionError(validation)
    reused = validate_q4nx_cache(
        "gemma3-1b",
        weights_dir=weights_dir,
        quantized_weights_dir=cache_dir,
        plan=plan,
        source_safetensors=(source_file,),
    )
    if not reused.valid:
        raise AssertionError(reused)

    for tensor in manifest.tensors:
        payload = load_q4nx_payload(cache_dir, tensor)
        activation = rng.uniform(-0.5, 0.5, size=tensor.shape[1]).astype(np.float32)
        native = cpu_q4nx_projection(payload, tensor, activation)
        dense = (q4nx_dense_dequantize(payload, tensor).astype(np.float32) @ activation).astype(bfloat16)
        if not np.allclose(native.astype(np.float32), dense.astype(np.float32), atol=0.0625):
            raise AssertionError(tensor.source_tensor_key)

    first = manifest.tensors[0]
    first_path = cache_dir / first.relative_path
    original_payload = first_path.read_bytes()
    first_path.write_bytes(original_payload[:-1] + bytes([original_payload[-1] ^ 0x01]))
    corrupt = validate_q4nx_cache(
        "gemma3-1b",
        weights_dir=weights_dir,
        quantized_weights_dir=cache_dir,
        plan=plan,
        source_safetensors=(source_file,),
    )
    if corrupt.valid or "payload-sha256-mismatch" not in corrupt.reason:
        raise AssertionError(corrupt)
    first_path.write_bytes(original_payload)
    first_path.unlink()
    missing = validate_q4nx_cache(
        "gemma3-1b",
        weights_dir=weights_dir,
        quantized_weights_dir=cache_dir,
        plan=plan,
        source_safetensors=(source_file,),
    )
    if missing.valid or "missing-payload" not in missing.reason:
        raise AssertionError(missing)
    first_path.write_bytes(original_payload)
    source_file.write_bytes(b"synthetic source v2")
    stale = validate_q4nx_cache(
        "gemma3-1b",
        weights_dir=weights_dir,
        quantized_weights_dir=cache_dir,
        plan=plan,
        source_safetensors=(source_file,),
    )
    if stale.valid or stale.reason != "source-safetensor-digest-mismatch":
        raise AssertionError(stale)

    print(format_manifest_summary(manifest))
    print("GEMMA3_Q4NX_CACHE_REUSE_SELF_TEST: PASS")
    print("GEMMA3_Q4NX_CACHE_INVALIDATION_SELF_TEST: PASS")
    print("GEMMA3_Q4NX_CPU_PROJECTION_SELF_TEST: PASS")
    print("GEMMA3_Q4NX_QUANTIZED_WEIGHTS_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 reusable Q4NX projection-weight cache")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default="gemma3-1b")
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--quantized-weights-dir", type=Path)
    parser.add_argument("--ensure", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-tensors", action="store_true")
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        help="print manifest JSON, or write it to the provided path",
    )
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0

    start = perf_counter()
    if args.weights_dir is None:
        inventory = load_real_model_artifacts(args.model_variant, weights_dir=None, strict=True)
        if inventory.weights_dir is None:
            raise RuntimeError("resolved weights_dir is missing")
        weights_dir = Path(inventory.weights_dir)
    else:
        weights_dir = args.weights_dir.expanduser()
    quantized_dir = args.quantized_weights_dir or default_quantized_weights_dir(weights_dir)
    if args.ensure:
        manifest = ensure_q4nx_cache(
            args.model_variant,
            weights_dir=weights_dir,
            quantized_weights_dir=quantized_dir,
            force=args.force,
        )
        print(format_manifest_summary(manifest, elapsed_seconds=perf_counter() - start))
    else:
        validation = validate_q4nx_cache(
            args.model_variant,
            weights_dir=weights_dir,
            quantized_weights_dir=quantized_dir,
        )
        print(validation.format())
        if not validation.valid:
            return 2
        manifest = validation.manifest
        if manifest is None:
            return 2
    if args.json:
        assert manifest is not None
        data: dict[str, object]
        if args.include_tensors:
            data = manifest.to_json_dict()
        else:
            data = _manifest_summary(manifest)
        text = json.dumps(data, indent=2, sort_keys=True) + "\n"
        if args.json == "-":
            print(text, end="")
        else:
            path = Path(args.json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"GEMMA3_Q4NX_MANIFEST_JSON: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
