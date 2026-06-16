#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 real-model artifact contracts for paper reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from gemma3.core.common import Q4NX_COLS, Q4NX_ROWS, pack_int4_low_first, q4nx_dequant_reference
from ml_dtypes import bfloat16


class Gemma3ArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class Gemma3ModelSpec:
    model_variant: str
    paper_name: str
    has_vision: bool
    prefill_lengths: tuple[int, ...]
    decode_lengths: tuple[int, ...]
    max_decode_context: int
    attention_pattern: str
    q4nx_rows: int = Q4NX_ROWS
    q4nx_cols: int = Q4NX_COLS
    config_status: str = "requires_real_model_config"

    def validate_sequence_length(self, sequence_length: int, *, phase: str) -> None:
        lengths = self.prefill_lengths if phase == "prefill" else self.decode_lengths
        if sequence_length not in lengths:
            raise ValueError(
                f"{self.model_variant} {phase} length {sequence_length} is not a paper target; "
                f"supported lengths: {lengths}"
            )


DEFAULT_MODEL_ROOT_ENV = "GEMMA3_MODEL_ROOT"
Q4NX_PROJECTION_FAMILIES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


DEFAULT_MODEL_DIR_NAMES = {
    "gemma3-1b": "gemma-3-1b-pt",
    "gemma3-4b": "gemma-3-4b-pt",
    "gemma3-4b-vision": "gemma-3-4b-pt",
}


OFFICIAL_MODEL_REPOS = {
    "gemma3-1b": "google/gemma-3-1b-pt",
    "gemma3-4b": "google/gemma-3-4b-pt",
    "gemma3-4b-vision": "google/gemma-3-4b-pt",
}


MODEL_SPECS: dict[str, Gemma3ModelSpec] = {
    "gemma3-1b": Gemma3ModelSpec(
        model_variant="gemma3-1b",
        paper_name="Gemma3-1B text",
        has_vision=False,
        prefill_lengths=(1024, 2048, 4096, 8192, 16384, 32768),
        decode_lengths=(1024, 2048, 4096, 8192, 16384, 32768),
        max_decode_context=32768,
        attention_pattern="paper text local/global pattern once real config is loaded",
    ),
    "gemma3-4b": Gemma3ModelSpec(
        model_variant="gemma3-4b",
        paper_name="Gemma3-4B text",
        has_vision=False,
        prefill_lengths=(1024, 2048, 4096, 8192, 16384, 32768),
        decode_lengths=(1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072),
        max_decode_context=131072,
        attention_pattern="5 local SWA layers followed by 1 global layer, repeated",
    ),
    "gemma3-4b-vision": Gemma3ModelSpec(
        model_variant="gemma3-4b-vision",
        paper_name="Gemma3-4B vision tower",
        has_vision=True,
        prefill_lengths=(1024, 2048, 4096, 8192, 16384, 32768),
        decode_lengths=(1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072),
        max_decode_context=131072,
        attention_pattern="vision_nca prefill plus 5-local/1-global text pattern",
    ),
}


@dataclass(frozen=True)
class Q4NXPackingContract:
    rows: int = Q4NX_ROWS
    cols: int = Q4NX_COLS
    packed_nibble_order: str = "low_nibble_first"
    scale_dtype: str = "bfloat16"
    min_offset_dtype: str = "bfloat16"
    logical_formula: str = "w_bf16[row, col] = scale[col] * q4[row, col] + min[col]"
    matrix_order: str = "projection weights are stored as out_dim x in_dim blocks"


@dataclass(frozen=True)
class Gemma3ArtifactInventory:
    model_variant: str
    weights_dir: str | None
    source_repo: str
    has_vision: bool
    default_weights_dir: str | None
    default_weights_dir_used: bool
    config_path: str | None
    tokenizer_path: str | None
    processor_path: str | None
    safetensors: tuple[str, ...]
    config_exists: bool
    tokenizer_exists: bool
    processor_exists: bool
    optional_packages: dict[str, bool]
    q4nx_contract: Q4NXPackingContract

    @property
    def has_weight_files(self) -> bool:
        return bool(self.safetensors)

    @property
    def can_load_real_artifacts(self) -> bool:
        return (
            self.has_weight_files
            and self.config_exists
            and self.tokenizer_exists
            and (not self.has_vision or self.processor_exists)
            and self.optional_packages.get("safetensors", False)
            and (
                self.optional_packages.get("tokenizers", False)
                or self.optional_packages.get("sentencepiece", False)
                or self.optional_packages.get("transformers", False)
            )
        )

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["has_weight_files"] = self.has_weight_files
        data["can_load_real_artifacts"] = self.can_load_real_artifacts
        return data


@dataclass(frozen=True)
class Gemma3ArtifactManifest:
    model_variant: str
    source_repo: str
    has_vision: bool
    default_weights_dir: str | None
    default_weights_dir_used: bool
    weights_dir: str | None
    config_path: str | None
    tokenizer_path: str | None
    processor_path: str | None
    safetensors: tuple[str, ...]
    tensor_shapes: dict[str, tuple[int, ...]]
    validation_status: str
    blockers: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.blockers

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["ready"] = self.ready
        return data


def official_model_repo(model_variant: str) -> str:
    model_spec(model_variant)
    return OFFICIAL_MODEL_REPOS[model_variant]


def model_spec(model_variant: str) -> Gemma3ModelSpec:
    try:
        return MODEL_SPECS[model_variant]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"model_variant must be one of: {supported}") from exc


def optional_package_status() -> dict[str, bool]:
    status = {}
    for name in ("safetensors", "transformers", "tokenizers", "sentencepiece"):
        try:
            __import__(name)
        except Exception:
            status[name] = False
        else:
            status[name] = True
    return status


def default_model_root() -> Path:
    return Path(os.environ.get(DEFAULT_MODEL_ROOT_ENV, str(Path.home() / "models"))).expanduser()


def default_weights_dir(model_variant: str, *, model_root: Path | None = None) -> Path:
    model_spec(model_variant)
    root = model_root or default_model_root()
    return root / DEFAULT_MODEL_DIR_NAMES[model_variant]


def resolve_weights_dir(
    model_variant: str,
    weights_dir: Path | None,
    *,
    use_default: bool = True,
) -> tuple[Path | None, Path]:
    default_dir = default_weights_dir(model_variant)
    if weights_dir is not None:
        return weights_dir.expanduser(), default_dir
    if use_default and default_dir.exists():
        return default_dir, default_dir
    return None, default_dir


def _find_safetensors(weights_dir: Path | None) -> tuple[str, ...]:
    if weights_dir is None or not weights_dir.exists():
        return tuple()
    return tuple(str(path) for path in sorted(weights_dir.glob("*.safetensors")))


def _resolve_config(weights_dir: Path | None) -> Path | None:
    if weights_dir is None:
        return None
    candidate = weights_dir / "config.json"
    return candidate if candidate.exists() else None


def _resolve_processor(weights_dir: Path | None) -> Path | None:
    if weights_dir is None:
        return None
    for name in ("preprocessor_config.json", "processor_config.json", "image_processor_config.json"):
        candidate = weights_dir / name
        if candidate.exists():
            return candidate
    return None


def _resolve_tokenizer(weights_dir: Path | None, tokenizer: Path | None) -> Path | None:
    if tokenizer is not None:
        return tokenizer
    if weights_dir is None:
        return None
    for name in ("tokenizer.json", "tokenizer.model", "tokenizer.spm", "gemma3_tokenizer.spiece"):
        candidate = weights_dir / name
        if candidate.exists():
            return candidate
    return None


def discover_model_artifacts(
    model_variant: str,
    *,
    weights_dir: Path | None = None,
    tokenizer: Path | None = None,
    source_repo: str | None = None,
    use_default_weights_dir: bool = True,
) -> Gemma3ArtifactInventory:
    spec = model_spec(model_variant)
    resolved_weights_dir, default_dir = resolve_weights_dir(
        model_variant,
        weights_dir,
        use_default=use_default_weights_dir,
    )
    tokenizer_path = _resolve_tokenizer(resolved_weights_dir, tokenizer)
    config_path = _resolve_config(resolved_weights_dir)
    processor_path = _resolve_processor(resolved_weights_dir)
    return Gemma3ArtifactInventory(
        model_variant=model_variant,
        weights_dir=str(resolved_weights_dir) if resolved_weights_dir is not None else None,
        source_repo=source_repo or official_model_repo(model_variant),
        has_vision=spec.has_vision,
        default_weights_dir=str(default_dir),
        default_weights_dir_used=weights_dir is None and resolved_weights_dir == default_dir,
        config_path=str(config_path) if config_path is not None else None,
        tokenizer_path=str(tokenizer_path) if tokenizer_path is not None else None,
        processor_path=str(processor_path) if processor_path is not None else None,
        safetensors=_find_safetensors(resolved_weights_dir),
        config_exists=bool(config_path and config_path.exists()),
        tokenizer_exists=bool(tokenizer_path and tokenizer_path.exists()),
        processor_exists=bool(processor_path and processor_path.exists()),
        optional_packages=optional_package_status(),
        q4nx_contract=Q4NXPackingContract(),
    )


def _manifest_blockers(inventory: Gemma3ArtifactInventory) -> list[str]:
    blockers: list[str] = []
    if not inventory.has_weight_files:
        blockers.append("missing-safetensors")
    if not inventory.config_exists:
        blockers.append("missing-config-json")
    if not inventory.tokenizer_exists:
        blockers.append("missing-tokenizer")
    if inventory.has_vision and not inventory.processor_exists:
        blockers.append("missing-processor")
    if not inventory.optional_packages.get("safetensors", False):
        blockers.append("missing-python-safetensors")
    if not any(inventory.optional_packages.get(pkg, False) for pkg in ("tokenizers", "sentencepiece", "transformers")):
        blockers.append("missing-python-tokenizer-package")
    return blockers


def _safetensor_shapes(paths: Iterable[str], *, max_tensors: int = 8) -> dict[str, tuple[int, ...]]:
    try:
        from safetensors import safe_open
    except Exception:
        return {}
    shapes: dict[str, tuple[int, ...]] = {}
    for filename in paths:
        try:
            with safe_open(filename, framework="np") as handle:
                for key in handle.keys():
                    shapes[key] = tuple(handle.get_tensor(key).shape)
                    if len(shapes) >= max_tensors:
                        return shapes
        except Exception:
            shapes[f"INVALID:{Path(filename).name}"] = tuple()
            return shapes
    return shapes


def artifact_manifest(
    model_variant: str,
    *,
    weights_dir: Path | None = None,
    tokenizer: Path | None = None,
    source_repo: str | None = None,
    include_tensor_shapes: bool = False,
    use_default_weights_dir: bool = True,
) -> Gemma3ArtifactManifest:
    inventory = discover_model_artifacts(
        model_variant,
        weights_dir=weights_dir,
        tokenizer=tokenizer,
        source_repo=source_repo,
        use_default_weights_dir=use_default_weights_dir,
    )
    blockers = _manifest_blockers(inventory)
    tensor_shapes = _safetensor_shapes(inventory.safetensors) if include_tensor_shapes else {}
    if any(key.startswith("INVALID:") for key in tensor_shapes):
        blockers.append("invalid-safetensors")
    status = "READY" if not blockers else "BLOCKED"
    return Gemma3ArtifactManifest(
        model_variant=model_variant,
        source_repo=inventory.source_repo,
        has_vision=inventory.has_vision,
        default_weights_dir=inventory.default_weights_dir,
        default_weights_dir_used=inventory.default_weights_dir_used,
        weights_dir=inventory.weights_dir,
        config_path=inventory.config_path,
        tokenizer_path=inventory.tokenizer_path,
        processor_path=inventory.processor_path,
        safetensors=inventory.safetensors,
        tensor_shapes=tensor_shapes,
        validation_status=status,
        blockers=tuple(dict.fromkeys(blockers)),
    )


def download_model_snapshot(
    model_variant: str,
    *,
    output_dir: Path,
    source_repo: str | None = None,
) -> Path:
    repo_id = source_repo or official_model_repo(model_variant)
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise Gemma3ArtifactError("python:huggingface_hub is required for download") from exc
    return Path(
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(output_dir),
            allow_patterns=[
                "*.safetensors",
                "*.json",
                "*.model",
                "*.spiece",
                "tokenizer.*",
            ],
        )
    )


def load_config_summary(inventory: Gemma3ArtifactInventory) -> dict[str, Any]:
    if not inventory.config_path:
        return {"status": "BLOCKED", "blocker": "missing-config-json"}
    data = json.loads(Path(inventory.config_path).read_text(encoding="utf-8"))
    text_config = data.get("text_config", data)
    summary: dict[str, Any] = {
        "status": "READY",
        "model_variant": inventory.model_variant,
        "model_type": data.get("model_type"),
        "architectures": data.get("architectures", []),
        "text_model_type": text_config.get("model_type"),
        "hidden_size": text_config.get("hidden_size"),
        "intermediate_size": text_config.get("intermediate_size"),
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "num_attention_heads": text_config.get("num_attention_heads"),
        "num_key_value_heads": text_config.get("num_key_value_heads"),
        "head_dim": text_config.get("head_dim"),
        "vocab_size": text_config.get("vocab_size", data.get("vocab_size")),
        "sliding_window": text_config.get("sliding_window", data.get("sliding_window")),
        "torch_dtype": data.get("torch_dtype", text_config.get("torch_dtype")),
        "max_position_embeddings": text_config.get("max_position_embeddings", data.get("max_position_embeddings")),
    }
    vision_config = data.get("vision_config")
    if isinstance(vision_config, dict):
        summary.update(
            {
                "vision_model_type": vision_config.get("model_type"),
                "vision_hidden_size": vision_config.get("hidden_size"),
                "vision_intermediate_size": vision_config.get("intermediate_size"),
                "vision_num_hidden_layers": vision_config.get("num_hidden_layers"),
                "vision_num_attention_heads": vision_config.get("num_attention_heads"),
                "vision_image_size": vision_config.get("image_size"),
                "vision_patch_size": vision_config.get("patch_size"),
                "mm_tokens_per_image": data.get("mm_tokens_per_image"),
                "image_token_index": data.get("image_token_index"),
            }
        )
    return summary


def real_tokenizer_prompt_ids(
    model_variant: str,
    sequence_length: int,
    *,
    weights_dir: Path | None = None,
    tokenizer: Path | None = None,
    seed_text: str = "gemma3 paper reproduction prompt",
) -> tuple[np.ndarray, dict[str, Any]]:
    spec = model_spec(model_variant)
    phase = "decode" if sequence_length in spec.decode_lengths else "prefill"
    spec.validate_sequence_length(sequence_length, phase=phase)
    inventory = load_real_model_artifacts(
        model_variant,
        weights_dir=weights_dir,
        tokenizer=tokenizer,
        strict=True,
    )
    if not inventory.tokenizer_path:
        raise Gemma3ArtifactError("real tokenizer prompt generation requires tokenizer.json")
    try:
        from tokenizers import Tokenizer
    except Exception as exc:
        raise Gemma3ArtifactError("python:tokenizers is required for real prompt generation") from exc

    tokenizer_obj = Tokenizer.from_file(inventory.tokenizer_path)
    base = (seed_text.strip() or "gemma3") + " "
    seed_ids = tokenizer_obj.encode(base).ids
    if not seed_ids:
        raise Gemma3ArtifactError("real tokenizer produced no seed tokens")
    repeat_ids = seed_ids[1:] if len(seed_ids) > 1 else seed_ids
    ids = list(seed_ids)
    while len(ids) < sequence_length:
        ids.extend(repeat_ids[: sequence_length - len(ids)])
    trimmed = np.asarray(ids[:sequence_length], dtype=np.int64)
    sample = trimmed[: min(64, len(trimmed))].tolist()
    decoded = tokenizer_obj.decode(sample) if sample else ""
    roundtrip = tokenizer_obj.encode(decoded).ids if decoded else []
    metadata = {
        "model_variant": model_variant,
        "sequence_length": sequence_length,
        "tokenizer_path": inventory.tokenizer_path,
        "weights_dir": inventory.weights_dir,
        "seed_token_count": len(seed_ids),
        "prompt_repetitions": (sequence_length + max(1, len(repeat_ids)) - 1) // max(1, len(repeat_ids)),
        "checksum": int(trimmed.sum()),
        "first_token_ids": trimmed[:8].tolist(),
        "roundtrip_nonempty": bool(roundtrip),
        "roundtrip_token_count": len(roundtrip),
    }
    return trimmed, metadata


def deterministic_prompt_token_ids(model_variant: str, sequence_length: int, *, seed_text: str = "gemma3-paper") -> np.ndarray:
    spec = model_spec(model_variant)
    phase = "decode" if sequence_length in spec.decode_lengths else "prefill"
    spec.validate_sequence_length(sequence_length, phase=phase)
    digest = hashlib.sha256(f"{model_variant}:{seed_text}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)
    # Token IDs are placeholders until a real tokenizer is available; keep them
    # deterministic and nonzero so prompt-length plumbing can be validated.
    return rng.integers(1, 32000, size=sequence_length, dtype=np.int64)


def load_real_model_artifacts(
    model_variant: str,
    *,
    weights_dir: Path | None = None,
    tokenizer: Path | None = None,
    strict: bool = True,
) -> Gemma3ArtifactInventory:
    inventory = discover_model_artifacts(model_variant, weights_dir=weights_dir, tokenizer=tokenizer)
    if strict and not inventory.can_load_real_artifacts:
        missing = []
        if not inventory.has_weight_files:
            missing.append("*.safetensors")
        if not inventory.config_exists:
            missing.append("config.json")
        if not inventory.tokenizer_exists:
            missing.append("tokenizer.json/tokenizer.model/tokenizer.spm")
        if inventory.has_vision and not inventory.processor_exists:
            missing.append("processor_config.json/image_processor_config.json")
        for package, available in inventory.optional_packages.items():
            if package == "safetensors" and not available:
                missing.append("python:safetensors")
        if not any(inventory.optional_packages.get(pkg, False) for pkg in ("tokenizers", "sentencepiece", "transformers")):
            missing.append("python tokenizer package")
        raise Gemma3ArtifactError(
            "real Gemma3 artifact loading is blocked; missing " + ", ".join(missing)
        )
    return inventory


def q4nx_roundtrip_sample(
    matrix: np.ndarray,
    *,
    rows: int = Q4NX_ROWS,
    cols: int = Q4NX_COLS,
) -> dict[str, Any]:
    source = np.asarray(matrix, dtype=np.float32)
    sample = np.zeros((rows, cols), dtype=np.float32)
    copy_rows = min(rows, source.shape[0])
    copy_cols = min(cols, source.shape[1])
    sample[:copy_rows, :copy_cols] = source[:copy_rows, :copy_cols]
    col_min = sample.min(axis=0)
    col_max = sample.max(axis=0)
    scale = (col_max - col_min) / 15.0
    scale = np.where(scale == 0.0, 1.0, scale)
    q = np.rint((sample - col_min[None, :]) / scale[None, :])
    q = np.clip(q, 0, 15).astype(np.uint8)
    packed = pack_int4_low_first(q).view(np.int8)
    dequant = q4nx_dequant_reference(
        packed,
        scale.astype(bfloat16),
        col_min.astype(bfloat16),
        rows,
        cols,
    ).astype(np.float32)
    compared = dequant[:copy_rows, :copy_cols] - source[:copy_rows, :copy_cols]
    abs_error = np.abs(compared)
    return {
        "sample_rows": rows,
        "sample_cols": cols,
        "copied_rows": copy_rows,
        "copied_cols": copy_cols,
        "max_abs_error": float(abs_error.max()) if abs_error.size else 0.0,
        "mean_abs_error": float(abs_error.mean()) if abs_error.size else 0.0,
        "packed_bytes": int(packed.size),
    }


def _projection_key_for_family(keys: Iterable[str], family: str) -> str | None:
    suffix = f".{family}.weight"
    matches = [key for key in keys if key.endswith(suffix)]
    if not matches:
        matches = [key for key in keys if suffix in key]
    return sorted(matches)[0] if matches else None


def _load_safetensor_matrix(paths: Iterable[str], key: str) -> np.ndarray:
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise Gemma3ArtifactError("python:safetensors is required for Q4NX round-trip") from exc
    for filename in paths:
        with safe_open(filename, framework="np") as handle:
            if key in handle.keys():
                return np.asarray(handle.get_tensor(key), dtype=np.float32)
    raise Gemma3ArtifactError(f"tensor key not found in safetensors: {key}")


def q4nx_roundtrip_report(
    model_variant: str,
    *,
    weights_dir: Path | None = None,
    tokenizer: Path | None = None,
) -> list[dict[str, Any]]:
    inventory = load_real_model_artifacts(
        model_variant,
        weights_dir=weights_dir,
        tokenizer=tokenizer,
        strict=True,
    )
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise Gemma3ArtifactError("python:safetensors is required for Q4NX round-trip") from exc
    keys: list[str] = []
    for filename in inventory.safetensors:
        with safe_open(filename, framework="np") as handle:
            keys.extend(handle.keys())
    records: list[dict[str, Any]] = []
    for family in Q4NX_PROJECTION_FAMILIES:
        key = _projection_key_for_family(keys, family)
        if key is None:
            records.append({"family": family, "status": "MISSING_TENSOR"})
            continue
        matrix = _load_safetensor_matrix(inventory.safetensors, key)
        sample = q4nx_roundtrip_sample(matrix)
        rows, cols = matrix.shape
        records.append(
            {
                "family": family,
                "status": "PASS",
                "tensor_key": key,
                "shape": (int(rows), int(cols)),
                "full_shape_requires_padding": (
                    rows % Q4NX_ROWS != 0 or cols % Q4NX_COLS != 0
                ),
                **sample,
            }
        )
    return records


def format_q4nx_roundtrip(records: list[dict[str, Any]]) -> str:
    lines = []
    for record in records:
        if record.get("status") != "PASS":
            lines.append(
                f"q4nx_roundtrip family={record['family']} status={record['status']}"
            )
            continue
        shape = "x".join(str(dim) for dim in record["shape"])
        lines.append(
            f"q4nx_roundtrip family={record['family']} status=PASS "
            f"shape={shape} sample={record['sample_rows']}x{record['sample_cols']} "
            f"max_abs_error={record['max_abs_error']:.6f} "
            f"mean_abs_error={record['mean_abs_error']:.6f} "
            f"padding={record['full_shape_requires_padding']}"
        )
    return "\n".join(lines)


def _q4nx_roundtrip_self_test() -> None:
    rng = np.random.default_rng(19)
    matrix = rng.normal(size=(Q4NX_ROWS, Q4NX_COLS)).astype(np.float32)
    record = q4nx_roundtrip_sample(matrix)
    if record["copied_rows"] != Q4NX_ROWS or record["copied_cols"] != Q4NX_COLS:
        raise AssertionError(record)
    if record["max_abs_error"] <= 0.0:
        raise AssertionError("expected lossy int4 round-trip error")
    print(
        f"q4nx_roundtrip_self_test sample={record['sample_rows']}x{record['sample_cols']} "
        f"max_abs_error={record['max_abs_error']:.6f}"
    )
    print("GEMMA3_Q4NX_ROUNDTRIP_SELF_TEST: PASS")


def format_model_specs() -> str:
    lines = []
    for spec in MODEL_SPECS.values():
        lines.append(
            f"model {spec.model_variant} vision={spec.has_vision} "
            f"prefill={','.join(str(v) for v in spec.prefill_lengths)} "
            f"decode={','.join(str(v) for v in spec.decode_lengths)} "
            f"max_context={spec.max_decode_context}"
        )
    return "\n".join(lines)


def _self_test() -> None:
    for variant in MODEL_SPECS:
        prompt = deterministic_prompt_token_ids(variant, 1024)
        if prompt.shape != (1024,):
            raise AssertionError(f"prompt shape mismatch for {variant}: {prompt.shape}")
        inv = discover_model_artifacts(variant)
        if inv.q4nx_contract.rows != Q4NX_ROWS or inv.q4nx_contract.cols != Q4NX_COLS:
            raise AssertionError("Q4NX contract mismatch")
        manifest = artifact_manifest(variant)
        if manifest.source_repo != official_model_repo(variant):
            raise AssertionError(f"source repo mismatch for {variant}: {manifest.source_repo}")
    print(format_model_specs())
    print("GEMMA3_ARTIFACT_CONTRACT_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 real artifact contract and discovery")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default="gemma3-1b")
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--tensor-shapes", action="store_true")
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--source-repo")
    parser.add_argument("--no-default-model-dir", action="store_true")
    parser.add_argument("--config-summary", action="store_true")
    parser.add_argument("--q4nx-roundtrip", action="store_true")
    parser.add_argument("--q4nx-roundtrip-self-test", action="store_true")
    parser.add_argument("--real-prompt", action="store_true")
    parser.add_argument("--seed-text", default="gemma3 paper reproduction prompt")
    parser.add_argument("--print-specs", action="store_true")
    parser.add_argument("--prompt-len", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    if args.q4nx_roundtrip_self_test:
        _q4nx_roundtrip_self_test()
        return 0
    if args.print_specs:
        print(format_model_specs())
    if args.prompt_len is not None:
        if args.real_prompt:
            prompt, metadata = real_tokenizer_prompt_ids(
                args.model_variant,
                args.prompt_len,
                weights_dir=args.weights_dir,
                tokenizer=args.tokenizer,
                seed_text=args.seed_text,
            )
            print(
                f"GEMMA3_REAL_PROMPT_IDS: model={args.model_variant} "
                f"len={len(prompt)} checksum={metadata['checksum']} "
                f"roundtrip_nonempty={metadata['roundtrip_nonempty']} "
                f"tokenizer={metadata['tokenizer_path']}"
            )
        else:
            prompt = deterministic_prompt_token_ids(args.model_variant, args.prompt_len)
            print(f"GEMMA3_PROMPT_IDS: model={args.model_variant} len={len(prompt)} checksum={int(prompt.sum())}")
    if args.download:
        try:
            target_dir = args.download_dir or Path("gemma3_artifacts") / args.model_variant
            downloaded = download_model_snapshot(
                args.model_variant,
                output_dir=target_dir,
                source_repo=args.source_repo,
            )
        except Gemma3ArtifactError as exc:
            print(f"GEMMA3_ARTIFACT_DOWNLOAD_BLOCKED: {exc}")
            return 2
        print(f"GEMMA3_ARTIFACT_DOWNLOAD: model={args.model_variant} dir={downloaded}")
    if args.manifest:
        manifest = artifact_manifest(
            args.model_variant,
            weights_dir=args.weights_dir,
            tokenizer=args.tokenizer,
            source_repo=args.source_repo,
            include_tensor_shapes=args.tensor_shapes,
            use_default_weights_dir=not args.no_default_model_dir,
        )
        print(json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True))
    if args.config_summary:
        inventory = discover_model_artifacts(
            args.model_variant,
            weights_dir=args.weights_dir,
            tokenizer=args.tokenizer,
            source_repo=args.source_repo,
            use_default_weights_dir=not args.no_default_model_dir,
        )
        print(json.dumps(load_config_summary(inventory), indent=2, sort_keys=True))
    if args.q4nx_roundtrip:
        try:
            records = q4nx_roundtrip_report(
                args.model_variant,
                weights_dir=args.weights_dir,
                tokenizer=args.tokenizer,
            )
        except Gemma3ArtifactError as exc:
            print(f"GEMMA3_Q4NX_ROUNDTRIP_BLOCKED: {exc}")
            return 2
        print(format_q4nx_roundtrip(records))
    if args.discover or args.strict_load:
        try:
            if args.strict_load:
                inventory = load_real_model_artifacts(
                    args.model_variant,
                    weights_dir=args.weights_dir,
                    tokenizer=args.tokenizer,
                    strict=True,
                )
            else:
                inventory = discover_model_artifacts(
                    args.model_variant,
                    weights_dir=args.weights_dir,
                    tokenizer=args.tokenizer,
                    source_repo=args.source_repo,
                    use_default_weights_dir=not args.no_default_model_dir,
                )
        except Gemma3ArtifactError as exc:
            print(f"GEMMA3_ARTIFACT_LOAD_BLOCKED: {exc}")
            return 2
        print(json.dumps(inventory.to_json_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
