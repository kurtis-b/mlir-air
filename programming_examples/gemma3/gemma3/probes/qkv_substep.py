#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gemma3 real-weight decode RMSNorm + Q/K/V substep probe.

This diagnostic extends the q-projection substep proof point to the full layer-0
Q/K/V decode projection stage. It is staged correctness evidence only: each
projection is split into FusedDQP column-block launches and accumulated on the
host, so the result is not a full-layer launch, TTFT/TPS timing, pseudo-NPU
power, or a paper-parity result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time

from gemma3.core.artifacts import MODEL_SPECS
from gemma3.paths import EXAMPLE_ROOT, RESULTS_DIR
from gemma3.probes.substep import (
    DEFAULT_INPUT_DISTRIBUTION,
    DEFAULT_LAYER,
    DEFAULT_MODEL,
    DEFAULT_NORM_TENSOR_KEY,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_PHASE,
    DEFAULT_THRESHOLD,
    _activate_probe_env,
    _correlation,
    _git_info,
    _load_safetensor_array,
    _load_static_norm_payload,
    _repack_q_proj_for_fused_dqp,
    _repo_root,
    _resolve_weights_dir,
    _run_elf_with_runner_bos,
    _shape_text,
    _tail,
)

DEFAULT_SEQUENCE_KIND = "decode-rmsnorm-qkv"
DEFAULT_QKV_SUBSTEP_PROBE_EVIDENCE = (
    RESULTS_DIR / "gemma3_1b_decode_rmsnorm_qkv_substep_probe.json"
)
PROJECTION_FAMILIES = ("q_proj", "k_proj", "v_proj")
EXPECTED_PROJECTION_SHAPES = {
    "q_proj": (1024, 1152),
    "k_proj": (256, 1152),
    "v_proj": (256, 1152),
}
EXPECTED_PROJECTION_OUTPUTS = {
    "q_proj": (1024,),
    "k_proj": (256,),
    "v_proj": (256,),
}


@dataclass(frozen=True)
class Gemma3QKVSubstepProbeResult:
    schema_version: int
    model_variant: str
    status: str
    sequence_kind: str
    phase: str
    layer_index: int
    stages: tuple[str, ...]
    input_shape: tuple[int, ...]
    norm_shape: tuple[int, ...]
    activation_shape: tuple[int, ...]
    q_projection_shape: tuple[int, ...]
    k_projection_shape: tuple[int, ...]
    v_projection_shape: tuple[int, ...]
    output_format: str
    bo_binding_mode: str
    norm_tensor_key: str
    static_norm_tensor_offset_bytes: int | None
    static_norm_bo_bytes: int | None
    projection_tensor_keys: dict[str, str]
    projection_weight_layout: str
    input_distribution: str
    rms_correlation: float | None
    q_projection_correlation: float | None
    k_projection_correlation: float | None
    v_projection_correlation: float | None
    dense_q_projection_correlation: float | None
    dense_k_projection_correlation: float | None
    dense_v_projection_correlation: float | None
    threshold: float
    remaining_model_runner_gaps: tuple[str, ...]
    command: tuple[str, ...]
    returncode: int | None
    elapsed_seconds: float | None
    blockers: tuple[str, ...]
    git_commit: str | None
    dirty_worktree: bool | None
    stdout_tail: tuple[str, ...]
    stderr_tail: tuple[str, ...]

    def _corr_text(self, value: float | None) -> str:
        return "n/a" if value is None else f"{value:.6f}"

    def format(self) -> str:
        blockers = ",".join(self.blockers) if self.blockers else "none"
        gaps = (
            ",".join(self.remaining_model_runner_gaps)
            if self.remaining_model_runner_gaps
            else "none"
        )
        stages = "|".join(self.stages) if self.stages else "none"
        projection_keys = "|".join(
            f"{family}:{self.projection_tensor_keys.get(family, 'missing')}"
            for family in PROJECTION_FAMILIES
        )
        return (
            f"qkv_substep_probe model={self.model_variant} status={self.status} "
            f"sequence={self.sequence_kind} phase={self.phase} layer=L{self.layer_index} "
            f"stages={stages} input={_shape_text(self.input_shape)} "
            f"norm={_shape_text(self.norm_shape)} activation={_shape_text(self.activation_shape)} "
            f"q_projection={_shape_text(self.q_projection_shape)} "
            f"k_projection={_shape_text(self.k_projection_shape)} "
            f"v_projection={_shape_text(self.v_projection_shape)} "
            f"output_format={self.output_format} bo_binding={self.bo_binding_mode} "
            f"norm_tensor={self.norm_tensor_key}@{self.static_norm_tensor_offset_bytes}/bo={self.static_norm_bo_bytes} "
            f"projection_tensors={projection_keys} weight_layout={self.projection_weight_layout} "
            f"input_distribution={self.input_distribution} "
            f"rms_correlation={self._corr_text(self.rms_correlation)} "
            f"q_projection_correlation={self._corr_text(self.q_projection_correlation)} "
            f"k_projection_correlation={self._corr_text(self.k_projection_correlation)} "
            f"v_projection_correlation={self._corr_text(self.v_projection_correlation)} "
            f"dense_q_projection_correlation={self._corr_text(self.dense_q_projection_correlation)} "
            f"dense_k_projection_correlation={self._corr_text(self.dense_k_projection_correlation)} "
            f"dense_v_projection_correlation={self._corr_text(self.dense_v_projection_correlation)} "
            f"threshold={self.threshold:g} model_runner_gaps={gaps} blockers={blockers}"
        )

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def _is_decode_qkv_substep_evidence(data: object, *, model_variant: str) -> bool:
    if not isinstance(data, dict):
        return False
    keys = data.get("projection_tensor_keys")
    return (
        data.get("schema_version") == 1
        and data.get("model_variant") == model_variant
        and data.get("status") == "QKV_SUBSTEP_SEQUENCE_PASS"
        and data.get("sequence_kind") == DEFAULT_SEQUENCE_KIND
        and data.get("phase") == DEFAULT_PHASE
        and data.get("layer_index") == DEFAULT_LAYER
        and data.get("output_format") == DEFAULT_OUTPUT_FORMAT
        and data.get("bo_binding_mode") == "runner-owned-persistent-bo"
        and isinstance(keys, dict)
        and all(family in keys for family in PROJECTION_FAMILIES)
        and not data.get("blockers")
        and "full-layer-not-wired" in tuple(data.get("remaining_model_runner_gaps", ()))
    )


def has_decode_qkv_substep_evidence(
    model_variant: str,
    path: Path | None = None,
) -> bool:
    evidence_path = path or DEFAULT_QKV_SUBSTEP_PROBE_EVIDENCE
    try:
        data = json.loads(evidence_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return _is_decode_qkv_substep_evidence(data, model_variant=model_variant)


def _projection_tensor_keys(model_variant: str, weights_dir: Path) -> dict[str, str]:
    from gemma3.npu.weight_plan import build_weight_plan

    plan = build_weight_plan(model_variant, weights_dir=weights_dir)
    keys = {}
    for record in plan.records:
        if record.layer_index == DEFAULT_LAYER and record.family in PROJECTION_FAMILIES:
            keys[record.family] = record.tensor_key
    missing = [family for family in PROJECTION_FAMILIES if family not in keys]
    if missing:
        raise RuntimeError(f"missing layer-0 projection tensors: {missing}")
    return keys


def _projection_backend_options() -> dict[str, object]:
    return dict(
        verbose=False,
        omit_pingpong=True,
        output_format=DEFAULT_OUTPUT_FORMAT,
        instance_name="fused_dqp_paper",
        target_device="npu2",
        runtime_loop_tiling_sizes=[1, 1],
        use_lock_race_condition_fix=True,
    )


def _run_projection_family(
    *,
    family: str,
    weight,
    activation,
    activation_padded,
    object_file: Path,
):
    import numpy as np
    from ml_dtypes import bfloat16
    from gemma3.core.common import fused_dqp_paper_reference
    from gemma3.kernels.fused_dqp import _pack_l3_inputs, build_paper_module

    expected_shape = EXPECTED_PROJECTION_SHAPES[family]
    if tuple(weight.shape) != expected_shape:
        raise RuntimeError(f"expected {family} shape {expected_shape}, got {weight.shape}")
    packed, scale, min_offset, padded_weight = _repack_q_proj_for_fused_dqp(weight)
    row_blocks = packed.shape[0]
    qkv_module = build_paper_module(
        32,
        256,
        "fused_dqp_accum_block_opt",
        str(object_file),
        row_blocks,
        1,
        2,
        4,
        "direct",
    )
    expected = fused_dqp_paper_reference(
        packed,
        scale,
        min_offset,
        activation,
        32,
        256,
    )
    accum = np.zeros(expected.shape, dtype=np.float32)
    for col_block in range(5):
        cb_slice = slice(col_block, col_block + 1)
        params = np.empty((row_blocks, 1, 512), dtype=bfloat16)
        params[..., :256] = scale[:, cb_slice, :]
        params[..., 256:] = min_offset[:, cb_slice, :]
        packed_l3 = _pack_l3_inputs(packed[:, cb_slice, :], params).reshape(
            row_blocks // 4,
            4,
            1,
            -1,
        )
        partial = _run_elf_with_runner_bos(
            mlir_module=qkv_module,
            backend_options=_projection_backend_options(),
            inputs=[packed_l3, activation[cb_slice, :]],
            output_shape=expected.shape,
            output_dtype=bfloat16,
        )
        accum += partial.astype(np.float32)
    actual = accum.astype(bfloat16)
    projection_correlation = _correlation(actual, expected)
    dense_expected = (
        padded_weight.astype(np.float32) @ activation_padded.astype(np.float32)
    ).astype(bfloat16)
    dense_correlation = _correlation(actual.reshape(-1), dense_expected)
    return actual, projection_correlation, dense_correlation


def _run_hardware_sequence(args: argparse.Namespace) -> Gemma3QKVSubstepProbeResult:
    _activate_probe_env()
    import numpy as np
    from ml_dtypes import bfloat16
    from weighted_rms_norm import build_module as build_rms_module
    from weighted_rms_norm import rms_norm_reference

    repo = _repo_root()
    git_commit, dirty = _git_info(repo)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    blockers: list[str] = []
    projection_keys: dict[str, str] = {}
    correlations: dict[str, float | None] = {family: None for family in PROJECTION_FAMILIES}
    dense_correlations: dict[str, float | None] = {family: None for family in PROJECTION_FAMILIES}
    start = time.perf_counter()

    try:
        weights_dir = _resolve_weights_dir(args.model_variant, args.weights_dir)
        norm_payload, norm_weight, norm_offset = _load_static_norm_payload(
            weights_dir,
            args.model_variant,
            args.norm_tensor_key,
        )
        projection_keys = _projection_tensor_keys(args.model_variant, weights_dir)
        rng = np.random.default_rng(0)
        x_input = rng.uniform(-0.5, 0.5, size=(1, 1152)).astype(bfloat16)
        rms_expected = rms_norm_reference(x_input, norm_weight)
        rms_module = build_rms_module(1, 1152, bfloat16, 16, herd_x=1)
        rms_actual = _run_elf_with_runner_bos(
            mlir_module=rms_module,
            backend_options=dict(
                verbose=False,
                omit_while_true_loop=False,
                output_format=DEFAULT_OUTPUT_FORMAT,
                instance_name="weighted_rms_norm",
                runtime_loop_tiling_sizes=[4, 4],
            ),
            inputs=[x_input, norm_payload],
            output_shape=rms_expected.shape,
            output_dtype=bfloat16,
        )
        rms_correlation = _correlation(rms_actual, rms_expected)
        stdout_lines.append(f"RMSNorm correlation: {rms_correlation:.6f}")
        if rms_correlation < DEFAULT_THRESHOLD:
            blockers.append("decode-rmsnorm-correlation-low")

        activation_padded = np.zeros((5 * 256,), dtype=bfloat16)
        activation_padded[:1152] = rms_actual.reshape(-1)
        activation = activation_padded.reshape(5, 256)
        object_file = EXAMPLE_ROOT / "build_peano" / "fused_dqp.o"
        if not object_file.exists():
            raise RuntimeError(f"missing FusedDQP object file: {object_file}")

        for family in PROJECTION_FAMILIES:
            weight = _load_safetensor_array(weights_dir, projection_keys[family])
            _, corr, dense_corr = _run_projection_family(
                family=family,
                weight=weight,
                activation=activation,
                activation_padded=activation_padded,
                object_file=object_file,
            )
            correlations[family] = corr
            dense_correlations[family] = dense_corr
            stdout_lines.append(f"{family} correlation: {corr:.6f}")
            stdout_lines.append(f"dense {family} correlation: {dense_corr:.6f}")
            if corr < DEFAULT_THRESHOLD:
                blockers.append(f"decode-{family}-correlation-low")
        returncode = 0 if not blockers else 1
    except Exception as exc:
        blockers.append(f"decode-qkv-substep-probe-failed:{exc}")
        rms_correlation = None
        norm_offset = None
        norm_payload = None
        returncode = 1
        stderr_lines.append(str(exc))

    elapsed = time.perf_counter() - start
    status = "QKV_SUBSTEP_SEQUENCE_PASS" if not blockers else "QKV_SUBSTEP_SEQUENCE_BLOCKED"
    return Gemma3QKVSubstepProbeResult(
        schema_version=1,
        model_variant=args.model_variant,
        status=status,
        sequence_kind=DEFAULT_SEQUENCE_KIND,
        phase=DEFAULT_PHASE,
        layer_index=DEFAULT_LAYER,
        stages=(
            "decode:L0:pre_attention_norm",
            "decode:L0:q_projection",
            "decode:L0:k_projection",
            "decode:L0:v_projection",
        ),
        input_shape=(1, 1152),
        norm_shape=(1, 1152),
        activation_shape=(5, 256),
        q_projection_shape=EXPECTED_PROJECTION_OUTPUTS["q_proj"],
        k_projection_shape=EXPECTED_PROJECTION_OUTPUTS["k_proj"],
        v_projection_shape=EXPECTED_PROJECTION_OUTPUTS["v_proj"],
        output_format=DEFAULT_OUTPUT_FORMAT,
        bo_binding_mode="runner-owned-persistent-bo",
        norm_tensor_key=args.norm_tensor_key,
        static_norm_tensor_offset_bytes=norm_offset,
        static_norm_bo_bytes=None if norm_payload is None else int(norm_payload.nbytes),
        projection_tensor_keys=projection_keys,
        projection_weight_layout="fused-dqp-paper-repacked-qkv-colblock-loop",
        input_distribution=DEFAULT_INPUT_DISTRIBUTION,
        rms_correlation=rms_correlation,
        q_projection_correlation=correlations["q_proj"],
        k_projection_correlation=correlations["k_proj"],
        v_projection_correlation=correlations["v_proj"],
        dense_q_projection_correlation=dense_correlations["q_proj"],
        dense_k_projection_correlation=dense_correlations["k_proj"],
        dense_v_projection_correlation=dense_correlations["v_proj"],
        threshold=DEFAULT_THRESHOLD,
        remaining_model_runner_gaps=("full-layer-not-wired",),
        command=tuple(sys.argv),
        returncode=returncode,
        elapsed_seconds=elapsed,
        blockers=tuple(dict.fromkeys(blockers)),
        git_commit=git_commit,
        dirty_worktree=dirty,
        stdout_tail=_tail("\n".join(stdout_lines)),
        stderr_tail=_tail("\n".join(stderr_lines)),
    )


def _self_test() -> None:
    result = Gemma3QKVSubstepProbeResult(
        schema_version=1,
        model_variant=DEFAULT_MODEL,
        status="QKV_SUBSTEP_SEQUENCE_PASS",
        sequence_kind=DEFAULT_SEQUENCE_KIND,
        phase=DEFAULT_PHASE,
        layer_index=DEFAULT_LAYER,
        stages=(
            "decode:L0:pre_attention_norm",
            "decode:L0:q_projection",
            "decode:L0:k_projection",
            "decode:L0:v_projection",
        ),
        input_shape=(1, 1152),
        norm_shape=(1, 1152),
        activation_shape=(5, 256),
        q_projection_shape=(1024,),
        k_projection_shape=(256,),
        v_projection_shape=(256,),
        output_format=DEFAULT_OUTPUT_FORMAT,
        bo_binding_mode="runner-owned-persistent-bo",
        norm_tensor_key=DEFAULT_NORM_TENSOR_KEY,
        static_norm_tensor_offset_bytes=0,
        static_norm_bo_bytes=266240,
        projection_tensor_keys={
            "q_proj": "model.layers.0.self_attn.q_proj.weight",
            "k_proj": "model.layers.0.self_attn.k_proj.weight",
            "v_proj": "model.layers.0.self_attn.v_proj.weight",
        },
        projection_weight_layout="fused-dqp-paper-repacked-qkv-colblock-loop",
        input_distribution=DEFAULT_INPUT_DISTRIBUTION,
        rms_correlation=0.999991,
        q_projection_correlation=1.0,
        k_projection_correlation=1.0,
        v_projection_correlation=1.0,
        dense_q_projection_correlation=0.994609,
        dense_k_projection_correlation=0.995000,
        dense_v_projection_correlation=0.995500,
        threshold=DEFAULT_THRESHOLD,
        remaining_model_runner_gaps=("full-layer-not-wired",),
        command=("python3", "-m", "gemma3.probes.qkv_substep", "--self-test"),
        returncode=0,
        elapsed_seconds=0.125,
        blockers=(),
        git_commit="fixture",
        dirty_worktree=False,
        stdout_tail=("RMSNorm correlation: 0.999991", "q_proj correlation: 1.000000"),
        stderr_tail=(),
    )
    if result.status != "QKV_SUBSTEP_SEQUENCE_PASS":
        raise AssertionError(result)
    if not _is_decode_qkv_substep_evidence(result.to_json_dict(), model_variant=DEFAULT_MODEL):
        raise AssertionError(result.to_json_dict())
    stale = dict(result.to_json_dict())
    stale["remaining_model_runner_gaps"] = ["full-qkv-substep-not-wired"]
    if _is_decode_qkv_substep_evidence(stale, model_variant=DEFAULT_MODEL):
        raise AssertionError(stale)
    print(result.format())
    print("GEMMA3_QKV_SUBSTEP_PROBE_SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma3 decode RMSNorm/QKV substep probe")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-hardware", action="store_true")
    parser.add_argument("--model-variant", choices=sorted(MODEL_SPECS), default=DEFAULT_MODEL)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--norm-tensor-key", default=DEFAULT_NORM_TENSOR_KEY)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    if not args.run_hardware:
        raise SystemExit("pass --run-hardware to touch the NPU; --self-test is hardware-free")
    result = _run_hardware_sequence(args)
    print(result.format())
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(result.to_json_dict(), indent=2, sort_keys=True) + "\n")
        print(f"GEMMA3_QKV_SUBSTEP_PROBE_JSON: {args.result_json}")
    return 0 if result.status == "QKV_SUBSTEP_SEQUENCE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
