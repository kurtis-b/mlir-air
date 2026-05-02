# SPDX-License-Identifier: MIT

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from compile import resolve_air_sources
from executors import CpuExecutor, GpuExecutor, NpuExecutor, StageExecutors
from kernels import KernelConfig
from manifest import artifact_root, load_json
from numerics import encoded_array_summary
from reference import (
    DEFAULT_INPUT_SCALE,
    DEFAULT_ROUTING_PROFILE,
    DEFAULT_WEIGHT_SCALE,
    optional_torch_validation,
    pack_expert_outputs,
    random_weights,
    routed_inputs,
    run_reference,
    topk_weights,
)
from results import edge_study_limitations, stage_metrics
from trace import TraceRecorder, summarize_device_events
from transfer import TransferManager


class MoERuntime:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = manifest
        self.cfg = KernelConfig(
            batch_tokens=manifest["model"]["batch_tokens"],
            hidden_size=manifest["model"]["hidden_size"],
            ffn_size=manifest["model"]["ffn_size"],
            dtype=manifest["model"]["dtype"],
        )
        self.input_scale = float(manifest.get("inputs", {}).get("scale", DEFAULT_INPUT_SCALE))
        self.weight_scale = float(manifest.get("weights", {}).get("scale", DEFAULT_WEIGHT_SCALE))
        self.routing_profile = manifest.get("workload", {}).get("routing_profile", DEFAULT_ROUTING_PROFILE)
        self.weights = random_weights(
            self.cfg,
            manifest["weights"]["seed"],
            scale=self.weight_scale,
            routing_profile=self.routing_profile,
        )
        self.expert_weights = {
            "expert0": (self.weights.expert0_w1, self.weights.expert0_w2),
            "expert1": (self.weights.expert1_w1, self.weights.expert1_w2),
        }
        self.transfer = TransferManager(manifest["runtime"]["transfer_mode"])
        self.artifact_root = artifact_root(manifest)
        self._sources: dict[str, dict[str, Path]] = {}
        self.executors = self._make_executors()
        self._expert_pool = ThreadPoolExecutor(max_workers=2)

    def logical_batch_tokens(self) -> int:
        return int(self.manifest.get("workload", {}).get("routed_tokens", self.cfg.batch_tokens))

    def _sources_for(self, backend: str) -> dict[str, Path]:
        if backend not in self._sources:
            self._sources[backend] = resolve_air_sources(self.manifest, backend)
        return self._sources[backend]

    def _make_executor(self, kind: str, backend: str) -> Any:
        if backend == "cpu":
            return CpuExecutor(kind, self.cfg.dtype)

        artifact = self.manifest["artifacts"].get(kind, {}).get(backend, {})
        source = self._sources_for(backend)[kind]
        if backend == "npu":
            return NpuExecutor(
                kind,
                source,
                artifact,
                self.artifact_root,
                self.manifest["compiler"]["npu_device"],
                self.cfg.dtype,
                self.cfg,
            )
        if backend == "gpu":
            function_name = {
                "router": "router_math",
                "expert": "expert_mlp",
                "aggregation": "aggregate_outputs",
            }[kind]
            return GpuExecutor(
                kind,
                source,
                artifact,
                self.artifact_root,
                self.manifest["compiler"]["gpu_arch"],
                function_name,
                self.cfg.dtype,
            )
        raise ValueError(f"Unsupported backend: {backend}")

    def _make_executors(self) -> StageExecutors:
        stages = self.manifest["runtime"]["stage_backends"]
        return StageExecutors(
            router=self._make_executor("router", stages["router"]),
            expert0=self._make_executor("expert", stages["expert0"]),
            expert1=self._make_executor("expert", stages["expert1"]),
            aggregation=self._make_executor("aggregation", stages["aggregation"]),
        )

    def prepare(self) -> None:
        for executor in (
            self.executors.router,
            self.executors.expert0,
            self.executors.expert1,
            self.executors.aggregation,
        ):
            prepare = getattr(executor, "prepare", None)
            if prepare:
                prepare()

    def _npu_stage_executed(self) -> bool:
        return any(backend == "npu" for backend in self.manifest["runtime"]["stage_backends"].values())

    def _npu_sources_report(self) -> dict[str, str]:
        if "npu" not in self._sources:
            return {}
        return {name: str(path) for name, path in self._sources["npu"].items()}

    def _npu_development_report(
        self,
        inputs: np.ndarray,
        reference: dict[str, np.ndarray],
        route_weights: np.ndarray,
        packed_experts: np.ndarray,
        *,
        executed: bool,
    ) -> dict[str, Any]:
        notes = [
            "Includes buffer layout, encoded dtypes, and selected artifact/source paths."
        ]
        if executed:
            notes.append("NPU execution completed for this run.")
        else:
            notes.append("Host-side report only: no NPU execution was performed.")
        return {
            "executed": bool(executed),
            "dtype": self.cfg.dtype,
            "device": self.manifest["compiler"]["npu_device"],
            "input_scale": self.input_scale,
            "weight_scale": self.weight_scale,
            "routing_profile": self.routing_profile,
            "sources": self._npu_sources_report(),
            "artifacts": self.manifest.get("artifacts", {}),
            "router": {
                "input": encoded_array_summary(inputs, self.cfg.dtype),
                "weights": encoded_array_summary(self.weights.router, self.cfg.dtype),
                "expected_output": encoded_array_summary(reference["logits"], self.cfg.dtype),
            },
            "expert0": {
                "input": encoded_array_summary(reference["expert0_input"], self.cfg.dtype),
                "w1": encoded_array_summary(self.expert_weights["expert0"][0], self.cfg.dtype),
                "w2": encoded_array_summary(self.expert_weights["expert0"][1], self.cfg.dtype),
                "expected_output": encoded_array_summary(reference["expert0_output"], self.cfg.dtype),
            },
            "expert1": {
                "input": encoded_array_summary(reference["expert1_input"], self.cfg.dtype),
                "w1": encoded_array_summary(self.expert_weights["expert1"][0], self.cfg.dtype),
                "w2": encoded_array_summary(self.expert_weights["expert1"][1], self.cfg.dtype),
                "expected_output": encoded_array_summary(reference["expert1_output"], self.cfg.dtype),
            },
            "aggregation": {
                "input": encoded_array_summary(packed_experts, self.cfg.dtype),
                "weights": encoded_array_summary(route_weights, self.cfg.dtype),
                "expected_output": encoded_array_summary(reference["output"], self.cfg.dtype),
            },
            "notes": [
                *notes,
                "Use this report to verify packed buffer layout, encoded dtypes, and artifact/source selection.",
            ],
        }

    def _run_expert(
        self,
        executor: Any,
        inputs: np.ndarray,
        w1: np.ndarray,
        w2: np.ndarray,
        trace: TraceRecorder | None,
        name: str,
        backend: str,
    ) -> np.ndarray:
        if trace is None:
            return executor.run(inputs, w1, w2)
        with trace.span(name, "stage", name, {"backend": backend}):
            return executor.run(inputs, w1, w2)

    def _run_single(
        self,
        inputs: np.ndarray,
        router_mode: str | None = None,
        *,
        validate: bool = True,
        capture_details: bool = True,
    ) -> dict[str, Any]:
        router_mode = router_mode or self.manifest["runtime"]["router_mode"]
        stages = self.manifest["runtime"]["stage_backends"]
        trace = TraceRecorder() if capture_details else None

        if trace is None:
            logits = self.executors.router.run(inputs, self.weights.router)
        else:
            with trace.span("router_math", "stage", "router", {"backend": stages["router"]}):
                logits = self.executors.router.run(inputs, self.weights.router)
        logits_cpu = self.transfer.transfer(stages["router"], "cpu", logits, trace, "router_to_cpu")

        if trace is None:
            route_weights = topk_weights(logits_cpu, router_mode, self.cfg.dtype)
            expert0_in, expert1_in = routed_inputs(inputs, route_weights, self.cfg.dtype)
        else:
            with trace.span("topk_select", "control", "cpu", {"mode": router_mode}):
                route_weights = topk_weights(logits_cpu, router_mode, self.cfg.dtype)
                expert0_in, expert1_in = routed_inputs(inputs, route_weights, self.cfg.dtype)

        expert0_arg = self.transfer.transfer("cpu", stages["expert0"], expert0_in, trace, "cpu_to_expert0")
        expert1_arg = self.transfer.transfer("cpu", stages["expert1"], expert1_in, trace, "cpu_to_expert1")
        expert0_w1 = self.transfer.transfer(
            "cpu",
            stages["expert0"],
            self.expert_weights["expert0"][0],
            trace,
            "expert0_w1_to_backend",
        )
        expert0_w2 = self.transfer.transfer(
            "cpu",
            stages["expert0"],
            self.expert_weights["expert0"][1],
            trace,
            "expert0_w2_to_backend",
        )
        expert1_w1 = self.transfer.transfer(
            "cpu",
            stages["expert1"],
            self.expert_weights["expert1"][0],
            trace,
            "expert1_w1_to_backend",
        )
        expert1_w2 = self.transfer.transfer(
            "cpu",
            stages["expert1"],
            self.expert_weights["expert1"][1],
            trace,
            "expert1_w2_to_backend",
        )

        future0 = self._expert_pool.submit(
            self._run_expert,
            self.executors.expert0,
            expert0_arg,
            expert0_w1,
            expert0_w2,
            trace,
            "expert0",
            stages["expert0"],
        )
        future1 = self._expert_pool.submit(
            self._run_expert,
            self.executors.expert1,
            expert1_arg,
            expert1_w1,
            expert1_w2,
            trace,
            "expert1",
            stages["expert1"],
        )
        expert0_out = future0.result()
        expert1_out = future1.result()

        aggregation_backend = stages["aggregation"]
        if trace is None:
            packed_experts = pack_expert_outputs(expert0_out, expert1_out, self.cfg.dtype)
        else:
            with trace.span("pack_aggregation_inputs", "control", "cpu", {"source0": stages["expert0"], "source1": stages["expert1"]}):
                packed_experts = pack_expert_outputs(expert0_out, expert1_out, self.cfg.dtype)
        agg_experts = self.transfer.transfer("cpu", aggregation_backend, packed_experts, trace, "experts_to_aggregation")
        agg_weights = self.transfer.transfer("cpu", aggregation_backend, route_weights, trace, "weights_to_aggregation")

        if trace is None:
            output = self.executors.aggregation.run(agg_experts, agg_weights)
        else:
            with trace.span("aggregation", "stage", "aggregation", {"backend": aggregation_backend}):
                output = self.executors.aggregation.run(agg_experts, agg_weights)
        output_cpu = self.transfer.transfer(aggregation_backend, "cpu", output, trace, "aggregation_to_cpu")

        if not capture_details:
            return {"output": output_cpu}

        actual_bundle = {
            "logits": logits_cpu,
            "weights": route_weights,
            "expert0_input": expert0_in,
            "expert1_input": expert1_in,
            "expert0_output": expert0_out,
            "expert1_output": expert1_out,
            "packed_expert_outputs": packed_experts,
            "output": output_cpu,
        }
        max_abs_error = None
        torch_validation = {"ran": False, "ok": False, "message": "skipped"}
        if validate:
            reference = run_reference(self.cfg, inputs, self.weights, router_mode)
            per_stage_metrics = stage_metrics(actual_bundle, reference)
            max_abs_error = float(per_stage_metrics["output"]["max_abs_error"])
            torch_validation = optional_torch_validation(
                inputs,
                self.weights,
                router_mode,
                self.cfg.dtype,
                actual=actual_bundle,
                quantized_reference=reference,
            )
        else:
            reference = {name: np.asarray(value, dtype=np.float32) for name, value in actual_bundle.items()}
            per_stage_metrics = {}
        npu_dev_report = self._npu_development_report(
            inputs,
            reference,
            route_weights,
            packed_experts,
            executed=self._npu_stage_executed(),
        )
        assert trace is not None
        trace_summary = trace.summary()
        transfer_events = self.transfer.snapshot()
        transfer_summary = self.transfer.summary()

        return {
            "inputs": inputs,
            "logits": logits_cpu,
            "weights": route_weights,
            "expert0_input": expert0_in,
            "expert1_input": expert1_in,
            "expert0_output": expert0_out,
            "expert1_output": expert1_out,
            "packed_expert_outputs": packed_experts,
            "output": output_cpu,
            "reference": reference["output"],
            "max_abs_error": max_abs_error,
            "workload": {
                "shape": {
                    "batch_tokens": self.cfg.batch_tokens,
                    "hidden_size": self.cfg.hidden_size,
                    "ffn_size": self.cfg.ffn_size,
                    "dtype": self.cfg.dtype,
                },
                "routing_profile": self.routing_profile,
                "input_scale": self.input_scale,
                "weight_scale": self.weight_scale,
                "kernel_chunk_tokens": self.cfg.batch_tokens,
                "context_length": self.manifest.get("workload", {}).get("context_length"),
                "routed_tokens": int(inputs.shape[0]),
                "chunk_count": 1,
            },
            "stage_metrics": per_stage_metrics,
            "torch_validation": torch_validation,
            "npu_development": npu_dev_report,
            "trace": trace,
            "trace_summary": trace_summary,
            "transfer_events": transfer_events,
            "transfer_summary": transfer_summary,
            "device_events": summarize_device_events(trace),
            "limitations": edge_study_limitations(self.manifest, transfer_summary),
        }

    def run(
        self,
        inputs: np.ndarray,
        router_mode: str | None = None,
        *,
        validate: bool = True,
        capture_details: bool = True,
    ) -> dict[str, Any]:
        if capture_details:
            self.transfer.reset_events()
        if inputs.shape[0] == self.cfg.batch_tokens:
            return self._run_single(inputs, router_mode, validate=validate, capture_details=capture_details)

        if not capture_details:
            router_mode = router_mode or self.manifest["runtime"]["router_mode"]
            total_tokens = int(inputs.shape[0])
            chunk_tokens = self.cfg.batch_tokens
            start = 0
            while start < total_tokens:
                end = min(start + chunk_tokens, total_tokens)
                valid_tokens = end - start
                chunk = np.asarray(inputs[start:end], dtype=np.float32)
                if valid_tokens < chunk_tokens:
                    padded = np.zeros((chunk_tokens, self.cfg.hidden_size), dtype=np.float32)
                    padded[:valid_tokens] = chunk
                    chunk = padded
                self._run_single(chunk, router_mode, validate=False, capture_details=False)
                start = end
            return {"output": None}

        router_mode = router_mode or self.manifest["runtime"]["router_mode"]
        total_tokens = int(inputs.shape[0])
        chunk_tokens = self.cfg.batch_tokens
        merged_trace = TraceRecorder()
        trace_offset_us = 0.0
        chunk_results: list[dict[str, Any]] = []
        actual_chunks: dict[str, list[np.ndarray]] = {
            "logits": [],
            "weights": [],
            "expert0_input": [],
            "expert1_input": [],
            "expert0_output": [],
            "expert1_output": [],
            "packed_expert_outputs": [],
            "output": [],
        }

        start = 0
        while start < total_tokens:
            end = min(start + chunk_tokens, total_tokens)
            valid_tokens = end - start
            chunk = np.asarray(inputs[start:end], dtype=np.float32)
            if valid_tokens < chunk_tokens:
                padded = np.zeros((chunk_tokens, self.cfg.hidden_size), dtype=np.float32)
                padded[:valid_tokens] = chunk
                chunk = padded
            chunk_result = self._run_single(chunk, router_mode, validate=False)
            chunk_results.append(chunk_result)
            merged_trace.extend(chunk_result["trace"].snapshot(), ts_offset_us=trace_offset_us)
            trace_offset_us += float(chunk_result["trace_summary"]["span_us"]) + 1.0

            actual_chunks["logits"].append(np.asarray(chunk_result["logits"][:valid_tokens], dtype=np.float32))
            actual_chunks["weights"].append(np.asarray(chunk_result["weights"][:valid_tokens], dtype=np.float32))
            actual_chunks["expert0_input"].append(np.asarray(chunk_result["expert0_input"][:valid_tokens], dtype=np.float32))
            actual_chunks["expert1_input"].append(np.asarray(chunk_result["expert1_input"][:valid_tokens], dtype=np.float32))
            actual_chunks["expert0_output"].append(np.asarray(chunk_result["expert0_output"][:valid_tokens], dtype=np.float32))
            actual_chunks["expert1_output"].append(np.asarray(chunk_result["expert1_output"][:valid_tokens], dtype=np.float32))
            actual_chunks["packed_expert_outputs"].append(
                np.asarray(chunk_result["packed_expert_outputs"][:valid_tokens], dtype=np.float32)
            )
            actual_chunks["output"].append(np.asarray(chunk_result["output"][:valid_tokens], dtype=np.float32))
            start = end

        actual_bundle = {name: np.concatenate(chunks, axis=0) for name, chunks in actual_chunks.items()}
        logical_cfg = KernelConfig(
            batch_tokens=total_tokens,
            hidden_size=self.cfg.hidden_size,
            ffn_size=self.cfg.ffn_size,
            dtype=self.cfg.dtype,
        )
        if validate:
            reference = run_reference(logical_cfg, inputs, self.weights, router_mode)
            per_stage_metrics = stage_metrics(actual_bundle, reference)
            max_abs_error = float(per_stage_metrics["output"]["max_abs_error"])
            torch_validation = optional_torch_validation(
                inputs,
                self.weights,
                router_mode,
                self.cfg.dtype,
                actual=actual_bundle,
                quantized_reference=reference,
            )
        else:
            reference = {name: np.asarray(value, dtype=np.float32) for name, value in actual_bundle.items()}
            per_stage_metrics = {}
            max_abs_error = None
            torch_validation = {"ran": False, "ok": False, "message": "skipped"}
        npu_dev_report = self._npu_development_report(
            inputs,
            reference,
            actual_bundle["weights"],
            actual_bundle["packed_expert_outputs"],
            executed=self._npu_stage_executed(),
        )
        transfer_events = self.transfer.snapshot()
        transfer_summary = self.transfer.summary()

        return {
            "inputs": inputs,
            "logits": actual_bundle["logits"],
            "weights": actual_bundle["weights"],
            "expert0_input": actual_bundle["expert0_input"],
            "expert1_input": actual_bundle["expert1_input"],
            "expert0_output": actual_bundle["expert0_output"],
            "expert1_output": actual_bundle["expert1_output"],
            "packed_expert_outputs": actual_bundle["packed_expert_outputs"],
            "output": actual_bundle["output"],
            "reference": reference["output"],
            "max_abs_error": max_abs_error,
            "workload": {
                "shape": {
                    "batch_tokens": total_tokens,
                    "hidden_size": self.cfg.hidden_size,
                    "ffn_size": self.cfg.ffn_size,
                    "dtype": self.cfg.dtype,
                },
                "routing_profile": self.routing_profile,
                "input_scale": self.input_scale,
                "weight_scale": self.weight_scale,
                "kernel_chunk_tokens": chunk_tokens,
                "context_length": self.manifest.get("workload", {}).get("context_length"),
                "routed_tokens": total_tokens,
                "chunk_count": len(chunk_results),
            },
            "stage_metrics": per_stage_metrics,
            "torch_validation": torch_validation,
            "npu_development": npu_dev_report,
            "trace": merged_trace,
            "trace_summary": merged_trace.summary(),
            "transfer_events": transfer_events,
            "transfer_summary": transfer_summary,
            "device_events": summarize_device_events(merged_trace),
            "limitations": edge_study_limitations(self.manifest, transfer_summary),
        }


def load_runtime(manifest_path: Path) -> MoERuntime:
    manifest = load_json(manifest_path)
    return MoERuntime(manifest)
