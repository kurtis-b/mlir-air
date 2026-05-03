# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

import compile_llm_linear
import run_llm_linear_suite
from llm_linear.manifest import load_json
from llm_linear.direct_bridge import DirectBridgeRunResult
from numerics import decode_npu_array, encode_npu_array
from llm_linear.reference import decode_gemv, prefill_gemm, random_inputs
from llm_linear.results import (
    benchmark_runtime,
    build_case_result,
    correctness_failure_message,
    result_csv_row,
    validate_runtime,
)
from llm_linear.runtime import LinearRuntime
from llm_linear.transfer import (
    DeviceResidentTensor,
    DirectTransferUnsupported,
    LinearTransferManager,
)


def _tiny_cpu_manifest(moe_dir: Path, tmp_path: Path) -> dict:
    manifest = load_json(moe_dir / "llm_linear" / "default_linear_manifest.json")
    manifest["model"].update({"M": 2, "K": 8, "H": 6, "N": 4, "dtype": "f16"})
    manifest["paths"]["artifacts"] = str(tmp_path / "artifacts")
    manifest["paths"]["generated_air_sources"] = str(tmp_path / "air")
    manifest["runtime"]["stage_backends"] = {"prefill": "cpu", "decode": "cpu"}
    manifest["runtime"]["transfer_mode"] = "host"
    return manifest


def test_linear_runtime_result_schema(moe_dir: Path, tmp_path: Path) -> None:
    manifest = _tiny_cpu_manifest(moe_dir, tmp_path)
    with LinearRuntime(manifest) as runtime:
        inputs = random_inputs(runtime.cfg, seed=manifest["inputs"]["seed"])
        timing = benchmark_runtime(
            runtime, inputs, iterations=2, warmup=1, measurement_mode="both"
        )
        last_run, validation_ms = validate_runtime(runtime, inputs)
    result = build_case_result(
        metadata={"command_line": ["test"]},
        case_name="cpu_only",
        manifest=manifest,
        iterations=2,
        requested_warmup=1,
        measurement_mode="both",
        timing=timing,
        last_run=last_run,
        validation_ms=validation_ms,
        phase_timings_ms={"compile_load_setup_ms": 0.0, "input_generation_ms": 0.0},
        suite="tiny_ci",
        workload_name="unit",
    )
    assert result["measurement"]["validation_timed"] is False
    assert result["measurement"]["compile_load_excluded"] is True
    assert result["stage_latency_ms"]["prefill"]["mean"] >= 0.0
    assert result["transfer_summary"]["device_resident_buffers"] is False
    assert result["correctness"]["validation_status"] == "pass"
    assert correctness_failure_message(result) is None
    row = result_csv_row(result)
    assert row["case_name"] == "cpu_only"
    assert row["validation_status"] == "pass"


def test_linear_runtime_quantized_decode_schema(moe_dir: Path, tmp_path: Path) -> None:
    manifest = _tiny_cpu_manifest(moe_dir, tmp_path)
    manifest["weights"]["decode"] = {
        "storage": "int4",
        "block_size": 4,
        "quant_axis": 0,
    }
    with LinearRuntime(manifest) as runtime:
        inputs = random_inputs(runtime.cfg, seed=manifest["inputs"]["seed"])
        last_run, _validation_ms = validate_runtime(runtime, inputs)
    assert last_run["numpy_validation"]["status"] == "pass"
    assert last_run["quantized_decode"]["enabled"] is True
    assert last_run["quantized_decode"]["detail"]["dequant_ms"] >= 0.0


def test_linear_direct_transfer_fails_closed(moe_dir: Path, tmp_path: Path) -> None:
    manifest = _tiny_cpu_manifest(moe_dir, tmp_path)
    manifest["runtime"]["transfer_mode"] = "direct"
    with LinearRuntime(manifest) as runtime:
        inputs = random_inputs(runtime.cfg, seed=manifest["inputs"]["seed"])
        with pytest.raises(DirectTransferUnsupported, match="GPU/NPU"):
            runtime.run(inputs)


def test_linear_direct_runtime_records_native_handoff(
    monkeypatch, moe_dir: Path, tmp_path: Path
) -> None:
    manifest = _tiny_cpu_manifest(moe_dir, tmp_path)
    manifest["runtime"]["stage_backends"] = {"prefill": "gpu", "decode": "npu"}
    manifest["runtime"]["transfer_mode"] = "direct"
    manifest["artifacts"] = {
        "prefill": {"gpu_direct": {"so": "prefill.so"}},
        "decode": {"npu": {"xclbin": "decode.xclbin", "insts": "decode.insts.bin"}},
    }

    class FakeStatus:
        available = True
        library_path = "fake_bridge.so"
        diagnostic = "ok"

    class FakeBridge:
        library_path = Path("fake_bridge.so")

        def __init__(self, library_path=None):
            pass

        def run(
            self,
            *,
            direction,
            dtype,
            shape,
            input_buffer,
            prefill_weights,
            decode_weights,
            output_buffer,
            prefill_output_buffer,
            decode_input_buffer,
            artifacts,
        ):
            assert direction == "gpu_prefill_npu_decode"
            x = decode_npu_array(input_buffer, dtype)
            wp = decode_npu_array(prefill_weights, dtype)
            wd = decode_npu_array(decode_weights, dtype)
            prefill = prefill_gemm(x, wp, dtype)
            decode_input = prefill[-1, :]
            output = decode_gemv(decode_input, wd, dtype)
            prefill_output_buffer[...] = encode_npu_array(prefill, dtype)
            decode_input_buffer[...] = encode_npu_array(decode_input, dtype)
            output_buffer[...] = encode_npu_array(output, dtype)
            return DirectBridgeRunResult(
                prefill_ms=1.0,
                decode_ms=2.0,
                handoff_us=0.0,
                direct_bytes=int(prefill_output_buffer.nbytes),
                subview_offset_bytes=int((shape[0] - 1) * shape[2] * 2),
                mechanism="xrt_bo_export_import_hip_vmem_fd:test",
                bo_flag=0,
                import_method=2,
                sync_events=[
                    {"event": "hipDeviceSynchronize"},
                    {"event": "xrtRunWait"},
                ],
                diagnostic="ok",
            )

    import llm_linear.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "probe_direct_bridge", lambda: FakeStatus())
    monkeypatch.setattr(runtime_module, "DirectBridge", FakeBridge)
    with LinearRuntime(manifest) as runtime:
        inputs = random_inputs(runtime.cfg, seed=manifest["inputs"]["seed"])
        result = runtime.run(inputs, validate=True)
    assert result["numpy_validation"]["status"] == "pass"
    assert result["transfer_summary"]["device_resident_buffers"] is True
    assert result["transfer_summary"]["direct_handoff"]["supported"] is True
    assert result["transfer_events"][0]["numpy_host_materializations"] == 0


def test_linear_direct_handoff_summary_schema() -> None:
    transfer = LinearTransferManager("direct")
    tensor = DeviceResidentTensor(
        owner="prefill",
        backend="gpu",
        dtype="uint16",
        shape=(4, 8),
        strides=(8, 1),
        byte_size=64,
        exported_handle_type="dma_buf_fd",
        sync_state="gpu_event_recorded",
        trace_id="edge0",
    )
    transfer.record_direct_handoff(
        producer="gpu",
        consumer="npu",
        tensor=tensor,
        elapsed_us=12.5,
        label="prefill_to_decode",
        mechanism="xrt_bo_export_import_hip_vmem_fd",
        sync_events=[{"producer": "gpu", "consumer": "npu"}],
        numpy_host_materializations=0,
    )
    summary = transfer.summary()
    assert summary["model"] == "device_resident_direct_handoff"
    assert summary["device_resident_buffers"] is True
    assert summary["numpy_host_materializations"] == 0
    assert summary["direct_handoff"]["edges"][0]["numpy_host_materializations"] == 0


def test_compile_llm_linear_cli(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_populate(manifest, backends):
        captured["backends"] = backends
        manifest["artifacts"]["prefill"]["gpu"] = {"so": "prefill.so"}
        return manifest

    monkeypatch.setattr(compile_llm_linear, "populate_artifacts", fake_populate)
    out_path = tmp_path / "compiled.json"
    assert (
        compile_llm_linear.main(["--backends", "gpu", "--manifest-out", str(out_path)])
        == 0
    )
    assert captured["backends"] == {"gpu"}
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["artifacts"]["prefill"]["gpu"]["so"] == "prefill.so"


def test_run_llm_linear_suite_cpu_filter_and_direct_failure(tmp_path: Path) -> None:
    out_dir = tmp_path / "suite"
    assert (
        run_llm_linear_suite.main(
            [
                "--suite",
                "tiny_ci",
                "--workload-filter",
                "tiny_m1",
                "--case-filter",
                "cpu_only",
                "--iterations",
                "1",
                "--warmup",
                "0",
                "--require-correctness",
                "--output-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["workloads"][0]["cases"][0]["case_name"] == "cpu_only"
    assert (out_dir / "summary.csv").read_text(encoding="utf-8").startswith("suite,")
    assert "audited device-resident buffers" in (out_dir / "report.md").read_text(
        encoding="utf-8"
    )

    with pytest.raises(SystemExit, match="GPU/NPU"):
        run_llm_linear_suite.main(["--transfer-mode", "direct"])
