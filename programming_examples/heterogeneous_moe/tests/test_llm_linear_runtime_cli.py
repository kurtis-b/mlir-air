# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import ctypes
import json
from pathlib import Path

import pytest
import numpy as np

import compile_llm_linear
import run_llm_linear_milestone2
import run_llm_linear_milestone3
import run_llm_linear_suite
from llm_linear.manifest import load_json, resolve_package_path
from llm_linear.direct_bridge import (
    DIRECT_CLASS_DEVICE_RESIDENT,
    DIRECT_CLASS_SHARED_HOST,
    DIRECT_CONTRACT,
    DirectBridgeMechanismReport,
    DirectBridgeProbeReport,
    DirectBridgeRunResult,
)
import llm_linear.direct_bridge as direct_bridge_module
from numerics import decode_npu_array, encode_npu_array
from llm_linear.reference import decode_gemv, prefill_gemm, random_inputs
from llm_linear.quantization import decode_gemv_fused_dequant
from llm_linear.results import (
    benchmark_runtime,
    build_case_result,
    correctness_failure_message,
    result_csv_row,
    validate_runtime,
)
from llm_linear.runtime import LinearRuntime, NpuLinearExecutor
from llm_linear.transfer import (
    DeviceResidentTensor,
    DirectTransferUnsupported,
    LinearTransferManager,
)

DIRECT_MECHANISM = "hip_vmem_export_xrt_bo_import_fd"


def _tiny_cpu_manifest(moe_dir: Path, tmp_path: Path) -> dict:
    manifest = load_json(moe_dir / "llm_linear" / "default_linear_manifest.json")
    manifest["model"].update({"M": 2, "K": 8, "H": 6, "N": 4, "dtype": "f16"})
    manifest["paths"]["artifacts"] = str(tmp_path / "artifacts")
    manifest["paths"]["generated_air_sources"] = str(tmp_path / "air")
    manifest["runtime"]["stage_backends"] = {"prefill": "cpu", "decode": "cpu"}
    manifest["runtime"]["transfer_mode"] = "host"
    return manifest


def _direct_probe_report(*, npu_verified: bool = False) -> DirectBridgeProbeReport:
    sync_events = [
        {"event": "hipDeviceSynchronize"},
        {"event": "xrtBoSyncToDevice"},
        {"event": "xrtRunWait"},
    ]
    mechanism = DirectBridgeMechanismReport(
        mechanism=DIRECT_MECHANISM,
        supported=True,
        direct_eligible=True,
        direct_class=DIRECT_CLASS_DEVICE_RESIDENT,
        ownership="hip_vmem",
        handle_type="posix_fd",
        import_view="xrt_bo",
        bidirectional_visibility=True,
        npu_kernel_verification=npu_verified,
        sync_events=sync_events if npu_verified else [],
        host_materialization_count=0,
        zero_host_copy=True,
        device_resident_buffers=True,
        diagnostic="ok",
    )
    return DirectBridgeProbeReport(
        contract=DIRECT_CONTRACT,
        direct_supported=True,
        selected_mechanism=DIRECT_MECHANISM,
        mechanisms=[mechanism],
        diagnostic="ok",
        library_path="fake_bridge.so",
    )


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


def test_linear_prefixed_output_paths_resolve_to_llm_linear_package(
    moe_dir: Path,
) -> None:
    assert (
        resolve_package_path("llm_linear/artifacts/benchmarks/example")
        == moe_dir / "llm_linear" / "artifacts" / "benchmarks" / "example"
    )


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


def test_linear_runtime_gpu_quantized_decode_receives_packed_buffers(
    monkeypatch, moe_dir: Path, tmp_path: Path
) -> None:
    manifest = _tiny_cpu_manifest(moe_dir, tmp_path)
    manifest["model"].update({"H": 8, "N": 8})
    manifest["weights"]["decode"] = {
        "storage": "int4",
        "block_size": 4,
        "quant_axis": 0,
    }
    manifest["runtime"]["stage_backends"] = {"prefill": "cpu", "decode": "gpu"}
    captured: dict[str, object] = {}

    class FakeGpuDecode:
        def __init__(
            self,
            kind,
            source,
            artifact,
            artifact_root_path,
            arch,
            dtype_name,
            decode_quantized=None,
            kernel_key=None,
        ):
            assert kind == "decode"
            assert kernel_key == "decode_int4"
            self.decode_quantized = decode_quantized
            self.dtype_name = dtype_name
            self.last_quantized_detail = None

        def run(self, decode_input, packed, scales):
            captured["packed_shape"] = packed.shape
            captured["packed_dtype"] = packed.dtype
            captured["scale_shape"] = scales.shape
            captured["scale_dtype"] = scales.dtype
            output, detail = decode_gemv_fused_dequant(
                decode_input,
                self.decode_quantized,
                self.dtype_name,
            )
            self.last_quantized_detail = detail
            return output

    import llm_linear.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "GpuLinearExecutor", FakeGpuDecode)
    with LinearRuntime(manifest) as runtime:
        inputs = random_inputs(runtime.cfg, seed=manifest["inputs"]["seed"])
        last_run, _validation_ms = validate_runtime(runtime, inputs)

    assert captured["packed_shape"] == (8, 1)
    assert captured["packed_dtype"] == np.dtype("uint32")
    assert captured["scale_shape"] == (2, 8)
    assert captured["scale_dtype"] == np.dtype("float32")
    assert last_run["numpy_validation"]["status"] == "pass"
    assert last_run["quantized_decode"]["hardware_fused"] is True
    labels = [event["label"] for event in last_run["transfer_events"]]
    assert "decode_packed_weights_to_backend" in labels
    assert "decode_scales_to_backend" in labels
    assert "decode_weights_to_backend" not in labels


def test_npu_prefill_executor_stages_multirow_inputs(tmp_path: Path) -> None:
    dtype = "bf16"
    inputs = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        dtype=np.float32,
    )
    weights = np.ones((4, 2), dtype=np.float32)
    staged_rows: list[np.ndarray] = []

    def fake_invoker(encoded_input, encoded_weights, output):
        decoded_input = decode_npu_array(np.asarray(encoded_input), dtype)
        staged_rows.append(decoded_input)
        row_sum = float(np.sum(decoded_input[0, :]))
        output_view = np.asarray(output).reshape((1, 2))
        output_view[...] = 0
        output_view[0, :] = encode_npu_array(
            np.asarray([[row_sum, row_sum + 1.0]], dtype=np.float32),
            dtype,
        )[0, :]
        return (encoded_input, encoded_weights, output)

    executor = NpuLinearExecutor(
        "prefill",
        Path("unused.mlir"),
        {},
        tmp_path,
        "npu2",
        dtype,
    )
    executor._invoker = fake_invoker

    output = executor.run(inputs, weights)

    expected = np.asarray([[10.0, 11.0], [26.0, 27.0], [42.0, 43.0]], dtype=np.float32)
    np.testing.assert_allclose(output, expected)
    assert len(staged_rows) == inputs.shape[0]
    for row, staged in enumerate(staged_rows):
        assert staged.shape == (1, inputs.shape[1])
        np.testing.assert_allclose(staged[0, :], inputs[row, :])


def test_linear_direct_transfer_fails_closed(moe_dir: Path, tmp_path: Path) -> None:
    manifest = _tiny_cpu_manifest(moe_dir, tmp_path)
    manifest["runtime"]["transfer_mode"] = "direct"
    with LinearRuntime(manifest) as runtime:
        inputs = random_inputs(runtime.cfg, seed=manifest["inputs"]["seed"])
        with pytest.raises(DirectTransferUnsupported, match="GPU/NPU"):
            runtime.run(inputs)


def test_direct_bridge_probe_report_supported(monkeypatch) -> None:
    payload = _direct_probe_report().to_dict()

    class FakeProbe:
        def __call__(self, buffer, capacity):
            encoded = json.dumps(payload).encode("utf-8")
            assert len(encoded) + 1 <= int(capacity)
            ctypes.memmove(buffer, encoded, len(encoded) + 1)
            return 0

    class FakeLibrary:
        llm_linear_direct_bridge_probe_report = FakeProbe()

    monkeypatch.setattr(
        direct_bridge_module,
        "_load_library",
        lambda path=None: (Path("fake_bridge.so"), FakeLibrary(), None),
    )

    status = direct_bridge_module.probe_direct_bridge()
    assert status.available is True
    assert status.probe_report is not None
    assert status.probe_report.contract == DIRECT_CONTRACT
    assert status.probe_report.selected_mechanism == DIRECT_MECHANISM
    selected = status.probe_report.selected_report()
    assert selected is not None
    assert selected.zero_host_copy is True
    assert selected.host_materialization_count == 0


def test_direct_bridge_probe_report_failed(monkeypatch) -> None:
    payload = {
        "schema_version": 1,
        "contract": DIRECT_CONTRACT,
        "direct_supported": False,
        "selected_mechanism": None,
        "diagnostic": "no audited path",
        "mechanisms": [
            {
                "mechanism": "numpy_host_staged_baseline",
                "supported": True,
                "direct_eligible": False,
                "direct_class": "host_staged_copy",
                "ownership": "host_numpy",
                "handle_type": "host_pointer",
                "import_view": "host_array",
                "bidirectional_visibility": True,
                "npu_kernel_verification": False,
                "sync_events": [],
                "host_materialization_count": 1,
                "zero_host_copy": False,
                "device_resident_buffers": False,
                "diagnostic": "baseline only",
            }
        ],
    }

    class FakeProbe:
        def __call__(self, buffer, capacity):
            encoded = json.dumps(payload).encode("utf-8")
            ctypes.memmove(buffer, encoded, len(encoded) + 1)
            return 1

    class FakeLibrary:
        llm_linear_direct_bridge_probe_report = FakeProbe()

    monkeypatch.setattr(
        direct_bridge_module,
        "_load_library",
        lambda path=None: (Path("fake_bridge.so"), FakeLibrary(), None),
    )

    status = direct_bridge_module.probe_direct_bridge()
    assert status.available is False
    assert status.probe_report is not None
    assert status.probe_report.direct_supported is False
    assert "no audited path" in status.diagnostic


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
        probe_report = _direct_probe_report()

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
            decode_packed_weights=None,
            decode_scales=None,
            output_buffer,
            prefill_output_buffer,
            decode_input_buffer,
            artifacts,
            decode_storage="dense",
            decode_block_size=0,
            decode_quant_axis=0,
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
            row_bytes = int(shape[2] * 2)
            row_offset = int((shape[0] - 1) * shape[2] * 2)
            return DirectBridgeRunResult(
                prefill_ms=1.0,
                decode_ms=2.0,
                handoff_us=0.0,
                direct_bytes=row_bytes,
                subview_offset_bytes=row_offset,
                mechanism="hip_vmem_export_xrt_bo_import_fd",
                bo_flag=0,
                import_method=3,
                sync_events=[
                    {"event": "hipDeviceSynchronize"},
                    {"event": "xrtBoSyncToDevice"},
                    {"event": "xrtBoSubview"},
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
    assert result["transfer_summary"]["direct_handoff"]["contract"] == DIRECT_CONTRACT
    assert result["transfer_summary"]["direct_handoff"]["mechanism"] == DIRECT_MECHANISM
    assert (
        result["transfer_summary"]["direct_handoff"]["direct_class"]
        == DIRECT_CLASS_DEVICE_RESIDENT
    )
    assert result["transfer_summary"]["direct_handoff"]["zero_host_copy"] is True
    probe_report = result["transfer_summary"]["direct_handoff"]["probe_report"]
    assert probe_report["selected_mechanism"] == DIRECT_MECHANISM
    selected = probe_report["mechanisms"][0]
    assert selected["bidirectional_visibility"] is True
    assert selected["npu_kernel_verification"] is True
    assert result["transfer_events"][0]["numpy_host_materializations"] == 0
    event = result["transfer_events"][0]
    assert event["bytes"] == manifest["model"]["H"] * 2
    assert event["direct_class"] == DIRECT_CLASS_DEVICE_RESIDENT
    assert event["zero_host_copy"] is True
    assert event["tensor"]["owner"] == "hip_vmem"
    assert event["tensor"]["imported_view"] == "xrt_bo"
    assert event["tensor"]["mechanism"] == DIRECT_MECHANISM
    assert event["tensor"]["direct_class"] == DIRECT_CLASS_DEVICE_RESIDENT
    assert (
        event["tensor"]["offset"]
        == (manifest["model"]["M"] - 1) * manifest["model"]["H"] * 2
    )
    sync_names = [item["event"] for item in event["sync_events"]]
    assert "xrtBoSubview" in sync_names
    assert "xrtBoCopy" not in sync_names


def test_linear_direct_runtime_records_quantized_decode_buffers(
    monkeypatch, moe_dir: Path, tmp_path: Path
) -> None:
    manifest = _tiny_cpu_manifest(moe_dir, tmp_path)
    manifest["model"].update({"H": 8, "N": 8})
    manifest["weights"]["decode"] = {
        "storage": "int4",
        "block_size": 4,
        "quant_axis": 0,
    }
    manifest["runtime"]["stage_backends"] = {"prefill": "gpu", "decode": "npu"}
    manifest["runtime"]["transfer_mode"] = "direct"
    manifest["artifacts"] = {
        "prefill": {"gpu_direct": {"so": "prefill.so"}},
        "decode_int4": {
            "npu": {"xclbin": "decode_int4.xclbin", "insts": "decode_int4.insts.bin"}
        },
    }
    captured: dict[str, object] = {}

    class FakeStatus:
        available = True
        library_path = "fake_bridge.so"
        diagnostic = "ok"
        probe_report = _direct_probe_report()

    class FakeBridge:
        library_path = Path("fake_bridge.so")

        def __init__(self, library_path=None):
            pass

        def run(self, **kwargs):
            captured.update(kwargs)
            dtype = kwargs["dtype"]
            shape = kwargs["shape"]
            x = decode_npu_array(kwargs["input_buffer"], dtype)
            wp = decode_npu_array(kwargs["prefill_weights"], dtype)
            prefill = prefill_gemm(x, wp, dtype)
            decode_input = prefill[-1, :]
            assert kwargs["decode_weights"] is None
            packed = kwargs["decode_packed_weights"]
            scales = kwargs["decode_scales"]
            assert packed.dtype == np.uint32
            assert packed.shape == (shape[2], shape[3] // 8)
            assert scales.dtype == np.float32
            output, _detail = decode_gemv_fused_dequant(
                decode_input,
                runtime.weights.decode_quantized,
                dtype,
            )
            kwargs["prefill_output_buffer"][...] = encode_npu_array(prefill, dtype)
            kwargs["decode_input_buffer"][...] = encode_npu_array(decode_input, dtype)
            kwargs["output_buffer"][...] = encode_npu_array(output, dtype)
            return DirectBridgeRunResult(
                prefill_ms=1.0,
                decode_ms=2.0,
                handoff_us=0.0,
                direct_bytes=int(shape[2] * 2),
                subview_offset_bytes=int((shape[0] - 1) * shape[2] * 2),
                mechanism=DIRECT_MECHANISM,
                bo_flag=0,
                import_method=3,
                sync_events=[
                    {"event": "hipDeviceSynchronize"},
                    {"event": "xrtBoSyncToDevice"},
                    {"event": "xrtBoSubview"},
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

    assert captured["decode_storage"] == "int4"
    assert captured["decode_block_size"] == 4
    assert captured["decode_quant_axis"] == 0
    assert result["numpy_validation"]["status"] == "pass"
    assert result["quantized_decode"]["enabled"] is True
    assert result["quantized_decode"]["quant_kind"] == "int4"
    assert result["quantized_decode"]["hardware_fused"] is True
    assert result["quantized_decode"]["packed_bytes"] == manifest["model"]["H"] * (
        manifest["model"]["N"] // 2
    )


def test_linear_direct_handoff_summary_schema() -> None:
    transfer = LinearTransferManager("direct")
    tensor = DeviceResidentTensor(
        owner="prefill",
        backend="gpu",
        dtype="uint16",
        shape=(4, 8),
        strides=(8, 1),
        byte_size=64,
        imported_view="xrt_bo",
        exported_handle_type="posix_fd",
        sync_state="gpu_event_recorded",
        trace_id="edge0",
        mechanism=DIRECT_MECHANISM,
        direct_class=DIRECT_CLASS_DEVICE_RESIDENT,
    )
    transfer.record_direct_handoff(
        producer="gpu",
        consumer="npu",
        tensor=tensor,
        elapsed_us=12.5,
        label="prefill_to_decode",
        mechanism=DIRECT_MECHANISM,
        sync_events=[{"producer": "gpu", "consumer": "npu"}],
        numpy_host_materializations=0,
        direct_class=DIRECT_CLASS_DEVICE_RESIDENT,
        probe_report=_direct_probe_report(npu_verified=True).to_dict(),
    )
    summary = transfer.summary()
    assert summary["model"] == "device_resident_direct_handoff"
    assert summary["device_resident_buffers"] is True
    assert summary["numpy_host_materializations"] == 0
    assert summary["direct_handoff"]["contract"] == DIRECT_CONTRACT
    assert summary["direct_handoff"]["mechanism"] == DIRECT_MECHANISM
    assert summary["direct_handoff"]["direct_class"] == DIRECT_CLASS_DEVICE_RESIDENT
    assert summary["direct_handoff"]["zero_host_copy"] is True
    assert summary["direct_handoff"]["edges"][0]["numpy_host_materializations"] == 0


def test_linear_direct_handoff_shared_host_zero_copy_summary() -> None:
    transfer = LinearTransferManager("direct")
    tensor = DeviceResidentTensor(
        owner="xrt_host_userptr",
        backend="gpu",
        dtype="uint16",
        shape=(8,),
        strides=(1,),
        byte_size=16,
        imported_view="hip_registered_host_pointer",
        exported_handle_type="host_pointer",
        mechanism="xrt_host_userptr_hip_registered_shared_host",
        direct_class=DIRECT_CLASS_SHARED_HOST,
        device_resident_buffers=False,
    )
    transfer.record_direct_handoff(
        producer="gpu",
        consumer="npu",
        tensor=tensor,
        elapsed_us=3.0,
        label="shared_host_probe",
        mechanism="xrt_host_userptr_hip_registered_shared_host",
        sync_events=[],
        numpy_host_materializations=0,
        direct_class=DIRECT_CLASS_SHARED_HOST,
        zero_host_copy=True,
        device_resident_buffers=False,
    )
    summary = transfer.summary()
    assert summary["direct_handoff"]["supported"] is True
    assert summary["direct_handoff"]["zero_host_copy"] is True
    assert summary["direct_handoff"]["direct_class"] == DIRECT_CLASS_SHARED_HOST
    assert summary["device_resident_buffers"] is False
    assert summary["model"] == "zero_host_copy_direct_handoff"


def test_linear_direct_transfer_rejects_host_staged_fallback() -> None:
    transfer = LinearTransferManager("direct")
    with pytest.raises(DirectTransferUnsupported, match="refuses host-staged"):
        transfer.transfer(
            "gpu",
            "npu",
            np.zeros((2,), dtype=np.float32),
            trace=None,
            label="gpu_to_npu",
        )


def test_milestone2_verifier_direct_contract(tmp_path: Path) -> None:
    probe_report = _direct_probe_report(npu_verified=True).to_dict()
    result = {
        "case_name": "gpu_prefill_npu_decode_direct",
        "dtype": "bf16",
        "shape": {"M": 4, "K": 128, "H": 128, "N": 64},
        "stage_backends": {"prefill": "gpu", "decode": "npu"},
        "correctness": {
            "validation_status": "pass",
            "prefill_allclose": True,
            "output_allclose": True,
        },
        "transfer_summary": {
            "numpy_host_materializations": 0,
            "direct_handoff_numpy_host_materializations": 0,
            "direct_handoff": {
                "supported": True,
                "contract": DIRECT_CONTRACT,
                "mechanism": DIRECT_MECHANISM,
                "direct_class": DIRECT_CLASS_DEVICE_RESIDENT,
                "zero_host_copy": True,
                "device_resident_buffers": True,
                "probe_report": probe_report,
            },
        },
        "execution_truth": {"numpy_host_materializations": 0},
        "direct_bridge": {
            "mechanism": DIRECT_MECHANISM,
            "direct_class": DIRECT_CLASS_DEVICE_RESIDENT,
            "zero_host_copy": True,
            "device_resident_buffers": True,
            "import_method": 3,
            "subview_offset_bytes": 768,
            "probe_report": probe_report,
        },
        "transfer_events": [
            {
                "actual_mode": "device_resident_direct_handoff",
                "mechanism": DIRECT_MECHANISM,
                "direct_class": DIRECT_CLASS_DEVICE_RESIDENT,
                "zero_host_copy": True,
                "device_resident_buffers": True,
                "bytes": 256,
                "numpy_host_materializations": 0,
                "tensor": {
                    "owner": "hip_vmem",
                    "imported_view": "xrt_bo",
                    "mechanism": DIRECT_MECHANISM,
                    "direct_class": DIRECT_CLASS_DEVICE_RESIDENT,
                    "zero_host_copy": True,
                    "byte_size": 256,
                    "offset": 768,
                },
                "sync_events": [
                    {"event": "hipDeviceSynchronize"},
                    {"event": "xrtBoSyncToDevice"},
                    {"event": "xrtBoSubview"},
                    {"event": "xrtRunWait"},
                ],
            }
        ],
    }
    assert (
        run_llm_linear_milestone2.validate_result_payload(result, require_direct=True)
        == []
    )

    stale = copy.deepcopy(result)
    stale["transfer_events"][0]["sync_events"][2] = {"event": "xrtBoCopy"}
    assert any(
        "xrtBoCopy" in message
        for message in run_llm_linear_milestone2.validate_result_payload(
            stale, require_direct=True
        )
    )

    log_path = tmp_path / "acceptance.log"
    log_path.write_text(
        "warning: Reverting to host copy of buffers (exec_buf: Operation not supported)\n",
        encoding="utf-8",
    )
    assert set(run_llm_linear_milestone2.scan_log_for_blockers(log_path)) == {
        "Reverting to host copy of buffers",
        "exec_buf: Operation not supported",
    }


def test_milestone3_verifier_requires_hardware_fused_int4() -> None:
    result = {
        "case_name": "gpu_only",
        "dtype": "bf16",
        "shape": {"M": 4, "K": 128, "H": 128, "N": 64},
        "stage_backends": {"prefill": "gpu", "decode": "gpu"},
        "correctness": {
            "validation_status": "pass",
            "prefill_allclose": True,
            "output_allclose": True,
        },
        "quantized_decode": {
            "enabled": True,
            "quant_kind": "int4",
            "hardware_fused": True,
            "packed_bytes": 4096,
            "scale_bytes": 1024,
            "metadata": {"quant_kind": "int4", "block_size": 32, "quant_axis": 0},
        },
    }
    assert (
        run_llm_linear_milestone3.validate_result_payload(result, require_direct=False)
        == []
    )
    stale = copy.deepcopy(result)
    stale["quantized_decode"]["hardware_fused"] = False
    assert any(
        "hardware_fused" in message
        for message in run_llm_linear_milestone3.validate_result_payload(
            stale, require_direct=False
        )
    )


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
