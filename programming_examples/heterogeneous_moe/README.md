# Heterogeneous MoE

This example implements a fixed-shape, two-expert MoE runtime on top of the MLIR-AIR repo. It is designed as a v1 research harness rather than a polished product surface:

- Router math, expert 0, expert 1, and aggregation are independently assignable to `cpu`, `npu`, or `gpu`.
- `top1` and `top2` routing are supported.
- Top-k selection remains on CPU in v1. The configurable router stage computes logits.
- Transfer mode can be `host`, `peer`, or `auto`.

The default kernel shape is intentionally small so the generated kernels stay explicit and easy to inspect:

- Tokens: `4`
- Hidden size: `16`
- FFN size: `32`
- Datatype: `bf16`

## Status

- CPU path: verified for both `top1` and `top2`; the benchmark reports `max_abs_error = 0.0` against the NumPy reference.
- NPU path: verified against the NumPy `bf16` reference on the default shape. On this machine the current benchmark reports `max_abs_error = 1.25` for `top1` and `1.5` for `top2`.
- GPU path: verified for both `top1` and `top2`; the benchmark reports `max_abs_error = 0.0` on `gfx1150`.
- Mixed NPU+GPU path: verified with `expert0=npu`, `expert1=gpu`, `router=cpu`, `aggregation=cpu`; the current `top2` benchmark reports `max_abs_error = 1.5`, matching the NPU side.

## Files

- `default_manifest.json`: default runtime and compiler configuration.
- `kernels.py`: emits the NPU AIR sources under `air/` and the GPU MLIR sources under `air_gpu/`.
- `compile_kernels.py`: compiles the NPU AIR sources and the GPU MLIR sources, then updates the manifest with artifact paths.
- `bench.py`: runs the benchmark and writes optional trace/results files.
- `reference.py`: NumPy reference implementation and optional PyTorch validation.

## CPU Smoke Test

```bash
cd /home/cj/mlir-air/programming_examples/heterogeneous_moe
python3 bench.py --iterations 1 --warmup 0 \
  --router-backend cpu \
  --expert0-backend cpu \
  --expert1-backend cpu \
  --aggregation-backend cpu \
  --router-mode top2
```

## Compile Kernels

The generated source files live in two trees:

- `air/`: AIR sources for the NPU flow.
- `air_gpu/`: GPU MLIR sources for the iGPU flow.

```bash
cd /home/cj/mlir-air/programming_examples/heterogeneous_moe
source /home/cj/mlir-air/utils/env_setup.sh \
  /home/cj/mlir-air/install-both \
  /home/cj/mlir-air/sandbox/lib/python3.12/site-packages/mlir_aie \
  /home/cj/mlir-air/sandbox/lib/python3.12/site-packages/llvm-aie \
  /home/cj/mlir-air/llvm/install-amdgpu

python3 compile_kernels.py --backends npu gpu
```

## Example Configurations

CPU-only:

```bash
python3 bench.py --router-backend cpu --expert0-backend cpu --expert1-backend cpu --aggregation-backend cpu
```

NPU-only:

```bash
python3 bench.py --router-backend npu --expert0-backend npu --expert1-backend npu --aggregation-backend npu --prepare
python3 bench.py --router-backend npu --expert0-backend npu --expert1-backend npu --aggregation-backend npu
```

GPU-only:

```bash
python3 bench.py --router-backend gpu --expert0-backend gpu --expert1-backend gpu --aggregation-backend gpu --router-mode top2
```

Mixed NPU+GPU:

```bash
python3 bench.py \
  --router-backend cpu \
  --expert0-backend npu \
  --expert1-backend gpu \
  --aggregation-backend cpu \
  --router-mode top2 \
  --trace-out artifacts/traces/npu_gpu_top2.json
```

## Notes

- The bias terms are omitted in v1 to keep the AIR kernel interface within the current Python XRT backend argument limit.
- `peer` transfer mode currently models copy elision on CPU-facing edges and same-backend edges. Direct `npu <-> gpu` peer transfer is intentionally reported as unsupported in `peer` mode and falls back only in `auto`.
- PyTorch validation is optional. If `torch` is unavailable, the benchmark still runs using the NumPy reference path.
- The example defaults to `bf16` because that matches the Ryzen NPU data path. The harness does not require `ml_dtypes`; it marshals `bf16` buffers as raw `uint16` bit patterns when talking to the device runtime.
- The iGPU path is compiled into per-kernel shared libraries under `artifacts/gpu/` and invoked through `_mlir_ciface_*_host` entrypoints.
- The GPU compile step uses the local LLVM/MLIR toolchain from `LLVM_INSTALL_DIR` (or `mlir-opt` / `mlir-translate` on `PATH`) and strips the generated module dtors before linking the shared libraries so the Python `ctypes` path exits cleanly.
- The current AIR-to-ROCDL path in this checkout is still not reliable for these MoE kernels, so the working iGPU path uses direct GPU MLIR generation under `air_gpu/` instead of lowering the AIR kernels to ROCDL.
