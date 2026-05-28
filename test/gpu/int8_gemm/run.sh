#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TMPDIR="${TMPDIR:-/tmp/air_int8_gemm}"
GPU_CHIP="${AIR_GPU_CHIP:-gfx1150}"
GPU_VARIANT="${AIR_INT8_GEMM_VARIANT:-lds_128x128_rocmlir_k32_pipe3}"
GPU_GROUP_SIZE="${AIR_INT8_GEMM_GROUP_SIZE:-8}"
export AIRGPU_USE_HIP_MALLOC="${AIRGPU_USE_HIP_MALLOC:-1}"
export AIRGPU_BENCHMARK_STREAM="${AIRGPU_BENCHMARK_STREAM:-1}"

if [ -z "${AIR_OPT:-}" ] && [ -x "$REPO_DIR/build-gpu/bin/air-opt" ]; then
  export AIR_OPT="$REPO_DIR/build-gpu/bin/air-opt"
fi
if [ -z "${MLIR_OPT:-}" ] && [ -x "$REPO_DIR/llvm/install-amdgpu/bin/mlir-opt" ]; then
  export MLIR_OPT="$REPO_DIR/llvm/install-amdgpu/bin/mlir-opt"
fi
[ -x "${AIR_OPT:-}" ] || { echo "ERROR: AIR_OPT not found" >&2; exit 1; }
[ -x "${MLIR_OPT:-}" ] || { echo "ERROR: MLIR_OPT not found" >&2; exit 1; }

mkdir -p "$TMPDIR"
ROCDL_MLIR="$TMPDIR/int8_gemm.rocdl.mlir"
OUTLINE_MLIR="$TMPDIR/int8_gemm.outline.mlir"
OUTLINE_LLVM_MLIR="$TMPDIR/int8_gemm.outline_llvm.mlir"
ISA_MLIR="$TMPDIR/int8_gemm.isa.mlir"
ISA_ASM="$TMPDIR/int8_gemm.isa.s"
FINAL_MLIR="$TMPDIR/int8_gemm.final.mlir"

AIR_TO_ROCDL="-air-to-rocdl=int8-gemm-variant=${GPU_VARIANT} int8-gemm-group-size=${GPU_GROUP_SIZE}"
AIR_GPU_OUTLINING="-air-gpu-outlining=int8-gemm-variant=${GPU_VARIANT} int8-gemm-group-size=${GPU_GROUP_SIZE}"
ROCDL_PIPELINE="rocdl-attach-target{chip=${GPU_CHIP} O=3},gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl{chipset=${GPU_CHIP} runtime=HIP},reconcile-unrealized-casts)"

"$AIR_OPT" "$SCRIPT_DIR/air_sync.mlir" "$AIR_TO_ROCDL" -o "$ROCDL_MLIR"
"$AIR_OPT" "$ROCDL_MLIR" "$AIR_GPU_OUTLINING" -o "$OUTLINE_MLIR"
"$MLIR_OPT" "--pass-pipeline=builtin.module(func.func(lower-affine,convert-linalg-to-loops,convert-scf-to-cf),gpu-kernel-outlining)" "$OUTLINE_MLIR" -o "$OUTLINE_LLVM_MLIR"
"$MLIR_OPT" "--pass-pipeline=builtin.module(${ROCDL_PIPELINE},gpu-module-to-binary{format=isa})" "$OUTLINE_LLVM_MLIR" -o "$ISA_MLIR"
"$MLIR_OPT" "--pass-pipeline=builtin.module(${ROCDL_PIPELINE},gpu-module-to-binary{format=bin},func.func(gpu-async-region,convert-scf-to-cf),gpu-to-llvm,convert-to-llvm,reconcile-unrealized-casts)" "$OUTLINE_LLVM_MLIR" -o "$FINAL_MLIR"

python3 - "$ISA_MLIR" "$ISA_ASM" <<'PY_EXTRACT'
import re
import sys
src, dst = sys.argv[1:]
text = open(src, encoding="utf-8").read()
matches = re.findall(r'assembly = "((?:[^"\\]|\\.)*)"', text, flags=re.S)
if not matches:
    raise SystemExit(f"no gpu.object assembly payload found in {src}")
hexdigits = set("0123456789abcdefABCDEF")
out = bytearray()
for value in matches:
    i = 0
    while i < len(value):
        if value[i] == "\\" and i + 2 < len(value) and value[i + 1] in hexdigits and value[i + 2] in hexdigits:
            out.append(int(value[i + 1:i + 3], 16)); i += 3
        elif value[i] == "\\" and i + 1 < len(value):
            i += 1; out.append(ord(value[i])); i += 1
        else:
            out.append(ord(value[i])); i += 1
    out.append(ord("\n"))
open(dst, "wb").write(out)
PY_EXTRACT

grep -Eq 'v_wmma_i32_16x16x16_iu8' "$ISA_ASM" || { echo "ERROR: expected WMMA INT8 instruction missing" >&2; exit 1; }
if grep -Eq 'v_wmma_.*16x16x64|v_swmmac|swmmac' "$ISA_ASM"; then
  echo "ERROR: forbidden GPU ISA marker found" >&2
  exit 1
fi

MLIR_RUNNER="${MLIR_RUNNER:-$(command -v mlir-runner || true)}"
[ -n "$MLIR_RUNNER" ] || MLIR_RUNNER="$REPO_DIR/llvm/install-amdgpu/bin/mlir-runner"
[ -x "$MLIR_RUNNER" ] || { echo "ERROR: mlir-runner not found" >&2; exit 1; }

AIRGPU_LIB="${AIRGPU_LIB:-${MLIR_AIR_INSTALL_DIR:+$MLIR_AIR_INSTALL_DIR/lib/libairgpu.so}}"
[ -n "$AIRGPU_LIB" ] || AIRGPU_LIB="$REPO_DIR/install-gpu/lib/libairgpu.so"
[ -f "$AIRGPU_LIB" ] || { echo "ERROR: libairgpu.so not found" >&2; exit 1; }

"$MLIR_RUNNER" --entry-point-result=void --shared-libs="$AIRGPU_LIB" "$FINAL_MLIR"
