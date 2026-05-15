#!/usr/bin/env bash
#===- gpu_disassemble.sh -----------------------------------------------===//
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
#===----------------------------------------------------------------------===//

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  gpu_disassemble.sh [options] <input.air.mlir>

Options:
  --gpu-arch <chip>     AMDGPU target chip (default: gfx1150)
  -o, --output-dir DIR  Directory for generated artifacts
  --prefix NAME         Output filename prefix (default: input basename)
  --opt-level N         ROCDL target optimization level (default: 3)
  --expect REGEX        Require REGEX to match extracted ISA; repeatable
  --forbid REGEX        Require REGEX not to match extracted ISA; repeatable
  -h, --help            Show this help

Tool overrides:
  AIR_OPT=/path/to/air-opt
  MLIR_OPT=/path/to/mlir-opt
  LLVM_READOBJ=/path/to/llvm-readobj
  MLIR_AIR_INSTALL_DIR=/path/to/mlir-air/install
  LLVM_INSTALL_DIR=/path/to/llvm/install
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU_CHIP="${AIR_GPU_CHIP:-gfx1150}"
OUTDIR=""
PREFIX=""
OPT_LEVEL="3"
EXPECT_PATTERNS=()
FORBID_PATTERNS=()
INPUT=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu-arch)
      GPU_CHIP="${2:?missing value for --gpu-arch}"
      shift 2
      ;;
    -o|--output-dir)
      OUTDIR="${2:?missing value for --output-dir}"
      shift 2
      ;;
    --prefix)
      PREFIX="${2:?missing value for --prefix}"
      shift 2
      ;;
    --opt-level)
      OPT_LEVEL="${2:?missing value for --opt-level}"
      shift 2
      ;;
    --expect)
      EXPECT_PATTERNS+=("${2:?missing value for --expect}")
      shift 2
      ;;
    --forbid)
      FORBID_PATTERNS+=("${2:?missing value for --forbid}")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [ -n "$INPUT" ]; then
        echo "ERROR: multiple input files: $INPUT and $1" >&2
        exit 2
      fi
      INPUT="$1"
      shift
      ;;
  esac
done

if [ -z "$INPUT" ]; then
  echo "ERROR: missing input AIR MLIR file" >&2
  usage >&2
  exit 2
fi

if [ ! -f "$INPUT" ]; then
  echo "ERROR: input file not found: $INPUT" >&2
  exit 1
fi

first_executable() {
  local candidate
  for candidate in "$@"; do
    [ -n "$candidate" ] || continue
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

require_tool() {
  local name="$1"
  local tool="$2"
  if [ -n "$tool" ] && { [ -x "$tool" ] || command -v "$tool" >/dev/null 2>&1; }; then
    return 0
  fi
  echo "ERROR: could not find $name. Set $name or source utils/env_setup_gpu.sh." >&2
  exit 1
}

if [ -z "${AIR_OPT:-}" ]; then
  AIR_OPT_CANDIDATES=()
  if [ -n "${MLIR_AIR_INSTALL_DIR:-}" ]; then
    AIR_OPT_CANDIDATES+=("$MLIR_AIR_INSTALL_DIR/bin/air-opt")
  fi
  AIR_OPT_CANDIDATES+=(
    "$REPO_DIR/install-gpu/bin/air-opt"
    "$REPO_DIR/install/bin/air-opt"
    air-opt
  )
  AIR_OPT="$(first_executable "${AIR_OPT_CANDIDATES[@]}" || true)"
fi

if [ -z "${MLIR_OPT:-}" ]; then
  MLIR_OPT_CANDIDATES=()
  if [ -n "${LLVM_INSTALL_DIR:-}" ]; then
    MLIR_OPT_CANDIDATES+=("$LLVM_INSTALL_DIR/bin/mlir-opt")
  fi
  MLIR_OPT_CANDIDATES+=(
    "$REPO_DIR/llvm/install-amdgpu/bin/mlir-opt"
    "$REPO_DIR/llvm/install/bin/mlir-opt"
    mlir-opt
  )
  MLIR_OPT="$(first_executable "${MLIR_OPT_CANDIDATES[@]}" || true)"
fi

require_tool AIR_OPT "$AIR_OPT"
require_tool MLIR_OPT "$MLIR_OPT"

if [ -z "${LLVM_READOBJ:-}" ]; then
  LLVM_READOBJ_CANDIDATES=("$(dirname "$MLIR_OPT")/llvm-readobj")
  if [ -n "${LLVM_INSTALL_DIR:-}" ]; then
    LLVM_READOBJ_CANDIDATES+=("$LLVM_INSTALL_DIR/bin/llvm-readobj")
  fi
  LLVM_READOBJ_CANDIDATES+=(
    "$REPO_DIR/llvm/install-amdgpu/bin/llvm-readobj"
    "$REPO_DIR/llvm/install/bin/llvm-readobj"
    llvm-readobj
  )
  LLVM_READOBJ="$(first_executable "${LLVM_READOBJ_CANDIDATES[@]}" || true)"
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to extract readable ISA from MLIR string escapes." >&2
  exit 1
fi

INPUT_BASENAME="$(basename "$INPUT")"
PREFIX="${PREFIX:-${INPUT_BASENAME%.*}}"
OUTDIR="${OUTDIR:-${TMPDIR:-/tmp}/air_gpu_isa_${PREFIX}}"
mkdir -p "$OUTDIR"

ROCDL_MLIR="$OUTDIR/${PREFIX}.rocdl.mlir"
OUTLINE_MLIR="$OUTDIR/${PREFIX}.outline.mlir"
OUTLINE_LLVM_MLIR="$OUTDIR/${PREFIX}.outline_llvm.mlir"
ISA_MLIR="$OUTDIR/${PREFIX}.isa.mlir"
ISA_ASM="$OUTDIR/${PREFIX}.isa.s"
BIN_MLIR="$OUTDIR/${PREFIX}.bin.mlir"
CODE_OBJECT="$OUTDIR/${PREFIX}.hsaco"
CODE_OBJECT_READOBJ="$OUTDIR/${PREFIX}.code_object.readobj.txt"
FINAL_MLIR="$OUTDIR/${PREFIX}.final.mlir"
SUMMARY="$OUTDIR/${PREFIX}.summary.txt"

echo "Lowering AIR to GPU/ROCDL for $GPU_CHIP"
"$AIR_OPT" "$INPUT" -air-to-rocdl -o "$ROCDL_MLIR"
"$AIR_OPT" "$ROCDL_MLIR" -air-gpu-outlining -o "$OUTLINE_MLIR"

"$MLIR_OPT" \
  "--pass-pipeline=builtin.module(func.func(lower-affine,convert-linalg-to-loops,convert-scf-to-cf),gpu-kernel-outlining)" \
  "$OUTLINE_MLIR" -o "$OUTLINE_LLVM_MLIR"

ROCDL_PIPELINE="rocdl-attach-target{chip=${GPU_CHIP} O=${OPT_LEVEL}},gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl{chipset=${GPU_CHIP} runtime=HIP},reconcile-unrealized-casts)"

"$MLIR_OPT" \
  "--pass-pipeline=builtin.module(${ROCDL_PIPELINE},gpu-module-to-binary{format=isa})" \
  "$OUTLINE_LLVM_MLIR" -o "$ISA_MLIR"

"$MLIR_OPT" \
  "--pass-pipeline=builtin.module(${ROCDL_PIPELINE},gpu-module-to-binary{format=bin})" \
  "$OUTLINE_LLVM_MLIR" -o "$BIN_MLIR"

"$MLIR_OPT" \
  "--pass-pipeline=builtin.module(${ROCDL_PIPELINE},gpu-module-to-binary{format=bin},func.func(gpu-async-region,convert-scf-to-cf),gpu-to-llvm,convert-to-llvm,reconcile-unrealized-casts)" \
  "$OUTLINE_LLVM_MLIR" -o "$FINAL_MLIR"

python3 - "$ISA_MLIR" "$ISA_ASM" <<'PY'
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()
matches = re.findall(r'assembly = "((?:[^"\\]|\\.)*)"', text, flags=re.S)
if not matches:
    raise SystemExit(f"no gpu.object assembly payload found in {src}")

hexdigits = set("0123456789abcdefABCDEF")

def decode_mlir_string(value):
    out = bytearray()
    i = 0
    while i < len(value):
        if value[i] == "\\" and i + 2 < len(value) and value[i + 1] in hexdigits and value[i + 2] in hexdigits:
            out.append(int(value[i + 1:i + 3], 16))
            i += 3
        elif value[i] == "\\" and i + 1 < len(value):
            i += 1
            out.append(ord(value[i]))
            i += 1
        else:
            out.append(ord(value[i]))
            i += 1
    return out.decode("utf-8", errors="replace")

with open(dst, "w", encoding="utf-8") as output:
    for index, match in enumerate(matches):
        if index:
            output.write("\n")
        output.write(decode_mlir_string(match))
PY

python3 - "$BIN_MLIR" "$CODE_OBJECT" <<'PY'
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()
match = re.search(r'bin = "((?:[^"\\]|\\.)*)"', text, flags=re.S)
if not match:
    raise SystemExit(f"no gpu.object binary payload found in {src}")

hexdigits = set("0123456789abcdefABCDEF")
value = match.group(1)
out = bytearray()
i = 0
while i < len(value):
    if value[i] == "\\" and i + 2 < len(value) and value[i + 1] in hexdigits and value[i + 2] in hexdigits:
        out.append(int(value[i + 1:i + 3], 16))
        i += 3
    elif value[i] == "\\" and i + 1 < len(value):
        i += 1
        out.append(ord(value[i]))
        i += 1
    else:
        out.append(ord(value[i]))
        i += 1

open(dst, "wb").write(out)
PY

if [ -n "$LLVM_READOBJ" ]; then
  "$LLVM_READOBJ" --file-headers --notes --sections --symbols "$CODE_OBJECT" > "$CODE_OBJECT_READOBJ"
else
  printf 'llvm-readobj not found; code object metadata dump skipped\n' > "$CODE_OBJECT_READOBJ"
fi

{
  echo "# GPU ISA summary"
  echo "input: $INPUT"
  echo "gpu_arch: $GPU_CHIP"
  echo "air_opt: $AIR_OPT"
  echo "mlir_opt: $MLIR_OPT"
  echo "llvm_readobj: ${LLVM_READOBJ:-unavailable}"
  echo
  echo "## Artifacts"
  printf '%s\n' \
    "$ROCDL_MLIR" \
    "$OUTLINE_MLIR" \
    "$OUTLINE_LLVM_MLIR" \
    "$ISA_MLIR" \
    "$ISA_ASM" \
    "$BIN_MLIR" \
    "$CODE_OBJECT" \
    "$CODE_OBJECT_READOBJ" \
    "$FINAL_MLIR"
  echo
  echo "## Final gpu.binary target"
  grep -aoE 'gpu\.binary @[A-Za-z0-9_.$-]+|chip = "[^"]+"|group_segment_fixed_size = [0-9]+ : i64|max_flat_workgroup_size = [0-9]+ : i64|reqd_workgroup_size = array<i32: [^>]+>|sgpr_count = [0-9]+ : i64|sgpr_spill_count = [0-9]+ : i64|vgpr_count = [0-9]+ : i64|vgpr_spill_count = [0-9]+ : i64|wavefront_size = [0-9]+ : i64' "$FINAL_MLIR" || true
  echo
  echo "## Code object metadata"
  grep -E 'Format:|Arch:|EF_AMDGPU|NT_AMDGPU_METADATA|amdhsa\.kernels|\.name:|\.sgpr_count|\.sgpr_spill_count|\.vgpr_count|\.vgpr_spill_count|\.wavefront_size|amdhsa\.target' "$CODE_OBJECT_READOBJ" || true
  echo
  echo "## ISA metadata markers"
  grep -E 'amdgcn_target|amdhsa_kernel|amdhsa_next_free_vgpr|amdhsa_next_free_sgpr|amdhsa_wavefront_size32|amdhsa\.target|\.sgpr_count|\.vgpr_count|\.wavefront_size' "$ISA_ASM" || true
} > "$SUMMARY"

for pattern in "${EXPECT_PATTERNS[@]}"; do
  if ! grep -Eq "$pattern" "$ISA_ASM"; then
    echo "ERROR: expected ISA pattern not found: $pattern" >&2
    echo "ISA file: $ISA_ASM" >&2
    exit 1
  fi
done

for pattern in "${FORBID_PATTERNS[@]}"; do
  if grep -Eq "$pattern" "$ISA_ASM"; then
    echo "ERROR: forbidden ISA pattern found: $pattern" >&2
    echo "ISA file: $ISA_ASM" >&2
    exit 1
  fi
done

echo "Wrote GPU ISA artifacts to $OUTDIR"
echo "Summary: $SUMMARY"
