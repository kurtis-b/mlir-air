#!/usr/bin/env bash
#===- npu_disassemble.sh -----------------------------------------------===//
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
#===----------------------------------------------------------------------===//

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  npu_disassemble.sh [options] <core.elf|air.insts.bin>

Options:
  -o, --output-dir DIR  Directory for generated artifacts
  --prefix NAME         Output filename prefix (default: input basename)
  --kind auto|elf|txn   Force input interpretation (default: auto)
  --mcpu NAME           AIE objdump CPU (default: aie2p)
  --triple TRIPLE       AIE objdump triple (default: aie2p-none-unknown-elf)
  --aiebu-mode MODE     aiebu-dump architecture mode (default: aie2txn)
  --no-profile          Skip aiebu-dump opcode frequency output
  --expect REGEX        Require REGEX to match disassembly output; repeatable
  -h, --help            Show this help

Tool overrides:
  PEANO_INSTALL_DIR=/path/to/llvm-aie
  LLVM_OBJDUMP=/path/to/llvm-objdump
  LLVM_READOBJ=/path/to/llvm-readobj
  AIEBU_DUMP=/opt/xilinx/xrt/bin/aiebu-dump
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUTDIR=""
PREFIX=""
KIND="auto"
MCPU="aie2p"
TRIPLE="aie2p-none-unknown-elf"
AIEBU_MODE="aie2txn"
RUN_PROFILE=1
EXPECT_PATTERNS=()
INPUT=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    -o|--output-dir)
      OUTDIR="${2:?missing value for --output-dir}"
      shift 2
      ;;
    --prefix)
      PREFIX="${2:?missing value for --prefix}"
      shift 2
      ;;
    --kind)
      KIND="${2:?missing value for --kind}"
      shift 2
      ;;
    --mcpu)
      MCPU="${2:?missing value for --mcpu}"
      shift 2
      ;;
    --triple)
      TRIPLE="${2:?missing value for --triple}"
      shift 2
      ;;
    --aiebu-mode)
      AIEBU_MODE="${2:?missing value for --aiebu-mode}"
      shift 2
      ;;
    --no-profile)
      RUN_PROFILE=0
      shift
      ;;
    --expect)
      EXPECT_PATTERNS+=("${2:?missing value for --expect}")
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

case "$KIND" in
  auto|elf|txn) ;;
  *)
    echo "ERROR: --kind must be auto, elf, or txn" >&2
    exit 2
    ;;
esac

if [ -z "$INPUT" ]; then
  echo "ERROR: missing input artifact" >&2
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

is_elf() {
  local magic
  magic="$(od -An -N4 -tx1 "$1" | tr -d ' \n')"
  [ "$magic" = "7f454c46" ]
}

if [ "$KIND" = "auto" ]; then
  if is_elf "$INPUT"; then
    KIND="elf"
  else
    KIND="txn"
  fi
fi

INPUT_BASENAME="$(basename "$INPUT")"
PREFIX="${PREFIX:-${INPUT_BASENAME%.*}}"
OUTDIR="${OUTDIR:-${TMPDIR:-/tmp}/air_npu_isa_${PREFIX}}"
mkdir -p "$OUTDIR"

SUMMARY="$OUTDIR/${PREFIX}.summary.txt"

if [ "$KIND" = "elf" ]; then
  shopt -s nullglob
  if [ -z "${LLVM_OBJDUMP:-}" ]; then
    PEANO_OBJDUMP_CANDIDATES=()
    if [ -n "${PEANO_INSTALL_DIR:-}" ]; then
      PEANO_OBJDUMP_CANDIDATES+=("$PEANO_INSTALL_DIR/bin/llvm-objdump")
    fi
    PEANO_OBJDUMP_CANDIDATES+=(
      "$REPO_DIR/sandbox"/lib/python*/site-packages/llvm-aie/bin/llvm-objdump
      "$REPO_DIR/my_install/mlir/bin/llvm-objdump"
      llvm-objdump
    )
    LLVM_OBJDUMP="$(first_executable "${PEANO_OBJDUMP_CANDIDATES[@]}" || true)"
  fi
  shopt -u nullglob

  if [ -z "${LLVM_OBJDUMP:-}" ]; then
    echo "ERROR: could not find llvm-objdump. Set PEANO_INSTALL_DIR or LLVM_OBJDUMP." >&2
    exit 1
  fi

  OBJDUMP_DIR="$(dirname "$LLVM_OBJDUMP")"
  LLVM_READOBJ="${LLVM_READOBJ:-$(first_executable "$OBJDUMP_DIR/llvm-readobj" llvm-readobj || true)}"

  DISASM="$OUTDIR/${PREFIX}.${MCPU}.disasm.s"
  HEADERS="$OUTDIR/${PREFIX}.headers.txt"

  "$LLVM_OBJDUMP" -d "--triple=${TRIPLE}" "--mcpu=${MCPU}" "$INPUT" > "$DISASM"

  if [ -n "$LLVM_READOBJ" ]; then
    "$LLVM_READOBJ" --file-headers --section-headers --symbols "$INPUT" > "$HEADERS"
  else
    printf 'llvm-readobj not found; header dump skipped\n' > "$HEADERS"
  fi

  {
    echo "# NPU core ISA summary"
    echo "input: $INPUT"
    echo "kind: elf"
    echo "triple: $TRIPLE"
    echo "mcpu: $MCPU"
    echo "llvm_objdump: $LLVM_OBJDUMP"
    echo "llvm_readobj: ${LLVM_READOBJ:-unavailable}"
    echo
    echo "## Artifacts"
    printf '%s\n' "$DISASM" "$HEADERS"
    echo
    echo "## Disassembly preview"
    sed -n '1,80p' "$DISASM"
  } > "$SUMMARY"

  CHECK_FILE="$DISASM"
else
  AIEBU_DUMP="${AIEBU_DUMP:-$(first_executable /opt/xilinx/xrt/bin/aiebu-dump aiebu-dump || true)}"
  if [ -z "$AIEBU_DUMP" ]; then
    echo "ERROR: could not find aiebu-dump. Source /opt/xilinx/xrt/setup.sh or set AIEBU_DUMP." >&2
    exit 1
  fi

  DISASM="$OUTDIR/${PREFIX}.${AIEBU_MODE}.disasm.txt"
  PROFILE="$OUTDIR/${PREFIX}.${AIEBU_MODE}.profile.txt"

  "$AIEBU_DUMP" "$INPUT" -d -m "$AIEBU_MODE" > "$DISASM"

  if [ "$RUN_PROFILE" -eq 1 ]; then
    "$AIEBU_DUMP" "$INPUT" -p -m "$AIEBU_MODE" > "$PROFILE"
  else
    printf 'aiebu-dump profile skipped by --no-profile\n' > "$PROFILE"
  fi

  {
    echo "# NPU transaction/control stream summary"
    echo "input: $INPUT"
    echo "kind: txn"
    echo "aiebu_mode: $AIEBU_MODE"
    echo "aiebu_dump: $AIEBU_DUMP"
    echo
    echo "## Artifacts"
    printf '%s\n' "$DISASM" "$PROFILE"
    echo
    echo "## Disassembly preview"
    sed -n '1,80p' "$DISASM"
    echo
    echo "## Opcode frequency preview"
    sed -n '1,80p' "$PROFILE"
  } > "$SUMMARY"

  CHECK_FILE="$DISASM"
fi

for pattern in "${EXPECT_PATTERNS[@]}"; do
  if ! grep -Eq "$pattern" "$CHECK_FILE"; then
    echo "ERROR: expected pattern not found: $pattern" >&2
    echo "Checked file: $CHECK_FILE" >&2
    exit 1
  fi
done

echo "Wrote NPU disassembly artifacts to $OUTDIR"
echo "Summary: $SUMMARY"
