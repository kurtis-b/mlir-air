#!/usr/bin/env bash
#===- cpu_disassemble.sh -----------------------------------------------===//
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
#===----------------------------------------------------------------------===//

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  cpu_disassemble.sh [options] <object|shared-library|executable>

Options:
  -o, --output-dir DIR  Directory for generated artifacts
  --prefix NAME         Output filename prefix (default: input basename)
  --symbol NAME         Disassemble only an exact symbol; repeatable
  --expect REGEX        Require REGEX to match disassembly output; repeatable
  -h, --help            Show this help

Tool overrides:
  LLVM_INSTALL_DIR=/path/to/llvm/install
  LLVM_READOBJ=/path/to/llvm-readobj
  LLVM_OBJDUMP=/path/to/llvm-objdump
  LLVM_NM=/path/to/llvm-nm
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUTDIR=""
PREFIX=""
SYMBOLS=()
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
    --symbol)
      SYMBOLS+=("${2:?missing value for --symbol}")
      shift 2
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

if [ -z "${LLVM_READOBJ:-}" ]; then
  LLVM_READOBJ_CANDIDATES=()
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

if [ -z "${LLVM_OBJDUMP:-}" ]; then
  LLVM_OBJDUMP_CANDIDATES=()
  if [ -n "${LLVM_INSTALL_DIR:-}" ]; then
    LLVM_OBJDUMP_CANDIDATES+=("$LLVM_INSTALL_DIR/bin/llvm-objdump")
  fi
  LLVM_OBJDUMP_CANDIDATES+=(
    "$REPO_DIR/llvm/install-amdgpu/bin/llvm-objdump"
    "$REPO_DIR/llvm/install/bin/llvm-objdump"
    llvm-objdump
  )
  LLVM_OBJDUMP="$(first_executable "${LLVM_OBJDUMP_CANDIDATES[@]}" || true)"
fi

if [ -z "${LLVM_NM:-}" ]; then
  LLVM_NM_CANDIDATES=()
  if [ -n "${LLVM_INSTALL_DIR:-}" ]; then
    LLVM_NM_CANDIDATES+=("$LLVM_INSTALL_DIR/bin/llvm-nm")
  fi
  LLVM_NM_CANDIDATES+=(
    "$REPO_DIR/llvm/install-amdgpu/bin/llvm-nm"
    "$REPO_DIR/llvm/install/bin/llvm-nm"
    llvm-nm
  )
  LLVM_NM="$(first_executable "${LLVM_NM_CANDIDATES[@]}" || true)"
fi

if [ -z "$LLVM_READOBJ" ]; then
  echo "ERROR: could not find llvm-readobj. Set LLVM_INSTALL_DIR or LLVM_READOBJ." >&2
  exit 1
fi

if [ -z "$LLVM_OBJDUMP" ]; then
  echo "ERROR: could not find llvm-objdump. Set LLVM_INSTALL_DIR or LLVM_OBJDUMP." >&2
  exit 1
fi

INPUT_BASENAME="$(basename "$INPUT")"
PREFIX="${PREFIX:-${INPUT_BASENAME%.*}}"
OUTDIR="${OUTDIR:-${TMPDIR:-/tmp}/air_cpu_isa_${PREFIX}}"
mkdir -p "$OUTDIR"

HEADERS="$OUTDIR/${PREFIX}.headers.txt"
DYNAMIC="$OUTDIR/${PREFIX}.dynamic.txt"
SYMBOL_TABLE="$OUTDIR/${PREFIX}.symbols.txt"
DISASM="$OUTDIR/${PREFIX}.disasm.s"
SUMMARY="$OUTDIR/${PREFIX}.summary.txt"

"$LLVM_READOBJ" --file-headers --section-headers --symbols "$INPUT" > "$HEADERS"

if ! "$LLVM_READOBJ" --dynamic-table --needed-libs "$INPUT" > "$DYNAMIC" 2>&1; then
  printf 'dynamic table unavailable for this artifact\n' > "$DYNAMIC"
fi

if [ -n "$LLVM_NM" ]; then
  "$LLVM_NM" -C --defined-only "$INPUT" > "$SYMBOL_TABLE" 2>&1 || true
else
  printf 'llvm-nm not found; symbol table skipped\n' > "$SYMBOL_TABLE"
fi

OBJDUMP_ARGS=(-d -C --x86-asm-syntax=intel)
for symbol in "${SYMBOLS[@]}"; do
  OBJDUMP_ARGS+=(--disassemble-symbols="$symbol")
done

"$LLVM_OBJDUMP" "${OBJDUMP_ARGS[@]}" "$INPUT" > "$DISASM"

{
  echo "# CPU disassembly summary"
  echo "input: $INPUT"
  echo "llvm_readobj: $LLVM_READOBJ"
  echo "llvm_objdump: $LLVM_OBJDUMP"
  echo "llvm_nm: ${LLVM_NM:-unavailable}"
  if [ "${#SYMBOLS[@]}" -gt 0 ]; then
    echo "symbols: ${SYMBOLS[*]}"
  fi
  echo
  echo "## Artifacts"
  printf '%s\n' "$HEADERS" "$DYNAMIC" "$SYMBOL_TABLE" "$DISASM"
  echo
  echo "## Header preview"
  sed -n '1,80p' "$HEADERS"
  echo
  echo "## Disassembly preview"
  sed -n '1,80p' "$DISASM"
} > "$SUMMARY"

for pattern in "${EXPECT_PATTERNS[@]}"; do
  if ! grep -Eq "$pattern" "$DISASM"; then
    echo "ERROR: expected pattern not found: $pattern" >&2
    echo "Checked file: $DISASM" >&2
    exit 1
  fi
done

echo "Wrote CPU disassembly artifacts to $OUTDIR"
echo "Summary: $SUMMARY"
