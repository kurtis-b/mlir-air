#!/usr/bin/env bash
# SPDX-License-Identifier: MIT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: disassemble.sh <cpu|gpu|npu> [options] <input>
Run `disassemble.sh <backend> --help` for backend options.
EOF
}

cpu_usage() {
  cat <<'EOF'
Usage: disassemble.sh cpu [options] <object|shared-library|executable>

  -o, --output-dir DIR  artifact directory
  --prefix NAME         output prefix (default: input basename)
  --symbol NAME         exact symbol; repeatable
  --expect REGEX        required disassembly pattern; repeatable
  -h, --help            show this help
Tool overrides: LLVM_INSTALL_DIR, LLVM_READOBJ, LLVM_OBJDUMP, LLVM_NM
EOF
}

gpu_usage() {
  cat <<'EOF'
Usage: disassemble.sh gpu [options] <input.air.mlir>

  --gpu-arch CHIP       AMDGPU chip (default: AIR_GPU_CHIP or gfx1150)
  -o, --output-dir DIR  artifact directory
  --prefix NAME         output prefix (default: input basename)
  --opt-level N         ROCDL optimization level (default: 3)
  --expect REGEX        required extracted-ISA pattern; repeatable
  --forbid REGEX        forbidden extracted-ISA pattern; repeatable
  -h, --help            show this help
Tool overrides: AIR_OPT, MLIR_OPT, LLVM_READOBJ, MLIR_AIR_INSTALL_DIR, LLVM_INSTALL_DIR
EOF
}

npu_usage() {
  cat <<'EOF'
Usage: disassemble.sh npu [options] <core.elf|air.insts.bin>

  -o, --output-dir DIR  artifact directory
  --prefix NAME         output prefix (default: input basename)
  --kind auto|elf|txn   input interpretation (default: auto)
  --mcpu NAME           AIE objdump CPU (default: aie2p)
  --triple TRIPLE       AIE objdump triple (default: aie2p-none-unknown-elf)
  --aiebu-mode MODE     aiebu-dump mode (default: aie2txn)
  --no-profile          skip aiebu-dump opcode frequency output
  --expect REGEX        required disassembly pattern; repeatable
  -h, --help            show this help
Tool overrides: PEANO_INSTALL_DIR, LLVM_OBJDUMP, LLVM_READOBJ, AIEBU_DUMP
EOF
}

die() { local code="${2:-1}"; echo "ERROR: $1" >&2; exit "$code"; }

first_executable() {
  local candidate
  for candidate in "$@"; do
    [ -n "$candidate" ] || continue
    if [ -x "$candidate" ]; then printf '%s\n' "$candidate"; return 0; fi
    if command -v "$candidate" >/dev/null 2>&1; then command -v "$candidate"; return 0; fi
  done
  return 1
}

find_tool() {
  local outvar="$1"
  shift
  local current="${!outvar:-}"
  [ -z "$current" ] || return 0
  printf -v "$outvar" '%s' "$(first_executable "$@" || true)"
}

require_tool() {
  local envvar="$1"
  local hint="$2"
  local tool="${!envvar:-}"
  [ -z "$tool" ] || { [ ! -x "$tool" ] && ! command -v "$tool" >/dev/null 2>&1; } || return 0
  die "could not find $envvar. $hint"
}

require_input() {
  local input="$1"
  local message="$2"
  local usage_fn="$3"
  if [ -z "$input" ]; then echo "ERROR: $message" >&2; "$usage_fn" >&2; exit 2; fi
  [ -f "$input" ] || die "input file not found: $input"
}

set_prefix_outdir() {
  local backend="$1"
  local input="$2"
  local basename
  basename="$(basename "$input")"
  PREFIX="${PREFIX:-${basename%.*}}"
  OUTDIR="${OUTDIR:-${TMPDIR:-/tmp}/air_${backend}_isa_${PREFIX}}"
  mkdir -p "$OUTDIR"
}

check_expected() {
  local file="$1"
  shift
  local pattern
  for pattern in "$@"; do
    if grep -Eq "$pattern" "$file"; then continue; fi
    echo "ERROR: expected pattern not found: $pattern" >&2
    echo "Checked file: $file" >&2
    exit 1
  done
}

extract_mlir_attr() {
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import re
import sys
src, dst, attr, mode = sys.argv[1:]
text = open(src, encoding="utf-8").read()
matches = re.findall(rf'{re.escape(attr)} = "((?:[^"\\]|\\.)*)"', text, flags=re.S)
if not matches:
    raise SystemExit(f"no gpu.object {attr} payload found in {src}")
hexdigits = set("0123456789abcdefABCDEF")
def decode(value):
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
    return bytes(out)
if mode == "bin":
    open(dst, "wb").write(decode(matches[0]))
else:
    with open(dst, "wb") as output:
        for index, match in enumerate(matches):
            if index:
                output.write(b"\n")
            output.write(decode(match))
PY
}

run_cpu() {
  OUTDIR=""
  PREFIX=""
  local input=""
  local -a symbols=() expect=()
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -o|--output-dir) OUTDIR="${2:?missing value for --output-dir}"; shift 2 ;;
      --prefix) PREFIX="${2:?missing value for --prefix}"; shift 2 ;;
      --symbol) symbols+=("${2:?missing value for --symbol}"); shift 2 ;;
      --expect) expect+=("${2:?missing value for --expect}"); shift 2 ;;
      -h|--help) cpu_usage; exit 0 ;;
      -*) echo "ERROR: unknown CPU option: $1" >&2; cpu_usage >&2; exit 2 ;;
      *) [ -z "$input" ] || die "multiple input files: $input and $1" 2; input="$1"; shift ;;
    esac
  done
  require_input "$input" "missing input artifact" cpu_usage

  local -a readobj_candidates=() objdump_candidates=() nm_candidates=()
  [ -z "${LLVM_INSTALL_DIR:-}" ] || readobj_candidates+=("$LLVM_INSTALL_DIR/bin/llvm-readobj")
  readobj_candidates+=("$REPO_DIR/llvm/install-amdgpu/bin/llvm-readobj" "$REPO_DIR/llvm/install/bin/llvm-readobj" llvm-readobj)
  [ -z "${LLVM_INSTALL_DIR:-}" ] || objdump_candidates+=("$LLVM_INSTALL_DIR/bin/llvm-objdump")
  objdump_candidates+=("$REPO_DIR/llvm/install-amdgpu/bin/llvm-objdump" "$REPO_DIR/llvm/install/bin/llvm-objdump" llvm-objdump)
  [ -z "${LLVM_INSTALL_DIR:-}" ] || nm_candidates+=("$LLVM_INSTALL_DIR/bin/llvm-nm")
  nm_candidates+=("$REPO_DIR/llvm/install-amdgpu/bin/llvm-nm" "$REPO_DIR/llvm/install/bin/llvm-nm" llvm-nm)
  find_tool LLVM_READOBJ "${readobj_candidates[@]}"
  find_tool LLVM_OBJDUMP "${objdump_candidates[@]}"
  find_tool LLVM_NM "${nm_candidates[@]}"
  require_tool LLVM_READOBJ "Set LLVM_INSTALL_DIR or LLVM_READOBJ."
  require_tool LLVM_OBJDUMP "Set LLVM_INSTALL_DIR or LLVM_OBJDUMP."

  set_prefix_outdir cpu "$input"
  local headers="$OUTDIR/${PREFIX}.headers.txt"
  local dynamic="$OUTDIR/${PREFIX}.dynamic.txt"
  local symtab="$OUTDIR/${PREFIX}.symbols.txt"
  local disasm="$OUTDIR/${PREFIX}.disasm.s"
  local summary="$OUTDIR/${PREFIX}.summary.txt"

  "$LLVM_READOBJ" --file-headers --section-headers --symbols "$input" > "$headers"
  "$LLVM_READOBJ" --dynamic-table --needed-libs "$input" > "$dynamic" 2>&1 || printf 'dynamic table unavailable for this artifact\n' > "$dynamic"
  if [ -n "${LLVM_NM:-}" ]; then
    "$LLVM_NM" -C --defined-only "$input" > "$symtab" 2>&1 || true
  else
    printf 'llvm-nm not found; symbol table skipped\n' > "$symtab"
  fi

  local -a objdump_args=(-d -C --x86-asm-syntax=intel)
  local symbol
  for symbol in "${symbols[@]}"; do
    objdump_args+=(--disassemble-symbols="$symbol")
  done
  "$LLVM_OBJDUMP" "${objdump_args[@]}" "$input" > "$disasm"

  {
    echo "# CPU disassembly summary"
    echo "input: $input"
    echo "llvm_readobj: $LLVM_READOBJ"
    echo "llvm_objdump: $LLVM_OBJDUMP"
    echo "llvm_nm: ${LLVM_NM:-unavailable}"
    [ "${#symbols[@]}" -eq 0 ] || echo "symbols: ${symbols[*]}"
    echo; echo "## Artifacts"; printf '%s\n' "$headers" "$dynamic" "$symtab" "$disasm"
    echo; echo "## Header preview"; sed -n '1,80p' "$headers"
    echo; echo "## Disassembly preview"; sed -n '1,80p' "$disasm"
  } > "$summary"
  check_expected "$disasm" "${expect[@]}"
  echo "Wrote CPU disassembly artifacts to $OUTDIR"
  echo "Summary: $summary"
}

run_gpu() {
  OUTDIR=""
  PREFIX=""
  local gpu_chip="${AIR_GPU_CHIP:-gfx1150}"
  local opt_level="3"
  local input=""
  local -a expect=() forbid=()
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --gpu-arch) gpu_chip="${2:?missing value for --gpu-arch}"; shift 2 ;;
      -o|--output-dir) OUTDIR="${2:?missing value for --output-dir}"; shift 2 ;;
      --prefix) PREFIX="${2:?missing value for --prefix}"; shift 2 ;;
      --opt-level) opt_level="${2:?missing value for --opt-level}"; shift 2 ;;
      --expect) expect+=("${2:?missing value for --expect}"); shift 2 ;;
      --forbid) forbid+=("${2:?missing value for --forbid}"); shift 2 ;;
      -h|--help) gpu_usage; exit 0 ;;
      -*) echo "ERROR: unknown GPU option: $1" >&2; gpu_usage >&2; exit 2 ;;
      *) [ -z "$input" ] || die "multiple input files: $input and $1" 2; input="$1"; shift ;;
    esac
  done
  require_input "$input" "missing input AIR MLIR file" gpu_usage

  local -a air_opt_candidates=() mlir_opt_candidates=() readobj_candidates=()
  [ -z "${MLIR_AIR_INSTALL_DIR:-}" ] || air_opt_candidates+=("$MLIR_AIR_INSTALL_DIR/bin/air-opt")
  air_opt_candidates+=("$REPO_DIR/install-gpu/bin/air-opt" "$REPO_DIR/install/bin/air-opt" air-opt)
  [ -z "${LLVM_INSTALL_DIR:-}" ] || mlir_opt_candidates+=("$LLVM_INSTALL_DIR/bin/mlir-opt")
  mlir_opt_candidates+=("$REPO_DIR/llvm/install-amdgpu/bin/mlir-opt" "$REPO_DIR/llvm/install/bin/mlir-opt" mlir-opt)
  find_tool AIR_OPT "${air_opt_candidates[@]}"
  find_tool MLIR_OPT "${mlir_opt_candidates[@]}"
  require_tool AIR_OPT "Set AIR_OPT or source utils/env_setup_gpu.sh."
  require_tool MLIR_OPT "Set MLIR_OPT or source utils/env_setup_gpu.sh."
  readobj_candidates=("$(dirname "$MLIR_OPT")/llvm-readobj")
  [ -z "${LLVM_INSTALL_DIR:-}" ] || readobj_candidates+=("$LLVM_INSTALL_DIR/bin/llvm-readobj")
  readobj_candidates+=("$REPO_DIR/llvm/install-amdgpu/bin/llvm-readobj" "$REPO_DIR/llvm/install/bin/llvm-readobj" llvm-readobj)
  find_tool LLVM_READOBJ "${readobj_candidates[@]}"
  command -v python3 >/dev/null 2>&1 || die "python3 is required to extract readable ISA from MLIR string escapes."

  set_prefix_outdir gpu "$input"
  local rocdl_mlir="$OUTDIR/${PREFIX}.rocdl.mlir"
  local outline_mlir="$OUTDIR/${PREFIX}.outline.mlir"
  local outline_llvm_mlir="$OUTDIR/${PREFIX}.outline_llvm.mlir"
  local isa_mlir="$OUTDIR/${PREFIX}.isa.mlir"
  local isa_asm="$OUTDIR/${PREFIX}.isa.s"
  local bin_mlir="$OUTDIR/${PREFIX}.bin.mlir"
  local code_object="$OUTDIR/${PREFIX}.hsaco"
  local readobj="$OUTDIR/${PREFIX}.code_object.readobj.txt"
  local final_mlir="$OUTDIR/${PREFIX}.final.mlir"
  local summary="$OUTDIR/${PREFIX}.summary.txt"

  echo "Lowering AIR to GPU/ROCDL for $gpu_chip"
  "$AIR_OPT" "$input" -air-to-rocdl -o "$rocdl_mlir"
  "$AIR_OPT" "$rocdl_mlir" -air-gpu-outlining -o "$outline_mlir"
  "$MLIR_OPT" "--pass-pipeline=builtin.module(func.func(lower-affine,convert-linalg-to-loops,convert-scf-to-cf),gpu-kernel-outlining)" "$outline_mlir" -o "$outline_llvm_mlir"

  local rocdl_pipeline="rocdl-attach-target{chip=${gpu_chip} O=${opt_level}},gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl{chipset=${gpu_chip} runtime=HIP},reconcile-unrealized-casts)"
  "$MLIR_OPT" "--pass-pipeline=builtin.module(${rocdl_pipeline},gpu-module-to-binary{format=isa})" "$outline_llvm_mlir" -o "$isa_mlir"
  "$MLIR_OPT" "--pass-pipeline=builtin.module(${rocdl_pipeline},gpu-module-to-binary{format=bin})" "$outline_llvm_mlir" -o "$bin_mlir"
  "$MLIR_OPT" "--pass-pipeline=builtin.module(${rocdl_pipeline},gpu-module-to-binary{format=bin},func.func(gpu-async-region,convert-scf-to-cf),gpu-to-llvm,convert-to-llvm,reconcile-unrealized-casts)" "$outline_llvm_mlir" -o "$final_mlir"

  extract_mlir_attr "$isa_mlir" "$isa_asm" assembly text
  extract_mlir_attr "$bin_mlir" "$code_object" bin bin
  if [ -n "${LLVM_READOBJ:-}" ]; then
    "$LLVM_READOBJ" --file-headers --notes --sections --symbols "$code_object" > "$readobj"
  else
    printf 'llvm-readobj not found; code object metadata dump skipped\n' > "$readobj"
  fi

  {
    echo "# GPU ISA summary"
    echo "input: $input"
    echo "gpu_arch: $gpu_chip"
    echo "air_opt: $AIR_OPT"
    echo "mlir_opt: $MLIR_OPT"
    echo "llvm_readobj: ${LLVM_READOBJ:-unavailable}"
    echo; echo "## Artifacts"; printf '%s\n' "$rocdl_mlir" "$outline_mlir" "$outline_llvm_mlir" "$isa_mlir" "$isa_asm" "$bin_mlir" "$code_object" "$readobj" "$final_mlir"
    echo; echo "## Final gpu.binary target"
    grep -aoE 'gpu\.binary @[A-Za-z0-9_.$-]+|chip = "[^"]+"|group_segment_fixed_size = [0-9]+ : i64|max_flat_workgroup_size = [0-9]+ : i64|reqd_workgroup_size = array<i32: [^>]+>|sgpr_count = [0-9]+ : i64|sgpr_spill_count = [0-9]+ : i64|vgpr_count = [0-9]+ : i64|vgpr_spill_count = [0-9]+ : i64|wavefront_size = [0-9]+ : i64' "$final_mlir" || true
    echo; echo "## Code object metadata"
    grep -E 'Format:|Arch:|EF_AMDGPU|NT_AMDGPU_METADATA|amdhsa\.kernels|\.name:|\.sgpr_count|\.sgpr_spill_count|\.vgpr_count|\.vgpr_spill_count|\.wavefront_size|amdhsa\.target' "$readobj" || true
    echo; echo "## ISA metadata markers"
    grep -E 'amdgcn_target|amdhsa_kernel|amdhsa_next_free_vgpr|amdhsa_next_free_sgpr|amdhsa_wavefront_size32|amdhsa\.target|\.sgpr_count|\.vgpr_count|\.wavefront_size' "$isa_asm" || true
  } > "$summary"

  check_expected "$isa_asm" "${expect[@]}"
  local pattern
  for pattern in "${forbid[@]}"; do
    if grep -Eq "$pattern" "$isa_asm"; then
      echo "ERROR: forbidden ISA pattern found: $pattern" >&2
      echo "ISA file: $isa_asm" >&2
      exit 1
    fi
  done
  echo "Wrote GPU ISA artifacts to $OUTDIR"
  echo "Summary: $summary"
}

is_elf() {
  local magic
  magic="$(od -An -N4 -tx1 "$1" | tr -d ' \n')"
  [ "$magic" = "7f454c46" ]
}

run_npu() {
  OUTDIR=""
  PREFIX=""
  local kind="auto"
  local mcpu="aie2p"
  local triple="aie2p-none-unknown-elf"
  local aiebu_mode="aie2txn"
  local run_profile=1
  local input=""
  local -a expect=()
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -o|--output-dir) OUTDIR="${2:?missing value for --output-dir}"; shift 2 ;;
      --prefix) PREFIX="${2:?missing value for --prefix}"; shift 2 ;;
      --kind) kind="${2:?missing value for --kind}"; shift 2 ;;
      --mcpu) mcpu="${2:?missing value for --mcpu}"; shift 2 ;;
      --triple) triple="${2:?missing value for --triple}"; shift 2 ;;
      --aiebu-mode) aiebu_mode="${2:?missing value for --aiebu-mode}"; shift 2 ;;
      --no-profile) run_profile=0; shift ;;
      --expect) expect+=("${2:?missing value for --expect}"); shift 2 ;;
      -h|--help) npu_usage; exit 0 ;;
      -*) echo "ERROR: unknown NPU option: $1" >&2; npu_usage >&2; exit 2 ;;
      *) [ -z "$input" ] || die "multiple input files: $input and $1" 2; input="$1"; shift ;;
    esac
  done
  case "$kind" in auto|elf|txn) ;; *) die "--kind must be auto, elf, or txn" 2 ;; esac
  require_input "$input" "missing input artifact" npu_usage
  [ "$kind" != "auto" ] || { if is_elf "$input"; then kind="elf"; else kind="txn"; fi; }
  set_prefix_outdir npu "$input"
  local summary="$OUTDIR/${PREFIX}.summary.txt"
  local check_file

  if [ "$kind" = "elf" ]; then
    local -a objdump_candidates=()
    [ -z "${PEANO_INSTALL_DIR:-}" ] || objdump_candidates+=("$PEANO_INSTALL_DIR/bin/llvm-objdump")
    shopt -s nullglob
    objdump_candidates+=("$REPO_DIR"/sandbox/lib/python*/site-packages/llvm-aie/bin/llvm-objdump)
    shopt -u nullglob
    objdump_candidates+=("$REPO_DIR/my_install/mlir/bin/llvm-objdump" llvm-objdump)
    find_tool LLVM_OBJDUMP "${objdump_candidates[@]}"
    require_tool LLVM_OBJDUMP "Set PEANO_INSTALL_DIR or LLVM_OBJDUMP."
    local objdump_dir
    objdump_dir="$(dirname "$LLVM_OBJDUMP")"
    [ -n "${LLVM_READOBJ:-}" ] || LLVM_READOBJ="$(first_executable "$objdump_dir/llvm-readobj" llvm-readobj || true)"
    local disasm="$OUTDIR/${PREFIX}.${mcpu}.disasm.s"
    local headers="$OUTDIR/${PREFIX}.headers.txt"
    "$LLVM_OBJDUMP" -d "--triple=${triple}" "--mcpu=${mcpu}" "$input" > "$disasm"
    if [ -n "${LLVM_READOBJ:-}" ]; then
      "$LLVM_READOBJ" --file-headers --section-headers --symbols "$input" > "$headers"
    else
      printf 'llvm-readobj not found; header dump skipped\n' > "$headers"
    fi
    {
      echo "# NPU core ISA summary"
      echo "input: $input"
      echo "kind: elf"
      echo "triple: $triple"
      echo "mcpu: $mcpu"
      echo "llvm_objdump: $LLVM_OBJDUMP"
      echo "llvm_readobj: ${LLVM_READOBJ:-unavailable}"
      echo; echo "## Artifacts"; printf '%s\n' "$disasm" "$headers"
      echo; echo "## Disassembly preview"; sed -n '1,80p' "$disasm"
    } > "$summary"
    check_file="$disasm"
  else
    [ -n "${AIEBU_DUMP:-}" ] || AIEBU_DUMP="$(first_executable /opt/xilinx/xrt/bin/aiebu-dump aiebu-dump || true)"
    require_tool AIEBU_DUMP "Source /opt/xilinx/xrt/setup.sh or set AIEBU_DUMP."
    local disasm="$OUTDIR/${PREFIX}.${aiebu_mode}.disasm.txt"
    local profile="$OUTDIR/${PREFIX}.${aiebu_mode}.profile.txt"
    "$AIEBU_DUMP" "$input" -d -m "$aiebu_mode" > "$disasm"
    if [ "$run_profile" -eq 1 ]; then
      "$AIEBU_DUMP" "$input" -p -m "$aiebu_mode" > "$profile"
    else
      printf 'aiebu-dump profile skipped by --no-profile\n' > "$profile"
    fi
    {
      echo "# NPU transaction/control stream summary"
      echo "input: $input"
      echo "kind: txn"
      echo "aiebu_mode: $aiebu_mode"
      echo "aiebu_dump: $AIEBU_DUMP"
      echo; echo "## Artifacts"; printf '%s\n' "$disasm" "$profile"
      echo; echo "## Disassembly preview"; sed -n '1,80p' "$disasm"
      echo; echo "## Opcode frequency preview"; sed -n '1,80p' "$profile"
    } > "$summary"
    check_file="$disasm"
  fi
  check_expected "$check_file" "${expect[@]}"
  echo "Wrote NPU disassembly artifacts to $OUTDIR"
  echo "Summary: $summary"
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  cpu) shift; run_cpu "$@" ;;
  gpu) shift; run_gpu "$@" ;;
  npu) shift; run_npu "$@" ;;
  "") usage >&2; exit 2 ;;
  *) echo "ERROR: unknown backend: $1" >&2; usage >&2; exit 2 ;;
esac
