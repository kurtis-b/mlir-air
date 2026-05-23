#!/usr/bin/env bash
# SPDX-License-Identifier: MIT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() { cat <<'EOF'
Usage: disassemble.sh <cpu|gpu|npu> [options] <input>
Run `disassemble.sh <backend> --help` for backend options.
EOF
}

cpu_usage() { cat <<'EOF'
Usage: disassemble.sh cpu [options] <object|shared-library|executable>
  -o, --output-dir DIR  artifact directory
  --prefix NAME         output prefix (default: input basename)
  --symbol NAME         exact symbol; repeatable
  --expect REGEX        required disassembly pattern; repeatable
  -h, --help            show this help
Tool overrides: LLVM_INSTALL_DIR, LLVM_READOBJ, LLVM_OBJDUMP, LLVM_NM
EOF
}

gpu_usage() { cat <<'EOF'
Usage: disassemble.sh gpu [options] <input.air.mlir>
  --gpu-arch CHIP       AMDGPU chip (default: AIR_GPU_CHIP or gfx1150)
  -o, --output-dir DIR  artifact directory
  --prefix NAME         output prefix (default: input basename)
  --opt-level N         ROCDL optimization level (default: 3)
  --int8-gemm-variant V AIR GPU INT8 GEMM variant for marked launches
  --int8-gemm-group-size N AIR GPU INT8 GEMM grouped M size
  --gpu-bare-ptr-kernels use bare-pointer GPU kernel ABI/lowering
  --expect REGEX        required extracted-ISA pattern; repeatable
  --forbid REGEX        forbidden extracted-ISA pattern; repeatable
  -h, --help            show this help
Tool overrides: AIR_OPT, MLIR_OPT, LLVM_READOBJ, MLIR_AIR_INSTALL_DIR, LLVM_INSTALL_DIR
EOF
}

npu_usage() { cat <<'EOF'
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
  local outvar="$1" found
  shift
  [ -z "${!outvar:-}" ] || return 0
  found="$(first_executable "$@" || true)"
  printf -v "$outvar" '%s' "$found"
}

require_tool() {
  local envvar="$1" hint="$2" tool
  tool="${!envvar:-}"
  if [ -n "$tool" ] && { [ -x "$tool" ] || command -v "$tool" >/dev/null 2>&1; }; then return 0; fi
  die "could not find $envvar. $hint"
}

require_input() {
  local input="$1" message="$2" usage_fn="$3"
  if [ -z "$input" ]; then echo "ERROR: $message" >&2; "$usage_fn" >&2; exit 2; fi
  [ -f "$input" ] || die "input file not found: $input"
}

set_prefix_outdir() {
  local backend="$1" input="$2" base
  base="$(basename "$input")"
  PREFIX="${PREFIX:-${base%.*}}"
  OUTDIR="${OUTDIR:-${TMPDIR:-/tmp}/air_${backend}_isa_${PREFIX}}"
  mkdir -p "$OUTDIR"
}

check_expected() {
  local file="$1" pattern
  shift
  for pattern in "$@"; do
    grep -Eq "$pattern" "$file" || die "expected pattern not found: $pattern"$'\n'"Checked file: $file"
  done
}

check_forbidden() {
  local file="$1" pattern
  shift
  for pattern in "$@"; do
    grep -Eq "$pattern" "$file" && die "forbidden ISA pattern found: $pattern"$'\n'"ISA file: $file"
  done
  return 0
}

extract_mlir_attr() {
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import re, sys
src, dst, attr, mode = sys.argv[1:]
text = open(src, encoding="utf-8").read()
matches = re.findall(rf'{re.escape(attr)} = "((?:[^"\\]|\\.)*)"', text, flags=re.S)
if not matches:
    raise SystemExit(f"no gpu.object {attr} payload found in {src}")
hexdigits = set("0123456789abcdefABCDEF")
def decode(value):
    out = bytearray(); i = 0
    while i < len(value):
        if value[i] == "\\" and i + 2 < len(value) and value[i + 1] in hexdigits and value[i + 2] in hexdigits:
            out.append(int(value[i + 1:i + 3], 16)); i += 3
        elif value[i] == "\\" and i + 1 < len(value):
            i += 1; out.append(ord(value[i])); i += 1
        else:
            out.append(ord(value[i])); i += 1
    return bytes(out)
if mode == "bin":
    open(dst, "wb").write(decode(matches[0]))
else:
    open(dst, "wb").write(b"\n".join(decode(match) for match in matches))
PY
}

run_cpu() {
  OUTDIR=""; PREFIX=""
  local input="" symbol headers dynamic symtab disasm summary
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
  find_tool LLVM_READOBJ "${LLVM_INSTALL_DIR:+$LLVM_INSTALL_DIR/bin/llvm-readobj}" "$REPO_DIR/llvm/install-amdgpu/bin/llvm-readobj" "$REPO_DIR/llvm/install/bin/llvm-readobj" llvm-readobj
  find_tool LLVM_OBJDUMP "${LLVM_INSTALL_DIR:+$LLVM_INSTALL_DIR/bin/llvm-objdump}" "$REPO_DIR/llvm/install-amdgpu/bin/llvm-objdump" "$REPO_DIR/llvm/install/bin/llvm-objdump" llvm-objdump
  find_tool LLVM_NM "${LLVM_INSTALL_DIR:+$LLVM_INSTALL_DIR/bin/llvm-nm}" "$REPO_DIR/llvm/install-amdgpu/bin/llvm-nm" "$REPO_DIR/llvm/install/bin/llvm-nm" llvm-nm
  require_tool LLVM_READOBJ "Set LLVM_INSTALL_DIR or LLVM_READOBJ."
  require_tool LLVM_OBJDUMP "Set LLVM_INSTALL_DIR or LLVM_OBJDUMP."
  set_prefix_outdir cpu "$input"
  headers="$OUTDIR/${PREFIX}.headers.txt"; dynamic="$OUTDIR/${PREFIX}.dynamic.txt"
  symtab="$OUTDIR/${PREFIX}.symbols.txt"; disasm="$OUTDIR/${PREFIX}.disasm.s"; summary="$OUTDIR/${PREFIX}.summary.txt"
  "$LLVM_READOBJ" --file-headers --section-headers --symbols "$input" > "$headers"
  "$LLVM_READOBJ" --dynamic-table --needed-libs "$input" > "$dynamic" 2>&1 || printf 'dynamic table unavailable for this artifact\n' > "$dynamic"
  if [ -n "${LLVM_NM:-}" ]; then "$LLVM_NM" -C --defined-only "$input" > "$symtab" 2>&1 || true; else printf 'llvm-nm not found; symbol table skipped\n' > "$symtab"; fi
  local -a objdump_args=(-d -C --x86-asm-syntax=intel)
  for symbol in "${symbols[@]}"; do objdump_args+=(--disassemble-symbols="$symbol"); done
  "$LLVM_OBJDUMP" "${objdump_args[@]}" "$input" > "$disasm"
  { echo "# CPU disassembly summary"; echo "input: $input"; echo "llvm_readobj: $LLVM_READOBJ"; echo "llvm_objdump: $LLVM_OBJDUMP"; echo "llvm_nm: ${LLVM_NM:-unavailable}"; [ "${#symbols[@]}" -eq 0 ] || echo "symbols: ${symbols[*]}"; echo; echo "## Artifacts"; printf '%s\n' "$headers" "$dynamic" "$symtab" "$disasm"; } > "$summary"
  check_expected "$disasm" "${expect[@]}"
  echo "Wrote CPU disassembly artifacts to $OUTDIR"; echo "Summary: $summary"
}

run_gpu() {
  OUTDIR=""; PREFIX=""
  local gpu_chip="${AIR_GPU_CHIP:-gfx1150}" opt_level="3" int8_gemm_variant="${AIR_INT8_GEMM_VARIANT:-}" int8_gemm_group_size="${AIR_INT8_GEMM_GROUP_SIZE:-}" input="" rocdl_mlir outline_mlir outline_llvm_mlir isa_mlir isa_asm bin_mlir code_object readobj final_mlir summary rocdl_pipeline rocdl_convert_options gpu_to_llvm_pass air_to_rocdl_pass air_gpu_outlining_pass air_to_rocdl_options air_gpu_outlining_options
  local use_bare_ptr_kernel=0
  local -a expect=() forbid=()
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --gpu-arch) gpu_chip="${2:?missing value for --gpu-arch}"; shift 2 ;;
      -o|--output-dir) OUTDIR="${2:?missing value for --output-dir}"; shift 2 ;;
      --prefix) PREFIX="${2:?missing value for --prefix}"; shift 2 ;;
      --opt-level) opt_level="${2:?missing value for --opt-level}"; shift 2 ;;
      --int8-gemm-variant) int8_gemm_variant="${2:?missing value for --int8-gemm-variant}"; shift 2 ;;
      --int8-gemm-group-size) int8_gemm_group_size="${2:?missing value for --int8-gemm-group-size}"; shift 2 ;;
      --gpu-bare-ptr-kernels) use_bare_ptr_kernel=1; shift ;;
      --expect) expect+=("${2:?missing value for --expect}"); shift 2 ;;
      --forbid) forbid+=("${2:?missing value for --forbid}"); shift 2 ;;
      -h|--help) gpu_usage; exit 0 ;;
      -*) echo "ERROR: unknown GPU option: $1" >&2; gpu_usage >&2; exit 2 ;;
      *) [ -z "$input" ] || die "multiple input files: $input and $1" 2; input="$1"; shift ;;
    esac
  done
  require_input "$input" "missing input AIR MLIR file" gpu_usage
  find_tool AIR_OPT "${MLIR_AIR_INSTALL_DIR:+$MLIR_AIR_INSTALL_DIR/bin/air-opt}" "$REPO_DIR/install-gpu/bin/air-opt" "$REPO_DIR/install/bin/air-opt" air-opt
  find_tool MLIR_OPT "${LLVM_INSTALL_DIR:+$LLVM_INSTALL_DIR/bin/mlir-opt}" "$REPO_DIR/llvm/install-amdgpu/bin/mlir-opt" "$REPO_DIR/llvm/install/bin/mlir-opt" mlir-opt
  require_tool AIR_OPT "Set AIR_OPT or source utils/env_setup_gpu.sh."
  require_tool MLIR_OPT "Set MLIR_OPT or source utils/env_setup_gpu.sh."
  find_tool LLVM_READOBJ "$(dirname "$MLIR_OPT")/llvm-readobj" "${LLVM_INSTALL_DIR:+$LLVM_INSTALL_DIR/bin/llvm-readobj}" "$REPO_DIR/llvm/install-amdgpu/bin/llvm-readobj" "$REPO_DIR/llvm/install/bin/llvm-readobj" llvm-readobj
  command -v python3 >/dev/null 2>&1 || die "python3 is required to extract readable ISA from MLIR string escapes."
  set_prefix_outdir gpu "$input"
  rocdl_mlir="$OUTDIR/${PREFIX}.rocdl.mlir"; outline_mlir="$OUTDIR/${PREFIX}.outline.mlir"; outline_llvm_mlir="$OUTDIR/${PREFIX}.outline_llvm.mlir"
  isa_mlir="$OUTDIR/${PREFIX}.isa.mlir"; isa_asm="$OUTDIR/${PREFIX}.isa.s"; bin_mlir="$OUTDIR/${PREFIX}.bin.mlir"; code_object="$OUTDIR/${PREFIX}.hsaco"
  readobj="$OUTDIR/${PREFIX}.code_object.readobj.txt"; final_mlir="$OUTDIR/${PREFIX}.final.mlir"; summary="$OUTDIR/${PREFIX}.summary.txt"
  echo "Lowering AIR to GPU/ROCDL for $gpu_chip"
  air_to_rocdl_pass="-air-to-rocdl"
  air_gpu_outlining_pass="-air-gpu-outlining"
  air_to_rocdl_options=""
  air_gpu_outlining_options=""
  if [ -n "$int8_gemm_variant" ]; then
    air_to_rocdl_options="int8-gemm-variant=${int8_gemm_variant}"
    air_gpu_outlining_options="int8-gemm-variant=${int8_gemm_variant}"
  fi
  if [ -n "$int8_gemm_group_size" ]; then
    air_to_rocdl_options="${air_to_rocdl_options:+${air_to_rocdl_options} }int8-gemm-group-size=${int8_gemm_group_size}"
    air_gpu_outlining_options="${air_gpu_outlining_options:+${air_gpu_outlining_options} }int8-gemm-group-size=${int8_gemm_group_size}"
  fi
  if [ -n "$air_to_rocdl_options" ]; then
    air_to_rocdl_pass="-air-to-rocdl=${air_to_rocdl_options}"
    air_gpu_outlining_pass="-air-gpu-outlining=${air_gpu_outlining_options}"
  fi
  case "$int8_gemm_variant" in
    *rawptr*) use_bare_ptr_kernel=1 ;;
  esac
  "$AIR_OPT" "$input" "$air_to_rocdl_pass" -o "$rocdl_mlir"
  "$AIR_OPT" "$rocdl_mlir" "$air_gpu_outlining_pass" -o "$outline_mlir"
  "$MLIR_OPT" "--pass-pipeline=builtin.module(func.func(lower-affine,convert-linalg-to-loops,convert-scf-to-cf),gpu-kernel-outlining)" "$outline_mlir" -o "$outline_llvm_mlir"
  rocdl_convert_options="chipset=${gpu_chip} runtime=HIP"
  gpu_to_llvm_pass="gpu-to-llvm"
  if [ "$use_bare_ptr_kernel" -eq 1 ]; then
    rocdl_convert_options="${rocdl_convert_options} index-bitwidth=32 use-bare-ptr-memref-call-conv=true"
    gpu_to_llvm_pass="gpu-to-llvm{use-bare-pointers-for-kernels=true}"
  fi
  rocdl_pipeline="rocdl-attach-target{chip=${gpu_chip} O=${opt_level}},gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl{${rocdl_convert_options}},reconcile-unrealized-casts)"
  "$MLIR_OPT" "--pass-pipeline=builtin.module(${rocdl_pipeline},gpu-module-to-binary{format=isa})" "$outline_llvm_mlir" -o "$isa_mlir"
  "$MLIR_OPT" "--pass-pipeline=builtin.module(${rocdl_pipeline},gpu-module-to-binary{format=bin})" "$outline_llvm_mlir" -o "$bin_mlir"
  "$MLIR_OPT" "--pass-pipeline=builtin.module(${rocdl_pipeline},gpu-module-to-binary{format=bin},func.func(gpu-async-region,convert-scf-to-cf),${gpu_to_llvm_pass},convert-to-llvm,reconcile-unrealized-casts)" "$outline_llvm_mlir" -o "$final_mlir"
  extract_mlir_attr "$isa_mlir" "$isa_asm" assembly text
  extract_mlir_attr "$bin_mlir" "$code_object" bin bin
  if [ -n "${LLVM_READOBJ:-}" ]; then "$LLVM_READOBJ" --file-headers --notes --sections --symbols "$code_object" > "$readobj"; else printf 'llvm-readobj not found; code object metadata dump skipped\n' > "$readobj"; fi
  {
    echo "# GPU ISA summary"; echo "input: $input"; echo "gpu_arch: $gpu_chip"; [ -z "$int8_gemm_variant" ] || echo "int8_gemm_variant: $int8_gemm_variant"; [ -z "$int8_gemm_group_size" ] || echo "int8_gemm_group_size: $int8_gemm_group_size"; echo "bare_ptr_kernel_abi: $use_bare_ptr_kernel"; echo "air_opt: $AIR_OPT"; echo "mlir_opt: $MLIR_OPT"; echo "llvm_readobj: ${LLVM_READOBJ:-unavailable}"
    echo; echo "## Artifacts"; printf '%s\n' "$rocdl_mlir" "$outline_mlir" "$outline_llvm_mlir" "$isa_mlir" "$isa_asm" "$bin_mlir" "$code_object" "$readobj" "$final_mlir"
    echo; grep -aoE 'gpu\.binary @[A-Za-z0-9_.$-]+|chip = "[^"]+"|sgpr_spill_count = [0-9]+ : i64|vgpr_spill_count = [0-9]+ : i64|sgpr_count = [0-9]+ : i64|vgpr_count = [0-9]+ : i64|wavefront_size = [0-9]+ : i64' "$final_mlir" || true
    grep -E 'Format:|Arch:|EF_AMDGPU|NT_AMDGPU_METADATA|\.sgpr_count|\.sgpr_spill_count|\.vgpr_count|\.vgpr_spill_count|\.wavefront_size|amdhsa\.target' "$readobj" || true
  } > "$summary"
  check_expected "$isa_asm" "${expect[@]}"; check_forbidden "$isa_asm" "${forbid[@]}"
  echo "Wrote GPU ISA artifacts to $OUTDIR"; echo "Summary: $summary"
}

is_elf() {
  local magic
  magic="$(od -An -N4 -tx1 "$1" | tr -d ' \n')"
  [ "$magic" = "7f454c46" ]
}

run_npu() {
  OUTDIR=""; PREFIX=""
  local kind="auto" mcpu="aie2p" triple="aie2p-none-unknown-elf" aiebu_mode="aie2txn" run_profile=1 input="" summary check_file
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
  set_prefix_outdir npu "$input"; summary="$OUTDIR/${PREFIX}.summary.txt"
  if [ "$kind" = "elf" ]; then
    shopt -s nullglob
    find_tool LLVM_OBJDUMP "${PEANO_INSTALL_DIR:+$PEANO_INSTALL_DIR/bin/llvm-objdump}" "$REPO_DIR"/sandbox/lib/python*/site-packages/llvm-aie/bin/llvm-objdump "$REPO_DIR/my_install/mlir/bin/llvm-objdump" llvm-objdump
    shopt -u nullglob
    require_tool LLVM_OBJDUMP "Set PEANO_INSTALL_DIR or LLVM_OBJDUMP."
    [ -n "${LLVM_READOBJ:-}" ] || LLVM_READOBJ="$(first_executable "$(dirname "$LLVM_OBJDUMP")/llvm-readobj" llvm-readobj || true)"
    local disasm="$OUTDIR/${PREFIX}.${mcpu}.disasm.s" headers="$OUTDIR/${PREFIX}.headers.txt"
    "$LLVM_OBJDUMP" -d "--triple=${triple}" "--mcpu=${mcpu}" "$input" > "$disasm"
    if [ -n "${LLVM_READOBJ:-}" ]; then "$LLVM_READOBJ" --file-headers --section-headers --symbols "$input" > "$headers"; else printf 'llvm-readobj not found; header dump skipped\n' > "$headers"; fi
    { echo "# NPU core ISA summary"; echo "input: $input"; echo "kind: elf"; echo "triple: $triple"; echo "mcpu: $mcpu"; echo "llvm_objdump: $LLVM_OBJDUMP"; echo "llvm_readobj: ${LLVM_READOBJ:-unavailable}"; echo; echo "## Artifacts"; printf '%s\n' "$disasm" "$headers"; } > "$summary"
    check_file="$disasm"
  else
    [ -n "${AIEBU_DUMP:-}" ] || AIEBU_DUMP="$(first_executable /opt/xilinx/xrt/bin/aiebu-dump aiebu-dump || true)"
    require_tool AIEBU_DUMP "Source /opt/xilinx/xrt/setup.sh or set AIEBU_DUMP."
    local disasm="$OUTDIR/${PREFIX}.${aiebu_mode}.disasm.txt" profile="$OUTDIR/${PREFIX}.${aiebu_mode}.profile.txt"
    "$AIEBU_DUMP" "$input" -d -m "$aiebu_mode" > "$disasm"
    if [ "$run_profile" -eq 1 ]; then "$AIEBU_DUMP" "$input" -p -m "$aiebu_mode" > "$profile"; else printf 'aiebu-dump profile skipped by --no-profile\n' > "$profile"; fi
    { echo "# NPU transaction/control stream summary"; echo "input: $input"; echo "kind: txn"; echo "aiebu_mode: $aiebu_mode"; echo "aiebu_dump: $AIEBU_DUMP"; echo; echo "## Artifacts"; printf '%s\n' "$disasm" "$profile"; } > "$summary"
    check_file="$disasm"
  fi
  check_expected "$check_file" "${expect[@]}"
  echo "Wrote NPU disassembly artifacts to $OUTDIR"; echo "Summary: $summary"
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  cpu) shift; run_cpu "$@" ;;
  gpu) shift; run_gpu "$@" ;;
  npu) shift; run_npu "$@" ;;
  "") usage >&2; exit 2 ;;
  *) echo "ERROR: unknown backend: $1" >&2; usage >&2; exit 2 ;;
esac
