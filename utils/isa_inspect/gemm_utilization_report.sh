#!/usr/bin/env bash
# SPDX-License-Identifier: MIT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DISASSEMBLE="$SCRIPT_DIR/disassemble.sh"

OUTDIR="${TMPDIR:-/tmp}/air_gemm_isa_utilization"
GPU_CHIP="${AIR_GPU_CHIP:-gfx1150}"
CPU_WARMUPS=0
CPU_ITERATIONS=1
RUN_CPU=0
RUN_GPU=0
RUN_NPU=0
STRICT=0
SKIP_CPU=0
SKIP_GPU=0
SKIP_NPU=0
NPU_BUILD_DIR_OVERRIDE=""

usage() {
  cat <<'EOF'
Usage: gemm_utilization_report.sh [options]

Build and disassemble fair int8 GEMM kernels for CPU, GPU, and NPU.
The comparison contract is M=N=K=1024, int8 inputs, int32 output.

Options:
  -o, --output-dir DIR    report/artifact directory
  --gpu-arch CHIP         AMDGPU chip (default: AIR_GPU_CHIP or gfx1150)
  --run-cpu               run the CPU benchmark after disassembly
  --run-gpu               run test/gpu/int8_gemm/run.sh after disassembly
  --run-npu               run the NPU profile target after disassembly
  --cpu-warmups N         CPU benchmark warmup iterations for --run-cpu
  --cpu-iterations N      CPU benchmark timed iterations for --run-cpu
  --npu-build-dir DIR     reuse an existing NPU build directory instead of compiling
  --skip-cpu              omit CPU
  --skip-gpu              omit GPU
  --skip-npu              omit NPU
  --strict                exit non-zero when any selected backend fails
  -h, --help              show this help

Environment:
  CXX, CXXFLAGS           CPU compiler overrides
  AIR_GPU_CHIP            default GPU chip
  MLIR_AIR_INSTALL_DIR    GPU runtime/library discovery
  PEANO_INSTALL_DIR       NPU core ELF objdump discovery
  XILINX_XRT, AIEBU_DUMP  NPU transaction stream and profile support
EOF
}

die() { echo "ERROR: $1" >&2; exit "${2:-1}"; }

parse_nonnegative_int() {
  local value="$1"
  local name="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be a non-negative integer" 2
  printf '%s\n' "$value"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -o|--output-dir) OUTDIR="${2:?missing value for --output-dir}"; shift 2 ;;
    --gpu-arch) GPU_CHIP="${2:?missing value for --gpu-arch}"; shift 2 ;;
    --run-cpu) RUN_CPU=1; shift ;;
    --run-gpu) RUN_GPU=1; shift ;;
    --run-npu) RUN_NPU=1; shift ;;
    --cpu-warmups) CPU_WARMUPS="$(parse_nonnegative_int "${2:?missing value for --cpu-warmups}" "--cpu-warmups")"; shift 2 ;;
    --cpu-iterations) CPU_ITERATIONS="$(parse_nonnegative_int "${2:?missing value for --cpu-iterations}" "--cpu-iterations")"; shift 2 ;;
    --npu-build-dir) NPU_BUILD_DIR_OVERRIDE="${2:?missing value for --npu-build-dir}"; shift 2 ;;
    --skip-cpu) SKIP_CPU=1; shift ;;
    --skip-gpu) SKIP_GPU=1; shift ;;
    --skip-npu) SKIP_NPU=1; shift ;;
    --strict) STRICT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ "$CPU_ITERATIONS" -gt 0 ] || die "--cpu-iterations must be greater than zero" 2

mkdir -p "$OUTDIR/logs"
REPORT="$OUTDIR/gemm_utilization_report.md"

run_capture() {
  local log="$1"
  shift
  mkdir -p "$(dirname "$log")"
  {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  } > "$log"
  "$@" >> "$log" 2>&1
}

run_capture_in_dir() {
  local dir="$1"
  local log="$2"
  shift 2
  mkdir -p "$(dirname "$log")"
  {
    printf '+ cd %q &&' "$dir"
    printf ' %q' "$@"
    printf '\n'
  } > "$log"
  (cd "$dir" && "$@") >> "$log" 2>&1
}

count_regex() {
  local file="$1"
  local pattern="$2"
  if [ -f "$file" ]; then
    grep -Eic "$pattern" "$file" || true
  else
    printf '0\n'
  fi
}

first_match() {
  local file="$1"
  local pattern="$2"
  if [ -f "$file" ]; then
    grep -Eim1 "$pattern" "$file" | sed 's/^[[:space:]]*//' || true
  fi
}

last_kv_value() {
  local file="$1"
  local key="$2"
  if [ -f "$file" ]; then
    grep -Eo "${key}=[^[:space:]]+" "$file" | tail -n1 | cut -d= -f2- || true
  fi
}

timing_field() {
  local file="$1"
  local domain="$2"
  local field="$3"
  if [ -f "$file" ]; then
    grep -E "timing_domain=${domain}\b" "$file" | tail -n1 |
      sed -E "s/.*${field}=([^[:space:]]+).*/\1/" || true
  fi
}

to_tops() {
  local gops="$1"
  if [ -z "$gops" ]; then
    echo "n/a"
  else
    awk -v value="$gops" 'BEGIN { printf "%.6f", value / 1000.0 }'
  fi
}

microseconds_to_ms() {
  local us="$1"
  if [ -z "$us" ]; then
    echo "n/a"
  else
    awk -v value="$us" 'BEGIN { printf "%.6f", value / 1000.0 }'
  fi
}

append_file_preview() {
  local file="$1"
  local lines="${2:-40}"
  if [ -f "$file" ]; then
    sed -n "1,${lines}p" "$file"
  else
    echo "unavailable: $file"
  fi
}

cpu_status="SKIP"
cpu_evidence="not requested"
cpu_artifacts="not generated"
cpu_run_note="not run"
cpu_log="$OUTDIR/logs/cpu_build.log"
cpu_disasm_log="$OUTDIR/logs/cpu_disassemble.log"
cpu_run_log="$OUTDIR/logs/cpu_run.log"
cpu_bin="$OUTDIR/cpu_build/int8_gemm_cpu"
cpu_disasm="$OUTDIR/cpu/cpu_int8_gemm.disasm.s"
cpu_perf_domain="not run"
cpu_perf_count="n/a"
cpu_perf_latency="n/a"
cpu_perf_throughput="n/a"
cpu_perf_note="not run"

parse_cpu_perf() {
  local avg_us min_us max_us gops validation tops
  avg_us="$(last_kv_value "$cpu_run_log" avg_us)"
  min_us="$(last_kv_value "$cpu_run_log" min_us)"
  max_us="$(last_kv_value "$cpu_run_log" max_us)"
  gops="$(last_kv_value "$cpu_run_log" gops)"
  validation="$(last_kv_value "$cpu_run_log" validation)"
  tops="$(to_tops "$gops")"
  cpu_perf_domain="$(last_kv_value "$cpu_run_log" timing_domain)"
  cpu_perf_count="$CPU_ITERATIONS"
  cpu_perf_latency="mean ${avg_us:-n/a} us, min ${min_us:-n/a} us, max ${max_us:-n/a} us"
  cpu_perf_throughput="${gops:-n/a} GOPS (${tops} TOPS)"
  cpu_perf_note="validation=${validation:-unknown}; sampled reference check"
}

run_cpu_path() {
  local cpu_dir="$REPO_DIR/test/cpu/int8_gemm"
  if ! run_capture "$cpu_log" make -C "$cpu_dir" BUILD_DIR="$OUTDIR/cpu_build"; then
    cpu_status="WARN"
    cpu_evidence="CPU benchmark build failed; see $cpu_log"
    return
  fi

  if ! run_capture "$cpu_disasm_log" "$DISASSEMBLE" cpu \
      --output-dir "$OUTDIR/cpu" --prefix cpu_int8_gemm \
      --symbol cpu_i8_gemm_vnni --expect 'vpdpbusd' "$cpu_bin"; then
    cpu_status="FAIL"
    cpu_evidence="CPU disassembly did not show required VNNI marker; see $cpu_disasm_log"
    return
  fi

  local vnni_count
  local zmm_count
  vnni_count="$(count_regex "$cpu_disasm" '\bvpdpbusd\b')"
  zmm_count="$(count_regex "$cpu_disasm" '\bzmm[0-9]+')"
  cpu_artifacts="$OUTDIR/cpu"
  if [ "$vnni_count" -gt 0 ]; then
    cpu_status="PASS"
    cpu_evidence="vpdpbusd=$vnni_count, zmm_refs=$zmm_count"
  else
    cpu_status="FAIL"
    cpu_evidence="vpdpbusd=0, zmm_refs=$zmm_count"
  fi

  if [ "$RUN_CPU" -eq 1 ]; then
    if run_capture "$cpu_run_log" "$cpu_bin" --warmups "$CPU_WARMUPS" --iterations "$CPU_ITERATIONS"; then
      cpu_run_note="ran; see $cpu_run_log"
      parse_cpu_perf
    else
      cpu_run_note="run failed; see $cpu_run_log"
      cpu_perf_note="run failed; see $cpu_run_log"
      [ "$cpu_status" = "PASS" ] && cpu_status="WARN"
    fi
  fi
}

gpu_status="SKIP"
gpu_evidence="not requested"
gpu_artifacts="not generated"
gpu_run_note="not run"
gpu_log="$OUTDIR/logs/gpu_disassemble.log"
gpu_run_log="$OUTDIR/logs/gpu_run.log"
gpu_isa="$OUTDIR/gpu/gpu_int8_gemm.isa.s"
gpu_summary="$OUTDIR/gpu/gpu_int8_gemm.summary.txt"
gpu_perf_domain="not run"
gpu_perf_count="n/a"
gpu_perf_latency="n/a"
gpu_perf_throughput="n/a"
gpu_perf_note="not run"

parse_gpu_perf() {
  local count min_ms mean_ms max_ms tops host_mean peak_pct
  count="$(timing_field "$gpu_run_log" kernel_event count)"
  min_ms="$(timing_field "$gpu_run_log" kernel_event min_ms)"
  mean_ms="$(timing_field "$gpu_run_log" kernel_event mean_ms)"
  max_ms="$(timing_field "$gpu_run_log" kernel_event max_ms)"
  tops="$(timing_field "$gpu_run_log" kernel_event tops)"
  host_mean="$(timing_field "$gpu_run_log" host_dispatch_wait mean_ms)"
  peak_pct="$(last_kv_value "$gpu_run_log" kernel_event_peak_pct)"
  gpu_perf_domain="kernel_event"
  gpu_perf_count="${count:-n/a}"
  gpu_perf_latency="mean ${mean_ms:-n/a} ms, min ${min_ms:-n/a} ms, max ${max_ms:-n/a} ms"
  gpu_perf_throughput="${tops:-n/a} TOPS"
  gpu_perf_note="host_dispatch_wait_mean_ms=${host_mean:-n/a}; peak_pct=${peak_pct:-n/a}"
}

run_gpu_path() {
  local gpu_input="$REPO_DIR/test/gpu/int8_gemm/air_sync.mlir"
  if ! run_capture "$gpu_log" "$DISASSEMBLE" gpu --gpu-arch "$GPU_CHIP" \
      --output-dir "$OUTDIR/gpu" --prefix gpu_int8_gemm \
      --expect 'v_wmma_i32_16x16x16_iu8' \
      --forbid 'v_wmma_.*16x16x64|v_swmmac|swmmac' "$gpu_input"; then
    gpu_status="WARN"
    gpu_evidence="GPU lowering/disassembly failed or required marker was absent; see $gpu_log"
    return
  fi

  local wmma_count barrier_count scratch spills wavefront vgprs sgprs lds
  wmma_count="$(count_regex "$gpu_isa" '\bv_wmma_i32_16x16x16_iu8\b')"
  barrier_count="$(count_regex "$gpu_isa" '\bs_barrier\b')"
  scratch="$(count_regex "$gpu_isa" 'uses_flat_scratch\s+1')"
  spills="$(count_regex "$gpu_summary" 'spill_count = [1-9]|_spill_count: [1-9]')"
  wavefront="$(first_match "$gpu_summary" 'wavefront_size|amdhsa_wavefront_size32')"
  vgprs="$(first_match "$gpu_summary" 'vgpr_count|amdhsa_next_free_vgpr')"
  sgprs="$(first_match "$gpu_summary" 'sgpr_count|amdhsa_next_free_sgpr')"
  lds="$(first_match "$gpu_isa" 'amdhsa_group_segment_fixed_size')"
  gpu_artifacts="$OUTDIR/gpu"
  if [ "$wmma_count" -gt 0 ] && [ "$scratch" -eq 0 ] && [ "$spills" -eq 0 ]; then
    gpu_status="PASS"
  else
    gpu_status="FAIL"
  fi
  gpu_evidence="wmma=$wmma_count, barriers=$barrier_count, scratch_markers=$scratch, spills=$spills"
  [ -z "$wavefront" ] || gpu_evidence+=", $wavefront"
  [ -z "$vgprs" ] || gpu_evidence+=", $vgprs"
  [ -z "$sgprs" ] || gpu_evidence+=", $sgprs"
  [ -z "$lds" ] || gpu_evidence+=", $lds"

  if [ "$RUN_GPU" -eq 1 ]; then
    if run_capture "$gpu_run_log" "$REPO_DIR/test/gpu/int8_gemm/run.sh"; then
      gpu_run_note="ran; see $gpu_run_log"
      parse_gpu_perf
    else
      gpu_run_note="run failed; see $gpu_run_log"
      gpu_perf_note="run failed; see $gpu_run_log"
      [ "$gpu_status" = "PASS" ] && gpu_status="WARN"
    fi
  fi
}

npu_status="SKIP"
npu_evidence="not requested"
npu_artifacts="not generated"
npu_run_note="not run"
npu_build_log="$OUTDIR/logs/npu_build.log"
npu_build_test_log="$OUTDIR/logs/npu_build_test_exe.log"
npu_run_log="$OUTDIR/logs/npu_profile.log"
npu_disasm_dir="$OUTDIR/npu"
npu_build_dir="$OUTDIR/npu_build"
npu_elf_count=0
npu_disasm_failures=0
npu_txn_note="transaction stream not generated"
npu_compile_note="fresh compile"
npu_perf_domain="not run"
npu_perf_count="n/a"
npu_perf_latency="n/a"
npu_perf_throughput="n/a"
npu_perf_note="not run"

parse_npu_perf() {
  local avg_us min_us max_us gops tops
  avg_us="$(sed -n -E 's/.*Avg NPU matmul time: ([0-9.]+)us\..*/\1/p' "$npu_run_log" | tail -n1)"
  gops="$(sed -n -E 's/.*Avg NPU gflops: ([0-9.]+).*/\1/p' "$npu_run_log" | tail -n1)"
  min_us="$(sed -n -E 's/.*Min NPU matmul time: ([0-9.]+)us\..*/\1/p' "$npu_run_log" | tail -n1)"
  max_us="$(sed -n -E 's/.*Max NPU matmul time: ([0-9.]+)us\..*/\1/p' "$npu_run_log" | tail -n1)"
  tops="$(to_tops "$gops")"
  npu_perf_domain="host run.wait"
  npu_perf_count="20"
  npu_perf_latency="mean ${avg_us:-n/a} us ($(microseconds_to_ms "$avg_us") ms), min ${min_us:-n/a} us, max ${max_us:-n/a} us"
  npu_perf_throughput="${gops:-n/a} GOPS (${tops} TOPS)"
  npu_perf_note="excludes output BO sync; warmups=10"
}

run_npu_path() {
  local npu_dir="$REPO_DIR/test/xrt/46_triton_matmul_ver4_strix_8x4_i8_i8_i32"
  if [ -n "$NPU_BUILD_DIR_OVERRIDE" ]; then
    npu_build_dir="$NPU_BUILD_DIR_OVERRIDE"
    npu_compile_note="reused build_dir=$npu_build_dir"
    if [ ! -d "$npu_build_dir" ]; then
      npu_status="WARN"
      npu_evidence="NPU build directory not found: $npu_build_dir"
      return
    fi
    printf 'Reusing NPU build directory: %s\n' "$npu_build_dir" > "$npu_build_log"
  else
    if ! run_capture "$npu_build_log" make -C "$npu_dir" BUILD_DIR="$npu_build_dir" \
        AIE_TARGET=aie2p M=1024 K=1024 N=1024 compile-xclbin; then
      npu_status="WARN"
      npu_evidence="NPU compile-xclbin failed; see $npu_build_log"
      return
    fi
  fi

  mapfile -t npu_elves < <(find "$npu_build_dir" -type f -name 'bare_matmul*_core_*.elf' | sort)
  if [ "${#npu_elves[@]}" -eq 0 ]; then
    mapfile -t npu_elves < <(find "$npu_build_dir" -type f -name '*.elf' | sort)
  fi
  npu_elf_count="${#npu_elves[@]}"
  if [ "$npu_elf_count" -eq 0 ]; then
    npu_status="WARN"
    npu_evidence="NPU build produced no per-core ELF files under $npu_build_dir"
    return
  fi

  mkdir -p "$npu_disasm_dir"
  local elf prefix log
  for elf in "${npu_elves[@]}"; do
    prefix="$(basename "${elf%.elf}")"
    log="$OUTDIR/logs/npu_${prefix}.log"
    if ! run_capture "$log" "$DISASSEMBLE" npu --kind elf \
        --mcpu aie2p --triple aie2p-none-unknown-elf \
        --output-dir "$npu_disasm_dir" --prefix "$prefix" "$elf"; then
      npu_disasm_failures=$((npu_disasm_failures + 1))
    fi
  done

  local insts="$npu_build_dir/air.insts.bin"
  if [ -f "$insts" ]; then
    if run_capture "$OUTDIR/logs/npu_air_insts.log" "$DISASSEMBLE" npu \
        --kind txn --output-dir "$npu_disasm_dir" --prefix npu_air_insts "$insts"; then
      npu_txn_note="transaction stream disassembled"
    else
      npu_txn_note="transaction stream disassembly failed; see $OUTDIR/logs/npu_air_insts.log"
    fi
  fi

  local vmac_count vload_count vstore_count acq_count rel_count
  vmac_count="$(grep -Eh '\bvmac\b' "$npu_disasm_dir"/*.disasm.s 2>/dev/null | wc -l | tr -d ' ')"
  vload_count="$(grep -Eh '\bvld[ab]?\b|\bvlda\b|\bvldb\b' "$npu_disasm_dir"/*.disasm.s 2>/dev/null | wc -l | tr -d ' ')"
  vstore_count="$(grep -Eh '\bvst\b' "$npu_disasm_dir"/*.disasm.s 2>/dev/null | wc -l | tr -d ' ')"
  acq_count="$(grep -Eh '\bacq\b' "$npu_disasm_dir"/*.disasm.s 2>/dev/null | wc -l | tr -d ' ')"
  rel_count="$(grep -Eh '\brel\b' "$npu_disasm_dir"/*.disasm.s 2>/dev/null | wc -l | tr -d ' ')"
  npu_artifacts="$npu_disasm_dir"
  npu_evidence="$npu_compile_note, core_elves=$npu_elf_count, disasm_failures=$npu_disasm_failures, vmac=$vmac_count, vloads=$vload_count, vstores=$vstore_count, acq=$acq_count, rel=$rel_count, $npu_txn_note"
  if [ "$npu_disasm_failures" -eq 0 ] && [ "$vmac_count" -gt 0 ]; then
    npu_status="PASS"
  else
    npu_status="WARN"
  fi

  if [ "$RUN_NPU" -eq 1 ]; then
    if [ ! -x "$npu_build_dir/test.exe" ]; then
      run_capture "$npu_build_test_log" make -C "$npu_dir" BUILD_DIR="$npu_build_dir" \
        AIE_TARGET=aie2p build-test-exe || true
    fi
    if [ -x "$npu_build_dir/test.exe" ] && [ -f "$npu_build_dir/air.xclbin" ] &&
       [ -f "$npu_build_dir/air.insts.bin" ] &&
       run_capture_in_dir "$npu_build_dir" "$npu_run_log" ./test.exe \
        -x air.xclbin -k MLIR_AIE -i air.insts.bin -M 1024 -K 1024 -N 1024 -v 0; then
      npu_run_note="ran; see $npu_run_log"
      parse_npu_perf
    else
      npu_run_note="profile failed; see $npu_run_log"
      npu_perf_note="profile failed; see $npu_run_log"
      [ "$npu_status" = "PASS" ] && npu_status="WARN"
    fi
  fi
}

echo "Writing artifacts under $OUTDIR"
[ "$SKIP_CPU" -eq 1 ] || run_cpu_path
[ "$SKIP_GPU" -eq 1 ] || run_gpu_path
[ "$SKIP_NPU" -eq 1 ] || run_npu_path

{
  echo "# GEMM ISA Utilization Report"
  echo
  echo "Artifacts: \`$OUTDIR\`"
  echo
  echo "## Comparison Contract"
  echo
  echo "| Field | Value |"
  echo "| --- | --- |"
  echo "| Shape | M=N=K=1024 |"
  echo "| Inputs | int8 A, int8 B, values 0..7 |"
  echo "| Output | int32 C |"
  echo "| Operation count | 2 * M * N * K integer ops |"
  echo "| CPU note | CPU hot loop uses a transposed B buffer to expose contiguous VNNI dot products. |"
  echo "| Timing note | Timing logs are optional context; the verdicts below are based on disassembled machine code. |"
  echo
  echo "## Verdicts"
  echo
  echo "| Backend | Status | Evidence | Artifacts | Runtime |"
  echo "| --- | --- | --- | --- | --- |"
  echo "| CPU | $cpu_status | $cpu_evidence | \`$cpu_artifacts\` | $cpu_run_note |"
  echo "| GPU | $gpu_status | $gpu_evidence | \`$gpu_artifacts\` | $gpu_run_note |"
  echo "| NPU | $npu_status | $npu_evidence | \`$npu_artifacts\` | $npu_run_note |"
  echo
  echo "## Performance"
  echo
  echo "| Backend | Timing domain | Count | Latency | Throughput | Notes |"
  echo "| --- | --- | --- | --- | --- | --- |"
  echo "| CPU | $cpu_perf_domain | $cpu_perf_count | $cpu_perf_latency | $cpu_perf_throughput | $cpu_perf_note |"
  echo "| GPU | $gpu_perf_domain | $gpu_perf_count | $gpu_perf_latency | $gpu_perf_throughput | $gpu_perf_note |"
  echo "| NPU | $npu_perf_domain | $npu_perf_count | $npu_perf_latency | $npu_perf_throughput | $npu_perf_note |"
  echo
  echo "## Source Kernels"
  echo
  echo "- CPU: \`test/cpu/int8_gemm/int8_gemm.cpp\`, symbol \`cpu_i8_gemm_vnni\`."
  echo "- GPU: \`test/gpu/int8_gemm/air_sync.mlir\`, compiled for \`$GPU_CHIP\`."
  echo "- NPU: \`test/xrt/46_triton_matmul_ver4_strix_8x4_i8_i8_i32\`, compiled with \`AIE_TARGET=aie2p M=1024 K=1024 N=1024\`."
  echo
  echo "## CPU Disassembly Preview"
  echo
  echo '```asm'
  append_file_preview "$cpu_disasm" 60
  echo '```'
  echo
  echo "## GPU Summary Preview"
  echo
  echo '```text'
  append_file_preview "$gpu_summary" 80
  echo '```'
  echo
  echo "## NPU Disassembly Preview"
  echo
  echo '```asm'
  first_npu_disasm=""
  if [ -d "$npu_disasm_dir" ]; then
    first_npu_disasm="$(find "$npu_disasm_dir" -type f -name '*.disasm.s' | sort | head -n1 || true)"
  fi
  if [ -n "$first_npu_disasm" ]; then
    append_file_preview "$first_npu_disasm" 80
  else
    echo "No NPU core disassembly available."
  fi
  echo '```'
  echo
  echo "## Logs"
  echo
  echo "- CPU build: \`$cpu_log\`"
  echo "- CPU disassembly: \`$cpu_disasm_log\`"
  echo "- GPU disassembly: \`$gpu_log\`"
  echo "- NPU build: \`$npu_build_log\`"
  echo "- CPU runtime: \`$cpu_run_log\`"
  echo "- GPU runtime: \`$gpu_run_log\`"
  echo "- NPU runtime: \`$npu_run_log\`"
} > "$REPORT"

echo "Report: $REPORT"

if [ "$STRICT" -eq 1 ]; then
  for status in "$cpu_status" "$gpu_status" "$npu_status"; do
    case "$status" in
      PASS|SKIP) ;;
      *) exit 1 ;;
    esac
  done
fi
