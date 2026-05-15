# Backend ISA And Disassembly Workflow

`utils/isa_inspect/disassemble.sh` provides one local CLI for compile-time ISA
inspection across CPU, GPU, and NPU artifacts.

| Backend | Local target | Primary artifact | Primary tool |
|---------|--------------|------------------|--------------|
| CPU | `x86_64` | host object, shared library, executable | `llvm-readobj`, `llvm-objdump` |
| GPU | `gfx1150` | AMDGPU code object embedded in `gpu.binary` | `mlir-opt` with `gpu-module-to-binary` |
| NPU core | `aie2p` | per-core AIE ELF | Peano `llvm-objdump` |
| NPU runtime stream | `aie2txn` | `air.insts.bin` transaction/control binary | XRT `aiebu-dump` |

Run `utils/isa_inspect/disassemble.sh --help` or
`utils/isa_inspect/disassemble.sh <cpu|gpu|npu> --help` for option details.

## Setup

Use the existing environment setup scripts directly.

```bash
source utils/env_setup_gpu.sh install-gpu llvm/install-amdgpu
source utils/env_setup.sh install-xrt <mlir-aie-install> <llvm-aie-install> my_install/mlir
source /opt/xilinx/xrt/setup.sh
```

Useful overrides include:

```bash
export AIR_GPU_CHIP=gfx1150
export AIR_OPT=/path/to/air-opt
export MLIR_OPT=/path/to/mlir-opt
export LLVM_INSTALL_DIR=/path/to/llvm/install
export LLVM_READOBJ=/path/to/llvm-readobj
export LLVM_OBJDUMP=/path/to/llvm-objdump
export LLVM_NM=/path/to/llvm-nm
export PEANO_INSTALL_DIR=/path/to/llvm-aie
export AIEBU_DUMP=/opt/xilinx/xrt/bin/aiebu-dump
```

The CLI also checks common repo-local install locations before falling back to
tools on `PATH`.

## Examples

CPU host artifact:

```bash
utils/isa_inspect/disassemble.sh cpu install-gpu/lib/libairgpu.so \
  --output-dir /tmp/air_cpu_isa \
  --symbol mgpuLaunchKernel \
  --expect 'Disassembly of section|file format elf64-x86-64'
```

GPU AIR input:

```bash
utils/isa_inspect/disassemble.sh gpu test/gpu/int8_gemm/air_sync.mlir \
  --gpu-arch gfx1150 \
  --output-dir /tmp/air_gpu_isa \
  --expect v_wmma_i32_16x16x16_iu8 \
  --forbid 'v_wmma_.*16x16x64|v_swmmac|swmmac'
```

NPU core ELF or transaction stream:

```bash
utils/isa_inspect/disassemble.sh npu path/to/segment_0_core_0_2.elf \
  --kind elf \
  --output-dir /tmp/air_npu_core_isa \
  --expect movxm
```

For `air.insts.bin`, use the same `npu` subcommand with `--kind txn` or the
default `--kind auto`, and expect `XAIE_IO_` control-stream operations.

## Outputs

Each subcommand writes backend-specific artifacts plus `*.summary.txt`. GPU
inspection preserves `${prefix}.final.mlir` for runtime smoke scripts and emits
readable ISA as `${prefix}.isa.s`.

## References

- Intel 64 and IA-32 Architectures Software Developer's Manual:
  <https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html>
- AMD64 Architecture Programmer's Manual:
  <https://www.amd.com/en/developer/resources/developer-guides-manuals.html>
- LLVM command guides:
  <https://llvm.org/docs/CommandGuide/>
- MLIR GPU dialect serialization:
  <https://mlir.llvm.org/docs/Dialects/GPU/>
- LLVM AMDGPU backend and code-object metadata:
  <https://llvm.org/docs/AMDGPUUsage.html>
- Xilinx AIEBU:
  <https://github.com/Xilinx/aiebu>
