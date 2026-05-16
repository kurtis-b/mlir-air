# Backend ISA And Disassembly Workflow

`utils/isa_inspect/disassemble.sh` inspects CPU, GPU, and NPU artifacts.

## Setup

Source the matching backend environment before use:

```bash
source utils/env_setup_gpu.sh install-gpu llvm/install-amdgpu
source utils/env_setup.sh install-xrt <mlir-aie-install> <llvm-aie-install> my_install/mlir
source /opt/xilinx/xrt/setup.sh
```

Useful overrides: `AIR_GPU_CHIP`, `AIR_OPT`, `MLIR_OPT`, `MLIR_AIR_INSTALL_DIR`,
`LLVM_INSTALL_DIR`, `LLVM_READOBJ`, `LLVM_OBJDUMP`, `LLVM_NM`, `PEANO_INSTALL_DIR`, and `AIEBU_DUMP`.

## Examples

```bash
utils/isa_inspect/disassemble.sh cpu path/to/int8_gemm_cpu --symbol cpu_i8_gemm_vnni --expect vpdpbusd
utils/isa_inspect/disassemble.sh gpu test/gpu/int8_gemm/air_sync.mlir --gpu-arch gfx1150 --expect v_wmma_i32_16x16x16_iu8 --forbid 'v_wmma_.*16x16x64|v_swmmac|swmmac'
utils/isa_inspect/disassemble.sh npu path/to/core.elf --kind elf --expect movxm
utils/isa_inspect/disassemble.sh npu path/to/air.insts.bin --kind txn --expect XAIE_IO_
```

Run `utils/isa_inspect/disassemble.sh --help` or `<backend> --help` for all options.

## Output Contract
Each mode writes backend artifacts plus `${prefix}.summary.txt`. CPU emits headers,
dynamic info, symbols, and disassembly; GPU emits lowered MLIR, ISA, HSACO,
code-object metadata, and final MLIR; NPU emits AIE ELF or transaction-stream disassembly/profile files.
