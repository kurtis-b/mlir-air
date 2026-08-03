# 10 — Phase G: Unattended Runner and CI

Drive the whole suite unattended, across reboots, with resumable checkpointing. Then wire what
can safely run in CI.

## The runner

Port `unattended_reboot.py` (2494 lines) and `test_unattended_reboot.py` (1790).

The job plan is a data table of `(job_id, description, module, argv, privileged_setup)` dicts
built by `build_job_plan()` over families x sequence ladder x block kinds, then families x ladder
x execution modes, then the helper studies, exports, plots and manifest. **Retargeting is
confined to that function.**

Behaviour to preserve:

- **State** — `<results_root>/automation/state.json` holding `current_job_index`, per-job
  status, `reboot_command`, `baseline_temperature_c`, `normal_ttm_pages_limit`, and pending
  reboot actions. Per-job logs under `automation/logs/`.
- **Resume idempotence** — every job is passed `--resume-input <its own output CSV>`, so resume
  is idempotent at both the job and the row level. `resume` re-arms the hook, retries the failed
  job, and continues.
- **The `@reboot` crontab hook** — installed in the *invoking user's* crontab, tagged
  `# transformer-layer-unattended:<state path>`, removed on completion, on `stop`, and on
  permanent failure. Running `start` under `sudo` puts it in root's crontab instead — document
  that as a footgun.
- **Thermal gating** between jobs via `sensors`, falling back to `rocm-smi`.
- **Power-mode enforcement** — `npu_runtime_checks.require_npu_power_mode_turbo()` parses
  `xrt-smi --batch examine -f JSON -r all`, falling back to a text regex.

### Two fixes iron made that must carry over

1. **TTM page-limit comparison within 1%, not one page.** The kernel derives that default from
   boot-time available memory, so a reboot onto a different kernel shifted it by 602 pages, the
   clear-TTM step could never be satisfied, and the queue halted at job 885 of 888 to avoid a
   reboot loop. A 1% band still rejects the deliberate 26 GB override.
2. **Guard the empty-mask column drop** in `plot_selected_component_groups_vs_pattern`. Selecting
   rows with a plain list mask means that when no row matches the execution mode, the empty list
   is read as a *column* selection, dropping every column, and the next lookup raises
   `KeyError: '_pair_key'`. Sparse result trees hit this routinely.

### Convention rule 5

At 2494 lines, this module is three times the repository's norm. Split it along the seams it
already has: job planning, state persistence, crontab hook, thermal gate, TTM transitions,
reboot orchestration, CLI.

## Job counts

`[Codex]` **Do not hard-code iron's 888 / 834 / 21 / 3 counts as acceptance criteria.** Those
describe iron's case matrix; this port's matrix will differ, and a hard-coded number becomes
either a false failure or a rubber stamp.

Generate expected counts from the checked-in suite profile and validate them in the manifest.

For context, iron's profiles were:

| Profile | Jobs | Observed wall clock |
|---|---|---|
| `full` (default) | 888 | 11 h to 2 days |
| `paper` | 834 | ~20 h; helper studies restricted to seq_len 512/2048/8192 |
| `execution-smoke-test` | 21 | minutes |
| `smoke-test` | 3 | seconds; measures nothing |

Wall clock swings by a factor of four depending on how warm `build/` is — compilation, not
measurement, dominates a cold run. Preserve `build/` between runs.

## Host prerequisites

Document these in the example README. The runner shells out to all of them, and a missing tool
fails mid-suite rather than at start unless checked.

| Tool | Package | Used for |
|---|---|---|
| `xrt-smi` | XRT | NPU power mode; probed at `/opt/xilinx/xrt/bin`, `/usr/local/bin`, `/usr/bin` |
| `amd-ttm` | `amd-debug-tools` | TTM page-limit transitions |
| `turbostat` | `linux-tools-$(uname -r)` | Package power for every end-to-end row |
| `sensors` | `lm-sensors` | Thermal gate between jobs |
| `rocm-smi` | ROCm | iGPU power, and the thermal-gate fallback |
| `crontab` | `cron` | The `@reboot` continuation hook |

`turbostat` is tied to the running kernel — `/usr/bin/turbostat` is only a dispatcher, and the
real binary ships per-kernel. Since this suite performs reboots, a kernel change mid-run can
leave every power row failing. Check with `sudo -n turbostat --version`.

Passwordless sudo is required because the runner executes from crontab and cannot answer a
prompt:

```
<user> ALL=(root) NOPASSWD: /usr/bin/xrt-smi, /opt/xilinx/xrt/bin/xrt-smi, \
                            /usr/local/bin/amd-ttm, /usr/bin/turbostat, \
                            /usr/sbin/reboot, /usr/bin/systemctl reboot
```

## NPU serialization

Wrap hardware jobs in the repository-wide convention:

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock <command>
```

`KernelCache` deliberately uses a *different* inode (`/tmp/npu.lock`) to avoid flock
self-deadlock. **Do not unify them.**

## CI wiring

Add to `programming_examples/CMakeLists.txt`, following the existing `add_lit_testsuite`
pattern:

```cmake
add_lit_testsuite(check-programming-examples-transformer-layer
  "Running transformer-layer execution-study tests (compile-only)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS} -j1 --filter "transformer_layer/.*/run_npu2_compile"
)
```

- **Compile-only is PR-gate-safe** — no NPU, no HF token, no secrets.
- **The measurement suite stays opt-in.** It runs 11 h to 2 days and needs NPU + iGPU on one
  host with passwordless sudo and reboot rights. That belongs to a dedicated runner invoked by
  `workflow_dispatch`, not to PR CI.
- `-j1` on any hardware suite is mandatory — concurrent NPU work causes OOM and contention.

## Work items

1. Split `unattended_reboot.py` along its seams (rule 5).
2. Retarget `build_job_plan()` to the ported studies.
3. Carry over the TTM 1% comparison and the empty-mask plot guard.
4. Derive expected job counts from the profile; validate in the manifest.
5. Write the prerequisites and recovery sections of the example README.
6. Register the compile-only lit suite in CMake.
7. Decide the opt-in workflow for the measurement suite.

## Gate

A full profile run completes, and `results_manifest.json` shows no missing files or rows —
against counts derived from the profile itself, not hard-coded.

## Risks

- **The suite needs a dedicated machine.** NPU + iGPU on one host, passwordless sudo, reboot
  rights, ~2.4 GB per results root, and up to two days of wall clock.
- A healthy iron run rebooted exactly twice: once into a 26 GB TTM page limit before the six
  16384-token iGPU jobs, and once back. Reboot handling is genuinely exercised, not theoretical.
- Installing a crontab hook on a shared machine affects other users of that machine. Confirm
  before enabling.
