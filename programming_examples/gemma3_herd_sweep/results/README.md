# Gemma3 Paper Result Artifacts

This directory is reserved for small Gemma3 paper-comparison result artifacts:
JSON result cells, Markdown summaries, and CSV summaries. Large model weights,
tokenizer caches, xclbins, ELFs, trace dumps, and debug IR should stay out of
source control unless they are reviewed as compact fixtures.

## Initial 1k CPU/iGPU/NPU Paper-Cell Evidence

The initial 1B 1k baseline cells use prompt length 1024, one warmup iteration,
three timed iterations, and 16 decode tokens. The timed region excludes model
load, tokenizer work, input construction, device placement, compile, BO
creation/preload, xclbin/ELF load, and kernel argument setup.

| File | Backend | Metric | Local | Paper | Classification | Power |
| --- | --- | --- | ---: | ---: | --- | --- |
| `gemma3_1b_cpu_prefill_1k_initial.json` | CPU/HF | Prefill TTFT | 1.430773033 s | 4.06 s | `EXPLAINED_DEVIATION` | 45.643 W RAPL package/total |
| `gemma3_1b_cpu_decode_1k_initial.json` | CPU/HF | Decode TPS | 12.400321286 | 41.9 | `EXPLAINED_DEVIATION` | 45.727 W RAPL package/total |
| `gemma3_1b_igpu_prefill_1k_initial.json` | iGPU/HF ROCm | Prefill TTFT | 0.527177805 s | 0.51 s | `PAPER_MATCH` | 37.273 W ROCm SMI GPU rail |
| `gemma3_1b_igpu_decode_1k_initial.json` | iGPU/HF ROCm | Decode TPS | 13.738045814 | 38.0 | `EXPLAINED_DEVIATION` | 42.871 W ROCm SMI GPU rail |
| `gemma3_1b_npu_prefill_1k_blocked_initial.json` | NPU | Prefill TTFT | blocked | 0.95 s | `REAL_MODEL_EXECUTION_NOT_IMPLEMENTED` | staged diagnostic payload attached |
| `gemma3_1b_npu_decode_1k_blocked_initial.json` | NPU | Decode TPS | blocked | 41.1 | `REAL_MODEL_EXECUTION_NOT_IMPLEMENTED` | staged diagnostic payload attached |

The iGPU cells set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` and use ROCm SMI
for timed-window GPU rail sampling. CPU cells use direct RAPL sysfs package
energy through the `power` group. iGPU CPU/total rails remain
`MISSING_POWER_FIELD`; official NPU timing and pseudo-NPU paper-cell power remain
blocked by full 1B loop wiring and fresh paper-shape hardware reruns. The prior
1B staged nonlinear blocker is retired for the current full-layer diagnostic
because it records `host_fallbacks=[]`; 4B and vision still need equivalent
composed evidence before that blocker can be narrowed. Kernel argument-layout
validation is complete for the real 1B 1k/32k context model-runner plan: 728
NPU candidate layouts, 2,236 positional
arguments, and zero binding blockers. Staged decode layer-0 correctness and
segmented kernel-only timing evidence is now present and attached under
`npu_staged_diagnostic` in the blocked NPU paper-cell JSONs, but official NPU
`local_value` remains null because this is not a timed paper-cell run.

`gemma3_1b_initial_1k_results.json` bundles these six cells, and
`gemma3_1b_initial_1k_summary.md` / `gemma3_1b_initial_1k_summary.csv` contain
the generated paper-target comparison summary.



## Nonlinear Kernel Evidence

- `gemma3_residual_add_smoke.json`: compact Strix/XRT evidence that the
  Gemma3 BF16 residual-add AIR wrapper compiles and runs as an ELF hardware
  smoke for the 1B hidden-size vector shape (`n=1152`, `tile_n=288`). This
  promotes residual add from pure host fallback to standalone NPU candidate
  status in the wiring manifest, but it is not yet model-loop timing or
  paper-parity evidence.
- `gemma3_rope_halfsplit_smoke.json`: compact Strix/XRT evidence that the
  Gemma3 half-split RoPE AIR wrapper compiles and runs as an ELF hardware
  smoke for `rows=4`, `head_dim=256`, and `herd_x=4`. This promotes RoPE to
  standalone NPU candidate status in the wiring manifest, but it is not yet
  model-loop timing or paper-parity evidence.
- `gemma3_geglu_smoke.json`: compact Strix/XRT evidence that the Gemma3 GeGLU
  AIR wrapper compiles and runs as an ELF hardware smoke for the 1B MLP
  activation vector shape (`n=6912`, `tile_n=288`). This promotes GeGLU from a
  compile-only candidate to 1B-sized standalone NPU candidate status and feeds
  the staged layer-0 launch evidence below.

## First Kernel Launch Probe Evidence

- `gemma3_1b_first_kernel_launch_probe.json`: compact Strix/XRT evidence that
  the promoted Gemma3 1B pre-attention RMSNorm shape (`1024x1152`) launches as
  an ELF on the NPU with the validated first-stage positional layout
  (`layer_input`, `static_norm_weights`, `prefill_L0_pre_attention_norm`). The
  worker passes the full contiguous `static_norm_weights` payload as argument 1,
  allocates/binds the three pyxrt BOs directly, and uses the actual layer-0
  `input_layernorm.weight` vector at byte offset 0. It validates with output
  correlation 0.999983 against the standalone CPU reference. This is first-stage
  launch evidence only; it is not a substep sequence, full model-runner launch,
  TTFT/TPS timing, pseudo-NPU power, or paper-parity result.


## Decode Substep Probe Evidence

- `gemma3_1b_decode_rmsnorm_qproj_substep_probe.json`: compact Strix/XRT
  evidence for a real Gemma3 1B decode substep. It launches layer-0 RMSNorm
  with the full contiguous `static_norm_weights` payload, then launches five
  FusedDQP q-projection col blocks with real `q_proj.weight` and accumulates the
  partial outputs on the host. It validates RMSNorm correlation 0.999991,
  accumulated q-projection correlation 1.000000 against the quantized FusedDQP
  reference, and dense original-weight correlation 0.994609. This is staged
  correctness evidence only; it is not full QKV, a full layer, TTFT/TPS timing,
  pseudo-NPU power, or paper-parity evidence.


## Q/K/V Substep Probe Evidence

- `gemma3_1b_decode_rmsnorm_qkv_substep_probe.json`: compact Strix/XRT
  evidence for the full Gemma3 1B decode RMSNorm-to-Q/K/V projection substep.
  It launches layer-0 RMSNorm with the full contiguous `static_norm_weights`
  payload, then launches five FusedDQP col blocks for each of q/k/v using real
  layer-0 projection weights and host accumulation. It validates RMSNorm
  correlation 0.999991, Q/K/V projection correlations
  1.000000/1.000000/1.000000 against the quantized FusedDQP references, and
  dense original-weight correlations 0.994609/0.995959/0.995720. This is staged
  correctness evidence only; it is not a full layer, TTFT/TPS timing,
  pseudo-NPU power, or paper-parity evidence.


## Stitched Decode Ingress Evidence

- `gemma3_1b_stitched_decode_ingress_probe.json`: clean-provenance Strix/XRT
  evidence for the first real Gemma3 1B stitched decode subgraph. It runs the
  eight-launch ELF `gemma3_decode_ingress_rms_qkv_qknorm_rope` for layer 0 with
  real norm and Q/K/V projection weights. The ABI has 18 public BO arguments and
  binds the RMSNorm output view (`1x1152`) and padded activation view (`5x256`)
  to the same zero-tailed BO, so no host activation packing or pad-copy launch
  is in the stitched path. It validates correlations of 0.999991 for input
  RMSNorm, 0.999990 for the padded activation view, 0.999969/0.999974/0.999973
  for Q/K/V projections, 0.999957/0.999966 for Q/K norm, and
  0.999957/0.999966 for Q/K RoPE. The single diagnostic `run.start()/wait2()`
  window is 0.009281 s, with compile, ELF load, BO creation/writes, and argument
  binding excluded. This is stitched-subgraph correctness evidence only; it is
  not a full decode layer, TTFT/TPS timing, pseudo-NPU power, or paper-parity
  evidence.


## Stitched Attention/O Evidence

- `gemma3_1b_stitched_attention_o_probe.json`: clean-provenance Strix/XRT
  evidence for the first post-ingress stitched decode slice. It runs
  `gemma3_decode_attention_o_projection`, which stitches single-token FlowQKV
  attention directly into the full-column-block O projection. The ABI has seven
  public BO arguments and aliases the attention output as both `1x4x256` for
  FlowQKV and `4x256` for FusedDQP, avoiding an attention-output layout copy.
  With real layer-0 `o_proj.weight`, it validates attention correlation
  0.999999, O-projection correlation 0.999991 against the quantized FusedDQP
  reference, and dense original-weight correlation 0.997821. The single
  diagnostic `run.start()/wait2()` window is 0.007962 s, with compile, ELF
  load, BO creation/writes, and argument binding excluded. This standalone
  slice is now also integrated in the decode-loop artifact below, but it is
  still not paper-cell timing.


## Stitched Post-Attention Residual Evidence

- `gemma3_1b_stitched_post_attention_residual_probe.json`: clean-provenance
  Strix/XRT evidence for the first post-attention tail stitched slice. It runs
  `gemma3_decode_post_attention_residual`, which stitches post-attention
  RMSNorm directly into the attention residual add. The ABI has six public BO
  arguments and aliases the RMSNorm output as both `1x1152` for weighted
  RMSNorm and `1152` for residual-add RHS, avoiding a norm-output layout copy.
  With real layer-0 `post_attention_layernorm.weight`, it validates
  post-attention RMSNorm correlation 0.999989 and attention-residual
  correlation 0.999955. The single diagnostic `run.start()/wait2()` window is
  0.000291 s, with compile, ELF load, BO creation/writes, and argument binding
  excluded. This standalone slice is now also integrated in the decode-loop
  artifact below, but it is still not paper-cell timing.


## Stitched FFN Gate/Up Evidence

- `gemma3_1b_stitched_ffn_gate_up_probe.json`: clean-provenance Strix/XRT
  evidence for the first stitched FFN slice. It runs
  `gemma3_decode_ffn_gate_up`, which stitches pre-feedforward RMSNorm directly
  into full-column-block gate/up FusedDQP projections. The ABI has eight public
  BO arguments and aliases the RMSNorm output as both `1x1152` for weighted
  RMSNorm and `5x256` for padded FusedDQP activation, avoiding host activation
  packing for the FFN ingress. With real layer-0 `pre_feedforward_layernorm`,
  `gate_proj.weight`, and `up_proj.weight`, it validates pre-FF RMSNorm
  correlation 0.999992, padded activation correlation 0.999991, gate/up
  projection correlations 0.999975/0.999975 against quantized FusedDQP
  references, and dense original-weight correlations 0.996628/0.996745. The
  single diagnostic `run.start()/wait2()` window is 0.071125 s, with compile,
  ELF load, BO creation/writes, and argument binding excluded. This standalone
  slice is now also integrated in the decode-loop artifact below, but it is
  still not paper-cell timing.


## Stitched GeGLU/Down Evidence

- `gemma3_1b_stitched_geglu_down_probe.json`: clean-provenance Strix/XRT
  evidence for the second stitched FFN slice. It runs
  `gemma3_decode_geglu_down`, which stitches the Gemma3 GeGLU activation into
  the down projection. The ABI has six public BO arguments and aliases the
  GeGLU output as both a contiguous `6912` vector and a `27x256` FusedDQP
  activation view, avoiding a host activation-layout copy. The down projection
  uses the paper-style FusedDQP builder with a trimmed 36-row-block,
  1x3-herd down-projection layout plus a streamed L1 col-block path, so the
  full 27-column-block down weight fits L1 without emitting duplicate
  memtile-to-core routes. With real layer-0 `down_proj.weight`, it validates
  GeGLU correlation 0.999975, down-projection correlation 0.999945 against the
  quantized FusedDQP reference, and dense original-weight correlation 0.996399.
  The single diagnostic `run.start()/wait2()` window is 0.087960 s, with
  compile, ELF load, BO creation/writes, and argument binding excluded. This
  standalone slice is not yet integrated into the decode-loop artifact, and it
  is still not paper-cell timing.


## Full Layer Probe Evidence

- `gemma3_1b_decode_full_layer_probe.json`: compact Strix/XRT evidence for one
  staged Gemma3 1B decode layer-0 pass. It launches pre-attention RMSNorm, Q/K
  RMSNorm, post-attention RMSNorm, pre/post-feedforward RMSNorm, single-token
  FlowQKV attention, GeGLU, and q/k/v/o/gate/up/down projection families on the
  NPU through split weighted RMSNorm, FlowQKV, GeGLU, and FusedDQP wrappers with
  real weights and runner-owned BOs. RMSNorm uses `norm_arg=selected-vector`,
  which passes the 2304-byte BF16 norm vector directly while preserving the
  recorded contiguous norm BO offset contract. It validates Q/K/post-attention/
  pre-FF/post-FF norm correlations of
  0.999988/0.999990/0.999985/0.999881/0.999983, all seven projection
  correlations at 1.000000 against quantized staged references, dense
  original-weight correlations
  0.994609/0.995959/0.995720/0.997553/0.996694/0.996806/0.997571, single-token
  FlowQKV attention correlation at 0.999998, GeGLU NPU activation correlation
  at 0.999992, and final layer-output correlation 0.999953. The refreshed JSON
  also launches Gemma half-split RoPE for Q/K at identity position 0 and both
  residual adds through the Gemma residual-add wrapper, validating RoPE
  correlations at 1.000000/1.000000 and residual correlations at
  0.999952/0.999953. It records 68 segmented NPU `run.start()/wait2()` launch
  windows totaling 0.154274 s for one staged layer in reused-ELF mode, or
  6.481976 staged layer passes/s. Its 26-layer kernel-only extrapolation is
  0.249307 decode TPS, far below the paper's 41.1 TPS 1B/1k NPU decode target
  and not a measured full-model decode TPS. Direct RAPL under `sg power`
  reports 19.027 W segmented package power and a 4.598 W pseudo-NPU
  package-delta from a 14.429 W quiescent package sample. The JSON records
  `full-1b-loop-not-wired` as the remaining model runner gap and
  `host_fallbacks=[]`; the wiring, model-runner, and reproduction-blocker
  manifests consume that evidence to drop the stale 1B nonlinear promotion
  blocker.

- `gemma3_1b_decode_full_layer_L1_probe.json`: compact Strix/XRT evidence that
  the same staged route works for layer 1 after fixing the nonzero-layer norm
  argument. The layer-1 `input_layernorm.weight` vector starts at byte offset
  10240 in the contiguous norm BO and is passed as a 2304-byte selected-vector
  argument. It validates RMSNorm correlation 0.999991, all seven projection
  correlations at 1.000000, and final layer-output correlation 1.000000. The
  JSON records 57 segmented launch windows totaling 0.142578 s, or 7.013715
  staged layer passes/s; the 26-layer kernel-only extrapolation is 0.269758
  decode TPS. Direct RAPL reports 17.420 W segmented package power and a
  5.918 W pseudo-NPU package-delta from an 11.502 W quiescent package sample.

These are staged correctness and diagnostic kernel-only timing artifacts only;
they are not a repeated model-runner loop, TTFT/TPS timing, pseudo-NPU paper
power, or paper-parity evidence.


## Decode Loop Probe Evidence

- `gemma3_1b_decode_loop_stitched_ingress_attention_o_post_attention_ffn_gate_up_geglu_down_probe.json`:
  clean-provenance Strix/XRT evidence that the stitched decode-loop route now
  covers ingress, `attention -> O projection`, `post-attention RMSNorm ->
  residual add`, `pre-FF RMSNorm -> gate/up projection`, and `GeGLU -> down
  projection` across all 26 real Gemma3 1B layers. The run uses
  `--ingress-mode stitched --attention-o-mode stitched --post-attention-mode
  stitched --ffn-gate-up-mode stitched --ffn-geglu-down-mode stitched`, preloads
  26 stitched BO sets for each integrated slice, leaves zero staged projection
  BO sets, and records `dirty_worktree=false` at commit
  `a94bd88b3412b73a08bf104b1c6eb5a7f2032e3f`. It validates every layer with no
  blockers. The measured post-warmup loop wall window is 6.331318 s, or
  0.157945 diagnostic TPS. The summed NPU `run.start()/wait2()` windows total
  5.975497 s across 182 launches, or 0.167350 kernel-only diagnostic TPS. This
  removes another 702 timed launches versus the gate/up-only stitched loop, but
  it is a performance regression from that artifact's 0.233325 loop-wall TPS
  and 0.288779 kernel-only TPS because the current streamed down-projection
  route dominates time. Full-window direct RAPL reports 11.504 W package power
  and a 1.764 W pseudo-NPU package-delta; segmented RAPL reports 11.254 W
  package power, with segmented pseudo-NPU delta clipped to 0.000 W because the
  immediate quiescent segment sample was higher than segmented package watts.
  It is still not a paper cell because attention is single-token, KV cache is
  not prefill-produced, logits/sampling are absent, and post-FF/final-residual
  work remains staged after the down projection.

- `gemma3_1b_decode_loop_stitched_ingress_attention_o_post_attention_ffn_gate_up_probe.json`:
  clean-provenance Strix/XRT evidence that the stitched decode-loop route now
  covers ingress, `attention -> O projection`, `post-attention RMSNorm ->
  residual add`, and `pre-FF RMSNorm -> gate/up projection` across all 26 real
  Gemma3 1B layers. The run uses `--ingress-mode stitched --attention-o-mode
  stitched --post-attention-mode stitched --ffn-gate-up-mode stitched`, preloads
  26 stitched BO sets for each integrated slice plus 702 remaining staged
  projection BO sets before timing, and records `dirty_worktree=false` at commit
  `d627bf804dbbccbcfc6a616dd93ea3fd7c943145`. It validates every layer with no
  blockers. The measured post-warmup loop wall window is 4.285861 s, or
  0.233325 diagnostic TPS. The summed NPU `run.start()/wait2()` windows total
  3.462861 s across 884 launches, or 0.288779 kernel-only diagnostic TPS. This
  removes 884 launches from the staged diagnostic, improves loop-wall TPS from
  0.184746 to 0.233325, and is the first stitched decode-loop diagnostic here
  whose kernel-only TPS exceeds the staged 0.274698 TPS artifact. Full-window
  direct RAPL reports 14.749 W package power and a 4.895 W pseudo-NPU
  package-delta; segmented RAPL reports 14.830 W package power, with segmented
  pseudo-NPU delta clipped to 0.000 W because the immediate quiescent segment
  sample was higher than segmented package watts. It is still not a paper cell
  because attention is single-token, KV cache is not prefill-produced,
  logits/sampling are absent, and GeGLU/down/post-FF/final-residual work remains
  staged after the gate/up projections.

- `gemma3_1b_decode_loop_stitched_ingress_attention_o_post_attention_probe.json`:
  clean-provenance Strix/XRT evidence that the stitched decode-loop route now
  covers ingress, `attention -> O projection`, and `post-attention RMSNorm ->
  residual add` across all 26 real Gemma3 1B layers. The run uses
  `--ingress-mode stitched --attention-o-mode stitched --post-attention-mode
  stitched`, preloads 26 stitched-ingress BO sets, 26 stitched attention/O BO
  sets, 26 stitched post-attention-residual BO sets, and 962 remaining staged
  projection BO sets before timing, and records `dirty_worktree=false` at commit
  `7d7da135846308cc1b171498b398cf492f1163c5`. It validates every layer with no
  blockers. The measured post-warmup loop wall window is 4.869055 s, or
  0.205379 diagnostic TPS. The summed NPU `run.start()/wait2()` windows total
  3.706862 s across 1,144 launches, or 0.269770 kernel-only diagnostic TPS.
  This removes 624 launches from the staged diagnostic and improves loop-wall
  TPS from 0.184746 to 0.205379, but kernel-only TPS remains slightly below the
  staged 0.274698 TPS artifact. Full-window direct RAPL reports 10.875 W
  package power; the pseudo-NPU package-delta is clipped to 0.000 W because the
  immediate quiescent package sample was higher than the timed-window package
  watts. It is still not a paper cell because attention is single-token, KV
  cache is not prefill-produced, logits/sampling are absent, and the FFN tail
  remains staged after the attention residual.

- `gemma3_1b_decode_loop_stitched_ingress_attention_o_probe.json`: clean-provenance
  Strix/XRT evidence that the stitched decode-loop route now covers both the
  ingress and the `attention -> O projection` slice across all 26 real Gemma3
  1B layers. The run uses `--ingress-mode stitched --attention-o-mode stitched`,
  preloads 26 stitched-ingress BO sets, 26 stitched attention/O BO sets, and
  962 remaining staged projection BO sets before timing, and records
  `dirty_worktree=false` at commit
  `885e1ad7072243e37802eaa0cc44d0dae8e5f40d`. It validates every layer with no
  blockers; final-output correlation is 0.999975 on layer 0 and 0.999706 on
  layer 25. The measured post-warmup loop wall window is 5.152988 s, or
  0.194062 diagnostic TPS. The summed NPU `run.start()/wait2()` windows total
  3.763859 s across 1,170 launches, or 0.265685 kernel-only diagnostic TPS.
  This removes 598 launches from the staged diagnostic and improves loop-wall
  TPS from 0.184746 to 0.194062, but kernel-only TPS remains slightly below the
  staged 0.274698 TPS artifact. Full-window direct RAPL reports 10.656 W
  package power and a 4.204 W pseudo-NPU package-delta. It is still not a paper
  cell because attention is single-token, KV cache is not prefill-produced,
  logits/sampling are absent, and residual/FFN/down-projection work remains
  staged after O projection.

- `gemma3_1b_decode_loop_stitched_ingress_probe.json`: clean-provenance
  Strix/XRT evidence that `--ingress-mode stitched` scales across one decode
  token and all 26 real Gemma3 1B layers. The run preloads 26 aliased
  stitched-ingress BO sets and 1,066 remaining staged projection BO sets before
  timing, validates every layer with no blockers, and records
  `dirty_worktree=false` at commit
  `671f213ba3c9c6e4d7bfe16e6bd8b46fad2be0f3`. The measured post-warmup loop
  wall window is 5.385961 s, or 0.185668 diagnostic TPS. The summed NPU
  `run.start()/wait2()` windows total 3.791450 s across 1,274 launches, or
  0.263751 kernel-only diagnostic TPS. This removes 494 launches from the
  staged 26-layer single-token diagnostic, but kernel-only TPS is still slightly
  worse than the staged 0.274698 TPS artifact, so the remaining performance work
  is post-ingress stitching and/or tuning the stitched ingress route rather than
  claiming paper parity. Full-window direct RAPL reports 10.912 W package power
  and a 3.826 W pseudo-NPU package-delta. The result is not a paper cell because
  attention remains single-token, KV cache is not prefill-produced, logits and
  sampling are absent, and the rest of the decode layer remains staged after
  ingress.

- `gemma3_1b_decode_loop_stitched_ingress_L1_probe.json`: clean-provenance
  Strix/XRT evidence that the staged decode-loop probe can now replace the
  RMSNorm/Q/K/V/QK-Norm/RoPE ingress launches with the stitched ELF
  `gemma3_decode_ingress_rms_qkv_qknorm_rope`. The run covers one real Gemma3
  1B layer and one decode token with `--ingress-mode stitched`, preloads one
  aliased stitched-ingress BO set plus 41 remaining staged projection BO sets
  before timing, and records `dirty_worktree=false` at commit
  `8dfc8524cc0ed0705c6123f4bc7b614ba3120572`. It validates stitched-ingress
  RMSNorm correlation 0.999991, Q/K/V projection correlations
  0.999971/0.999977/0.999978, and final layer-output correlation 0.999967.
  The measured post-warmup loop wall window is 0.213629 s, or 4.681005
  diagnostic layer-token/s, and the summed NPU `run.start()/wait2()` windows
  total 0.146627 s across 49 launches, or 6.820013 kernel-only
  layer-token/s. Full-window direct RAPL reports 10.792 W package power and a
  3.896 W pseudo-NPU package-delta. This is not a paper cell: it covers one
  layer rather than all 26 layers, still uses single-token attention, and leaves
  attention, O projection, residual, FFN, logits/sampling, and prefill-produced
  KV-cache work outside the stitched production path.

- `gemma3_1b_decode_loop_probe.json`: compact Strix/XRT evidence for one staged
  Gemma3 1B decode token across all 26 real layers. The probe preloads packed
  projection inputs into 1,456 runner-owned BO sets before timing, warms one
  layer to compile/load reusable ELF runners and allocate runner-owned BOs, then
  measures the post-warmup loop with static projection arguments represented by
  no-allocation metadata placeholders. It validates every layer with RMSNorm and
  final-output correlations above 0.99998 and all projection correlations at
  approximately 1.000000. The refreshed loop launches QK-Norm, RoPE,
  single-token FlowQKV attention, post/pre/post RMSNorm, GeGLU, and both
  residual adds on the NPU for every layer and records `host_fallbacks=[]`. It
  also runs a 6.902014 s untimed all-layer reference pass before the measured
  loop, so loop-wall timing excludes CPU reference/correlation checks. The
  measured post-warmup loop wall window is 5.412830 s, or 0.184746 diagnostic
  TPS, and includes dynamic BO writes plus output sync/readback. The summed NPU
  `run.start()/wait2()` windows total 3.640356 s across 1,768 launches, or
  0.274698 kernel-only diagnostic TPS. Compared with the paper's 41.1 TPS 1B/1k
  NPU decode target, those diagnostics are 99.550% and 99.332% low,
  respectively. Full-window direct RAPL reports 16.790 W package power and a
  7.494 W pseudo-NPU package-delta from a 9.295 W quiescent package sample. The
  segmented kernel-only pseudo-NPU delta is not usable in this run because its
  quiescent sample was taken while preparation was already busy. This remains
  diagnostic, not a paper cell: the default result bundle still uses the
  single-token attention path, logits/sampling are not wired, and the production
  contiguous static-weight BO route is still not complete. A separate
  `tiled-stats-1k` decode-loop artifact below integrates host-batched 1k
  tiled-stat attention in diagnostic mode, but it uses a synthetic prefill-shaped KV
  cache and host-side reduction.

- `gemma3_1b_decode_loop_tiled_stats_probe.json`: compact Strix/XRT evidence
  for one staged Gemma3 1B decode token across all 26 real layers with
  `attention_mode=tiled-stats-1k`. The probe keeps the same reusable ELF runner
  and preloaded projection BO-set path as the default decode-loop diagnostic,
  but replaces the single-token attention launch with 16 host-batched tiled-stat
  attention launches per layer over a synthetic prefill-shaped KV cache. It
  validates all 26 layers with `host_fallbacks=[]`, records a 37.129594 s
  untimed all-layer reference pass outside the measured window, and measures a
  35.815148 s post-warmup loop wall window, or 0.027921 diagnostic TPS. The
  summed NPU `run.start()/wait2()` windows total 32.934816 s across 2,158
  launches, or 0.030363 kernel-only diagnostic TPS. Compared with the paper's
  41.1 TPS 1B/1k NPU decode target, those diagnostics are 99.932% and 99.926%
  low, respectively. Direct RAPL reports 17.577 W package power and a 0.473 W
  pseudo-NPU package-delta; the delta is low because the quiescent package
  sample was close to the timed-window package average in this run. This is not
  a paper cell because the KV cache is synthetic rather than prefill-produced,
  the tiled softmax-stat reduction is host-side, logits/sampling are not wired,
  and the production contiguous static-weight BO route is still not complete.


## FlowQKV Tiled Stats Evidence

- `gemma3_flowqkv_tiled_stats_1k_smoke.json`: compact Strix/XRT evidence for a
  diagnostic Gemma3 1B 1k decode-attention shape. The wrapper compiles a
  two-tile direct-output ELF module (`q_chunk=4`, `kv_tile=32`, `head_dim=256`,
  `herd=1x2`) and reuses it across 16 host batches to cover `kv_len=1024`.
  Hardware output stats correlate 1.000000 with the tile-stat reference, with
  0.0% mismatches and max stats error 0.0000038. Merging the real NPU tile
  stats gives 0.999958 correlation against exact CPU attention and max output
  error 0.000183. This proves the full-cache L1 allocation blocker is avoidable
  for 1k attention, but it is still diagnostic because the tile-stat reduction
  is host-side. The diagnostic decode-loop probe can now call this path through
  `--attention-mode tiled-stats-1k`, but production paper-cell execution still
  needs a real prefill-constructed KV cache and NPU-side reduction. Full
  8x4 direct output remains a shim S2MM resource limit, and the attempted
  full-herd L2-gather route is still an AIE routing packet-id-0 blocker.


## Static Preload Evidence

- `gemma3_static_preload_evidence.json`: compact Strix/XRT evidence that
  `gemma3-1b`, `gemma3-4b`, and the `gemma3-4b-vision` text stack
  serialized and wrote all planned text projection tensors into one contiguous
  XRT BO per model variant. This is static-weight preload evidence only; it is not a model
  kernel launch, correctness, timing, or paper-parity result.


## BO Allocation Evidence

- `gemma3_bo_allocation_evidence.json`: compact Strix/XRT evidence for full
  paper-shape BO allocation. The current benchmark-cell entries validate
  `gemma3-1b` at 32k prompt/32k decode context with 69 BOs totaling
  1,998,196,224 bytes, and `gemma3-4b` plus `gemma3-4b-vision` at 32k
  prompt/128k decode context with 85 BOs totaling 7,261,614,080 bytes. The
  ledger also preserves earlier monolithic-KV failures where 4B text and vision
  hit the local XRT host-memory allocation limit at the first
  9,126,805,504-byte KV-cache BO after allocating 4,454,893,568 bytes. This is
  allocation evidence only; it is not a kernel launch, correctness, timing, or
  paper-parity result.


## Norm Weight Plan Evidence

- `gemma3_norm_weight_plan_evidence.json`: compact safetensor-metadata evidence
  for the BF16 norm vectors needed by RMSNorm and QK-Norm promotion. It records
  tensor counts and byte totals only; it is not XRT preload, kernel launch,
  correctness, timing, or paper-parity evidence.


## Norm Preload Evidence

- `gemma3_norm_preload_evidence.json`: compact Strix/XRT evidence that the
  RMSNorm/QK-Norm BF16 vectors for `gemma3-1b`, `gemma3-4b`, and the
  `gemma3-4b-vision` text stack were serialized and written into one contiguous
  XRT BO per variant. This is norm-weight preload evidence only; it is not
  kernel launch, correctness, timing, or paper-parity evidence.
