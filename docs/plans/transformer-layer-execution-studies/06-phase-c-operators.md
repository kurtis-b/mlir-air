# 06 — Phase C: Operators

Re-express iron's six new operators as AIR builders.

The `design.py` files (~4.4k lines against `aie.iron` ObjectFifo / Worker / Runtime /
TensorAccessPattern / SequentialPlacer) are the part that must genuinely be rewritten — they
have no counterpart in AIR's launch-segment-herd model. The `reference.py` torch oracles port
verbatim.

## Shape of the result

Per convention rules 1, 3 and 4, each iron `<op>/{op,design}.py` pair collapses into a single
`build_<name>_module(...)` function returning an `air.ir.Module`. There is no operator class and
no artifact-DAG file-loading seam. The torch reference moves into the same module as a
module-level function, following
`programming_examples/weighted_rms_norm/weighted_rms_norm.py`.

## The operators

| iron operator | design.py | Approach in MLIR-AIR |
|---|---|---|
| `causal_mask` | — (86 L op.py) | **Not an operator.** Pure composition: `elementwise_add` with a torch-precomputed triangular mask registered as a static buffer. Becomes a builder keyword argument. Free once eltwise-add is wired. |
| `qkv_proj` | 561 | GEMM `A(M,K) @ B(K,3K)` with C split three ways at the runtime-sequence level. Closest existing analogue: `shared/builders/rms_gemms_rope_multi.py` minus RMSNorm and RoPE. Reuse `shared/builders/gemm_builder.py`. |
| `addnorm` | 382 | Weighted LayerNorm + residual over the new `add_layer_norm_rows` kernel; extend `programming_examples/layer_norm/`. **Change from iron:** it bakes weights into the MLIR via `np.load()` at generation time and hashes them into the artifact name. Pass weights as a runtime memref argument instead — otherwise every weight change forces a recompile. |
| `ffn` | 1096 | Staged up-projection → fused GeLU → down-projection with `down_proj_depth` memory-tile accumulation staging. `programming_examples/ffn_swiglu/` is SwiGLU-shaped; this is a new GeLU-shaped builder. |
| `mha_out_proj` | 1350 | Largest. Fused attention + output projection, optional causal masking, `parallel_seq` / `parallel_heads` / `o_proj_acc_depth` knobs. Compose from `flash_attention/kernel_fusion_based/` plus the O-projection half of `o_ffn_multi.py`. |
| `dynamic_gemm` | 1009 | Standalone GEMM with runtime-sequence M/N tail handling. **The structural blocker** — see below. iron's documented constraints: non-batched only, `num_aie_columns ∈ {4,8}`, `c_col_maj=False`, no bootstrap M-tail without a preceding full row block. |

Convention rule 5 applies: `mha_out_proj` (1350), `ffn` (1096) and `dynamic_gemm` (1009) all
exceed the repository's ~800-line norm and should be split along their internal staging seams.

## The shape-coverage problem

`registry_lookup.gemm_config()` **raises** on an unmeasured `(M, K, N)` rather than guessing —
deliberately, because hand-copied tile configs previously caused drift bugs
(`kernel_registry/registry_lookup.py`). The two registry JSONs hold **40 measured shapes total**
(33 bf16-out + 7 f32-out).

iron's case matrix is 6 families x 9 sequence lengths x ~8 GEMM roles — several hundred distinct
shapes, roughly an order of magnitude more than the registry has ever held.

### Resolution, in preference order

1. **Make the `block` study double as the registry sweep tool.** iron's `block/run.py` already
   sweeps per-operator candidates and records a winner per `(operator, shape)`. Wire its output
   into `kernel_registry/details/*.json` so ladder coverage accrues as a byproduct rather than
   as separate manual work. This is also convention rule 9 — registry as the single source of
   truth.
2. **Implement runtime M/N tail handling** (iron's `dynamic_gemm`) so one compiled kernel spans
   the ladder.
3. **Add an explicit unregistered-shape heuristic path** to `gemm_builder.py` that records what
   it guessed, so guesses are visible rather than silent.

`[Codex]` **Make the full planned shape matrix an explicit Phase C acceptance condition**, or
restrict the case matrix to registered shapes. Deciding this late will strand Phase E.

## Work items

1. Wire `causal_mask` as a builder keyword argument over the eltwise-add path.
2. `build_qkv_proj_module(...)` reusing `gemm_builder`.
3. `build_addnorm_module(...)` with runtime weight arguments.
4. `build_ffn_module(...)` — GeLU-shaped, split per rule 5.
5. `build_mha_out_proj_module(...)` — split per rule 5.
6. Resolve the shape-coverage decision; implement whichever of the three options is chosen.
7. Port the six torch reference oracles into their builder modules.
8. Append registry rows for every validated `(kernel, shape)`.

## Gate

Three conditions, all required:

1. **Numerics** — each operator matches an FP32 numpy reference under `np.isclose` at the
   registry's `rtol` / `atol`. This is the repository's stated standard; the `kernel_registry`
   README is explicit that the gate is element-wise closeness, **not** cosine similarity.
2. **Registration** — a row is appended to both `kernel_registry/supported_kernels.md` and
   `details/<Kernel>_bf16.md`, carrying `mean_rel_L1`, `Used by`, and status.
3. **Coverage** — every shape the case matrix needs is either registered or provably covered by
   the dynamic path.

## Risks

- **Shape coverage is the biggest unknown in this phase.** An order-of-magnitude registry
  expansion is a large sweep; the alternative (runtime tail handling) is a significant kernel
  design effort. Neither is small.
- `mha_out_proj` is the largest single rewrite in the port and depends on FlashAttention
  behaviour that Goal 1 will later modify. Coordinate the two.
- Touching `shared/builders/` affects the shipped LLM deployments — re-run `make verify`.
