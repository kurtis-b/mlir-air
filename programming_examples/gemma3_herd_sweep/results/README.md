# Gemma3 Paper Result Artifacts

This directory is reserved for small Gemma3 paper-comparison result artifacts:
JSON result cells, Markdown summaries, and CSV summaries. Large model weights,
tokenizer caches, xclbins, ELFs, trace dumps, and debug IR should stay out of
source control unless they are reviewed as compact fixtures.


## Static Preload Evidence

- `gemma3_static_preload_evidence.json`: compact Strix/XRT evidence that
  `gemma3-1b`, `gemma3-4b`, and the `gemma3-4b-vision` text stack
  serialized and wrote all planned text projection tensors into one contiguous
  XRT BO per model variant. This is static-weight preload evidence only; it is not a model
  kernel launch, correctness, timing, or paper-parity result.


## BO Allocation Evidence

- `gemma3_bo_allocation_evidence.json`: compact Strix/XRT evidence for full
  paper-shape BO allocation. The current entries validate `gemma3-1b` at 32k
  prompt and 32k decode context, and record that `gemma3-4b` and
  `gemma3-4b-vision` hit the local XRT host-memory allocation limit at the
  first 9,126,805,504-byte KV-cache BO after allocating 4,454,893,568 bytes.
  This is allocation evidence only; it is not a kernel launch, correctness,
  timing, or paper-parity result.
