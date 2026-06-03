# Gemma3 Paper Result Artifacts

This directory is reserved for small Gemma3 paper-comparison result artifacts:
JSON result cells, Markdown summaries, and CSV summaries. Large model weights,
tokenizer caches, xclbins, ELFs, trace dumps, and debug IR should stay out of
source control unless they are reviewed as compact fixtures.


## Static Preload Evidence

- `gemma3_static_preload_evidence.json`: compact Strix/XRT evidence that
  `gemma3-1b` serialized and wrote all planned text projection tensors into
  XRT BOs. This is static-weight preload evidence only; it is not a model
  kernel launch, correctness, timing, or paper-parity result.
