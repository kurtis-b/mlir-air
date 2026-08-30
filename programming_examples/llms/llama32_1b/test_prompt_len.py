# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host regression for the prompt-length seam of the two Llama drivers (no
device, no HF): the REAL prompt length must flow run_once -> generate ->
run_npu_prefill. Instruct templates put EOS inside the prompt (Llama-3's
<|eot_id|> after every message, SmolLM2's ChatML), so the old "count the
non-EOS tokens" recovery landed pred_pos two tokens early; the fake prompt
below carries an EOS mid-prompt so that count is one short. Each driver runs
in a child process (they chdir on import).

    python3 test_prompt_len.py          (from programming_examples/llms/llama32_1b)
"""

import contextlib
import inspect
import os
import subprocess
import sys
import types

DRIVERS = {
    "llama32_1b": "llama32_1b_inference",
    "llama32_1b_int4": "llama32_1b_int4_inference",
}
TOKENS = [1, 5, 6, 2, 1, 7, 8]  # EOS (=2) inside the prompt, template-style
SEQ_LEN = 16


class _Captured(Exception):
    def __init__(self, kw):
        self.kw = kw


def _raise(*_a, **kw):
    raise _Captured(kw)


class _FakeTok:
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "templated:" + messages[0]["content"]

    def encode(self, text):
        return list(TOKENS)


def _check_driver(sub):
    ddir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", sub)
    os.chdir(ddir)
    sys.path.insert(0, ddir)
    mod = __import__(DRIVERS[sub])
    tok = _FakeTok()
    nullctx = lambda name: contextlib.nullcontext()  # noqa: E731
    session = types.SimpleNamespace(
        tokenizer=tok,
        seq_len=SEQ_LEN,
        model_variant="instruct",
        prefill_cache=types.SimpleNamespace(
            profiler=types.SimpleNamespace(time_cpu=nullctx)
        ),
        weights=None,
        config=None,
        decode_cache=None,
        rope_lut_bf16=None,
    )
    params = inspect.signature(mod.run_npu_prefill).parameters
    assert "prompt_len" in params and params["prompt_len"].default is None, sub
    mod.generate = _raise  # run_once must hand generate the pre-pad length
    try:
        mod.run_once(session, "hi", n_tokens=4)
        raise AssertionError(f"{sub}: run_once did not call generate")
    except _Captured as e:
        got = e.kw.get("prompt_len")
        assert got == len(
            TOKENS
        ), f"{sub}: run_once passed prompt_len={got}, want {len(TOKENS)}"
    print(f"PASS  {sub}: run_once -> generate carries prompt_len={len(TOKENS)}")


def main():
    if len(sys.argv) > 1:  # child: one driver
        _check_driver(sys.argv[1])
        return 0
    n_pass = 0
    for sub in DRIVERS:
        p = subprocess.run(
            [sys.executable, __file__, sub], capture_output=True, text=True
        )
        print(p.stdout.strip() or p.stderr.strip()[-600:])
        n_pass += p.returncode == 0
    print(f"prompt_len seam tests: {n_pass}/{len(DRIVERS)} passed")
    return 0 if n_pass == len(DRIVERS) else 1


if __name__ == "__main__":
    sys.exit(main())
