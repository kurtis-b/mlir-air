# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only checks on the weights-injection seam (doc 58 phase M1).

``generate_golden_reference(..., weights=...)`` replaces the drawn weight set
without moving the input activation, the draw order or the oracle;
``layer_inputs(golden, names)`` builds any mode's ordered device-input list from
that mode's own NAMES tuple. This file pins the four properties the seam is only
useful if it has, and the two traps doc 58a section 3.2 named.

    python3 pattern/test_weights_injection.py

No hardware, no MLIR, a couple of seconds. It is a ``test_*.py`` so ordinary
pytest discovery finds it too (porting convention 11).

WHAT IS AND IS NOT CHECKED HERE
    The BYTE-identity claim -- injecting the generated weights reproduces the
    whole run digest for digest, at four configurations and five modes -- is the
    SHA control ``results/item31-4a-m1-20260827/control/seam_sha.py``, which
    also compares against a baseline recorded from the pre-M1 commit. This file
    is the part that belongs in the suite: the invariants a future edit could
    break without a baseline to compare against.

THE TWO TRAPS, MADE EXECUTABLE
    doc 58a section 3.2(d): a weights design that keys a static buffer by NAME,
    or that mutates a loaded array in place, leaves ``content_key_once``'s cached
    key intact -- the pool reuses the CLEAN buffer object and the injected run
    PASSES, which is precisely what the negative control exists to catch and
    which arrives as a red gate with a confusing message.
    ``test_a_perturbed_weight_keys_differently`` pins the working half;
    ``test_in_place_mutation_defeats_the_cached_key`` pins the broken half, so
    the trap is a demonstrated behaviour rather than a warning in prose.
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(_HERE)
_PROJ_ROOT = os.path.dirname(_EXAMPLE_ROOT)
for _p in (_EXAMPLE_ROOT, _PROJ_ROOT, os.path.join(_PROJ_ROOT, "llms")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from builders.block import BLOCK_INPUT_NAMES  # noqa: E402
from pattern.fused.fused import FUSED_INPUT_NAMES  # noqa: E402
from pattern.offload.offload import OFFLOAD_INPUT_NAMES  # noqa: E402
from pattern.reference import (  # noqa: E402
    WEIGHT_DRAW_ORDER,
    check_weights,
    generate_golden_reference,
    layer_inputs,
    weight_shapes,
)
from pattern.runlist.runlist import RUNLIST_INPUT_NAMES  # noqa: E402

#: Short, because every check here is about tensor plumbing rather than about
#: numerics: the widths are the block's and only `seq_len` is cut, which is free
#: (every operator in the chain is row-parallel). `test_reference.py` uses the
#: same shape for the same reason.
SHAPE = dict(seq_len=64, hidden_size=768, intermediate_size=3072, num_heads=12)

#: Every mode's tuple, imported from the mode rather than transcribed. A mode
#: whose tuple changes without `_INPUT_SOURCES` growing a matching entry fails
#: `test_every_mode_builds_a_list_as_long_as_its_names_tuple` rather than
#: mis-indexing the injection target at run time.
MODE_NAMES = {
    "block": BLOCK_INPUT_NAMES,
    "offload": OFFLOAD_INPUT_NAMES,
    "runlist": RUNLIST_INPUT_NAMES,
    "fused": FUSED_INPUT_NAMES,
}


def _bytes(array):
    return np.ascontiguousarray(array).tobytes()


def test_the_new_parameter_does_not_move_the_generated_path():
    """``weights=None`` and no argument at all are the same run."""
    a = generate_golden_reference(**SHAPE)
    b = generate_golden_reference(**SHAPE, weights=None)
    assert _bytes(a["input"]) == _bytes(b["input"])
    for name in WEIGHT_DRAW_ORDER:
        assert _bytes(a["weights"][name]) == _bytes(b["weights"][name]), name
    for name in a["boundaries"]:
        assert _bytes(a["boundaries"][name]) == _bytes(b["boundaries"][name]), name


def test_injecting_the_generated_weights_reproduces_the_run():
    """M1's defining gate, at one shape: same weights in, same everything out.

    The SHA control checks this at four configurations and five modes and against
    a pre-M1 baseline; this is the version that runs in the suite.
    """
    for variant in ("encoder_bert", "decoder_gpt2"):
        gen = generate_golden_reference(**SHAPE, workload_variant=variant)
        inj = generate_golden_reference(
            **SHAPE, workload_variant=variant, weights=gen["weights"]
        )
        assert _bytes(inj["input"]) == _bytes(gen["input"]), variant
        for name in gen["boundaries"]:
            assert _bytes(inj["boundaries"][name]) == _bytes(
                gen["boundaries"][name]
            ), f"{variant}/{name}"
        assert _bytes(inj["output"]) == _bytes(gen["output"]), variant


def test_the_seam_hands_through_the_callers_own_array_objects():
    """Identity, not equality -- ``content_key_once`` caches by ``id(array)``.

    A per-dispatch copy would be numerically invisible and would re-hash every
    weight on every forward, which is the cost the operator's S1 rule ("content
    key once per plan") exists to avoid.
    """
    gen = generate_golden_reference(**SHAPE)
    inj = generate_golden_reference(**SHAPE, weights=gen["weights"])
    for name in WEIGHT_DRAW_ORDER:
        assert inj["weights"][name] is gen["weights"][name], name


def test_the_input_activation_is_drawn_and_not_injected():
    """Injection replaces the tensors; the activation stays the seed's draw.

    Two claims in one: the drawn input is unchanged by injection, AND it does not
    depend on WHICH weights are injected -- which is what makes two injected runs
    at one seed comparable.
    """
    gen = generate_golden_reference(**SHAPE)
    zeros = {n: np.zeros(s, dtype=bfloat16) for n, s in weight_shapes(768, 3072).items()}
    inj = generate_golden_reference(**SHAPE, weights=zeros)
    assert _bytes(inj["input"]) == _bytes(gen["input"])


def test_injected_weights_actually_reach_the_layer():
    """A different weight set is a different layer -- injection is not ignored."""
    gen = generate_golden_reference(**SHAPE)
    scaled = {n: (np.asarray(a, np.float32) * 2.0).astype(bfloat16)
              for n, a in gen["weights"].items()}
    inj = generate_golden_reference(**SHAPE, weights=scaled)
    assert _bytes(inj["output"]) != _bytes(gen["output"])


def test_every_mode_builds_a_list_as_long_as_its_names_tuple():
    """doc 58a section 3.2(b): the injection index is derived, so keep it derivable.

    ``prepared["inject"]`` is ``NAMES.index("ln1_weight")``. That is only safe
    while the list and the tuple have the same length AND the same order, which
    is what building the list FROM the tuple guarantees and what this asserts.
    """
    gen = generate_golden_reference(**SHAPE)
    for mode, names in MODE_NAMES.items():
        built = layer_inputs(gen, names)
        assert len(built) == len(names), mode
        idx = names.index("ln1_weight")
        assert built[idx] is gen["weights"]["ln1_weight"], mode
        assert built[0] is gen["input"], mode


def test_the_fused_and_split_modes_disagree_only_about_the_qkv_weight():
    """The two tuple shapes are the same tensors, packed differently."""
    gen = generate_golden_reference(**SHAPE)
    fused = layer_inputs(gen, FUSED_INPUT_NAMES)
    split = layer_inputs(gen, OFFLOAD_INPUT_NAMES)
    assert len(fused) + 2 == len(split)
    w = gen["weights"]
    assert _bytes(fused[1]) == _bytes(
        np.concatenate([w["q_weight"], w["k_weight"], w["v_weight"]], axis=1)
    )
    for name in ("w_o", "ln1_weight", "w_up", "w_down", "ln2_weight"):
        assert (
            fused[FUSED_INPUT_NAMES.index(name)]
            is split[OFFLOAD_INPUT_NAMES.index(name)]
        ), name


def test_layer_inputs_refuses_a_name_it_has_no_source_for():
    """A mode that grows an input without a source is a refusal, not a KeyError."""
    gen = generate_golden_reference(**SHAPE)
    try:
        layer_inputs(gen, BLOCK_INPUT_NAMES + ("rope_lut",))
    except ValueError as exc:
        assert "rope_lut" in str(exc)
    else:
        raise AssertionError("an unknown input name was accepted")


def test_check_weights_refuses_the_three_ways_a_set_can_be_wrong():
    """Key set, shape and dtype, each checked and each named in the message."""
    gen = generate_golden_reference(**SHAPE)["weights"]
    cases = []

    short = dict(gen)
    short.pop("ln2_weight")
    cases.append((short, "ln2_weight"))

    extra = dict(gen, q_norm=np.zeros((64,), dtype=bfloat16))
    cases.append((extra, "q_norm"))

    misshaped = dict(gen, ln1_weight=np.zeros((32,), dtype=bfloat16))
    cases.append((misshaped, "ln1_weight"))

    f32 = dict(gen, ln1_weight=np.zeros((768,), dtype=np.float32))
    cases.append((f32, "bfloat16"))

    for bad, needle in cases:
        try:
            check_weights(bad, 768, 3072)
        except ValueError as exc:
            assert needle in str(exc), (needle, str(exc))
        else:
            raise AssertionError(f"check_weights accepted a set missing {needle}")


def test_a_perturbed_weight_keys_differently():
    """The working half of doc 58a's trap (d): the key is over CONTENT.

    ``opcheck.py::_inject`` builds a FRESH array and writes into it, so the
    clean array the pool already keyed is untouched and the perturbed one keys
    differently. That is the whole mechanism by which a fault-injected run
    re-uploads the weight instead of reusing the clean BO.
    """
    from shared.infra.bo_pool import content_key, content_key_once

    gen = generate_golden_reference(**SHAPE)
    clean = gen["weights"]["ln1_weight"]
    clean_key = content_key_once(clean)

    perturbed = np.array(clean, copy=True)
    perturbed[0] = np.asarray(float(perturbed[0]) + 2.0, dtype=perturbed.dtype)

    assert content_key(perturbed) != clean_key
    assert content_key_once(perturbed) != clean_key
    # And the clean array's key is still its own -- the two coexist, which is
    # what lets one gate run both halves against one ELF cache.
    assert content_key_once(clean) == clean_key


def test_in_place_mutation_defeats_the_cached_key():
    """The BROKEN half of trap (d), demonstrated rather than warned about.

    Mutating a keyed array in place leaves ``content_key_once``'s cache holding
    the pre-mutation digest, so a pool would reuse the clean BO and the injected
    run would PASS. ``forget_content_key`` is the documented escape and is
    checked here so the seam's "never mutate, never copy" rule has a test behind
    the reason for it.
    """
    from shared.infra.bo_pool import (
        content_key,
        content_key_once,
        forget_content_key,
    )

    array = np.array(generate_golden_reference(**SHAPE)["weights"]["ln1_weight"])
    key = content_key_once(array)
    array[0] = np.asarray(float(array[0]) + 2.0, dtype=array.dtype)

    assert content_key_once(array) == key, "the cache is not id-keyed any more"
    assert content_key(array) != key, "the bytes did not actually move"
    assert forget_content_key(array) is True
    assert content_key_once(array) != key


def test_band_excess_is_one_exactly_on_the_band():
    """``band_excess`` is the unit the derived fault delta is measured in."""
    from opcheck_layer import band_excess

    expected = np.zeros((4,), np.float32)
    atol, rtol = 1e-1, 1.6e-2
    on = np.full((4,), atol, np.float32)
    assert abs(band_excess(on, expected, atol, rtol) - 1.0) < 1e-6
    assert band_excess(on * 2, expected, atol, rtol) > 1.9


def test_the_derived_delta_keeps_the_shipped_one_when_it_already_trips():
    """A weight set the shipped constant discriminates on must not move it.

    That is the whole of the M1 encoder case: measured at 512x768x3072x12 with
    real Qwen3-0.6B and Llama-3.2-1B tensors, FAULT_DELTA = 2.0 still leaves the
    band by 19621 and 2295 elements. The derivation must be a no-op there, or it
    would quietly change a number the recorded calibration owns.
    """
    from opcheck_layer import derive_fault_delta

    kwargs = dict(SHAPE)
    gen = generate_golden_reference(**kwargs)
    delta, excess = derive_fault_delta(
        kwargs, gen["weights"], "ln1_weight", (0,), gen["output"], 1e-1, 1.6e-2, 2.0
    )
    assert delta == 2.0, delta
    assert excess >= 2.0, excess


def test_the_derived_delta_grows_when_the_band_swallows_the_response():
    """The decoder-at-real-weight-scale case, forced by widening the band.

    A band wide enough to swallow the shipped perturbation is exactly what a
    real-weight pre-norm layer produces (doc 58 M1 measured a Llama-3.2-1B
    decoder whose whole output absmax, 3.75e-1, sits INSIDE its own 4.5e-1
    atol). Reproducing it here by widening `atol` rather than by loading a model
    keeps the suite host-only and hardware-free, and tests the same branch.
    """
    from opcheck_layer import derive_fault_delta

    kwargs = dict(SHAPE)
    gen = generate_golden_reference(**kwargs)
    base_delta, base_excess = derive_fault_delta(
        kwargs, gen["weights"], "ln1_weight", (0,), gen["output"], 1e-1, 1.6e-2, 2.0
    )
    wide = 1e-1 * base_excess  # the shipped delta now lands exactly on the band
    delta, excess = derive_fault_delta(
        kwargs, gen["weights"], "ln1_weight", (0,), gen["output"], wide, 1.6e-2, 2.0
    )
    assert delta > base_delta, (delta, base_delta)
    assert delta == base_delta * 2 ** round(np.log2(delta / base_delta))
    assert excess >= 2.0, excess


def test_a_vacuous_band_is_a_refusal_and_not_a_wider_delta():
    """No delta at all is the answer when the tolerance cannot discriminate.

    Running on would be the silently-vacuous negative control doc 58a section
    3.2(c) names, so the cap raises -- and the message must say that the
    tolerance is what failed, not the delta.
    """
    from opcheck_layer import derive_fault_delta

    kwargs = dict(SHAPE)
    gen = generate_golden_reference(**kwargs)
    try:
        derive_fault_delta(
            kwargs, gen["weights"], "ln1_weight", (0,), gen["output"], 1e9, 1.6e-2, 2.0
        )
    except ValueError as exc:
        assert "tolerance table" in str(exc), str(exc)
    else:
        raise AssertionError("a vacuous band was accepted")


def test_the_delta_hook_is_absent_for_the_generated_path():
    """No injected weights, no hook -- the generated path pays nothing."""
    from opcheck_layer import fault_delta_hook

    gen = generate_golden_reference(**SHAPE)
    assert fault_delta_hook(dict(SHAPE), None, "ln1_weight", (0,), gen["output"], 1e-1) is None
    hook = fault_delta_hook(
        dict(SHAPE), gen["weights"], "ln1_weight", (0,), gen["output"], 1e-1
    )
    assert callable(hook)
    assert hook() == 2.0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"weights-injection tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
