# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-only: ffn_resident's dataflow, INTERPRETED OUT OF THE BUILT MODULE.

WHAT THIS PINS, AND WHY IT WAS REBUILT
    ``builders/ffn_resident.py`` moves every byte through hand-derived
    patterns: the w_up (sweep, k', column)-major packing, the shim's 4-D
    row-major->blocked retile of ``hidden``, the strided chunk extraction
    from the blocked H group, the w_down per-K-step refill, and a global K
    order assembled from three herds' loops and three segment-scope feed
    nests. A single off-by-one in any of them produces plausible wrong
    numbers on the device, hours after the mistake.

    UNTIL `[2026-08-12]` THIS FILE DID NOT LOOK AT THE MODULE. It imported
    ``ffn_resident_pack_w_up`` and re-derived every DMA pattern and every
    loop order BY HAND in numpy, so the builder and the check were two
    independent transcriptions of one design and the check could only ever
    disagree with itself. Re-imposing the exact c-major w_down order that
    route E1 deleted still printed 5/5 (queue item 17). The arm was blind to
    the whole class of defect it existed to catch.

    It now BUILDS ``build_ffn_resident_module`` and INTERPRETS the resulting
    ``air.ir.Module``: every ``air.dma_memcpy_nd`` and ``air.channel.put`` /
    ``air.channel.get`` executed with the offsets, sizes and strides the op
    actually carries, every ``scf.for`` at its actual bounds, every
    ``func.call`` dispatched by the symbol the builder named, the three
    herds as ``herd_x`` concurrent actors each, the four channels as FIFOs.
    Nothing about the dataflow is transcribed here any more: change a
    stride, a sub-channel index, a loop order or a nest shape in the builder
    and this file computes a different answer.

    f64 throughout, so ONLY ordering and addressing are under test (bf16
    rounding is the device's business and the numeric arm's), and the result
    must equal ``gelu(hidden @ w_up) @ w_down`` to f64 round-off.

THE TWO MODELS THIS INTERPRETER APPLIES, STATED SO THEY CAN BE ARGUED WITH

    (M1) ``air-dma-to-channel``'s HOIST. Every TEXTUAL segment-scope
    ``air.dma_memcpy_nd`` becomes its own auto channel whose L3-side reads
    are hoisted into their own launch-scope loop nest, cloned from the
    enclosing ``scf.for`` nest; sibling nests are CONCATENATED in textual
    order, and the L2-side landing stays where the dma was. This is the rule
    the builder is written against (its w_down comment; doc 19 step 1), it
    is what turned four Python-unrolled refills into four sibling per-c
    nests, and it is why route E1's fix was to make the refill ONE textual
    instance over a real ``scf.for c``.

    (M2) THE MEMTILE LOCK PAIRING. An L2 staging buffer is one allocation
    behind one lock pair: values land in it in the order the shim issued
    them, and the k-th consumption round reads the k-th value that landed.
    Modelled literally -- the landing blocks until the previous value has
    been released, and a reader acquires once per innermost-loop iteration.
    This is the pairing the E1 commit names ("the pairing of refill chunk i
    with put group i is carried by the channel's FIFO order and the memtile
    ring's backpressure").

    Under (M1) + (M2) a delivery order that disagrees with the consumption
    order is a DATA defect here, element-visible. On hardware the same
    disagreement stalls the ring and shows up as ERT_CMD_STATE_TIMEOUT
    instead. Both are rejections; this arm catches the mismatch, it does not
    predict its hardware symptom.

WHAT THIS ARM DOES *NOT* MODEL -- do not cite it for any of these
    - Timing, bandwidth, BD folding (``air-opt-shim-dma-bds``) or channel
      fusion (``air-fuse-channels``). The interpreter is untimed and the
      named channels' FIFOs are unbounded.
    - Wall 5's D1 (inter-channel starvation between coupled feeds) and
      wall 6's lock-conservation imbalance on ``l2_b_down`` (queue item 18).
      Both are properties of the LOWERED design's lock and BD counts; this
      arm reads the AIR module, so it can see neither. The device gate
      (``run_npu2_ffn_resident_peano.lit``) is what settles them.
    - bf16 rounding, kernel numerics, placement, routing, the column budget.
      ``ffn_resident_structure.py`` owns the structural half.

IT CARRIES ITS OWN NEGATIVE CONTROLS, and they run on every invocation
    An arm that cannot fail is not a check (queue items 10, 14 and 17 --
    three in one week). Two defects are re-imposed on the module this arm
    just built, and both must be REJECTED:

      (NC1) THE c-MAJOR w_down REFILL route E1 deleted -- the refill nest's
        ``c`` loop Python-unrolled back into ``herd_x`` sibling nests, which
        under (M1) concatenate into a c-major delivery order against the
        sweep-major consumer. This is the defect queue item 17 proved the
        old arm could not see. Its anchors are asserted, not assumed: if the
        refill nest is no longer one 3-deep nest over (sweeps, herd_x,
        chunks_per_group) the control reports STALE and the clause goes RED
        -- a control that stopped describing the module has not rejected
        anything, and scoring that as a pass is item 17 all over again.
      (NC2) THE RETILE SEAM: the ``hidden`` refill's 4-D L3 read walked with
        its two innermost strides swapped -- the row-major -> blocked
        microtile slip that seam 1 exists to get right.

    Both print a named line, so the lit recipe FileChecks that the controls
    fired rather than trusting that they exist.

    No framework dependency -- pytest is not in the sandbox venv -- so it
    runs as a plain script and prints a named pass count, exactly as
    ``test_block_cache.py`` does; the lit arm FileChecks the count so a
    check that stops running fails loudly.
"""

import itertools
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TL = os.path.dirname(_HERE)  # transformer_layer/
_PE = os.path.dirname(_TL)  # programming_examples/
for _p in (_PE, os.path.join(_PE, "llms"), _TL):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from air.ir import (  # noqa: E402
    AffineAddExpr,
    AffineConstantExpr,
    AffineMapAttr,
    AffineMulExpr,
    AffineSymbolExpr,
    BlockArgument,
    MemRefType,
)

from builders.ffn_accum import (  # noqa: E402
    FFN_ACCUM_MM_SYMBOL,
    FFN_ACCUM_ZERO_SYMBOL,
    MICRO,
    TILE_M,
    ffn_accum_pack_w,
)
from builders.ffn_resident import (  # noqa: E402
    build_ffn_resident_module,
    ffn_resident_pack_w_up,
)
from builders.gelu import GELU_SYMBOL  # noqa: E402

SEQ, FFN, EMB = 64, 3072, 768
HERD_X, TILE_K = 4, 32
GROUP_N = EMB // HERD_X  # 192
SWEEPS = FFN // (HERD_X * GROUP_N)  # 4
CPG = GROUP_N // TILE_K  # 6
K_UP = EMB // TILE_K  # 24
CHUNK = TILE_M * TILE_K  # 2048
UP_B = TILE_K * GROUP_N  # 6144
DOWN_CHUNK = TILE_K * EMB  # 24576

# ShapedType::kDynamic, as the static_*_{offsets,sizes,strides} arrays spell it.
_DYN = -9223372036854775808
_BLOCK = object()  # an actor yields this when it cannot make progress
_TOKEN = "__iteration_token__"  # (M2) release scope, see _run_op's scf.for

_passed = 0
_failed = 0


def _check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"FAIL: {name}")


class EmulationError(RuntimeError):
    """The interpreter refused: deadlock, an unhandled op, a dry stream."""


class ControlNotApplicable(EmulationError):
    """A negative control's anchor no longer describes the built module.

    Kept DISTINCT from EmulationError on purpose. A control that raises this
    has not rejected anything -- it has stopped being a control -- and the
    clause must go RED, not green. Folding the two together is exactly how a
    check ends up reading as present and unable to fail (queue item 17).
    """


# --------------------------------------------------------------------------
# Reading the IR: attributes, affine maps, strided patterns.
# --------------------------------------------------------------------------


def _int_array(op, name):
    """``array<i64: 8, 24, 8, 8>`` -> [8, 24, 8, 8]; ``array<i64>`` -> []."""
    text = str(op.attributes[name])
    inner = text[text.index("<") + 1 : text.rindex(">")]
    if ":" not in inner:
        return []
    return [int(x) for x in inner.split(":", 1)[1].split(",") if x.strip()]


def _sym(op, name="sym_name"):
    return str(op.attributes[name]).strip('"')


def _affine_eval(expr, syms):
    if isinstance(expr, AffineConstantExpr):
        return expr.value
    if isinstance(expr, AffineSymbolExpr):
        return syms[expr.position]
    if isinstance(expr, AffineAddExpr):
        return _affine_eval(expr.lhs, syms) + _affine_eval(expr.rhs, syms)
    if isinstance(expr, AffineMulExpr):
        return _affine_eval(expr.lhs, syms) * _affine_eval(expr.rhs, syms)
    raise EmulationError(f"unhandled affine expression {expr}")


def _eval(value, env):
    """Evaluate an index/i1 SSA value: loop IVs, constants, affine.apply."""
    if value in env:
        return env[value]
    if isinstance(value, BlockArgument):
        raise EmulationError(f"unbound block argument {value}")
    op = value.owner
    if op.name == "arith.constant":
        out = int(str(op.attributes["value"]).split(":")[0].strip())
    elif op.name == "affine.apply":
        amap = AffineMapAttr(op.attributes["map"]).value
        out = _affine_eval(amap.results[0], [_eval(o, env) for o in op.operands])
    elif op.name == "arith.cmpi":
        lhs, rhs = (_eval(o, env) for o in op.operands)
        pred = int(str(op.attributes["predicate"]).split(":")[0].strip())
        if pred != 0:  # only `eq` is emitted (the down herd's guarded zero)
            raise EmulationError(f"unhandled cmpi predicate {pred}")
        out = lhs == rhs
    else:
        raise EmulationError(f"cannot evaluate {op.name} as an index")
    env[value] = out
    return out


def _resolve(static, dyn_values, env):
    """Splice the dynamic operands into a static_* array's kDynamic slots."""
    out, it = [], iter(dyn_values)
    for entry in static:
        out.append(_eval(next(it), env) if entry == _DYN else entry)
    return out


def _seg(op):
    return _int_array(op, "operandSegmentSizes")


def _chan_operands(op, env):
    """(indices, buffer, (offsets, sizes, strides)) of air.channel.put/get."""
    seg = _seg(op)  # [async, indices, memref, offsets, sizes, strides]
    operands = list(op.operands)
    i = seg[0]
    indices = tuple(_eval(operands[i + j], env) for j in range(seg[1]))
    i += seg[1]
    buf = operands[i]
    i += seg[2]
    dyn = []
    for n in seg[3:6]:
        dyn.append(operands[i : i + n])
        i += n
    side = "src" if op.name.endswith("put") else "dst"
    pattern = tuple(
        _resolve(_int_array(op, f"static_{side}_{kind}"), dyn[j], env)
        for j, kind in enumerate(("offsets", "sizes", "strides"))
    )
    return indices, buf, pattern


def _dma_parts(op):
    """(dst value, src value, dynamic-operand groups) of air.dma_memcpy_nd."""
    seg = _seg(op)  # [async, dst, d_off, d_siz, d_str, src, s_off, s_siz, s_str]
    operands = list(op.operands)
    i = seg[0]
    dst = operands[i]
    i += seg[1]
    dst_dyn = []
    for n in seg[2:5]:
        dst_dyn.append(operands[i : i + n])
        i += n
    src = operands[i]
    i += seg[5]
    src_dyn = []
    for n in seg[6:9]:
        src_dyn.append(operands[i : i + n])
        i += n
    return dst, src, dst_dyn, src_dyn


def _dma_pattern(op, side, dyn, env):
    return tuple(
        _resolve(_int_array(op, f"static_{side}_{kind}"), dyn[j], env)
        for j, kind in enumerate(("offsets", "sizes", "strides"))
    )


_IDX_CACHE = {}


def _walk_indices(sizes, strides):
    """Linear element offsets a BD visits, in the order it visits them."""
    key = (tuple(sizes), tuple(strides))
    hit = _IDX_CACHE.get(key)
    if hit is None:
        hit = np.zeros(1, dtype=np.int64)
        for size, stride in zip(sizes, strides):
            hit = (hit[:, None] + np.arange(size, dtype=np.int64) * stride).ravel()
        _IDX_CACHE[key] = hit
    return hit


def _base(offsets, strides):
    return int(sum(o * s for o, s in zip(offsets, strides)))


def _gather(buf, pattern):
    offsets, sizes, strides = pattern
    if not sizes:
        return buf.copy()
    return buf[_base(offsets, strides) + _walk_indices(sizes, strides)]


def _scatter(buf, pattern, values):
    offsets, sizes, strides = pattern
    if not sizes:
        buf[:] = values
        return
    buf[_base(offsets, strides) + _walk_indices(sizes, strides)] = values


def _is_memref(value):
    return str(value.type).startswith("memref<")


def _elems(value):
    return int(np.prod(MemRefType(value.type).shape))


def _shape(value):
    return tuple(MemRefType(value.type).shape)


# --------------------------------------------------------------------------
# The kernels the module dispatches by name.
# --------------------------------------------------------------------------


def _unblock(flat, rows, cols):
    return (
        flat.reshape(rows // MICRO, cols // MICRO, MICRO, MICRO)
        .transpose(0, 2, 1, 3)
        .reshape(rows, cols)
    )


def _block(mat):
    r, c = mat.shape
    return np.ascontiguousarray(
        mat.reshape(r // MICRO, MICRO, c // MICRO, MICRO)
        .transpose(0, 2, 1, 3)
        .reshape(-1)
    )


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


def _kernel_mm(bufs, shapes):
    """The -D-baked in-place accumulate C += A @ B, all three microtiled."""
    a, b, c = bufs
    (m, k), (_, n), _ = shapes
    c[:] = _block(_unblock(c, m, n) + _unblock(a, m, k) @ _unblock(b, k, n))


def _kernel_zero(bufs, shapes):
    bufs[0][:] = 0.0


def _kernel_gelu(bufs, shapes):
    bufs[1][:] = _gelu(bufs[0])


_KERNELS = {
    FFN_ACCUM_MM_SYMBOL: _kernel_mm,
    FFN_ACCUM_ZERO_SYMBOL: _kernel_zero,
    GELU_SYMBOL: _kernel_gelu,
}


# --------------------------------------------------------------------------
# (M1): what the shim delivers into each staged buffer, and in what order.
# --------------------------------------------------------------------------


def _body(op, region=0):
    """The operations of ``op``'s region as Operations.

    Normalising matters: ``block.operations`` yields OpViews, and an OpView's
    ``.name`` is the op's OWN symbol (``FuncOp.name`` is ``ffn_resident``),
    not the operation name. Everything below dispatches on the operation
    name, so everything below goes through here.
    """
    return [child.operation for child in op.regions[region].blocks[0].operations]


def _find(op, name):
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                child = child.operation
                if child.name == name:
                    return child
                found = _find(child, name)
                if found is not None:
                    return found
    return None


def _collect_auto_channels(segment_body):
    """Every TEXTUAL segment-scope dma, with the scf.for nest enclosing it.

    Herd regions are skipped: a herd-internal dma is per-core and keeps its
    program order, so it needs no hoist model and is copied in place.
    """
    found = []

    def walk(ops, loops):
        for op in ops:
            if op.name == "scf.for":
                walk(_body(op), loops + [op])
            elif op.name == "scf.if":
                for region in op.regions:
                    for block in region.blocks:
                        walk([c.operation for c in block.operations], loops)
            elif op.name == "air.herd":
                continue
            elif op.name == "air.dma_memcpy_nd":
                found.append({"op": op, "loops": list(loops)})

    walk(segment_body, [])
    return found


def _loop_range(for_op, env):
    lb, ub, step = (_eval(o, env) for o in for_op.operands)
    return range(lb, ub, step)


def _deliveries(auto, env, unroll_loop=None, mis_retile=False):
    """The ordered stream one auto channel lands in its L2 destination.

    ``unroll_loop`` re-imposes the pre-E1 shape: Python-unrolling loop
    ``unroll_loop`` makes that many TEXTUAL dma instances, each hoisted into
    its own nest (the remaining loops) and the nests CONCATENATED -- so the
    unrolled loop becomes the OUTERMOST of the delivery order.
    ``mis_retile`` swaps the two innermost strides of the L3 read, which is
    seam 1's off-by-one: the microtile walked down its columns instead of
    across its rows, so the k' slice arrives transposed inside every 8x8.
    """
    op = auto["op"]
    _, src, _, src_dyn = _dma_parts(op)
    loops = auto["loops"]
    ranges = [_loop_range(f, env) for f in loops]
    order = list(range(len(loops)))
    if unroll_loop is not None:
        order = [unroll_loop] + [i for i in order if i != unroll_loop]
    ivs = [f.regions[0].blocks[0].arguments[0] for f in loops]
    buf = env[src]
    out = []
    for combo in itertools.product(*[ranges[i] for i in order]):
        local = dict(env)
        for slot, value in zip(order, combo):
            local[ivs[slot]] = value
        offsets, sizes, strides = _dma_pattern(op, "src", src_dyn, local)
        if not sizes:
            out.append(buf.copy())
            continue
        base = _base(offsets, strides)
        if mis_retile:
            strides = strides[:-2] + [strides[-1], strides[-2]]
        out.append(buf[base + _walk_indices(sizes, strides)])
    return out


# --------------------------------------------------------------------------
# The interpreter.
# --------------------------------------------------------------------------


class _Ctx:
    def __init__(self):
        self.chan = {}  # (channel symbol, indices) -> list used as a FIFO
        self.stage = {}  # staged L2 buffer -> the shim's delivery stream
        self.produced = {}  # (M2) values landed in that buffer
        self.released = {}  # (M2) consumption rounds finished on it
        self.held = {}  # (M2) buffer -> the iteration token holding it
        self.ops = 0
        self.calls = {}  # kernel symbol -> dispatches
        self.moved = {}  # channel symbol -> (puts, gets)

    def fifo(self, key):
        return self.chan.setdefault(key, [])


def _chan_name(op):
    return str(op.attributes["chan_name"]).lstrip("@")


def _run_ops(ops, env, ctx, in_herd):
    for op in ops:
        yield from _run_op(op, env, ctx, in_herd)


def _acquire(buf, env, ctx):
    """(M2): take the next value that landed, once per innermost iteration."""
    token = env.get(_TOKEN)
    if ctx.held.get(buf) is token and token is not None:
        return
    while ctx.produced[buf] <= ctx.released[buf]:
        yield _BLOCK
    ctx.held[buf] = token


def _release(token, ctx):
    for buf, held in list(ctx.held.items()):
        if held is token:
            del ctx.held[buf]
            ctx.released[buf] += 1


def _run_op(op, env, ctx, in_herd):
    op = op.operation
    name = op.name
    ctx.ops += 1
    if name == "scf.for":
        iv = op.regions[0].blocks[0].arguments[0]
        body = _body(op)
        for i in _loop_range(op, env):
            inner = dict(env)
            inner[iv] = i
            token = object()
            inner[_TOKEN] = token
            yield from _run_ops(body, inner, ctx, in_herd)
            _release(token, ctx)
    elif name == "scf.if":
        if _eval(op.operands[0], env):
            yield from _run_ops(_body(op), dict(env), ctx, in_herd)
    elif name == "memref.alloc":
        # NaN, not zero: anything that reads a buffer before it is filled
        # poisons the result instead of passing on a lucky zero.
        env[op.results[0]] = np.full(_elems(op.results[0]), np.nan)
    elif name == "memref.dealloc":
        buf = env.get(op.operands[0])
        if buf is not None and in_herd:
            buf[:] = np.nan  # use-after-dealloc poisons too
    elif name == "air.channel.put":
        indices, buf, pattern = _chan_operands(op, env)
        if buf in ctx.stage:
            yield from _acquire(buf, env, ctx)
        ctx.fifo((_chan_name(op), indices)).append(_gather(env[buf], pattern))
        ctx.moved[_chan_name(op)] = tuple(
            a + b for a, b in zip(ctx.moved.get(_chan_name(op), (0, 0)), (1, 0))
        )
    elif name == "air.channel.get":
        indices, buf, pattern = _chan_operands(op, env)
        queue = ctx.fifo((_chan_name(op), indices))
        while not queue:
            yield _BLOCK
        _scatter(env[buf], pattern, queue.pop(0))
        ctx.moved[_chan_name(op)] = tuple(
            a + b for a, b in zip(ctx.moved.get(_chan_name(op), (0, 0)), (0, 1))
        )
    elif name == "air.dma_memcpy_nd":
        dst, src, dst_dyn, src_dyn = _dma_parts(op)
        if in_herd:
            _scatter(
                env[dst],
                _dma_pattern(op, "dst", dst_dyn, env),
                _gather(env[src], _dma_pattern(op, "src", src_dyn, env)),
            )
        else:
            # (M1) hoisted the L3 side; this is the landing. (M2) makes it
            # wait for the previous value's consumption round to finish.
            while ctx.produced[dst] > ctx.released[dst]:
                yield _BLOCK
            stream = ctx.stage[dst]
            if not stream:
                raise EmulationError(
                    "the hoisted stream ran dry: the feed nest consumes more "
                    "landings than its loop nest issues"
                )
            _scatter(env[dst], _dma_pattern(op, "dst", dst_dyn, env), stream.pop(0))
            ctx.produced[dst] += 1
    elif name == "func.call":
        callee = str(op.attributes["callee"]).lstrip("@")
        if callee not in _KERNELS:
            raise EmulationError(f"module dispatches an unmodelled kernel {callee}")
        bufs, shapes = [], []
        for operand in op.operands:
            if _is_memref(operand):
                bufs.append(env[operand])
                shapes.append(_shape(operand))
        _KERNELS[callee](bufs, shapes)
        ctx.calls[callee] = ctx.calls.get(callee, 0) + 1
    elif name in (
        "arith.constant",
        "arith.cmpi",
        "affine.apply",
        "scf.yield",
        "air.herd_terminator",
        "air.segment_terminator",
        "air.launch_terminator",
        "func.return",
    ):
        pass
    else:
        raise EmulationError(f"unhandled op {name}")


def _bind_region_args(op, env):
    """Bind an air.launch/segment body's block arguments to its operands."""
    seg = _seg(op)
    operands = list(op.operands)[seg[0] + seg[1] :]
    block = op.regions[0].blocks[0]
    if len(block.arguments) != len(operands):
        raise EmulationError(f"{op.name} carries sizes this interpreter does not model")
    inner = dict(env)
    for arg, operand in zip(block.arguments, operands):
        inner[arg] = env[operand]
    return inner, block


def _segment_of(module):
    func = None
    for op in _body(module.operation):
        if op.name == "func.func" and _sym(op) == "ffn_resident":
            func = op
    if func is None:
        raise EmulationError("no @ffn_resident function in the module")
    launch = _find(func, "air.launch")
    segment = _find(launch, "air.segment") if launch is not None else None
    if segment is None:
        raise EmulationError("no air.launch/air.segment in @ffn_resident")
    return func, launch, segment


def _segment_env(module, hosts):
    """Bind the four host arrays through launch/segment, run the prologue."""
    func, launch, segment = _segment_of(module)
    env = {}
    for arg, host in zip(func.regions[0].blocks[0].arguments, hosts):
        env[arg] = host
    env, _ = _bind_region_args(launch, env)
    env, sblock = _bind_region_args(segment, env)
    body = [op.operation for op in sblock.operations]
    for op in body:
        if op.name == "memref.alloc":
            env[op.results[0]] = np.full(_elems(op.results[0]), np.nan)
        elif op.name == "arith.constant":
            _eval(op.results[0], env)
    return env, body


def _defect_targets(autos):
    """Which auto channel each negative control aims at, asserted not assumed."""
    refill = [i for i, a in enumerate(autos) if len(a["loops"]) == 3]
    retile = [
        i
        for i, a in enumerate(autos)
        if len(_int_array(a["op"], "static_src_sizes")) == 4
    ]
    return refill, retile


def emulate(module, hosts, defect=None):
    """Run the module over f64 host arrays; ``hosts[3]`` (y) is written.

    ``defect`` re-imposes a known builder defect on the module just built:
      ("c_major_refill",) -- E1's Python-unrolled c, propagated through (M1);
      ("mis_retile",)     -- the hidden retile's microtile walk transposed.
    """
    ctx = _Ctx()
    env, body = _segment_env(module, hosts)

    autos = _collect_auto_channels(body)
    if not autos:
        raise EmulationError("no segment-scope refill found: the feed nests moved")
    refill, retile = _defect_targets(autos)

    for i, auto in enumerate(autos):
        unroll, mis_retile = None, False
        if defect and defect[0] == "c_major_refill":
            if len(refill) != 1:
                raise ControlNotApplicable(
                    "NC1 no longer applies: expected exactly one 3-deep refill "
                    f"nest, found {len(refill)}"
                )
            if i == refill[0]:
                bounds = [len(_loop_range(f, env)) for f in auto["loops"]]
                if bounds != [SWEEPS, HERD_X, CPG]:
                    raise ControlNotApplicable(
                        f"NC1 no longer applies: refill nest bounds {bounds} are not "
                        f"(sweeps, herd_x, chunks_per_group) = {[SWEEPS, HERD_X, CPG]}"
                    )
                unroll = 1  # the `c` loop, Python-unrolled back into siblings
        if defect and defect[0] == "mis_retile":
            if len(retile) != 1:
                raise ControlNotApplicable(
                    "NC2 no longer applies: expected exactly one 4-D L3 retile, "
                    f"found {len(retile)}"
                )
            if i == retile[0]:
                mis_retile = True
        dst = _dma_parts(auto["op"])[0]
        # (M2): sibling auto channels landing in the SAME buffer share ONE
        # stream -- one allocation, one lock pair.
        ctx.stage.setdefault(dst, []).extend(
            _deliveries(auto, env, unroll_loop=unroll, mis_retile=mis_retile)
        )
        ctx.produced.setdefault(dst, 0)
        ctx.released.setdefault(dst, 0)

    actors = []
    for op in body:
        if op.name == "air.herd":
            seg = _seg(op)
            operands = list(op.operands)
            sizes = [_eval(o, env) for o in operands[seg[0] : seg[0] + seg[1]]]
            hblock = op.regions[0].blocks[0]
            kernel_operands = operands[seg[0] + seg[1] :]
            for tx in range(sizes[0]):
                for ty in range(sizes[1]):
                    henv = dict(env)
                    for arg, value in zip(hblock.arguments, [tx, ty] + sizes):
                        henv[arg] = value
                    for arg, operand in zip(
                        hblock.arguments[2 + len(sizes) :], kernel_operands
                    ):
                        henv[arg] = env[operand]
                    actors.append(
                        (
                            f"{_sym(op)}[{tx},{ty}]",
                            _run_ops(
                                [o.operation for o in hblock.operations],
                                henv,
                                ctx,
                                True,
                            ),
                        )
                    )
        elif op.name == "scf.for":
            actors.append(
                (f"feed_nest_{len(actors)}", _run_op(op, dict(env), ctx, False))
            )

    live = list(actors)
    while live:
        progressed = False
        for entry in list(live):
            before = ctx.ops
            try:
                next(entry[1])
            except StopIteration:
                live.remove(entry)
                progressed = True
                continue
            if ctx.ops > before:
                progressed = True
        if not progressed:
            raise EmulationError(
                "deadlock: no actor can advance -- blocked "
                + ", ".join(name for name, _ in live)
            )
    return ctx


# --------------------------------------------------------------------------
# The clauses.
# --------------------------------------------------------------------------


def _h_put_patterns(module):
    """The up herd's chunk puts on ffn_res_h, straight off the module."""
    out = []

    def walk(ops):
        for op in ops:
            if op.name == "air.channel.put" and _chan_name(op).endswith("_h"):
                out.append(
                    tuple(
                        _int_array(op, f"static_src_{kind}")
                        for kind in ("offsets", "sizes", "strides")
                    )
                )
            for region in op.regions:
                for block in region.blocks:
                    walk([c.operation for c in block.operations])

    walk(_body(module.operation))
    return out


def main():
    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((SEQ, EMB))
    w_up = rng.standard_normal((EMB, FFN))
    w_down = rng.standard_normal((FFN, EMB))
    wup_packed = ffn_resident_pack_w_up(w_up, HERD_X, TILE_K)
    wdown_packed = ffn_accum_pack_w(w_down, HERD_X, TILE_K)

    module = build_ffn_resident_module()

    def hosts():
        # y need not enter zeroed (the down ring's guarded zero owns that);
        # handing it noise is what proves the guard fires.
        return [
            np.ascontiguousarray(hidden.reshape(-1)),
            wup_packed.astype(np.float64),
            wdown_packed.astype(np.float64),
            rng.standard_normal(SEQ * EMB),
        ]

    # 1. The shim retile, WITH THE PATTERN READ OFF THE MODULE, delivers
    # exactly the k' column slice of the row-major hidden, blocked -- for
    # every one of the sweeps x k' refills the nest issues, in that order.
    env, body = _segment_env(module, hosts())
    autos = _collect_auto_channels(body)
    refill, retile = _defect_targets(autos)
    ok = len(retile) == 1
    if ok:
        delivered = _deliveries(autos[retile[0]], env)
        ok = len(delivered) == SWEEPS * K_UP
        for s in range(SWEEPS):
            for kp in range(K_UP):
                want = _block(hidden[:, kp * TILE_K : (kp + 1) * TILE_K])
                ok = ok and np.array_equal(delivered[s * K_UP + kp], want)
    _check("shim 4-D retile (pattern off the module) == k' column slice", ok)

    # 2. The w_up packing puts group g's k'-slice where the feed reads it.
    ok = True
    for s in range(SWEEPS):
        for kp in range(K_UP):
            w_off = (s * K_UP + kp) * HERD_X * UP_B
            for c in range(HERD_X):
                g = s * HERD_X + c
                got = _unblock(
                    wup_packed[w_off + c * UP_B : w_off + (c + 1) * UP_B],
                    TILE_K,
                    GROUP_N,
                )
                want = w_up[
                    kp * TILE_K : (kp + 1) * TILE_K, g * GROUP_N : (g + 1) * GROUP_N
                ]
                ok = ok and np.array_equal(got, want)
    _check("w_up pack: every (s, k', c) slice", ok)

    # 3. The chunk put, WITH THE PATTERNS READ OFF THE MODULE, extracts
    # column block jj of the blocked group -- all chunks_per_group of them,
    # in the down ring's K order.
    group = rng.standard_normal((TILE_M, GROUP_N))
    blocked = _block(group)
    patterns = _h_put_patterns(module)
    ok = len(patterns) == CPG
    if ok:
        for jj, pattern in enumerate(patterns):
            want = _block(group[:, jj * TILE_K : (jj + 1) * TILE_K])
            ok = ok and np.array_equal(_gather(blocked, pattern), want)
    _check("chunk put (patterns off the module) == blocked column block", ok)

    # 4. The w_down refill at K step j is rows [32j, 32j+32), per-tx sliced.
    ok = True
    for j in range(FFN // TILE_K):
        wchunk = wdown_packed[j * DOWN_CHUNK : (j + 1) * DOWN_CHUNK]
        for tx in range(HERD_X):
            got = _unblock(wchunk[tx * UP_B : (tx + 1) * UP_B], TILE_K, GROUP_N)
            want = w_down[
                j * TILE_K : (j + 1) * TILE_K, tx * GROUP_N : (tx + 1) * GROUP_N
            ]
            ok = ok and np.array_equal(got, want)
    _check("w_down pack: every (j, tx) slice", ok)

    ref = _gelu(hidden @ w_up) @ w_down

    def run(defect=None):
        args = hosts()
        ctx = emulate(module, args, defect=defect)
        return float(np.abs(args[3].reshape(SEQ, EMB) - ref).max()), ctx

    # 5. End to end, INTERPRETED OUT OF THE BUILT MODULE: three herds, four
    # channels, three feed nests, every pattern as the op carries it.
    err, ctx = run()
    print(
        f"module-interpreted dataflow: max |y - reference| = {err:.3e} "
        f"over {SEQ}x{EMB} f64 elements"
    )
    _check(f"module-interpreted dataflow exact (max err {err:.2e})", err < 1e-9)

    # 6. LIVENESS, so clause 5 cannot pass vacuously. Every count is DERIVED
    # from the design, not copied from a run: the two GEMM herds dispatch the
    # one in-place entry point herd_x*sweeps*k_steps + herd_x*(ffn/tile_k)
    # times, the zero fires once per up group plus once per down core (the
    # guarded first-K-step call), GeLU once per chunk, and every value the
    # shim delivered must have been consumed -- a staged stream left holding
    # data means a feed nest stopped reading it.
    want_calls = {
        FFN_ACCUM_MM_SYMBOL: HERD_X * SWEEPS * K_UP + HERD_X * (FFN // TILE_K),
        FFN_ACCUM_ZERO_SYMBOL: HERD_X * SWEEPS + HERD_X,
        GELU_SYMBOL: HERD_X * SWEEPS * CPG,
    }
    undrained = {
        i: len(stream) for i, stream in enumerate(ctx.stage.values()) if stream
    }
    census = ", ".join(f"{k} {v}" for k, v in sorted(ctx.calls.items()))
    moved = ", ".join(f"{k} {p}p/{g}g" for k, (p, g) in sorted(ctx.moved.items()))
    print(f"dispatch census: {census}")
    print(f"channel census: {moved}; undrained staged streams: {len(undrained)}")
    _check(
        "liveness: dispatch census and staged streams are as the design derives",
        ctx.calls == want_calls and not undrained,
    )

    # 7-8. NEGATIVE CONTROLS -- the defects re-imposed on the built module.
    for tag, defect, clause in (
        (
            "c-major w_down refill (pre-E1 c unroll)",
            ("c_major_refill",),
            "NEGATIVE CONTROL 1: the c-major refill order is REJECTED",
        ),
        (
            "hidden 4-D retile walked with its two inner strides swapped",
            ("mis_retile",),
            "NEGATIVE CONTROL 2: the mis-walked retile is REJECTED",
        ),
    ):
        try:
            got, _ = run(defect=defect)
            rejected = not (got < 1e-9)
            detail = f"max err {got:.2e}"
        except ControlNotApplicable as exc:
            # NOT a rejection: the control stopped describing the module, so
            # it tested nothing. Red, loudly, with the anchor that broke.
            rejected = False
            detail = f"STALE -- {exc}"
        except EmulationError as exc:
            # A refusal DURING the run (deadlock, dry stream) IS a rejection:
            # the defective module could not be executed to an answer.
            rejected = True
            detail = str(exc).split("\n")[0]
        print(f"negative control: {tag} -> {detail}")
        _check(clause, rejected)

    total = _passed + _failed
    print(f"ffn_resident emulation tests: {_passed}/{total} passed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
