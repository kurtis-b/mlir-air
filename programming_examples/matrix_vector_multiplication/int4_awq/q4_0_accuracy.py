# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measure what q4_0 costs in accuracy, against an HF bf16 checkpoint.

Host-only. Three lenses, in increasing order of relevance:

  --weights     per-tensor reconstruction error of the dequantized q4_0 weight
                against the HF bf16 original, over every pure-q4_0 linear in
                the model. Also splits out how much of it is the packer's own
                fp16 `d` -> bf16 rounding rather than 4-bit quantization.

  --activations error of the projection OUTPUT on activations captured from a
                real forward pass. This is the number that matters: weight
                Frobenius error cannot rank quantization formats, because an
                activation-aware format (AWQ) deliberately trades weight error
                for output error under the real activation distribution.

  --salient     why the two differ. Ranks input channels by activation
                magnitude and reports how much of the output error energy the
                top ones own -- the asymmetry AWQ's per-channel scaling
                exploits and q4_0, being activation-agnostic, cannot.

Usage:
  python3 q4_0_accuracy.py --gguf model.gguf --hf HuggingFaceTB/SmolLM2-1.7B-Instruct --weights
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_q4_0 import GGUFFile, dequant_q4_0_reference, llama_unpermute_rows

# (gguf role, HF module path, permuted-by-llama.cpp)
ROLES = [
    ("attn_q", "self_attn.q_proj", "q"),
    ("attn_k", "self_attn.k_proj", "kv"),
    ("attn_v", "self_attn.v_proj", None),
    ("attn_output", "self_attn.o_proj", None),
    ("ffn_gate", "mlp.gate_proj", None),
    ("ffn_up", "mlp.up_proj", None),
    ("ffn_down", "mlp.down_proj", None),
]


def _nh(kind, n_head, n_kv_head):
    return {"q": n_head, "kv": n_kv_head}.get(kind)


def _dequant(g, name, scale_dtype, nh):
    ti = g.tensors[name]
    K, M = int(ti.ne[0]), int(ti.ne[1])
    w = dequant_q4_0_reference(np.asarray(g.raw_bytes(name)), K, M, scale_dtype)
    return llama_unpermute_rows(w, nh) if nh else w


def _cos(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _rel(hat, ref):
    return float(np.linalg.norm(hat - ref) / np.linalg.norm(ref))


def cmd_weights(args, g, hf_weight, n_head, n_kv_head, n_layers):
    print("q4_0 vs HF bf16 -- WEIGHT reconstruction, all layers")
    print("%-13s %-5s %-11s %-11s %-11s %-11s"
          % ("role", "n", "rel med", "rel max", "cos min", "d-round add"))
    skipped = []
    allrel = []
    for role, hfname, kind in ROLES:
        rels, coss, deltas = [], [], []
        for li in range(n_layers):
            gname = "blk.%d.%s.weight" % (li, role)
            ti = g.tensors[gname]
            if ti.type_name != "Q4_0":
                skipped.append((li, role, ti.type_name))
                continue
            nh = _nh(kind, n_head, n_kv_head)
            ref = hf_weight(li, hfname)
            hb = _dequant(g, gname, "bf16", nh)
            hf_ = _dequant(g, gname, "fp16", nh)
            rb, rf = _rel(hb, ref), _rel(hf_, ref)
            rels.append(rb)
            coss.append(_cos(hb, ref))
            deltas.append(rb - rf)
        if not rels:
            print("%-13s %-5d (no pure-q4_0 layers)" % (role, 0))
            continue
        allrel.extend(rels)
        print("%-13s %-5d %-11.5f %-11.5f %-11.6f %-11.2e"
              % (role, len(rels), np.median(rels), max(rels), min(coss), np.median(deltas)))
    print("%-13s %-5d %-11.5f %-11.5f" % ("ALL", len(allrel), np.median(allrel), max(allrel)))
    print()
    print("  'd-round add' is how much the packer's fp16->bf16 scale rounding adds")
    print("  to rel_fro on top of 4-bit quantization. Measured, not assumed.")
    if skipped:
        print()
        print("  NOT q4_0 in this 'Q4_0' checkpoint (%d):" % len(skipped))
        for s in skipped:
            print("    layer %2d %-12s -> %s" % s)


def cmd_activations(args, g, model, torch, n_head, n_kv_head):
    layers = [int(x) for x in args.layers.split(",")]
    cap = {}

    def mk(key):
        def hook(m, i, o):
            # MUST return None; a non-None return replaces the module output.
            if key not in cap:
                x = i[0].detach().to(torch.float32)
                cap[key] = x.reshape(-1, x.shape[-1]).numpy()
        return hook

    hooks = []
    for li in layers:
        for role, hfname, _ in ROLES:
            mod = model.model.layers[li]
            for part in hfname.split("."):
                mod = getattr(mod, part)
            hooks.append(mod.register_forward_hook(mk("%d.%s" % (li, hfname))))
    _forward(args, model, torch)
    for h in hooks:
        h.remove()

    print("q4_0 vs HF bf16 -- PROJECTION OUTPUT on real activations")
    print("%-6s %-13s %-11s %-11s %-11s" % ("layer", "role", "out_cos", "out_rel", "w_rel"))
    rows = []
    for li in layers:
        for role, hfname, kind in ROLES:
            gname = "blk.%d.%s.weight" % (li, role)
            if g.tensors[gname].type_name != "Q4_0":
                print("%-6d %-13s (not q4_0)" % (li, role))
                continue
            Wq = _dequant(g, gname, "bf16", _nh(kind, n_head, n_kv_head))
            mod = model.model.layers[li]
            for part in hfname.split("."):
                mod = getattr(mod, part)
            Wr = mod.weight.detach().to(torch.float32).numpy()
            x = cap["%d.%s" % (li, hfname)]
            yq, yr = x @ Wq.T, x @ Wr.T
            rows.append((_cos(yq, yr), _rel(yq, yr), _rel(Wq, Wr)))
            print("%-6d %-13s %-11.6f %-11.5f %-11.5f" % (li, role, *rows[-1]))
    c = np.array([r[0] for r in rows])
    rl = np.array([r[1] for r in rows])
    wr = np.array([r[2] for r in rows])
    print()
    print("  n=%d  out_cos min %.6f median %.6f | out_rel max %.5f median %.5f "
          "| w_rel median %.5f" % (len(rows), c.min(), np.median(c), rl.max(),
                                   np.median(rl), np.median(wr)))


def cmd_salient(args, g, model, torch, n_head, n_kv_head):
    li, hfname = args.salient_layer, args.salient_proj
    role = dict((h, r) for r, h, _ in ROLES)[hfname]
    kind = dict((h, k) for _, h, k in ROLES)[hfname]
    cap = {}

    def hook(m, i, o):
        if "x" not in cap:
            x = i[0].detach().to(torch.float32)
            cap["x"] = x.reshape(-1, x.shape[-1]).numpy()

    mod = model.model.layers[li]
    for part in hfname.split("."):
        mod = getattr(mod, part)
    h = mod.register_forward_hook(hook)
    _forward(args, model, torch)
    h.remove()
    x = cap["x"]

    Wq = _dequant(g, "blk.%d.%s.weight" % (li, role), "bf16", _nh(kind, n_head, n_kv_head))
    Wr = mod.weight.detach().to(torch.float32).numpy()
    dW = Wq - Wr
    K = x.shape[1]
    amag = np.abs(x).mean(axis=0)
    order = np.argsort(-amag)
    contrib = (dW**2).sum(axis=0) * (x**2).sum(axis=0)
    tot = contrib.sum()
    print("layer %d %s -- activation magnitude per input channel (%d tokens)"
          % (li, hfname, x.shape[0]))
    print("  max %.3f | p99 %.3f | median %.3f | max/median %.1fx"
          % (amag.max(), np.percentile(amag, 99), np.median(amag),
             amag.max() / np.median(amag)))
    for frac in (0.001, 0.01, 0.05, 0.10):
        n = max(1, int(K * frac))
        top = order[:n]
        print("  top %5.1f%% of channels (%4d of %d): %5.1f%% of output error energy, "
              "%5.1f%% of activation energy"
              % (100 * frac, n, K, 100 * contrib[top].sum() / tot,
                 100 * (x[:, top] ** 2).sum() / (x**2).sum()))
    we = np.linalg.norm(dW, axis=0) / np.maximum(np.linalg.norm(Wr, axis=0), 1e-30)
    print("  per-channel relative WEIGHT error: median %.4f p99 %.4f max %.4f"
          % (np.median(we), np.percentile(we, 99), we.max()))


def _forward(args, model, torch):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.hf)
    with torch.no_grad():
        model(**tok(args.prompt, return_tensors="pt"))


def main():
    p = argparse.ArgumentParser(prog="q4_0_accuracy.py")
    p.add_argument("--gguf", required=True)
    p.add_argument("--hf", required=True, help="HF repo id of the bf16 original")
    p.add_argument("--weights", action="store_true")
    p.add_argument("--activations", action="store_true")
    p.add_argument("--salient", action="store_true")
    p.add_argument("--layers", default="0,11,23")
    p.add_argument("--salient-layer", type=int, default=0, dest="salient_layer")
    p.add_argument("--salient-proj", default="self_attn.v_proj", dest="salient_proj")
    p.add_argument(
        "--prompt",
        default="Introduce me what is GPU. Explain briefly how a graphics "
        "processing unit differs from a central processing unit, and why "
        "parallelism matters.",
    )
    args = p.parse_args()
    if not (args.weights or args.activations or args.salient):
        args.weights = True

    g = GGUFFile(args.gguf)
    n_head = g.metadata["llama.attention.head_count"]
    n_kv_head = g.metadata["llama.attention.head_count_kv"]
    n_layers = g.metadata["llama.block_count"]
    print("gguf: %s (arch=%s, %d layers, n_head=%d, n_kv_head=%d)"
          % (os.path.basename(args.gguf), g.architecture(), n_layers, n_head, n_kv_head))
    print("hf  : %s\n" % args.hf)

    if args.weights and not (args.activations or args.salient):
        # Weights-only needs safetensors, not a full torch model load.
        from safetensors import safe_open

        hub = os.path.expanduser("~/.cache/huggingface/hub")
        d = os.path.join(hub, "models--" + args.hf.replace("/", "--"), "snapshots")
        snap = os.path.join(d, os.listdir(d)[0])
        f = safe_open(os.path.join(snap, "model.safetensors"), framework="np")

        def hf_weight(li, hfname):
            return f.get_tensor("model.layers.%d.%s.weight" % (li, hfname)).astype(np.float32)

        cmd_weights(args, g, hf_weight, n_head, n_kv_head, n_layers)
        return 0

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.bfloat16).eval()
    if args.weights:
        def hf_weight(li, hfname):
            mod = model.model.layers[li]
            for part in hfname.split("."):
                mod = getattr(mod, part)
            return mod.weight.detach().to(torch.float32).numpy()

        cmd_weights(args, g, hf_weight, n_head, n_kv_head, n_layers)
        print()
    if args.activations:
        cmd_activations(args, g, model, torch, n_head, n_kv_head)
        print()
    if args.salient:
        cmd_salient(args, g, model, torch, n_head, n_kv_head)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
