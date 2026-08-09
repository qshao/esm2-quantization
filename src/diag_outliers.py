"""Why does int8_dyn collapse on some assays? Measure the activations.

int8_dyn already uses per-token activation scales and per-channel weight scales,
so "quantize at a finer granularity" is not an available explanation -- it is
already the configuration being used. What a per-token scale still cannot absorb
is a single CHANNEL that is huge within a token: the token's scale is set by that
one channel, and every other channel in the token collapses toward zero after
rounding. That is the activation-outlier regime, and it is what SmoothQuant
migrates into the weights.

This script tests whether the assays that collapse actually show that signature,
rather than assuming they do. It compares one collapsing assay against
length-matched controls that do not collapse, on the same model.

    python src/diag_outliers.py --model 3B

Reported per encoder Linear input, over the real masked-marginals batch:
  outlier_ratio  max over channels of |x|, divided by the median over channels.
                 This is the quantity SmoothQuant reduces. ~1 means the token's
                 range is shared; large means one channel owns it.
  kurtosis       heavy-tailedness of the per-channel magnitudes.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dms as dms_mod          # noqa: E402
import models                  # noqa: E402
import quant                   # noqa: E402

COLLAPSE = "UBR5_HUMAN_Tsuboyama_2023_1I2T"
CONTROLS = ["DNJA1_HUMAN_Tsuboyama_2023_2LO1", "PR40A_HUMAN_Tsuboyama_2023_1UZC",
            "SPTN1_CHICK_Tsuboyama_2023_1TUD", "YAIA_ECOLI_Tsuboyama_2023_2KVT"]


def masked_batch(tok, seq, variants, batch_size, device):
    """One chunk of the real masked-marginals input, not a synthetic probe."""
    positions = sorted({m.pos for v in variants for m in dms_mod.parse_variant(v)})
    enc = tok(seq, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"][0].to(device)
    chunk = positions[:batch_size]
    X = ids.unsqueeze(0).repeat(len(chunk), 1)
    for r, p in enumerate(chunk):
        X[r, p] = tok.mask_token_id
    attn = enc["attention_mask"][0].to(device).unsqueeze(0).expand(len(chunk), -1)
    return X, attn


@torch.no_grad()
def profile(model, tok, seq, variants, device, batch_size=8):
    stats = {}

    def hook(name):
        def fn(mod, args):
            x = args[0].detach()
            x = x.reshape(-1, x.shape[-1]).abs().float()
            per_channel = x.amax(dim=0)                    # absmax per channel
            med = per_channel.median().clamp_min(1e-9)
            v = per_channel / med
            n = v.numel()
            mu, sd = v.mean(), v.std().clamp_min(1e-9)
            stats.setdefault(name, []).append((
                float(per_channel.max() / med),            # outlier ratio
                float((((v - mu) / sd) ** 4).mean()),      # kurtosis
                float(x.max()),                            # raw absmax
            ))
        return fn

    handles = [m.register_forward_pre_hook(hook(n))
               for n, m in model.named_modules()
               if quant.encoder_linear_filter(m, n)]
    X, attn = masked_batch(tok, seq, variants, batch_size, device)
    model(input_ids=X, attention_mask=attn)
    for h in handles:
        h.remove()
    a = np.array([v[0] for vs in stats.values() for v in vs])
    k = np.array([v[1] for vs in stats.values() for v in vs])
    r = np.array([v[2] for vs in stats.values() for v in vs])
    return {"n_linears": len(stats), "outlier_ratio_median": float(np.median(a)),
            "outlier_ratio_p95": float(np.percentile(a, 95)),
            "outlier_ratio_max": float(a.max()),
            "kurtosis_median": float(np.median(k)), "absmax_max": float(r.max())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="3B")
    p.add_argument("--dms_batch", type=int, default=8)
    args = p.parse_args()
    import json

    import pandas as pd
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    idx = {a["id"]: a for a in json.load(open(os.path.join(
        root, "data", "proteingym_v1", "index.json")))["assays"]}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, _ = models.load(args.model, "bf16", device=device)
    print(f"{'assay':36s} {'role':9s} {'L':>4s} {'outlier ratio':>26s} "
          f"{'kurtosis':>9s} {'absmax':>9s}")
    print(f"{'':36s} {'':9s} {'':>4s} {'median':>8s}{'p95':>9s}{'max':>9s}")
    print("-" * 100)
    out = []
    for aid, role in [(COLLAPSE, "COLLAPSE")] + [(c, "control") for c in CONTROLS]:
        a = idx[aid]
        df = pd.read_csv(os.path.join(root, "data", "proteingym_v1", a["csv"]))
        s = profile(model, tok, a["wt"], df["mutant"].astype(str).tolist(),
                    device, args.dms_batch)
        s.update(assay=aid, role=role, seq_len=a["seq_len"])
        out.append(s)
        print(f"{aid[:36]:36s} {role:9s} {a['seq_len']:4d} "
              f"{s['outlier_ratio_median']:8.1f}{s['outlier_ratio_p95']:9.1f}"
              f"{s['outlier_ratio_max']:9.1f} {s['kurtosis_median']:9.1f} "
              f"{s['absmax_max']:9.1f}")
    with open(os.path.join(root, "results",
                           f"outliers_{args.model}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[done] results/outliers_{args.model}.json")


if __name__ == "__main__":
    main()
