"""Is a change in ground-truth correlation real, or is it resampling noise?

The 3B run produced ground-truth Spearman shifts of up to +-0.05 in
inconsistent directions -- int4_wo beat fp32 by +0.0495 on BLAT while losing
0.0167 on IF1. A single rho per config cannot separate "quantization changed
the answer" from "these happen to be 4996 particular variants", and that
distinction is the whole conclusion: whether quantization is safe for variant
effect work, or whether it is a coin flip you got lucky on.

Method: paired bootstrap over variants. Both configs are scored on the SAME
resampled variant set every iteration, so the shared assay noise cancels and
what remains is the part attributable to the config. Reporting an unpaired
interval here would be much wider and would hide a real effect.

    python src/analyze_dms.py results/dms_3B_*_<jobid>_scores.json
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy.stats import spearmanr

N_BOOT = 2000
SEED = 0


def analyse(path: str, n_boot: int = N_BOOT) -> None:
    d = json.load(open(path))
    expt = np.array([np.nan if v is None else v for v in d["experimental"]],
                    dtype=float)
    keep = ~np.isnan(expt)

    cols = {k: np.array([np.nan if x is None else x for x in v], dtype=float)
            for k, v in d["scores"].items()}
    for v in cols.values():
        keep &= ~np.isnan(v)

    expt = expt[keep]
    cols = {k: v[keep] for k, v in cols.items()}
    n = keep.sum()
    if "fp32" not in cols:
        print(f"{path}: no fp32 reference, skipping")
        return

    name = path.split("/")[-1].replace("_scores.json", "")
    print(f"\n=== {name}  (n={n} variants, {n_boot} bootstrap resamples) ===")
    print(f"{'config':10s} {'rho_expt':>9s} {'delta':>9s} "
          f"{'95% CI of delta':>22s} {'p':>8s}   verdict")
    print("-" * 88)

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(n_boot, n))

    base = cols["fp32"]
    rho_base = spearmanr(base, expt).statistic
    print(f"{'fp32':10s} {rho_base:9.4f} {'(ref)':>9s} {'':>22s} {'':>8s}")

    for k, v in cols.items():
        if k == "fp32":
            continue
        rho = spearmanr(v, expt).statistic
        # Paired: identical resample applied to both configs and to expt.
        deltas = np.empty(n_boot)
        for i in range(n_boot):
            j = idx[i]
            e = expt[j]
            deltas[i] = spearmanr(v[j], e).statistic - spearmanr(base[j], e).statistic
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        # Two-sided bootstrap p: how often the sign flips relative to the point
        # estimate. Small p = the shift is consistent, not sampling luck.
        p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
        verdict = "REAL" if lo > 0 or hi < 0 else "noise"
        print(f"{k:10s} {rho:9.4f} {rho - rho_base:+9.4f} "
              f"[{lo:+7.4f}, {hi:+7.4f}]{'':>4s} {p:8.4f}   {verdict}")


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    for p in paths:
        analyse(p)
