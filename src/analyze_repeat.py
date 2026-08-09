"""Run-to-run repeatability: the noise floor under every number in the paper.

We report differences in mean Spearman at 1e-4 resolution and a worst-assay
statistic that is a maximum over 201 assays. Neither is interpretable without
knowing what the pipeline does when nothing changes. Two specific claims depend
on it:

  * "bf16 moves the benchmark mean by +0.0002" is meaningless if re-running
    bf16 moves it by more.
  * The tail statistic is a MAXIMUM over 201 assays, which any noise inflates.
    The right null is not zero, it is the largest per-assay |drho| between two
    runs of the SAME configuration.

Compares results/proteingym (run 1) against results/proteingym_rep2 (run 2),
which differ only in job id and physical GPU.

    python src/analyze_repeat.py
"""

from __future__ import annotations

import glob
import gzip
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN1 = os.path.join(ROOT, "results", "proteingym")
RUN2 = os.path.join(ROOT, "results", "proteingym_rep2")


def load_scores(path):
    out = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                break
            if "error" not in r:
                out[r["assay"]] = np.array(r["scores"], dtype=float)
    return out


def main() -> None:
    idx = {a["id"]: a for a in json.load(open(os.path.join(
        ROOT, "data", "proteingym_v1", "index.json")))["assays"]}
    pairs = []
    for p2 in sorted(glob.glob(os.path.join(RUN2, "*.jsonl.gz"))):
        p1 = os.path.join(RUN1, os.path.basename(p2))
        if os.path.exists(p1):
            model, cfg = os.path.basename(p2).replace(".jsonl.gz", "").split("_", 1)
            pairs.append((model, cfg, p1, p2))
    if not pairs:
        raise SystemExit(f"no replicate pairs found under {RUN2}")

    print("=" * 100)
    print("RUN-TO-RUN REPEATABILITY  (identical inputs; different job, different GPU)")
    print(f"{'model':6s} {'config':10s} {'assays':>7s} {'variants':>10s} "
          f"{'identical':>10s} {'max|dscore|':>12s} {'max|drho|':>10s} {'d(mean rho)':>12s}")
    print("-" * 100)
    rows = []
    for model, cfg, p1, p2 in pairs:
        s1, s2 = load_scores(p1), load_scores(p2)
        common = sorted(set(s1) & set(s2))
        n_var = n_same = 0
        max_ds = 0.0
        drhos = {}
        for a in common:
            v1, v2 = s1[a], s2[a]
            if v1.shape != v2.shape:
                continue
            ok = ~np.isnan(v1) & ~np.isnan(v2)
            n_var += int(ok.sum())
            n_same += int((v1[ok] == v2[ok]).sum())
            if ok.sum():
                max_ds = max(max_ds, float(np.abs(v1[ok] - v2[ok]).max()))
            df = pd.read_csv(os.path.join(ROOT, "data", "proteingym_v1",
                                          idx[a]["csv"]))
            e = df["DMS_score"].to_numpy(float)
            m = ok & ~np.isnan(e)
            if m.sum() > 2:
                drhos[a] = (spearmanr(v2[m], e[m]).statistic
                            - spearmanr(v1[m], e[m]).statistic)
        d = np.array(list(drhos.values()))
        worst = max(drhos, key=lambda k: abs(drhos[k])) if drhos else None
        rows.append({"model": model, "config": cfg, "n_assays": len(common),
                     "n_variants": n_var, "frac_identical": n_same / max(n_var, 1),
                     "max_abs_dscore": max_ds,
                     "max_abs_drho": float(np.abs(d).max()) if len(d) else 0.0,
                     "worst_assay": worst,
                     "d_mean_rho": float(d.mean()) if len(d) else 0.0})
        r = rows[-1]
        print(f"{model:6s} {cfg:10s} {r['n_assays']:7d} {r['n_variants']:10,d} "
              f"{r['frac_identical']:9.4%} {r['max_abs_dscore']:12.2e} "
              f"{r['max_abs_drho']:10.5f} {r['d_mean_rho']:+12.6f}")

    print("\nInterpretation:")
    floor = max(r["max_abs_drho"] for r in rows)
    mean_floor = max(abs(r["d_mean_rho"]) for r in rows)
    print(f"  Largest per-assay |drho| between two runs of the same config: {floor:.5f}")
    print(f"    -> the null for the worst-assay tail statistic is this, not zero.")
    print(f"  Largest change in benchmark mean rho from a repeat run: {mean_floor:.6f}")
    print(f"    -> effects below this are not resolvable by this pipeline.")
    for r in rows:
        if r["max_abs_drho"] > 0:
            print(f"  {r['model']}/{r['config']}: worst-drifting assay {r['worst_assay']}")

    # The headline collapse must reproduce, or the paper's central example is a
    # one-off rather than a property of the configuration.
    for r in rows:
        if r["model"] == "3B" and r["config"] == "int8_dyn":
            s1, s2 = load_scores(os.path.join(RUN1, "3B_int8_dyn.jsonl.gz")), \
                     load_scores(os.path.join(RUN2, "3B_int8_dyn.jsonl.gz"))
            a = "UBR5_HUMAN_Tsuboyama_2023_1I2T"
            if a in s1 and a in s2:
                df = pd.read_csv(os.path.join(ROOT, "data", "proteingym_v1",
                                              idx[a]["csv"]))
                e = df["DMS_score"].to_numpy(float)
                m = ~np.isnan(s1[a]) & ~np.isnan(s2[a]) & ~np.isnan(e)
                print(f"\n  headline check -- {a}:")
                print(f"    run 1 rho = {spearmanr(s1[a][m], e[m]).statistic:.4f}   "
                      f"run 2 rho = {spearmanr(s2[a][m], e[m]).statistic:.4f}")

    with open(os.path.join(RUN1, "repeatability.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\n[done] results/proteingym/repeatability.json")


if __name__ == "__main__":
    main()
