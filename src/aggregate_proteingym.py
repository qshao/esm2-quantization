"""Benchmark-wide analysis of the ProteinGym sweep: accuracy, speed, RAM.

run_proteingym.py deliberately writes only raw per-variant scores, so every
cross-config comparison lives here. That split is what lets the array tasks run
in any order, and it means this script can be re-run with new statistics
without touching the GPU.

Three things this can answer that the three-assay run could not:

  1. A benchmark-level verdict per config. With 201 assays the unit of
     resampling becomes the ASSAY, not the variant. That is the right unit:
     the question is "does this config degrade variant-effect prediction in
     general", and answering it by resampling variants within three assays
     silently treats assay choice as fixed when assay choice was the dominant
     source of disagreement.

  2. Whether fidelity-vs-fp32 predicts ground-truth damage. The report claimed
     r = 0.87 between rho_fp32 and |delta rho_expt| from 15 assay/config pairs.
     Here that becomes ~1000 pairs, enough for the claim to actually survive or
     die rather than be suggestive.

  3. Whether damage concentrates by assay type. ProteinGym labels each assay
     with a function category and an MSA-depth bin; a config that is safe on
     stability assays and harmful on binding assays is a materially different
     recommendation from one that is uniformly mildly lossy.

    python src/aggregate_proteingym.py --model 3B
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_BOOT = 2000
SEED = 0
REF = "fp32"


def load_config(path: str) -> dict:
    out = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                break  # torn final line from a killed job
            if "error" not in r:
                out[r["assay"]] = r
    return out


def load_all(model: str, out_dir: str) -> dict:
    runs = {}
    for p in sorted(glob.glob(os.path.join(out_dir, f"{model}_*.jsonl.gz"))):
        cfg = os.path.basename(p)[len(model) + 1:].replace(".jsonl.gz", "")
        runs[cfg] = load_config(p)
    return runs


def per_assay_table(runs: dict, index: dict, meta: pd.DataFrame) -> pd.DataFrame:
    """One row per (assay, config): accuracy, speed, RAM.

    Assays are kept only where EVERY config produced a score, so all columns
    are compared on identical assay sets. Otherwise a config that crashed on
    the ten hardest assays would post the best average by dodging them.
    """
    common = set.intersection(*(set(r) for r in runs.values())) if runs else set()
    rows = []
    for aid in sorted(common):
        info = index[aid]
        df = pd.read_csv(os.path.join(ROOT, "data", "proteingym_v1", info["csv"]))
        expt = df["DMS_score"].to_numpy(float)
        ref = np.array(runs[REF][aid]["scores"], dtype=float) if REF in runs else None
        for cfg, r in runs.items():
            s = np.array(r[aid]["scores"], dtype=float)
            ok = ~np.isnan(s) & ~np.isnan(expt)
            row = {
                "assay": aid, "config": cfg, "seq_len": info["seq_len"],
                "n_variants": int(ok.sum()), "n_positions": info["n_positions"],
                "seconds": r[aid]["seconds"], "peak_mem_gb": r[aid]["peak_mem_gb"],
                "variants_per_s": r[aid]["variants_per_s"],
                "positions_per_s": round(info["n_positions"] / r[aid]["seconds"], 2),
                "rho_expt": spearmanr(s[ok], expt[ok]).statistic,
                "category": meta.at[aid, "coarse_selection_type"],
                "msa_depth": meta.at[aid, "MSA_Neff_L_category"],
            }
            if ref is not None:
                m = ok & ~np.isnan(ref)
                row["rho_fp32"] = (1.0 if cfg == REF
                                   else spearmanr(s[m], ref[m]).statistic)
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_over_assays(tab: pd.DataFrame, clusters: pd.Series | None = None,
                          n_boot: int = N_BOOT) -> pd.DataFrame:
    """Paired bootstrap for delta(mean rho_expt) vs fp32.

    Paired: one resample of indices is applied to the candidate and to fp32
    together, so assay-difficulty variance cancels instead of swamping the
    interval. That is not a refinement, it is the difference between a usable
    answer and none: per-assay rho spans ~0.0-0.9 while the config effect is
    ~0.006, and the unpaired interval comes out 8x wider, calling every config
    noise.

    `clusters` switches the resampling unit from the assay to the PROTEIN
    (a cluster bootstrap: whole clusters are drawn, keeping every assay of a
    drawn protein). The 201 assays are not 201 independent draws -- they cover
    173 proteins, and BLAT_ECOLX alone appears 4 times. Resampling assays treats
    correlated replicates as independent evidence and returns an interval that
    is too narrow. Resampling proteins is the honest unit, and the two are
    reported side by side so the difference is visible rather than assumed.
    """
    wide = tab.pivot(index="assay", columns="config", values="rho_expt").dropna()
    if REF not in wide:
        raise SystemExit(f"no {REF} column; cannot compute deltas")
    rng = np.random.default_rng(SEED)
    n = len(wide)

    if clusters is None:
        idx = rng.integers(0, n, size=(n_boot, n))
        draws = [idx]
    else:
        # Cluster resample: draw n_clusters clusters with replacement, then take
        # every assay belonging to each drawn cluster. Resample size therefore
        # varies between replicates, so they cannot share one 2-D index array.
        g = clusters.reindex(wide.index)
        members = [np.flatnonzero((g == k).to_numpy()) for k in g.unique()]
        k = len(members)
        pick = rng.integers(0, k, size=(n_boot, k))
        draws = [np.concatenate([members[j] for j in row]) for row in pick]

    base = wide[REF].to_numpy()
    rows = []
    for cfg in wide.columns:
        v = wide[cfg].to_numpy()
        if clusters is None:
            d = (v[draws[0]] - base[draws[0]]).mean(axis=1)
        else:
            d = np.array([(v[s] - base[s]).mean() for s in draws])
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        rows.append({
            "config": cfg, "n_assays": n, "mean_rho_expt": v.mean(),
            "median_rho_expt": np.median(v), "delta_mean": v.mean() - base.mean(),
            "ci_lo": lo, "ci_hi": hi, "p": p,
            "verdict": "-" if cfg == REF else ("REAL" if (lo > 0 or hi < 0) else "noise"),
            "n_assays_worse": int((v < base).sum()),
        })
    return pd.DataFrame(rows).set_index("config")


def fidelity_vs_damage(tab: pd.DataFrame) -> dict:
    """Does drift from fp32 predict loss of ground-truth accuracy?

    The three-assay report found fidelity predicted the MAGNITUDE of the
    ground-truth change but not its SIGN, from 15 pairs. Both halves are
    re-tested here at ~1000 pairs.
    """
    d = tab[tab.config != REF].copy()
    ref = tab[tab.config == REF].set_index("assay")["rho_expt"]
    d["delta"] = d["rho_expt"] - d["assay"].map(ref)
    d = d.dropna(subset=["rho_fp32", "delta"])
    infid = 1.0 - d["rho_fp32"]
    out = {"n_pairs": int(len(d))}
    if len(d) > 2:
        out["pearson_infidelity_vs_absdelta"] = float(pearsonr(infid, d["delta"].abs())[0])
        out["spearman_infidelity_vs_absdelta"] = float(spearmanr(infid, d["delta"].abs()).statistic)
        out["pearson_infidelity_vs_signeddelta"] = float(pearsonr(infid, d["delta"])[0])
        out["frac_delta_negative"] = float((d["delta"] < 0).mean())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="3B")
    p.add_argument("--out_dir", default="results/proteingym")
    p.add_argument("--index", default="data/proteingym_v1/index.json")
    p.add_argument("--meta", default="data/proteingym/DMS_substitutions.csv")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    runs = load_all(args.model, args.out_dir)
    if not runs:
        raise SystemExit(f"no {args.model}_*.jsonl.gz under {args.out_dir}")
    index = {a["id"]: a for a in json.load(open(args.index))["assays"]}
    meta = pd.read_csv(args.meta).set_index("DMS_id")

    print(f"configs: " + ", ".join(f"{k} ({len(v)} assays)" for k, v in runs.items()))
    tab = per_assay_table(runs, index, meta)
    n_assay = tab.assay.nunique()
    print(f"[common] {n_assay} assays scored by every config, "
          f"{int(tab[tab.config == list(runs)[0]].n_variants.sum()):,} variants\n")

    base = args.out or os.path.join(args.out_dir, f"summary_{args.model}")
    tab.to_csv(base + "_per_assay.csv", index=False)

    # ---- accuracy ----
    prot = meta.loc[tab.assay.unique(), "UniProt_ID"]
    boot = bootstrap_over_assays(tab)
    cboot = bootstrap_over_assays(tab, clusters=prot)
    print("=" * 104)
    print(f"ACCURACY -- mean Spearman vs experiment over {n_assay} assays, "
          f"{prot.nunique()} proteins ({N_BOOT} paired resamples)")
    print("  assay bootstrap: resamples assays.  protein bootstrap: resamples whole "
          "proteins, so\n  the repeated assays on one protein cannot count as "
          "independent evidence.")
    print(f"{'config':10s} {'mean_rho':>9s} {'delta':>9s} "
          f"{'95% CI (assay)':>21s} {'95% CI (protein)':>21s} {'p_prot':>7s} "
          f"{'worse/n':>9s}  verdict")
    print("-" * 104)
    for cfg, r in boot.iterrows():
        c = cboot.loc[cfg]
        d = "(ref)" if cfg == REF else f"{r.delta_mean:+.4f}"
        ci = "" if cfg == REF else f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]"
        cci = "" if cfg == REF else f"[{c.ci_lo:+.4f}, {c.ci_hi:+.4f}]"
        pv = "" if cfg == REF else f"{c.p:.4f}"
        w = "" if cfg == REF else f"{r.n_assays_worse}/{int(r.n_assays)}"
        print(f"{cfg:10s} {r.mean_rho_expt:9.4f} {d:>9s} {ci:>21s} {cci:>21s} "
              f"{pv:>7s} {w:>9s}  {c.verdict}")
    print("=" * 104)
    print("verdict uses the protein bootstrap -- the conservative of the two.")

    # ---- speed and RAM ----
    agg = tab.groupby("config").agg(
        total_sec=("seconds", "sum"), pos_per_s=("positions_per_s", "median"),
        var_per_s=("variants_per_s", "median"), peak_med=("peak_mem_gb", "median"),
        peak_max=("peak_mem_gb", "max"), rho_fp32_med=("rho_fp32", "median"),
        rho_fp32_min=("rho_fp32", "min"))
    print("\nSPEED / RAM over the full benchmark")
    print(f"{'config':10s} {'total_h':>8s} {'pos/s':>8s} {'var/s':>9s} "
          f"{'peakGB_med':>11s} {'peakGB_max':>11s} {'rho_fp32_med':>13s} {'min':>8s}")
    print("-" * 96)
    for cfg, r in agg.iterrows():
        print(f"{cfg:10s} {r.total_sec / 3600:8.3f} {r.pos_per_s:8.1f} "
              f"{r.var_per_s:9.1f} {r.peak_med:11.2f} {r.peak_max:11.2f} "
              f"{r.rho_fp32_med:13.5f} {r.rho_fp32_min:8.4f}")

    # ---- by category ----
    piv = tab.pivot_table(index="category", columns="config", values="rho_expt")
    print("\nMean rho vs experiment by function category")
    print(piv.round(4).to_string())
    piv2 = tab.pivot_table(index="msa_depth", columns="config", values="rho_expt")
    print("\nMean rho vs experiment by MSA depth")
    print(piv2.round(4).to_string())

    fd = fidelity_vs_damage(tab)
    print(f"\nFidelity vs ground-truth damage ({fd.get('n_pairs')} assay/config pairs)")
    for k, v in fd.items():
        if k != "n_pairs":
            print(f"   {k:38s} {v:+.4f}")

    with open(base + ".json", "w") as fh:
        json.dump({"model": args.model, "n_assays": int(n_assay),
                   "n_proteins": int(prot.nunique()),
                   "accuracy": json.loads(boot.reset_index().to_json(orient="records")),
                   "accuracy_cluster_by_protein":
                       json.loads(cboot.reset_index().to_json(orient="records")),
                   "speed_ram": json.loads(agg.reset_index().to_json(orient="records")),
                   "by_category": json.loads(piv.to_json()),
                   "by_msa_depth": json.loads(piv2.to_json()),
                   "fidelity_vs_damage": fd}, fh, indent=2)
    print(f"\n[done] {base}.json and {base}_per_assay.csv")


if __name__ == "__main__":
    main()
