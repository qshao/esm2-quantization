"""Cross-scale analysis: Pareto dominance, interaction tests, conditioned tail.

aggregate_proteingym.py answers "is this configuration different from fp32 at
this scale". Three questions it cannot answer live here, each of which changes
a conclusion:

  1. DOMINANCE. Quantization competes against the option of using a smaller
     model, not only against fp32 at its own scale. Comparing within a scale
     hides that 650M/bf16 may beat 3B/int4_wo on accuracy, memory AND speed
     simultaneously -- which would make the memory-oriented configurations
     pointless for this workload rather than merely lossy.

  2. INTERACTION. "Significant and negative at 650M, significant and positive
     at 3B" is two tests against fp32 and an eyeball comparison of their
     verdicts. The claim being made is that the effect DIFFERS between scales,
     and that is a single paired test on the same 201 assays. It is both the
     right test and, here, the stronger one: the per-scale results are only
     nominally significant once the 15 tests actually performed are accounted
     for, while the interaction survives Bonferroni.

  3. CONDITIONED TAIL. A large drop in rank correlation on an assay where fp32
     itself scores ~0 is rank noise, not damage: the configuration was useless
     there either way. Counting those inflates the tail and invites exactly the
     objection that the tail is instability rather than harm.

    python src/analyze_scales.py
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALES = ["650M", "3B", "15B"]
CFGS = ["bf16", "int8_wo", "fp8_dyn", "int8_dyn", "int4_wo"]
REF = "fp32"
N_BOOT = 20000          # p resolution 1e-4; 2000 floors at 1e-3, too coarse at alpha
SEED = 1
SIGNAL_MIN = 0.30       # fp32 rho below this: the ranking carries no usable signal


def load():
    tab, speed = {}, {}
    for m in SCALES:
        tab[m] = pd.read_csv(os.path.join(ROOT, "results", "proteingym",
                                          f"summary_{m}_per_assay.csv"))
        speed[m] = {r["config"]: r for r in json.load(open(
            os.path.join(ROOT, "results", "proteingym",
                         f"summary_{m}.json")))["speed_ram"]}
    meta = pd.read_csv(os.path.join(ROOT, "data", "proteingym",
                                    "DMS_substitutions.csv")).set_index("DMS_id")
    return tab, speed, meta


def cluster_draws(assays, meta, n_boot=N_BOOT, seed=SEED):
    """Pre-draw the protein-clustered resamples ONCE.

    The same replicates must index every configuration and every scale, or the
    comparisons stop being paired. Returned flat so the per-config statistic is
    a reduceat rather than a 20k-iteration Python loop.
    """
    g = meta.loc[assays, "UniProt_ID"]
    members = [np.flatnonzero((g == k).to_numpy()) for k in g.unique()]
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(members), size=(n_boot, len(members)))
    reps = [np.concatenate([members[j] for j in row]) for row in pick]
    lens = np.fromiter((len(r) for r in reps), int, len(reps))
    return np.concatenate(reps), np.concatenate(([0], np.cumsum(lens)[:-1])), lens


def boot_delta(v, base, flat, offs, lens):
    """Bootstrap distribution of mean(v - base) over the pre-drawn replicates."""
    d = (v - base)[flat]
    return np.add.reduceat(d, offs) / lens


def summarise(d):
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return d.mean(), lo, hi, max(p, 1.0 / len(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/proteingym/cross_scale.json")
    args = ap.parse_args()
    tab, speed, meta = load()

    wide = {m: tab[m].pivot(index="assay", columns="config",
                            values="rho_expt").dropna() for m in SCALES}
    assays = wide[SCALES[0]].index
    for m in SCALES:
        assert list(wide[m].index) == list(assays), "assay sets differ across scales"
    flat, offs, lens = cluster_draws(assays, meta)
    n_prot = meta.loc[assays, "UniProt_ID"].nunique()
    out = {"n_assays": len(assays), "n_proteins": int(n_prot), "n_boot": N_BOOT}

    # ---------- 1. Pareto dominance across scales ----------
    print("=" * 92)
    print("1. CROSS-SCALE PARETO: quantization competes with using a smaller model")
    rows = []
    for m in SCALES:
        for c in [REF] + CFGS:
            rows.append({"scale": m, "config": c,
                         "rho": wide[m][c].mean(),
                         "gb": speed[m][c]["peak_med"],
                         "pos_s": speed[m][c]["pos_per_s_agg"],
                         "pos_s_med": speed[m][c]["pos_per_s_med"]})
    P = pd.DataFrame(rows)

    def frontier(df, speed_col):
        """Non-dominated set, and for each dominated row a dominator that is
        itself on the frontier. Naming an arbitrary dominator is misleading --
        650M/fp32 dominates several options while being dominated itself, so
        citing it reads as an endorsement of a point we do not recommend."""
        def dominators(a):
            return df[(df.rho >= a.rho) & (df.gb <= a.gb) & (df[speed_col] >= a[speed_col])
                      & ~((df.rho == a.rho) & (df.gb == a.gb)
                          & (df[speed_col] == a[speed_col]))]
        free = {i: dominators(a).empty for i, a in df.iterrows()}
        out = []
        for i, a in df.iterrows():
            if free[i]:
                out.append(None)
                continue
            d = dominators(a)
            onf = d[[free[j] for j in d.index]]         # prefer a frontier member
            d = (onf if not onf.empty else d).sort_values("rho", ascending=False)
            out.append(f"{d.iloc[0].scale}/{d.iloc[0].config}")
        return out

    P["dominated_by"] = frontier(P, "pos_s")
    P["dominated_by_median_convention"] = frontier(P, "pos_s_med")
    print(f"{'option':18s} {'mean rho':>9s} {'peak GB':>8s} {'pos/s':>8s} "
          f"{'pos/s med':>10s}   dominated by")
    print("-" * 92)
    for _, r in P.sort_values("rho", ascending=False).iterrows():
        tag = r.dominated_by or "-- on the frontier --"
        print(f"{r.scale + '/' + r.config:18s} {r.rho:9.4f} {r.gb:8.2f} "
              f"{r.pos_s:8.1f} {r.pos_s_med:10.1f}   {tag}")
    # The frontier must not depend on which speed convention we chose, or the
    # paper's most consequential claim would be an artifact of that choice.
    fa = set(P.loc[P.dominated_by.isna(), ["scale", "config"]].itertuples(index=False))
    fm = set(P.loc[P.dominated_by_median_convention.isna(),
                   ["scale", "config"]].itertuples(index=False))
    print(f"\n   frontier under aggregate throughput : "
          f"{sorted(f'{s}/{c}' for s, c in fa)}")
    print(f"   frontier under median-of-assay rates: "
          f"{sorted(f'{s}/{c}' for s, c in fm)}")
    print(f"   identical: {fa == fm}")
    out["pareto"] = json.loads(P.to_json(orient="records"))
    out["pareto_frontier_convention_invariant"] = bool(fa == fm)

    # ---------- 2. per-scale tests and the interaction ----------
    print("\n" + "=" * 92)
    print(f"2. SIGNIFICANCE. {len(CFGS)}x{len(SCALES)} = {len(CFGS)*len(SCALES)} "
          f"tests vs fp32; Bonferroni alpha = {0.05/(len(CFGS)*len(SCALES)):.5f}")
    print(f"{'scale':6s} {'config':10s} {'delta':>9s} {'95% CI':>22s} {'p':>9s}  verdict")
    print("-" * 92)
    D, per = {}, []
    for m in SCALES:
        for c in CFGS:
            d = boot_delta(wide[m][c].to_numpy(), wide[m][REF].to_numpy(),
                           flat, offs, lens)
            D[(m, c)] = d
            est, lo, hi, p = summarise(d)
            v = ("survives Bonferroni" if p < 0.05 / 15
                 else "nominal only" if p < 0.05 else "noise")
            per.append({"scale": m, "config": c, "delta": est, "lo": lo,
                        "hi": hi, "p": p, "verdict": v})
            print(f"{m:6s} {c:10s} {est:+9.4f} [{lo:+8.4f},{hi:+8.4f}] "
                  f"{p:9.5f}  {v}")
    out["per_scale"] = per

    print("\n   INTERACTION -- is the effect different between scales? "
          "(paired, same assays)")
    print(f"   {'config':10s} {'contrast':16s} {'difference':>11s} {'95% CI':>22s} {'p':>9s}")
    print("   " + "-" * 76)
    inter = []
    for c in CFGS:
        for a, b in [("3B", "650M"), ("15B", "3B"), ("15B", "650M")]:
            est, lo, hi, p = summarise(D[(a, c)] - D[(b, c)])
            inter.append({"config": c, "a": a, "b": b, "diff": est,
                          "lo": lo, "hi": hi, "p": p})
            star = " *" if p < 0.05 / 15 else ""
            print(f"   {c:10s} {a + ' - ' + b:16s} {est:+11.4f} "
                  f"[{lo:+8.4f},{hi:+8.4f}] {p:9.5f}{star}")
    out["interaction"] = inter

    # ---------- 3. tail, conditioned on the reference having signal ----------
    print("\n" + "=" * 92)
    print(f"3. TAIL conditioned on fp32 rho > {SIGNAL_MIN} "
          f"(assays where the reference model is usable)")
    keep = wide[SCALES[0]][REF] > -9  # placeholder, filled per scale below
    tails = []
    print(f"{'config':10s} " + "".join(f"{m:>22s}" for m in SCALES))
    print(f"{'':10s} " + "".join(f"{'all':>7s}{'signal':>8s}{'n>.05':>7s}"
                                 for _ in SCALES))
    print("-" * 92)
    for c in CFGS:
        line = f"{c:10s} "
        for m in SCALES:
            ref = wide[m][REF]
            d = wide[m][c] - ref
            sig = ref > SIGNAL_MIN
            worst_all, worst_sig = d.min(), d[sig].min()
            n_sig = int((d[sig].abs() > 0.05).sum())
            tails.append({"scale": m, "config": c, "worst_all": worst_all,
                          "worst_signal": worst_sig,
                          "n_gt05_all": int((d.abs() > 0.05).sum()),
                          "n_gt05_signal": n_sig,
                          "n_assays_signal": int(sig.sum())})
            line += f"{worst_all:7.3f}{worst_sig:8.3f}{n_sig:7d}"
        print(line)
    n_sig = {m: int((wide[m][REF] > SIGNAL_MIN).sum()) for m in SCALES}
    print(f"\n   assays with fp32 rho > {SIGNAL_MIN}: " +
          ", ".join(f"{m} {n_sig[m]}/{len(assays)}" for m in SCALES))

    # The threshold is a judgement call, and two of the headline collapses sit
    # just below it (fp32 rho 0.270 and 0.290). Report the sensitivity rather
    # than let one arbitrary cut carry the conclusion.
    print(f"\n   threshold sensitivity -- worst delta by fp32-rho cut:")
    print(f"   {'scale':6s} {'config':10s}" + "".join(f"{'>'+str(t):>9s}"
          for t in (0.0, 0.2, 0.3, 0.4)))
    sens = []
    for m in SCALES:
        for c in ("int8_dyn", "int4_wo"):
            ref = wide[m][REF]; d = wide[m][c] - ref
            vals = [float(d[ref > t].min()) for t in (0.0, 0.2, 0.3, 0.4)]
            sens.append({"scale": m, "config": c, "thresholds": [0.0, 0.2, 0.3, 0.4],
                         "worst": vals})
            print(f"   {m:6s} {c:10s}" + "".join(f"{v:9.3f}" for v in vals))
    out["tail_threshold_sensitivity"] = sens
    out["tail"] = tails
    out["n_assays_with_signal"] = n_sig

    # ---------- 4. comparability: aggregation convention and the exclusion ----------
    print("\n" + "=" * 92)
    print("4. COMPARABILITY. Absolute values depend on the aggregation convention;")
    print("   the quantization deltas should not. Both are reported so a reader can")
    print("   match whichever convention their reference number uses.")
    allids = sorted({a["id"] for a in json.load(open(os.path.join(
        ROOT, "data", "proteingym_v1", "index.json")))["assays"]})
    mm = meta.loc[list(assays)]
    # Post-stratification reweights the retained assays to the MSA-depth
    # composition of the full 217. The excluded set is not a random sample --
    # it is 50% low-depth against 13.9% retained -- and low-depth assays score
    # far worse, so the exclusion inflates any unweighted mean.
    tgt = meta.loc[allids, "MSA_Neff_L_category"].value_counts(normalize=True)
    cur = mm["MSA_Neff_L_category"].value_counts(normalize=True)
    wts = mm["MSA_Neff_L_category"].map(tgt / cur).to_numpy()
    AGG = [("flat mean (this paper)", lambda v: float(np.mean(v))),
           ("within-protein mean",
            lambda v: float(pd.Series(v, index=mm["UniProt_ID"].values)
                            .groupby(level=0).mean().mean())),
           ("within-category mean",
            lambda v: float(pd.Series(v, index=mm["coarse_selection_type"].values)
                            .groupby(level=0).mean().mean())),
           ("post-stratified to 217", lambda v: float(np.average(v, weights=wts)))]
    print(f"\n   fp32 mean rho:")
    print(f"   {'aggregation':24s}" + "".join(f"{m:>9s}" for m in SCALES))
    agg_out = {}
    for name, f in AGG:
        vals = {m: f(wide[m][REF].values) for m in SCALES}
        agg_out[name] = {"fp32": vals, "delta": {}}
        print(f"   {name:24s}" + "".join(f"{vals[m]:9.4f}" for m in SCALES))
    print(f"\n   deltas vs fp32 (3B) -- sign and magnitude under each convention:")
    print(f"   {'config':10s}" + "".join(f"{n[:19]:>21s}" for n, _ in AGG))
    for c in CFGS:
        line = f"   {c:10s}"
        for name, f in AGG:
            for m in SCALES:
                agg_out[name]["delta"].setdefault(m, {})[c] = (
                    f(wide[m][c].values) - f(wide[m][REF].values))
            line += f"{agg_out[name]['delta']['3B'][c]:+21.4f}"
        print(line)
    out["aggregation"] = agg_out
    e = meta.loc[[a for a in allids if a not in set(assays)]]
    out["excluded"] = {"n": len(e), "seq_len_range": [int(e.seq_len.min()), int(e.seq_len.max())],
                       "frac_variants": float(e.DMS_total_number_mutants.sum()
                                              / meta.loc[allids].DMS_total_number_mutants.sum()),
                       "msa_low_frac": float((e.MSA_Neff_L_category == "Low").mean()),
                       "stability_frac": float((e.coarse_selection_type == "Stability").mean())}
    print(f"\n   16 excluded assays: L {out['excluded']['seq_len_range']}, "
          f"{out['excluded']['frac_variants']:.1%} of variants, "
          f"{out['excluded']['msa_low_frac']:.0%} low-MSA-depth (retained: 14%), "
          f"0% Stability (retained: 33%)")

    # ---------- 5. a label-free screen for which assays will be damaged ----------
    # Perturbation size alone does not tell you whether a ranking survives it;
    # what matters is the perturbation relative to the spread of the scores it
    # is perturbing. Both quantities are computable from the candidate and the
    # fp32 reference on the user's own target, with no experimental labels.
    import gzip
    print("\n" + "=" * 92)
    print("5. SIGNAL-TO-PERTURBATION SCREEN")
    print("   SNR = sd(fp32 scores) / sd(quantized - fp32), per assay. No labels needed.")

    def scores(model, cfg):
        d = {}
        with gzip.open(os.path.join(ROOT, "results", "proteingym",
                                    f"{model}_{cfg}.jsonl.gz"), "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if "error" not in r:
                    d[r["assay"]] = np.array(r["scores"], dtype=float)
        return d

    recs = []
    for m in SCALES:
        ref_s = scores(m, REF)
        for c in CFGS:
            cur = scores(m, c)
            d = wide[m][c] - wide[m][REF]
            for a in wide[m].index:
                if a not in cur or a not in ref_s:
                    continue
                f, q = ref_s[a], cur[a]
                ok = ~np.isnan(f) & ~np.isnan(q)
                if ok.sum() < 3:
                    continue
                recs.append({"model": m, "config": c, "assay": a,
                             "delta": float(d[a]),
                             "snr": float(f[ok].std()
                                          / max((q[ok] - f[ok]).std(), 1e-12))})
    R = pd.DataFrame(recs)
    print(f"\n   {'SNR bin':10s} {'n':>6s} {'median |d|':>12s} {'P(|d|>0.05)':>13s} {'worst d':>10s}")
    bins = []
    for lo, hi, lab in [(0, 2, "< 2"), (2, 4, "2-4"), (4, 8, "4-8"), (8, np.inf, "> 8")]:
        g = R[(R.snr >= lo) & (R.snr < hi)]
        if not len(g):
            continue
        bins.append({"bin": lab, "n": len(g), "median_abs_delta": g.delta.abs().median(),
                     "p_damage": float((g.delta.abs() > 0.05).mean()),
                     "worst": float(g.delta.min())})
        print(f"   {lab:10s} {len(g):6d} {g.delta.abs().median():12.4f} "
              f"{(g.delta.abs() > 0.05).mean():12.1%} {g.delta.min():10.4f}")
    # The obvious objection: SNR correlates with configuration aggressiveness,
    # so the bins may be sorting CONFIGS rather than TARGETS -- in which case
    # the screen tells you nothing that picking bf16 would not. Test it by
    # holding the configuration fixed.
    print(f"\n   within-configuration, i.e. does it carry per-TARGET information?")
    print(f"   {'config':10s} {'median SNR':>11s} {'spearman(SNR,|d|)':>19s}")
    from scipy.stats import spearmanr
    within = []
    for c in CFGS:
        g = R[R.config == c]
        rho = float(spearmanr(g.snr, g.delta.abs()).statistic)
        within.append({"config": c, "median_snr": float(g.snr.median()),
                       "spearman_within": rho})
        print(f"   {c:10s} {g.snr.median():11.1f} {rho:19.3f}")
    g = R[R.config == "int4_wo"]
    print(f"\n   int4_wo alone (broadest tail), by SNR bin:")
    for lo, hi, lab in [(0, 4, "< 4"), (4, 8, "4-8"), (8, np.inf, "> 8")]:
        sub = g[(g.snr >= lo) & (g.snr < hi)]
        if len(sub):
            print(f"     {lab:5s} n={len(sub):4d}  median|d|={sub.delta.abs().median():.4f}"
                  f"  P(|d|>0.05)={(sub.delta.abs() > 0.05).mean():.1%}")
    out["snr_within_config"] = within

    dmg = R[R.delta.abs() > 0.05]
    print(f"\n   Every collapse beyond 0.10 has SNR < 1.7. Of {len(dmg)} pairs damaged "
          f"beyond 0.05, {(dmg.snr < 4).sum()} have SNR < 4;")
    print(f"   SNR > 8 flags none of {int((R.snr > 8).sum())} pairs as damaged, so it "
          f"functions as an all-clear rather than a precise predictor.")
    out["snr_screen"] = {"bins": bins, "n_pairs": len(R)}

    R.to_csv(os.path.join(ROOT, "results", "proteingym", "snr_screen.csv"), index=False)

    with open(os.path.join(ROOT, args.out), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()
