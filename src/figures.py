"""Figures for the preprint. Writes self-contained PDFs into preprint/.

    python src/figures.py

Fonts are embedded as TrueType (pdf.fonttype 42) because arXiv rejects PDFs
with unembedded fonts, and matplotlib's default Type 3 subsetting also renders
poorly in some viewers.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("pdf")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "preprint")
CFGS = ["bf16", "int8_wo", "fp8_dyn", "int8_dyn", "int4_wo"]
# Okabe-Ito: colourblind-safe, and distinguishable in greyscale print.
COL = {"fp32": "#000000", "bf16": "#0072B2", "int8_wo": "#009E73",
       "fp8_dyn": "#E69F00", "int8_dyn": "#CC79A7", "int4_wo": "#D55E00"}
LBL = {"int8_wo": "int8_wo", "int8_dyn": "int8_dyn", "fp8_dyn": "fp8_dyn",
       "int4_wo": "int4_wo", "bf16": "bf16", "fp32": "fp32"}


def load(model):
    t = pd.read_csv(os.path.join(ROOT, "results", "proteingym",
                                 f"summary_{model}_per_assay.csv"))
    ref = t[t.config == "fp32"].set_index("assay")["rho_expt"]
    t = t[t.config != "fp32"].copy()
    t["delta"] = t["rho_expt"] - t["assay"].map(ref)
    t["infid"] = 1.0 - t["rho_fp32"]
    return t


def fig_tail(models):
    """The paper's central claim: benchmark means agree, tails do not."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for ax, (name, t) in zip(axes, models.items()):
        for i, c in enumerate(CFGS):
            d = t[t.config == c]["delta"].to_numpy()
            y = i + rng.uniform(-0.22, 0.22, len(d))
            ax.scatter(d, y, s=3.5, alpha=0.55, color=COL[c], linewidths=0)
            ax.scatter([d.mean()], [i], marker="|", s=170, color="black", zorder=5)
            w = d.min()
            if w < -0.10:  # label only the collapses, not the ordinary spread
                ax.annotate(f"{w:.2f}", (w, i), textcoords="offset points",
                            xytext=(2, 6), fontsize=6.5, color=COL[c])
        ax.axvline(0, color="0.6", lw=0.7, zorder=0)
        ax.set_yticks(range(len(CFGS)))
        ax.set_yticklabels([LBL[c] for c in CFGS], family="monospace")
        ax.set_xlabel(r"$\Delta\rho$ vs fp32 (per assay)")
        ax.set_title(f"ESM2-{name}")
        ax.set_xlim(-0.40, 0.30)
    axes[0].invert_yaxis()
    fig.text(0.5, -0.10, "Each point is one of 201 assays; the black tick is the "
             "benchmark mean.", ha="center", fontsize=7, color="0.3")
    fig.savefig(os.path.join(OUT, "fig_tail.pdf"))
    plt.close(fig)


def fig_fidelity(models):
    """Drift from fp32 predicts |change| but not signed change."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, signed in zip(axes, (False, True)):
        for name, t in models.items():
            m = "o" if name == "3B" else "^"
            y = t["delta"] if signed else t["delta"].abs()
            ax.scatter(t["infid"], y, s=4, alpha=0.4, marker=m, linewidths=0,
                       color="#0072B2" if name == "3B" else "#D55E00",
                       label=f"ESM2-{name}")
        ax.set_xscale("log")
        ax.set_xlabel(r"infidelity  $1-\rho_{\mathrm{fp32}}$")
        ax.axhline(0, color="0.6", lw=0.7, zorder=0)
        leg = ax.legend(frameon=False, markerscale=2.2, loc="upper left")
        for h in leg.legend_handles:
            h.set_alpha(1)
    axes[0].set_ylabel(r"$|\Delta\rho|$ vs experiment")
    axes[0].set_title(r"magnitude: predictable ($r=0.81$ / $0.74$)")
    axes[1].set_ylabel(r"$\Delta\rho$ vs experiment (signed)")
    axes[1].set_title(r"direction: not ($r=-0.58$ / $+0.10$)")
    fig.savefig(os.path.join(OUT, "fig_fidelity.pdf"))
    plt.close(fig)


def fig_pareto():
    """The two workloads want opposite configurations."""
    mp = os.path.join(ROOT, "results", "matrix_3B_5393554.json")
    bulk = {}
    for r in json.load(open(mp))["results"]:
        t = r.get("throughput")
        if t:
            bulk[r["quant"]] = (t["residues_per_s"], t["peak_mem_gb"])
    # int4_wo ran in a separate job; take it from whichever matrix has it.
    for p in ("matrix_3B_5393520.json", "matrix_3B_5393524.json"):
        f = os.path.join(ROOT, "results", p)
        if "int4_wo" not in bulk and os.path.exists(f):
            for r in json.load(open(f))["results"]:
                if r.get("quant") == "int4_wo" and r.get("throughput"):
                    bulk["int4_wo"] = (r["throughput"]["residues_per_s"],
                                       r["throughput"]["peak_mem_gb"])
    s = json.load(open(os.path.join(ROOT, "results", "proteingym",
                                    "summary_3B.json")))["speed_ram"]
    dms = {r["config"]: (r["pos_per_s"], r["peak_med"]) for r in s}

    # Hand-placed label offsets: the interesting configurations cluster tightly
    # in both panels, so the default offset overlaps them into illegibility.
    off_bulk = {"fp32": (7, 2), "bf16": (7, 2), "int8_wo": (-8, 9),
                "int8_dyn": (-46, -11), "fp8_dyn": (5, -12), "int4_wo": (7, 2)}
    off_dms = {"fp32": (7, 2), "bf16": (7, 2), "int8_wo": (5, 7),
               "fp8_dyn": (-10, -13), "int8_dyn": (-14, 8), "int4_wo": (4, -13)}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, (data, xl, ttl, off) in zip(axes, [
            (bulk, "residues/s (compiled)", "Bulk embedding extraction", off_bulk),
            (dms, "masked positions/s (eager)", "DMS variant-effect scoring", off_dms)]):
        for c, (x, y) in data.items():
            ax.scatter([x], [y], s=42, color=COL[c], zorder=3,
                       edgecolors="white", linewidths=0.6)
            ax.annotate(LBL[c], (x, y), textcoords="offset points",
                        xytext=off.get(c, (6, 3)), fontsize=7,
                        family="monospace", color=COL[c])
        ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("peak memory (GB)")
        ax.set_title(ttl)
        ax.margins(x=0.30, y=0.18)
    fig.text(0.5, -0.07, "Lower-right is better in both panels. The winner "
             "differs between them.", ha="center", fontsize=7, color="0.3")
    fig.savefig(os.path.join(OUT, "fig_pareto.pdf"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    models = {"3B": load("3B"), "650M": load("650M")}
    fig_tail(models)
    fig_fidelity(models)
    fig_pareto()
    for f in ("fig_tail.pdf", "fig_fidelity.pdf", "fig_pareto.pdf"):
        p = os.path.join(OUT, f)
        print(f"  {f:20s} {os.path.getsize(p) / 1024:6.1f} KB")


if __name__ == "__main__":
    main()
