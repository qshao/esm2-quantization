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
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharex=True, sharey=True)
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
            mk = {"650M": "^", "3B": "o", "15B": "s"}[name]
            cl = {"650M": "#D55E00", "3B": "#0072B2", "15B": "#009E73"}[name]
            y = t["delta"] if signed else t["delta"].abs()
            ax.scatter(t["infid"], y, s=4, alpha=0.35, marker=mk, linewidths=0,
                       color=cl, label=f"ESM2-{name}")
        ax.set_xscale("log")
        ax.set_xlabel(r"infidelity  $1-\rho_{\mathrm{fp32}}$")
        ax.axhline(0, color="0.6", lw=0.7, zorder=0)
        leg = ax.legend(frameon=False, markerscale=2.2, loc="upper left")
        for h in leg.legend_handles:
            h.set_alpha(1)
    for ax in axes:
        ax.margins(y=0.12)        # outliers must not sit on the spine
    # The single worst point in the study; a short steep leader keeps the
    # label off the point cloud without a long line across empty axes.
    # No leader line: matplotlib anchors it to the text bounding box, which
    # renders as an underline before the diagonal. Right-aligned just above the
    # point is unambiguous and cleaner, since nothing else is nearby.
    axes[1].annotate("int8_dyn, UBR5_HUMAN (3B)  ", (0.36, -0.369),
                     textcoords="offset points", xytext=(0, 9), fontsize=6,
                     color="0.25", ha="right")
    axes[0].set_ylabel(r"$|\Delta\rho|$ vs experiment")
    axes[0].set_title(r"magnitude: predictable ($r=0.74/0.81/0.56$)")
    axes[1].set_ylabel(r"$\Delta\rho$ vs experiment (signed)")
    axes[1].set_title(r"direction: not ($r=+0.10/-0.58/-0.40$)")
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


def fig_scale():
    """Three things that only a scale sweep shows."""
    ms = ["650M", "3B", "15B"]
    x = [0, 1, 2]
    spd, rho, tail = {}, {}, {}
    for m in ms:
        j = json.load(open(os.path.join(ROOT, "results", "proteingym",
                                        f"summary_{m}.json")))
        spd[m] = {r["config"]: r["pos_per_s"] for r in j["speed_ram"]}
        t = pd.read_csv(os.path.join(ROOT, "results", "proteingym",
                                     f"summary_{m}_per_assay.csv"))
        rho[m] = t[t.config == "fp32"]["rho_expt"].mean()
        tail[m] = {c: int((t[t.config == c]["rho_fp32"] < 0.99).sum())
                   for c in CFGS}

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

    # (a) speed relative to bf16. bf16 is a legend entry, not an inline label,
    # so no text sits on the line it describes.
    ax = axes[0]
    ax.axhline(1.0, color=COL["bf16"], lw=1.2, ls="--", label="bf16 (baseline)")
    for c in ["fp8_dyn", "int8_wo", "int8_dyn", "int4_wo"]:
        ax.plot(x, [spd[m][c] / spd[m]["bf16"] for m in ms], "o-",
                color=COL[c], lw=1.4, ms=4, label=LBL[c])
    ax.set_ylim(0, 1.85)          # headroom so the legend clears the 1.0 line
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylabel("DMS speed relative to bf16")
    ax.set_title("(a) quantization catches up")
    ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=0.7,
              handlelength=1.3, borderpad=0.1, fontsize=6, labelspacing=0.3)

    # (b) accuracy vs scale. margins keep the value labels off the spines.
    ax = axes[1]
    ax.plot(x, [rho[m] for m in ms], "o-", color="black", lw=1.4, ms=5)
    # Labels go up-and-right of each marker: the curve descends left to right,
    # so that quadrant is the one it never occupies. Centred-above collides
    # with the line for every point after the first.
    for i, m in enumerate(ms):
        ax.annotate(f"{rho[m]:.4f}", (i, rho[m]), textcoords="offset points",
                    xytext=(7, 6), fontsize=6.5, ha="left")
    ax.set_ylim(0.4280, 0.4530)
    ax.set_xlim(-0.30, 2.75)
    ax.set_ylabel(r"fp32 mean $\rho$ vs experiment")
    ax.set_title("(b) accuracy does not follow")

    # (c) the crossover: the safe configs degrade with scale while the most
    # aggressive one improves.
    ax = axes[2]
    # bf16 and int8_wo nearly coincide here (0/0/13 and 0/0/15). Dashing bf16
    # is not enough on its own -- CFGS puts it first, so int8_wo was drawn on
    # top of it and it vanished. It needs the higher zorder too.
    for c in CFGS:
        ax.plot(x, [tail[m][c] for m in ms], marker="o", color=COL[c], lw=1.4,
                ms=4, ls="--" if c == "bf16" else "-", label=LBL[c],
                zorder=5 if c == "bf16" else 3)
    ax.set_ylim(-14, 232)
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylabel("assays with " + r"$\rho_{\mathrm{fp32}}<0.99$")
    ax.set_title("(c) fidelity crossover")
    ax.legend(frameon=False, loc="upper right", fontsize=6, handlelength=1.4,
              borderpad=0.1, labelspacing=0.25)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(["650M", "3B", "15B"])
        ax.set_xlabel("model scale")
    fig.subplots_adjust(wspace=0.48)
    fig.savefig(os.path.join(OUT, "fig_scale.pdf"))
    plt.close(fig)


def fig_dominance():
    """Quantization competes with using a smaller model, not only with fp32."""
    P = pd.DataFrame(json.load(open(os.path.join(
        ROOT, "results", "proteingym", "cross_scale.json")))["pareto"])
    MK = {"650M": "o", "3B": "s", "15B": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, xcol, xl, logx in [(axes[0], "gb", "peak memory (GB)", True),
                               (axes[1], "pos_s", "masked positions/s", True)]:
        for _, r in P.iterrows():
            front = r.dominated_by is None
            ax.scatter(r[xcol], r.rho, s=54 if front else 26,
                       marker=MK[r.scale], color=COL[r.config],
                       edgecolors="black" if front else "none",
                       linewidths=0.9 if front else 0, zorder=4 if front else 3,
                       alpha=1.0 if front else 0.55)
        # Config name only: the caption and footer both say every frontier
        # point is 650M, and the "650M/" prefix made the two top labels
        # overlap. Offsets separate bf16 from int8_wo, which differ by 0.0003
        # in rho and would otherwise print on top of each other.
        off = {"bf16": (7, 5), "int8_wo": (7, -11), "int4_wo": (7, -3)}
        for _, r in P[P.dominated_by.isna()].iterrows():
            ax.annotate(LBL[r.config], (r[xcol], r.rho),
                        textcoords="offset points",
                        xytext=off.get(r.config, (7, -3)), fontsize=6.5,
                        family="monospace", color=COL[r.config])
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel(r"mean $\rho$ vs experiment")
        ax.margins(x=0.34, y=0.14)
    # Scale legend, drawn in neutral grey so it reads as shape not colour.
    h = [plt.Line2D([], [], ls="", marker=MK[m], color="0.35", ms=5,
                    label=f"ESM2-{m}") for m in ("650M", "3B", "15B")]
    h.append(plt.Line2D([], [], ls="", marker="o", mfc="none", mec="black",
                        ms=7, label="Pareto frontier"))
    axes[0].legend(handles=h, frameon=False, loc="lower left", fontsize=6,
                   handletextpad=0.4, labelspacing=0.3)
    fig.text(0.5, -0.06, "Upper-left is better in both panels. Every frontier "
             "point is ESM2-650M.", ha="center", fontsize=7, color="0.3")
    fig.subplots_adjust(wspace=0.32)
    fig.savefig(os.path.join(OUT, "fig_dominance.pdf"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    models = {m: load(m) for m in ("650M", "3B", "15B")}
    fig_tail(models)
    fig_fidelity(models)
    fig_pareto()
    fig_scale()
    fig_dominance()
    for f in ("fig_tail.pdf", "fig_fidelity.pdf", "fig_pareto.pdf",
              "fig_scale.pdf", "fig_dominance.pdf"):
        p = os.path.join(OUT, f)
        print(f"  {f:20s} {os.path.getsize(p) / 1024:6.1f} KB")


if __name__ == "__main__":
    main()
