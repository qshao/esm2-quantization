"""Per-assay comparison of the W8A8 variants: symmetric, asymmetric, SmoothQuant.

Emits results/proteingym/w8a8_remedies.csv so the table in the paper and the
figure are generated from one computation rather than two.

    python src/analyze_w8a8.py --model 3B
"""

from __future__ import annotations

import argparse
import gzip
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANTS = ["int8_dyn", "int8_dyn_asym", "int8_dyn_sq"]


def load(model: str, cfg: str) -> dict:
    d = {}
    with gzip.open(os.path.join(ROOT, "results", "proteingym",
                                f"{model}_{cfg}.jsonl.gz"), "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if "error" not in r:
                d[r["assay"]] = np.array(r["scores"], dtype=float)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3B")
    args = ap.parse_args()
    idx = {a["id"]: a for a in json.load(open(os.path.join(
        ROOT, "data", "proteingym_v1", "index.json")))["assays"]}

    S = {c: load(args.model, c) for c in ["fp32"] + VARIANTS}
    common = sorted(set.intersection(*(set(v) for v in S.values())))
    rows = []
    for a in common:
        e = pd.read_csv(os.path.join(ROOT, "data", "proteingym_v1",
                                     idx[a]["csv"]))["DMS_score"].to_numpy(float)
        m = ~np.isnan(e)
        for c in S:
            m &= ~np.isnan(S[c][a])
        if m.sum() < 3:
            continue
        r0 = spearmanr(S["fp32"][a][m], e[m]).statistic
        row = {"assay": a, "seq_len": idx[a]["seq_len"], "rho_fp32_expt": r0}
        for c in VARIANTS:
            row[f"delta_{c}"] = spearmanr(S[c][a][m], e[m]).statistic - r0
            row[f"fid_{c}"] = spearmanr(S[c][a][m], S["fp32"][a][m]).statistic
        rows.append(row)
    X = pd.DataFrame(rows)
    out = os.path.join(ROOT, "results", "proteingym", "w8a8_remedies.csv")
    X.to_csv(out, index=False)

    print(f"=== W8A8 variants, {args.model}, {len(X)} assays ===")
    print(f"{'config':16s} {'mean d':>9s} {'worst d':>9s} {'|d|>.05':>8s} "
          f"{'fid med':>9s} {'fid min':>9s}")
    for c in VARIANTS:
        print(f"{c:16s} {X['delta_' + c].mean():+9.4f} {X['delta_' + c].min():+9.4f} "
              f"{int((X['delta_' + c].abs() > .05).sum()):8d} "
              f"{X['fid_' + c].median():9.5f} {X['fid_' + c].min():9.4f}")
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
