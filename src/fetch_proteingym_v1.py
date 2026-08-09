"""Extract the full ProteinGym v1 substitution benchmark into per-assay files.

Why not the v0.1 fetcher (src/fetch_proteingym.py): that one pulls per-assay
CSVs from the ProteinGym_v0.1 HF repo (85 substitution assays) and takes
`target_seq` from the GitHub reference file on main -- which is v1.0, 217
assays. Mixing them works only for the 85 assays present in both, and silently
mismatches names for the rest (v1.0 renamed e.g. A0A140D2T1_ZIKV_Sourisseau_2019
-> ..._Sourisseau_growth_2019). The v1 HF repo ships everything as parquet with
`target_seq` inline, so scores and WT come from one version-consistent source
and there is nothing to cross-reference.

The Marks lab zip (marks.hms.harvard.edu) is unreachable from this cluster --
egress is blocked -- so the HF parquet is the only viable route here.

Every variant's stated WT residue is checked against target_seq before anything
is written. An off-by-one does not raise at scoring time; it quietly scores the
wrong positions and returns a plausible correlation, which is unrecoverable
after the fact. Assays that fail are dropped from the index and named in the
report, never written.

    python src/fetch_proteingym_v1.py --check    # validate, write nothing
    python src/fetch_proteingym_v1.py           # validate and write
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "proteingym_v1")
SHARD = ("https://huggingface.co/datasets/OATML-Markslab/ProteinGym_v1/"
         "resolve/main/DMS_substitutions/train-{i:05d}-of-00005.parquet")
N_SHARDS = 5


def download(raw_dir: str) -> list[str]:
    os.makedirs(raw_dir, exist_ok=True)
    paths = []
    for i in range(N_SHARDS):
        p = os.path.join(raw_dir, f"t{i}.parquet")
        if not os.path.exists(p):
            print(f"[get] shard {i}")
            with urllib.request.urlopen(SHARD.format(i=i), timeout=1800) as r, \
                    open(p, "wb") as fh:
                fh.write(r.read())
        paths.append(p)
    return paths


def collect(paths: list[str]):
    """Rows grouped by assay, one shard at a time.

    `mutated_sequence` is deliberately never read: it is one full-length protein
    string per row and would add ~1 GB of strings for data already implied by
    target_seq + mutant.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    frames, wt = {}, {}
    for p in paths:
        t = pq.read_table(p, columns=["DMS_id", "mutant", "DMS_score", "target_seq"])
        df = t.to_pandas()
        for aid, g in df.groupby("DMS_id", sort=False):
            seqs = g["target_seq"].unique()
            if len(seqs) != 1:
                raise SystemExit(f"{aid}: {len(seqs)} distinct target_seq values")
            if aid in wt and wt[aid] != seqs[0]:
                raise SystemExit(f"{aid}: target_seq differs across shards")
            wt[aid] = seqs[0]
            frames.setdefault(aid, []).append(g[["mutant", "DMS_score"]])
        del df, t
    return {a: pd.concat(v, ignore_index=True) for a, v in frames.items()}, wt


def validate(aid: str, seq: str, mutants) -> tuple[list, int, int]:
    """Return (errors, n_positions, n_multi).

    Checked on unique (position, wt-residue) pairs rather than per row: the
    guarantee is identical -- a row can only fail through the pair it contains --
    and it turns ~2.5M string comparisons into ~L per assay.
    """
    from dms import parse_variant

    pairs, n_multi, errs = set(), 0, []
    for v in mutants:
        try:
            muts = parse_variant(v)
        except ValueError as e:
            errs.append(str(e))
            continue
        if len(muts) > 1:
            n_multi += 1
        for m in muts:
            pairs.add((m.pos, m.wt))
    for pos, aa in sorted(pairs):
        if not (1 <= pos <= len(seq)):
            errs.append(f"{aa}{pos}: outside sequence of length {len(seq)}")
        elif seq[pos - 1] != aa:
            errs.append(f"{aa}{pos}: target_seq has {seq[pos - 1]!r}")
    return errs, len({p for p, _ in pairs}), n_multi


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=ROOT)
    p.add_argument("--check", action="store_true", help="validate, write nothing")
    args = p.parse_args()

    paths = download(os.path.join(args.root, "_raw"))
    print("[read] loading shards")
    frames, wt = collect(paths)
    print(f"[read] {len(frames)} assays, "
          f"{sum(len(d) for d in frames.values()):,} rows")

    assays, failed = [], {}
    for aid in sorted(frames):
        df, seq = frames[aid], wt[aid]
        errs, n_pos, n_multi = validate(aid, seq, df["mutant"].astype(str))
        if errs:
            failed[aid] = errs[:3]
            continue
        assays.append({"id": aid, "wt": seq, "seq_len": len(seq),
                       "csv": f"assays/{aid}.csv", "n_variants": len(df),
                       "n_multi": n_multi, "n_positions": n_pos})

    print(f"\n[check] {len(assays)} assays pass WT validation, {len(failed)} fail")
    for aid, e in failed.items():
        print(f"   FAIL {aid}: {e}")
    if args.check:
        return

    out = os.path.join(args.root, "assays")
    os.makedirs(out, exist_ok=True)
    for a in assays:
        frames[a["id"]].to_csv(os.path.join(args.root, a["csv"]), index=False)

    idx = {"source": "OATML-Markslab/ProteinGym_v1 (HF), DMS_substitutions parquet",
           "n_assays": len(assays), "failed": failed, "assays": assays}
    with open(os.path.join(args.root, "index.json"), "w") as fh:
        json.dump(idx, fh, indent=1)

    tot = sum(a["n_variants"] for a in assays)
    print(f"\n[done] wrote {len(assays)} assays ({tot:,} variants) to {out}")
    for lim in (512, 1022, 2048):
        s = [a for a in assays if a["seq_len"] <= lim]
        print(f"   L<={lim:5d}: {len(s):4d} assays, "
              f"{sum(a['n_variants'] for a in s):9,d} variants, "
              f"{sum(a['n_positions'] for a in s):7,d} masked positions")


if __name__ == "__main__":
    main()
