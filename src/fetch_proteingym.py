"""Download ProteinGym substitution assays and their wild-type sequences.

    python src/fetch_proteingym.py BLAT_ECOLX_Stiffler_2015 IF1_ECOLI_Kelsic_2016
    python src/fetch_proteingym.py --list          # browse what is available

Two sources, because neither has both halves:
  * the assay CSVs (mutant, DMS_score) live in the HF dataset
  * `target_seq` -- the WT the mutant positions are indexed against -- lives in
    the repo's reference file, NOT in the assay CSV

Every variant's stated WT residue is checked against `target_seq` before the
files are written. An off-by-one in that indexing does not raise; it silently
scores the wrong positions and yields a plausible-looking correlation, so the
check is the point of this script rather than a nicety.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REF_URL = ("https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/"
           "reference_files/DMS_substitutions.csv")
DMS_URL = ("https://huggingface.co/datasets/OATML-Markslab/ProteinGym_v0.1/"
           "resolve/main/ProteinGym_substitutions/{assay}.csv")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "proteingym")


def _get(url: str, timeout: int = 300) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def load_reference(out_dir: str):
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "DMS_substitutions.csv")
    if not os.path.exists(path):
        with open(path, "wb") as fh:
            fh.write(_get(REF_URL))
    return pd.read_csv(path).set_index("DMS_id")


def fetch(assay: str, ref, out_dir: str) -> None:
    import pandas as pd

    from dms import parse_variant, _check_wt

    if assay not in ref.index:
        raise SystemExit(f"unknown assay {assay!r}; try --list")

    wt = str(ref.loc[assay, "target_seq"]).strip().upper()
    raw = _get(DMS_URL.format(assay=assay))
    df = pd.read_csv(io.BytesIO(raw))

    singles = df[~df["mutant"].astype(str).str.contains(":")]
    bad = []
    for v in singles["mutant"].astype(str):
        try:
            _check_wt(wt, parse_variant(v))
        except ValueError as e:
            bad.append(str(e))
    if bad:
        raise SystemExit(
            f"{assay}: {len(bad)} variants disagree with target_seq, refusing to "
            f"write. First: {bad[0]}"
        )

    with open(os.path.join(out_dir, f"{assay}.csv"), "wb") as fh:
        fh.write(raw)
    with open(os.path.join(out_dir, f"{assay}.wt.txt"), "w") as fh:
        fh.write(wt)

    pos = sorted({m.pos for v in singles["mutant"].astype(str)
                  for m in parse_variant(v)})
    print(f"{assay}: L={len(wt)} variants={len(df)} singles={len(singles)} "
          f"positions={len(pos)} coverage={len(pos)/len(wt):.0%} | WT check OK")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("assays", nargs="*")
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--list", action="store_true",
                   help="print assays with length and single-mutant count")
    p.add_argument("--max_len", type=int, default=1000,
                   help="--list only; ESM-2 was trained at 1024 tokens")
    args = p.parse_args()

    ref = load_reference(args.out_dir)
    if args.list:
        cols = ["seq_len", "DMS_number_single_mutants", "selection_assay"]
        d = ref[ref.seq_len <= args.max_len]
        print(d[cols].sort_values("seq_len").to_string())
        return
    if not args.assays:
        raise SystemExit("give at least one assay id, or --list")
    for a in args.assays:
        fetch(a, ref, args.out_dir)


if __name__ == "__main__":
    main()
