# ProteinGym v1 — full substitution benchmark

217 deep-mutational-scanning assays, 2,465,767 variants. This is the whole
substitution benchmark, not the three-assay subset in `../proteingym/`.

Built by `src/fetch_proteingym_v1.py`. Re-run it to rebuild; `_raw/` is cached
so only the first run downloads.

## Source, and why this one

| | `../proteingym/` (v0.1) | here (v1) |
|---|---|---|
| assays | 3 of ~85 | 217 |
| scores | HF `ProteinGym_v0.1` per-assay CSVs | HF `ProteinGym_v1` parquet |
| WT sequence | GitHub `reference_files/DMS_substitutions.csv` | `target_seq` column, inline |

The older path stitches two sources that are **different releases**: the HF
`ProteinGym_v0.1` repo carries ~85 substitution assays, while the reference file
on GitHub `main` is v1.0 with 217. They agree for the 85 assays in both and
diverge elsewhere, including renames — `A0A140D2T1_ZIKV_Sourisseau_2019` became
`A0A140D2T1_ZIKV_Sourisseau_growth_2019`. That is fine for three hand-picked
assays and a trap at benchmark scale.

The v1 parquet carries `target_seq` on every row, so scores and wild-type come
from one release and there is nothing to cross-reference. `fetch_proteingym_v1`
also asserts each assay has exactly one distinct `target_seq`, including across
shard boundaries.

`marks.hms.harvard.edu` (the canonical zip) is unreachable from this cluster —
egress is blocked, `curl` fails instantly rather than timing out — so the HF
parquet is the only viable route here.

## Indexing is validated, not assumed

Every variant's stated wild-type residue is checked against `target_seq` before
anything is written. All 217 assays pass, over all 2,465,767 variants.

This matters more than it sounds. An off-by-one in mutant indexing raises no
error: it scores the wrong positions and returns a plausible correlation. There
is no way to detect it downstream, and every number built on it is wrong.

The check runs on unique `(position, wt-residue)` pairs rather than per row —
identical guarantee, since a row can only fail through a pair it contains, and
it turns ~2.5M string comparisons into ~L per assay.

## Layout

```
index.json      one entry per assay: id, wt, seq_len, csv, n_variants,
                n_multi, n_positions.  The scoring driver reads only this.
assays/<id>.csv mutant, DMS_score — row order is the canonical variant order
                that array tasks align on
_raw/           the five downloaded parquet shards (~110 MB, cached)
```

`mutated_sequence` is deliberately never read: it is one full-length protein
string per row, ~1 GB of strings for data already implied by `target_seq` plus
`mutant`.

## Coverage

| filter | assays | variants | masked positions |
|---|---|---|---|
| L ≤ 512 | 162 | 2,054,997 | 25,265 |
| **L ≤ 1022** (used) | **201** | **2,413,913** | **40,775** |
| all | 217 | 2,465,767 | — |

Masked positions is the number that sets runtime: masked-marginals scoring is
one forward per mutated position, so 40,775 is the real size of the job and
2.4M variants is nearly free on top of it.

The 16 excluded assays exceed ESM-2's 1024-token context (1022 + `<cls>` +
`<eos>`). They are **skipped and named** in each run's `.meta.json`, never
truncated: a truncation window is a change to the scoring protocol, and mixing
protocols across configs would confound the very comparison being made.
