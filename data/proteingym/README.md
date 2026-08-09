# ProteinGym assay data

Downloaded from the ProteinGym v0.1 HF dataset; metadata (including `target_seq`,
the wild-type used for indexing) from the ProteinGym repo's reference file.

    DMS_substitutions.csv    reference metadata for all assays
    <ASSAY>.csv              mutant, DMS_score, DMS_score_bin
    <ASSAY>.wt.txt           wild-type sequence for that assay

Every variant's stated WT residue was checked against `target_seq` before use --
all 19,557 singles across the three assays matched, confirming ProteinGym's
1-based positions line up with `dms.parse_variant`. That check is worth keeping:
an off-by-one here does not crash, it silently produces plausible scores.

Regenerate with `python src/fetch_proteingym.py <ASSAY> [<ASSAY> ...]`.
