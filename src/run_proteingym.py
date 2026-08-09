"""Score the whole ProteinGym substitution benchmark under ONE quant config.

run_matrix.py loads the model once per invocation. That is right for a
three-assay run and untenable for 217: load+quantize costs 7-19 s, so looping
assays outside the loader would spend ~1200 loads -- over three GPU-hours spent
re-materialising weights that never changed. Here the loop is inverted: one
config, one load, every assay.

Nothing in this script compares configs. Every per-variant score is written to
disk and all cross-config work (rho vs fp32, paired bootstrap) is deferred to
aggregate_proteingym.py. That is what makes the SLURM array order-free -- no
task waits on the fp32 reference, and one config dying does not invalidate the
other five. It also means a config can be re-run alone without redoing the rest.

    python src/run_proteingym.py --model 3B --config bf16

Resumable: results append to a gzip JSONL, one line per assay, and a restart
skips assays already present. A 217-assay job that dies at hour four resumes
rather than restarting.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import bench
import dms as dms_mod
import models
import quant

INDEX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "proteingym_v1", "index.json")


def load_index(path: str, max_len: int, only=None):
    with open(path) as fh:
        idx = json.load(fh)
    assays = idx["assays"]
    if only:
        want = set(only.split(","))
        assays = [a for a in assays if a["id"] in want]
    keep = [a for a in assays if a["seq_len"] <= max_len]
    skip = [a for a in assays if a["seq_len"] > max_len]
    return keep, skip, idx


def read_variants(a: dict, root: str) -> list[str]:
    """Variants in canonical order: the assay CSV's row order.

    Experimental scores are deliberately NOT read here. They are not needed to
    produce a prediction, and re-reading them per config invites the two halves
    drifting apart; aggregation pairs scores to DMS_score by row position from
    this same file.

    Aggregation aligns configs positionally, so this order is the contract
    between array tasks. It is deterministic (file order), never sorted, and
    never filtered here -- the fetch step already dropped anything that failed
    the WT check, and re-filtering at scoring time would silently desynchronise
    two configs if their pandas versions disagreed.
    """
    import pandas as pd

    df = pd.read_csv(os.path.join(root, a["csv"]))
    return df["mutant"].astype(str).tolist()


def done_assays(path: str) -> set:
    if not os.path.exists(path):
        return set()
    seen = set()
    try:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                try:
                    seen.add(json.loads(line)["assay"])
                except Exception:
                    # A torn final line from a job killed mid-write. Everything
                    # before it is still good; that assay simply gets redone.
                    break
    except OSError:
        return set()
    return seen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="650M")
    p.add_argument("--config", required=True, help="one quant config name")
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    p.add_argument("--index", default=INDEX)
    p.add_argument("--max_len", type=int, default=1022,
                   help="ESM-2 trained at 1024 tokens; +cls +eos leaves 1022. "
                        "Longer assays are skipped and reported, not truncated: "
                        "a truncation window is a scoring-protocol change and "
                        "would confound the quantization comparison.")
    p.add_argument("--only", default=None, help="comma-separated assay ids")
    p.add_argument("--dms_batch", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="first N assays (smoke)")
    p.add_argument("--out_dir", default="results/proteingym")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = os.path.dirname(args.index)
    keep, skip, idx = load_index(args.index, args.max_len, args.only)
    keep.sort(key=lambda a: a["seq_len"])  # cheap assays first: fail fast, and
    #                                        a truncated run still covers breadth
    if args.limit:
        keep = keep[: args.limit]

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out = os.path.join(args.out_dir, f"{args.model}_{args.config}{tag}.jsonl.gz")
    already = done_assays(out)
    todo = [a for a in keep if a["id"] not in already]

    print(f"[setup] model={args.model} config={args.config} device={device}")
    print(f"[setup] {len(keep)} assays within max_len={args.max_len}, "
          f"{len(skip)} skipped as too long, {len(already)} already done "
          f"-> {len(todo)} to score")
    print(f"[setup] out={out}")
    if not todo:
        print("[done] nothing to do")
        return

    cfg = quant.get(args.config)
    ok, why = cfg.check_supported()
    if not ok:
        raise SystemExit(f"[skip] {args.config}: {why}")

    t_load = time.time()
    model, tok, info = models.load(args.model, args.config, attn=args.attn,
                                   device=device, compile_model=False)
    t_load = time.time() - t_load
    print(f"[load] {t_load:.1f}s  weights {info['weight_bytes'] / 1024**3:.2f} GB")

    meta = {"model": args.model, "config": args.config, "device": device,
            "attn": args.attn, "weight_gb": round(info["weight_bytes"] / 1024**3, 3),
            "load_seconds": round(t_load, 1), "max_len": args.max_len,
            "dms_batch": args.dms_batch, "n_assays": len(keep),
            "skipped_too_long": [a["id"] for a in skip],
            # Filename stays {model}_{config} so a resubmission resumes into the
            # same file; the job id therefore has to live in the metadata. A
            # list, because a resumed sweep is legitimately several jobs.
            "slurm_jobs": [os.environ.get("SLURM_JOB_ID", "local")]}
    mp = out.replace(".jsonl.gz", ".meta.json")
    if os.path.exists(mp):
        try:
            prev = json.load(open(mp))
            meta["slurm_jobs"] = prev.get("slurm_jobs", []) + meta["slurm_jobs"]
        except Exception:
            pass
    with open(mp, "w") as fh:
        json.dump(meta, fh, indent=2)

    t_all = time.time()
    for i, a in enumerate(todo, 1):
        variants = read_variants(a, root)
        try:
            # Warm up on the first assay only. bench_dms otherwise scores every
            # assay twice to absorb torch.compile, which does not apply here
            # (uncompiled) and would turn a ~3 GPU-hour sweep into ~6. The first
            # pass still absorbs CUDA context init and cuBLAS heuristic
            # selection, which are per-process, not per-assay.
            bd = bench.bench_dms(model, tok, a["wt"], variants,
                                 batch_size=args.dms_batch, device=device,
                                 method="masked", use_compiled=False,
                                 warmup=(i == 1))
        except Exception as e:
            print(f"[{i}/{len(todo)}] {a['id']:45s} FAIL {type(e).__name__}: {e}")
            with gzip.open(out, "at") as fh:
                fh.write(json.dumps({"assay": a["id"],
                                     "error": f"{type(e).__name__}: {e}"}) + "\n")
            continue

        scores = bd.pop("_scores")
        # Rounded to 6 dp: the score is a log-ratio of order 1-10, so 6 dp is
        # ~1e-6 absolute -- three orders below the smallest quantization drift
        # we can resolve, and it roughly halves the on-disk size at 2.5M values.
        vec = [round(scores[v], 6) if v in scores else None for v in variants]
        rec = {"assay": a["id"], "n": len(variants), "seq_len": a["seq_len"],
               "n_positions": bd.get("n_masked_positions"),
               "seconds": bd.get("seconds"), "peak_mem_gb": bd.get("peak_mem_gb"),
               "variants_per_s": bd.get("variants_per_s"), "scores": vec}
        with gzip.open(out, "at") as fh:
            fh.write(json.dumps(rec) + "\n")

        el = time.time() - t_all
        rate = i / el
        print(f"[{i}/{len(todo)}] {a['id']:45s} L={a['seq_len']:5d} "
              f"n={len(variants):7d} pos={rec['n_positions']:5d} "
              f"{rec['seconds']:7.1f}s peak={rec['peak_mem_gb']:5.2f}GB "
              f"| eta {(len(todo) - i) / rate / 60:.0f} min", flush=True)

    print(f"\n[done] {args.config}: {len(todo)} assays in "
          f"{(time.time() - t_all) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
