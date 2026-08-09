"""Run the quantization config matrix and report speed + fidelity together.

Protocol: the fp32 model is loaded first and its outputs cached as ground truth,
then every other config is measured against it. Speed numbers without the
fidelity numbers beside them are how a broken config gets adopted, so this
script always emits both.

  python src/run_matrix.py --model 650M --configs bf16,int8_wo,int8_dyn,fp8_dyn
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import bench
import data
import dms as dms_mod
import models
import quant
import validate


def build_probe_set(args):
    """Sequences used for fidelity checks and for the throughput benchmark."""
    if args.fasta:
        seqs = data.read_fasta(args.fasta, max_len=args.max_len)[: args.n_seqs]
    else:
        lengths = [int(x) for x in args.lengths.split(",")]
        seqs = data.synthetic(args.n_seqs, lengths, seed=0)
    return seqs


def build_dms_case(args, seqs):
    """A DMS case: a wild-type sequence plus variants to score."""
    if args.dms_csv:
        import pandas as pd

        df = pd.read_csv(args.dms_csv)
        wt = open(args.dms_wt).read().strip() if args.dms_wt else None
        if wt is None:
            raise SystemExit("--dms_csv requires --dms_wt (file with the WT sequence)")
        variants = df[args.dms_variant_col].astype(str).tolist()[: args.n_variants]
        expt = None
        if args.dms_score_col and args.dms_score_col in df:
            expt = dict(
                zip(
                    df[args.dms_variant_col].astype(str),
                    df[args.dms_score_col].astype(float),
                )
            )
        return wt, variants, expt

    # Synthetic fallback: a deep mutational scan over the first probe sequence,
    # so the harness is runnable before real assay data is wired in.
    wt = seqs[0].seq[: args.dms_len]
    aa = "ACDEFGHIKLMNPQRSTVWY"
    variants = []
    for pos in range(1, len(wt) + 1):
        for mt in aa:
            if mt != wt[pos - 1]:
                variants.append(f"{wt[pos - 1]}{pos}{mt}")
    return wt, variants[: args.n_variants], None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="650M", help="650M | 3B | 150M | HF id")
    p.add_argument("--configs", default="fp32,bf16,int8_wo,int8_dyn")
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile_mode", default="max-autotune-no-cudagraphs",
                   help="default|reduce-overhead|max-autotune; "
                        "max-autotune is what makes W8A8 pay off")
    p.add_argument("--fasta", default=None)
    p.add_argument("--n_seqs", type=int, default=32)
    p.add_argument("--lengths", default="256,512,1024")
    p.add_argument("--max_len", type=int, default=None)
    p.add_argument("--token_budget", type=int, default=16384)
    p.add_argument("--fidelity_n", type=int, default=8,
                   help="sequences used for embedding/logit fidelity")
    p.add_argument("--skip_dms", action="store_true")
    p.add_argument("--skip_throughput", action="store_true",
                   help="DMS-only run. DMS scoring is deliberately uncompiled "
                        "(see bench.bench_dms), so skipping the throughput bench "
                        "also skips paying compile cost for numbers this run "
                        "does not use.")
    p.add_argument("--dms_csv", default=None)
    p.add_argument("--dms_wt", default=None)
    p.add_argument("--dms_variant_col", default="mutant")
    p.add_argument("--dms_score_col", default="DMS_score")
    p.add_argument("--dms_len", type=int, default=40)
    p.add_argument("--n_variants", type=int, default=200)
    p.add_argument("--dms_batch", type=int, default=16)
    p.add_argument("--dms_compiled", action="store_true",
                   help="Score DMS through the compiled module (needs --compile). "
                        "Tests whether autotune amortises over a real assay's "
                        "many masked positions; the default uncompiled path is "
                        "the right choice for short/few-position scans.")
    p.add_argument("--out", default="results/matrix.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] no CUDA device -- speed numbers will be meaningless")

    seqs = build_probe_set(args)
    batches = data.bucket_batches(seqs, token_budget=args.token_budget)
    fid_seqs = seqs[: args.fidelity_n]

    dms_wt, dms_variants, dms_expt = (None, None, None)
    if not args.skip_dms:
        dms_wt, dms_variants, dms_expt = build_dms_case(args, seqs)

    print(f"[setup] device={device} model={args.model} attn={args.attn}")
    print(f"[setup] {len(seqs)} sequences -> {len(batches)} bucketed batches, "
          f"padding waste {data.padding_waste(batches):.1%} "
          f"(naive fixed-size would be "
          f"{data.padding_waste(data.naive_batches(seqs, 8)):.1%})")
    if dms_variants:
        n_pos = len({m.pos for v in dms_variants for m in dms_mod.parse_variant(v)})
        print(f"[setup] DMS: {len(dms_variants)} variants over {n_pos} masked "
              f"positions (WT length {len(dms_wt)})")

    ref = {}
    results = []
    # Per-variant scores, kept so the ground-truth correlations can be tested
    # for significance afterwards. A single rho per config cannot distinguish
    # "quantization changed the answer" from "4996 variants resampled", and at
    # 3B the swings reach +-0.05 in inconsistent directions -- exactly the range
    # where that distinction decides the conclusion.
    all_dms_scores = {}

    for name in args.configs.split(","):
        name = name.strip()
        cfg = quant.get(name)
        ok, why = cfg.check_supported()
        if not ok:
            print(f"\n[skip] {name}: {why}")
            results.append({"quant": name, "skipped": why})
            continue

        print(f"\n=== {name} ===")
        t0 = time.time()
        try:
            model, tok, info = models.load(
                args.model, name, attn=args.attn,
                device=device, compile_model=args.compile,
                compile_mode=args.compile_mode,
            )
        except Exception as e:
            print(f"[fail] load: {type(e).__name__}: {e}")
            results.append({"quant": name, "error": f"{type(e).__name__}: {e}"})
            continue

        rec = dict(info)
        rec["weight_gb"] = round(info["weight_bytes"] / 1024**3, 3)

        try:
            # --- fidelity ---
            emb, logits = validate.embed_and_logits(
                model, tok, fid_seqs, device=device, max_len=args.max_len
            )
            dms_scores = None
            if dms_variants:
                bd = bench.bench_dms(
                    model, tok, dms_wt, dms_variants,
                    batch_size=args.dms_batch, device=device, method="masked",
                    use_compiled=args.dms_compiled,
                )
                dms_scores = bd.pop("_scores")
                rec["dms_bench"] = bd
                all_dms_scores[name] = dms_scores

            if name == "fp32":
                ref = {"emb": emb, "logits": logits, "dms": dms_scores}
                rec["reference"] = True
                # fp32 skips dms_metrics (it IS the reference), but its own
                # ground-truth correlation is the ceiling every quantized config
                # is judged against -- without it "rho_expt 0.42" is unreadable.
                if dms_scores and dms_expt:
                    from scipy.stats import spearmanr

                    ek = [k for k in dms_scores if k in dms_expt]
                    if len(ek) > 2:
                        rec["dms_spearman_ref_vs_expt"] = float(
                            spearmanr([dms_scores[k] for k in ek],
                                      [dms_expt[k] for k in ek]).statistic
                        )
                        rec["dms_n_expt"] = len(ek)
            elif ref:
                rec.update(validate.embedding_metrics(ref["emb"], emb))
                rec.update(validate.logit_metrics(ref["logits"], logits))
                if dms_scores and ref.get("dms"):
                    rec.update(validate.dms_metrics(ref["dms"], dms_scores, dms_expt))
            else:
                rec["fidelity"] = "no fp32 reference in this run"

            # --- speed ---
            if not args.skip_throughput:
                rec["throughput"] = bench.bench_forward(
                    model, tok, batches, device=device
                )

        except Exception as e:
            import traceback

            traceback.print_exc()
            rec["error"] = f"{type(e).__name__}: {e}"

        rec["total_seconds"] = round(time.time() - t0, 1)
        results.append(rec)
        _print_row(rec)
        bench.free(model)
        # Drop the caller's reference too. bench.free() can only delete its own
        # local name, so without this the model stays alive through the NEXT
        # config's load and lands inside its peak-memory measurement.
        model = None

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if all_dms_scores:
        # Aligned on a single variant order so downstream analysis is a plain
        # column-wise comparison; experimental scores ride along as a column.
        variants = sorted(set().union(*(s.keys() for s in all_dms_scores.values())))
        sp = args.out.replace(".json", "_scores.json")
        with open(sp, "w") as fh:
            json.dump(
                {
                    "variants": variants,
                    "experimental": [dms_expt.get(v) if dms_expt else None
                                     for v in variants],
                    "scores": {k: [s.get(v) for v in variants]
                               for k, s in all_dms_scores.items()},
                },
                fh,
            )
        print(f"[done] wrote {sp} ({len(variants)} variants x "
              f"{len(all_dms_scores)} configs)")

    with open(args.out, "w") as fh:
        json.dump(
            {"args": vars(args), "device": device, "results": results}, fh, indent=2
        )
    print(f"\n[done] wrote {args.out}")
    if not args.skip_throughput:
        _summary(results)
    if not args.skip_dms:
        _dms_summary(results)


def _print_row(rec):
    t = rec.get("throughput", {})
    print(f"  weights {rec.get('weight_gb', '?')} GB | "
          f"peak {t.get('peak_mem_gb', '?')} GB | "
          f"{t.get('residues_per_s', '?')} res/s")
    if "emb_cos_mean" in rec:
        print(f"  emb cos {rec['emb_cos_mean']:.6f} | "
              f"logit KL {rec.get('logit_kl_mean', float('nan')):.3e}")
    if "dms_spearman_vs_fp32" in rec:
        print(f"  DMS spearman vs fp32 {rec['dms_spearman_vs_fp32']:.4f} | "
              f"max drift {rec.get('dms_max_abs_drift', 0):.3f}")


def _dms_summary(results):
    """The three axes for the DMS workload: accuracy, speed, RAM.

    Two accuracy columns, and they answer different questions:

      rho_fp32 -- does quantization change the fp32 model's variant ranking?
                  A fidelity question, no ground truth involved.
      rho_expt -- does the config still predict the *experiment*? This is the
                  one that decides whether a config is usable, and it can only
                  be computed against a real assay.

    They can disagree: fp32 itself only reaches rho_expt ~0.4-0.5 on most
    assays, so quantization noise well below that floor is invisible in
    practice even when rho_fp32 shows it clearly.
    """
    print("\n" + "=" * 78)
    print("DMS workload: accuracy / speed / RAM")
    print(f"{'config':10s} {'wGB':>6s} {'peakGB':>7s} {'var/s':>9s} "
          f"{'sec':>7s} {'rho_fp32':>9s} {'rho_expt':>9s}")
    print("-" * 78)
    for r in results:
        if "skipped" in r:
            print(f"{r['quant']:10s} {r['skipped'][:60]}")
            continue
        b = r.get("dms_bench", {})
        if not b:
            print(f"{r['quant']:10s} {r.get('error', 'no dms result')[:60]}")
            continue
        if r.get("reference"):
            rho_fp32 = "(ref)"
            # fp32 is the reference, so its ground-truth correlation is stored
            # under the *_ref_vs_expt key by validate.dms_metrics.
            rho_expt = r.get("dms_spearman_ref_vs_expt", float("nan"))
        else:
            rho_fp32 = f"{r.get('dms_spearman_vs_fp32', float('nan')):.4f}"
            rho_expt = r.get("dms_spearman_vs_expt", float("nan"))
        print(f"{r['quant']:10s} {r.get('weight_gb', 0):6.2f} "
              f"{b.get('peak_mem_gb', 0):7.2f} {b.get('variants_per_s', 0):9.1f} "
              f"{b.get('seconds', 0):7.2f} {rho_fp32:>9s} {rho_expt:9.4f}")
    print("=" * 78)


def _summary(results):
    print("\n" + "=" * 78)
    hdr = f"{'config':10s} {'wGB':>6s} {'peakGB':>7s} {'res/s':>10s} {'speedup':>8s} " \
          f"{'embcos':>9s} {'DMSrho':>8s}"
    print(hdr)
    print("-" * 78)
    base = None
    for r in results:
        if r.get("quant") == "bf16" or (base is None and "throughput" in r):
            base = r.get("throughput", {}).get("residues_per_s")
    for r in results:
        if "skipped" in r or ("error" in r and "throughput" not in r):
            print(f"{r['quant']:10s} {r.get('skipped', r.get('error', ''))[:60]}")
            continue
        t = r.get("throughput", {})
        rs = t.get("residues_per_s", 0)
        sp = f"{rs / base:.2f}x" if base and rs else "-"
        print(f"{r['quant']:10s} {r.get('weight_gb', 0):6.2f} "
              f"{t.get('peak_mem_gb', 0):7.2f} {rs:10.1f} {sp:>8s} "
              f"{r.get('emb_cos_mean', float('nan')):9.6f} "
              f"{r.get('dms_spearman_vs_fp32', float('nan')):8.4f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
