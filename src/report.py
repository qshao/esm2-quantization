"""Render matrix results as a three-axis table: RAM, speed, accuracy.

  python src/report.py results/matrix_3B_*.json

RAM is reported as two numbers because they answer different questions:
  weights  -- what the model costs to hold; decides how many replicas fit
  peak     -- weights + transient activations; decides whether a job OOMs.
              Past roughly L=2000 the O(L^2) attention term dominates this and
              quantization stops helping at all.

Accuracy is reported as three tiers rather than one number, because their
sensitivity differs by orders of magnitude and the workloads care about
different ones:
  emb_cos  -- embedding extraction
  logit_KL -- per-residue tasks
  DMS_rho  -- variant effect scoring; the strictest, and the one that decides
              whether a config is safe for log-ratio scoring
"""

from __future__ import annotations

import glob
import json
import sys


def load(paths: list[str]) -> list[tuple[str, dict]]:
    out = []
    for pat in paths:
        for p in sorted(glob.glob(pat)):
            with open(p) as fh:
                out.append((p, json.load(fh)))
    return out


def fmt(v, spec=".3f", missing="-"):
    if v is None:
        return missing
    try:
        if v != v:  # NaN
            return missing
        return format(v, spec)
    except (TypeError, ValueError):
        return missing


def render(path: str, doc: dict) -> None:
    args = doc.get("args", {})
    rows = doc.get("results", [])
    model = args.get("model", "?")
    compiled = args.get("compile")
    mode = args.get("compile_mode") if compiled else "eager"

    print(f"\n{'=' * 96}")
    print(f"{path}")
    print(f"model ESM2-{model} | device {doc.get('device')} | "
          f"attn {args.get('attn')} | compile: {mode}")
    print("=" * 96)

    # bf16 is the practical baseline -- it is what you would actually run
    # without quantization, so speedups are quoted against it, not against fp32.
    base = None
    for r in rows:
        if r.get("quant") == "bf16":
            base = (r.get("throughput") or {}).get("residues_per_s")

    print(f"{'config':10s} | {'RAM: wt':>8s} {'peak':>7s} | "
          f"{'res/s':>9s} {'vs bf16':>8s} | "
          f"{'emb_cos':>9s} {'logitKL':>9s} {'DMS_rho':>8s}")
    print("-" * 96)

    for r in rows:
        q = r.get("quant", "?")
        if "skipped" in r:
            print(f"{q:10s} | skipped: {r['skipped']}")
            continue
        if "error" in r and "throughput" not in r:
            print(f"{q:10s} | ERROR: {str(r['error'])[:70]}")
            continue

        t = r.get("throughput") or {}
        rs = t.get("residues_per_s")
        sp = f"{rs / base:.2f}x" if (base and rs) else "-"
        ref = " (ref)" if r.get("reference") else ""

        print(f"{q:10s} | {fmt(r.get('weight_gb'), '8.2f')} "
              f"{fmt(t.get('peak_mem_gb'), '7.2f')} | "
              f"{fmt(rs, '9.0f')} {sp:>8s} | "
              f"{fmt(r.get('emb_cos_mean'), '9.6f')} "
              f"{fmt(r.get('logit_kl_mean'), '9.2e')} "
              f"{fmt(r.get('dms_spearman_vs_fp32'), '8.4f')}{ref}")

    # DMS cost is dominated by the number of masked forwards, not per-forward
    # cost, so surface it separately -- it is the real lever on that workload.
    dms = [(r.get("quant"), r.get("dms_bench")) for r in rows if r.get("dms_bench")]
    if dms:
        print(f"\n{'DMS scoring':10s} | {'variants/s':>11s} {'seconds':>9s} "
              f"{'masked positions':>18s}")
        print("-" * 55)
        for q, b in dms:
            print(f"{q:10s} | {b.get('variants_per_s', 0):11.1f} "
                  f"{b.get('seconds', 0):9.2f} {b.get('n_masked_positions', 0):18d}")

    print("\nAccuracy is vs the fp32 model. emb_cos forgives the most, DMS_rho the least:")
    print("  a config can be fine for embeddings and unusable for variant scoring.")


def main():
    paths = sys.argv[1:] or ["results/matrix_*.json"]
    docs = load(paths)
    if not docs:
        print("no result files matched", paths)
        return
    for p, d in docs:
        render(p, d)


if __name__ == "__main__":
    main()
