"""Fidelity metrics: how far does a quantized config drift from the fp32 model?

Three metrics, deliberately ordered by how sensitive they are, because a config
that looks perfect on the first can still be unusable on the third:

  1. embedding cosine  -- mean-pooled representations. Very forgiving: error
     averages out over the length of the sequence.
  2. logit KL          -- per-residue distribution drift. Moderately sensitive.
  3. DMS score drift   -- log p(mut) - log p(wt). MOST sensitive, because it is
     a difference of two near-equal numbers, so quantization noise does not
     cancel and gets amplified relative to the signal.

Checking only (1) and declaring a config good is the standard way to ship a
broken variant-effect pipeline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def embed_and_logits(model, tok, seqs, device="cuda", max_len=None):
    """Return (mean-pooled embeddings [B,H], per-residue logits list).

    Pooling excludes <cls>, <eos> and padding -- including them would dilute the
    very differences we are trying to measure.
    """
    texts = [s.seq if hasattr(s, "seq") else s for s in seqs]
    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=max_len is not None,
        max_length=max_len,
        add_special_tokens=True,
    ).to(device)

    # torch.compile's OptimizedModule does not reliably surface hidden_states,
    # so fidelity is measured through the underlying module. The quantization is
    # identical either way -- compile only fuses kernels -- and this path is not
    # on the timed hot loop.
    base = getattr(model, "_orig_mod", model)
    out = base(**enc, output_hidden_states=True)
    h = out.hidden_states[-1].float()          # [B, T, H]
    logits = out.logits.float()                # [B, T, V]

    # Mask out special tokens as well as padding.
    special = torch.zeros_like(enc["attention_mask"], dtype=torch.bool)
    for tid in (tok.cls_token_id, tok.eos_token_id, tok.pad_token_id):
        if tid is not None:
            special |= enc["input_ids"] == tid
    keep = enc["attention_mask"].bool() & ~special

    emb = (h * keep.unsqueeze(-1)).sum(1) / keep.sum(1, keepdim=True).clamp(min=1)

    per_seq_logits = [logits[i][keep[i]] for i in range(logits.shape[0])]
    return emb.cpu(), [t.cpu() for t in per_seq_logits]


def embedding_metrics(ref_emb: torch.Tensor, test_emb: torch.Tensor) -> dict:
    cos = F.cosine_similarity(ref_emb, test_emb, dim=-1)
    rel = (test_emb - ref_emb).norm(dim=-1) / ref_emb.norm(dim=-1).clamp(min=1e-9)
    return {
        "emb_cos_mean": float(cos.mean()),
        "emb_cos_min": float(cos.min()),
        "emb_rel_l2_mean": float(rel.mean()),
    }


def logit_metrics(ref_logits: list, test_logits: list) -> dict:
    kls, maes = [], []
    for r, t in zip(ref_logits, test_logits):
        n = min(r.shape[0], t.shape[0])
        r, t = r[:n], t[:n]
        lr = F.log_softmax(r, dim=-1)
        lt = F.log_softmax(t, dim=-1)
        # KL(ref || test), averaged over residues.
        kls.append(float((lr.exp() * (lr - lt)).sum(-1).mean()))
        maes.append(float((r - t).abs().mean()))
    return {
        "logit_kl_mean": sum(kls) / len(kls),
        "logit_kl_max": max(kls),
        "logit_mae_mean": sum(maes) / len(maes),
    }


def dms_metrics(ref_scores: dict, test_scores: dict, experimental: dict | None = None) -> dict:
    """Drift of variant-effect scores, plus ground-truth correlation if available."""
    from scipy.stats import spearmanr, pearsonr

    keys = sorted(set(ref_scores) & set(test_scores))
    r = [ref_scores[k] for k in keys]
    t = [test_scores[k] for k in keys]

    out = {
        "dms_n": len(keys),
        # How well the quantized model preserves the fp32 model's *ranking*.
        # This is the number that decides whether a config is safe for DMS.
        "dms_spearman_vs_fp32": float(spearmanr(r, t).statistic),
        "dms_pearson_vs_fp32": float(pearsonr(r, t).statistic),
        "dms_max_abs_drift": max(abs(a - b) for a, b in zip(r, t)) if keys else 0.0,
    }
    if experimental:
        ek = [k for k in keys if k in experimental]
        if len(ek) > 2:
            e = [experimental[k] for k in ek]
            out["dms_spearman_vs_expt"] = float(
                spearmanr([test_scores[k] for k in ek], e).statistic
            )
            out["dms_spearman_ref_vs_expt"] = float(
                spearmanr([ref_scores[k] for k in ek], e).statistic
            )
    return out
