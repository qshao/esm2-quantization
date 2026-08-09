"""Variant-effect (DMS) scoring for ESM-2.

Two scorers, deliberately both implemented so the accuracy/cost tradeoff is
measurable rather than assumed:

  masked_marginals : one forward per *mutated position*. The standard ESM
                     protocol and the more accurate one. Cost is O(n_positions),
                     which is what actually dominates DMS runtime -- not the
                     per-forward cost that quantization addresses.

  wt_marginals     : a single forward on the unmasked sequence. ~L times
                     cheaper, at some loss of rank correlation.

Two throughput points matter more than any quantization choice here:
  1. Only positions that actually carry a variant are masked (assays commonly
     cover a fraction of the sequence).
  2. The masked copies are independent, so they are batched. Reference
     implementations typically run them one at a time.

Token indexing: the ESM tokenizer prepends <cls>, so a 1-based protein position
p maps to token index p.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


@dataclass
class Mutation:
    wt: str
    pos: int  # 1-based position in the protein
    mt: str

    @classmethod
    def parse(cls, s: str) -> "Mutation":
        m = MUT_RE.match(s.strip())
        if not m:
            raise ValueError(f"cannot parse mutation {s!r}; expected e.g. 'A42G'")
        return cls(m.group(1), int(m.group(2)), m.group(3))

    def __str__(self) -> str:
        return f"{self.wt}{self.pos}{self.mt}"


def parse_variant(s: str) -> list[Mutation]:
    """Parse a variant string; ':'-separated for multi-mutants."""
    return [Mutation.parse(p) for p in s.split(":")]


def _check_wt(seq: str, muts: list[Mutation]) -> None:
    for m in muts:
        if not (1 <= m.pos <= len(seq)):
            raise ValueError(f"{m}: position outside sequence of length {len(seq)}")
        if seq[m.pos - 1] != m.wt:
            raise ValueError(
                f"{m}: sequence has {seq[m.pos - 1]!r} at position {m.pos}, "
                f"variant claims {m.wt!r} -- indexing mismatch"
            )


@torch.no_grad()
def masked_marginals(
    model,
    tok,
    seq: str,
    variants: list[str],
    batch_size: int = 16,
    device: str = "cuda",
    validate_wt: bool = True,
) -> dict[str, float]:
    """Score variants by masked marginals.

    Returns {variant_string: score}, score = sum over constituent mutations of
    log p(mt | masked context) - log p(wt | masked context).
    """
    parsed = {v: parse_variant(v) for v in variants}
    if validate_wt:
        for muts in parsed.values():
            _check_wt(seq, muts)

    # Only mask positions that some variant actually touches.
    positions = sorted({m.pos for muts in parsed.values() for m in muts})

    enc = tok(seq, return_tensors="pt", add_special_tokens=True)
    base_ids = enc["input_ids"][0].to(device)
    attn = enc["attention_mask"][0].to(device)
    mask_id = tok.mask_token_id

    # log p(. | context with position p masked), for each p, at token index p.
    logprobs: dict[int, torch.Tensor] = {}

    for i in range(0, len(positions), batch_size):
        chunk = positions[i : i + batch_size]
        ids = base_ids.unsqueeze(0).repeat(len(chunk), 1)
        for row, p in enumerate(chunk):
            ids[row, p] = mask_id
        out = model(
            input_ids=ids,
            attention_mask=attn.unsqueeze(0).expand(len(chunk), -1),
        ).logits
        # Take the row-specific masked position, in fp32 for a stable log_softmax.
        for row, p in enumerate(chunk):
            logprobs[p] = torch.log_softmax(out[row, p].float(), dim=-1).cpu()

    return _assemble(parsed, tok, logprobs)


@torch.no_grad()
def wt_marginals(
    model,
    tok,
    seq: str,
    variants: list[str],
    device: str = "cuda",
    validate_wt: bool = True,
) -> dict[str, float]:
    """Score variants from a single unmasked forward pass (~L x cheaper)."""
    parsed = {v: parse_variant(v) for v in variants}
    if validate_wt:
        for muts in parsed.values():
            _check_wt(seq, muts)

    enc = tok(seq, return_tensors="pt", add_special_tokens=True).to(device)
    out = model(**enc).logits[0]
    lp = torch.log_softmax(out.float(), dim=-1).cpu()
    positions = sorted({m.pos for muts in parsed.values() for m in muts})
    logprobs = {p: lp[p] for p in positions}
    return _assemble(parsed, tok, logprobs)


def _assemble(parsed, tok, logprobs) -> dict[str, float]:
    """Sum per-mutation log-ratios into a per-variant score.

    NOTE: this log-ratio is a difference of two near-equal numbers, so
    quantization error here does not cancel -- it is amplified relative to the
    signal. This is the metric most worth watching across quant configs.
    """
    scores = {}
    for v, muts in parsed.items():
        total = 0.0
        for m in muts:
            lp = logprobs[m.pos]
            total += float(lp[tok.convert_tokens_to_ids(m.mt)]
                           - lp[tok.convert_tokens_to_ids(m.wt)])
        scores[v] = total
    return scores


def spearman(a: list[float], b: list[float]) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)
