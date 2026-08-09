"""Sequence loading and batching for ESM-2.

Length-bucketed batching is here because on real protein sets it is usually a
larger throughput win than any quantization config: a naive "loop over FASTA,
pad to the longest in the batch" pipeline commonly wastes 50-80% of its FLOPs on
padding. Batches are formed under a *token budget* rather than a fixed sequence
count, so short-sequence batches get large and long-sequence batches stay within
memory.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Seq:
    id: str
    seq: str

    def __len__(self) -> int:
        return len(self.seq)


def read_fasta(path: str, max_len: int | None = None) -> list[Seq]:
    out, sid, buf = [], None, []

    def flush():
        if sid is not None:
            s = "".join(buf)
            out.append(Seq(sid, s[:max_len] if max_len else s))

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                sid, buf = line[1:].split()[0], []
            else:
                buf.append(line)
    flush()
    return out


def synthetic(n: int, lengths: list[int], seed: int = 0) -> list[Seq]:
    """Deterministic synthetic sequences, for benchmarking without real data."""
    rng = random.Random(seed)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    out = []
    for i in range(n):
        L = lengths[i % len(lengths)]
        out.append(Seq(f"synth_{i}_L{L}", "".join(rng.choice(aa) for _ in range(L))))
    return out


def bucket_batches(
    seqs: list[Seq],
    token_budget: int = 16384,
    max_batch: int = 256,
) -> list[list[Seq]]:
    """Sort by length, then greedily fill batches under a padded-token budget.

    Cost of a batch is (batch size) x (longest member), since that is what
    actually gets computed after padding.
    """
    ordered = sorted(seqs, key=len)
    batches, cur, cur_max = [], [], 0
    for s in ordered:
        nmax = max(cur_max, len(s))
        if cur and ((len(cur) + 1) * nmax > token_budget or len(cur) >= max_batch):
            batches.append(cur)
            cur, cur_max = [s], len(s)
        else:
            cur, cur_max = cur + [s], nmax
    if cur:
        batches.append(cur)
    return batches


def naive_batches(seqs: list[Seq], batch_size: int = 8) -> list[list[Seq]]:
    """Unsorted fixed-size batching -- the baseline that bucketing improves on."""
    return [seqs[i : i + batch_size] for i in range(0, len(seqs), batch_size)]


def padding_waste(batches: list[list[Seq]]) -> float:
    """Fraction of padded tokens that carry no information."""
    real = sum(len(s) for b in batches for s in b)
    padded = sum(len(b) * max(len(s) for s in b) for b in batches)
    return 1.0 - real / padded if padded else 0.0
