"""Throughput / latency / memory measurement.

Measures what the workloads actually cost, not FLOP counts:
  * embedding extraction -- residues per second over length-bucketed batches
  * DMS scoring          -- variants per second, which is dominated by the
                            number of masked forwards, not the per-forward cost

Peak memory is read from torch.cuda.max_memory_allocated, reset per measurement,
so the reported number includes transient activations (the O(L^2) attention
term) and not just the weights.
"""

from __future__ import annotations

import time

import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak():
    """Establish a clean floor before a peak-memory measurement.

    The gc.collect() is load-bearing, not hygiene. nn.Modules contain reference
    cycles, so dropping the last reference to a model does NOT free it by
    refcounting alone -- it becomes cyclic garbage that survives until a
    collection runs. Without this, each config's peak silently included the
    PREVIOUS config's weights, producing the tell-tale signature of a memory
    column that is off-by-one-config:

        fp32     w 2.49  peak 2.56   (+0.07)
        bf16     w 1.21  peak 3.74   (+2.53 ~= fp32's weights)
        int8_wo  w 0.61  peak 1.86   (+1.25 ~= bf16's weights)

    i.e. bf16 appearing to need MORE peak memory than fp32, which is the kind of
    result that gets explained away as "allocator noise" instead of fixed.
    """
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


@torch.no_grad()
def bench_forward(
    model,
    tok,
    batches,
    device: str = "cuda",
    warmup: int = 2,
    repeats: int = 3,
    pad_multiple: int = 128,
) -> dict:
    """Time a forward pass over pre-formed batches of Seq objects."""
    # Warm up over EVERY batch, not a prefix. Each length bucket is a distinct
    # input shape, so under torch.compile an unseen bucket triggers a
    # recompilation. Warming only a prefix leaves that compile cost inside the
    # timed region, which charges it to whichever config happens to run first
    # and produces meaningless speedups.
    for _ in range(max(1, warmup)):
        for b in batches:
            texts = [s.seq for s in b]
            enc = tok(texts, return_tensors="pt", padding=True,
                      pad_to_multiple_of=pad_multiple).to(device)
            model(**enc)
    _sync()

    _reset_peak()

    n_res = n_tok = 0
    t0 = time.perf_counter()
    for _ in range(repeats):
        for b in batches:
            texts = [s.seq for s in b]
            enc = tok(texts, return_tensors="pt", padding=True,
                      pad_to_multiple_of=pad_multiple).to(device)
            model(**enc)
            n_res += sum(len(s) for s in b)
            n_tok += enc["input_ids"].numel()
    _sync()
    dt = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return {
        "seconds": round(dt, 3),
        "n_sequences": len(batches) and sum(len(b) for b in batches) * repeats,
        "residues_per_s": round(n_res / dt, 1),
        "padded_tokens_per_s": round(n_tok / dt, 1),
        # Fraction of compute spent on padding -- the length-bucketing payoff.
        "padding_overhead": round(1 - n_res / n_tok, 4) if n_tok else 0.0,
        "peak_mem_gb": round(peak / 1024**3, 3),
    }


@torch.no_grad()
def bench_dms(
    model,
    tok,
    seq: str,
    variants: list[str],
    batch_size: int = 16,
    device: str = "cuda",
    method: str = "masked",
    use_compiled: bool = False,
    warmup: bool = True,
) -> dict:
    """Time a full DMS scoring run and report variants/sec.

    Runs against the UNCOMPILED module by design. Measured on 650M/A100 with a
    40-residue WT over 11 masked positions:

        eager    one forward at (11, 42):        27.9 ms
        eager    full masked_marginals pass:      0.03 s
        compiled one forward at (11, 42):     15138.1 ms

    DMS scoring uses a much smaller input shape than the bulk-throughput batches,
    and under max-autotune every new shape costs ~15 s to autotune -- a cost that
    recurs rather than amortising at this batch size. That made the timed region
    measure autotune cost, not inference, and ranked configs by their number of
    Triton kernel candidates: fp32 (fewest) looked 'fastest', fp8 (98 candidates)
    'slowest', exactly inverting the throughput column.

    Practical consequence beyond the benchmark: for short-WT / few-position DMS
    work, max-autotune is actively counterproductive -- the pass is latency-bound
    and compilation never pays for itself. Compile is worth it for DMS only when
    the assay has enough masked positions to amortise it.

    use_compiled=True keeps the compiled module instead, which is how that last
    sentence gets tested rather than assumed: a real assay has one fixed shape
    (n_positions x L) repeated ceil(n_positions/batch) times, so autotune is paid
    once and amortises over the whole run -- the opposite of the synthetic case.
    The untimed warmup pass below absorbs the compile, so the timed pass measures
    steady-state either way.

    warmup=False skips that extra pass. It is only safe when use_compiled is
    False AND the process has already run at least one assay -- there is then no
    compilation to absorb, and CUDA context / cuBLAS heuristics are already warm.
    It exists for the full-benchmark sweep (run_proteingym.py), where warming
    every one of 201 assays would double a 3 GPU-hour job to 6 for no gain.
    """
    import dms as dms_mod

    if not use_compiled:
        model = getattr(model, "_orig_mod", model)

    # Run the WHOLE scoring pass once, untimed, then time the second pass.
    #
    # A cheaper warmup on a slice of the variants does not work: masked_marginals
    # batches over distinct *positions*, and a leading slice of a DMS variant list
    # is all at the same position (variants are enumerated position by position).
    # That warms a batch of 1 while the real pass uses a batch of n_positions, so
    # the real shape still compiles inside the timed region. The symptom is a DMS
    # column ranked by compile cost rather than inference cost -- fp32 appearing
    # fastest and fp8 slowest, exactly inverting the throughput column.
    #
    # Doubling the work is worth it: this is the only way to be sure the timed
    # pass sees zero compilation, whatever the position/batch structure is.
    def _score():
        if method == "masked":
            return dms_mod.masked_marginals(
                model, tok, seq, variants, batch_size=batch_size, device=device
            )
        return dms_mod.wt_marginals(model, tok, seq, variants, device=device)

    if warmup:
        _score()
        _sync()

    _reset_peak()

    t0 = time.perf_counter()
    scores = _score()
    _sync()
    dt = time.perf_counter() - t0

    n_pos = len({m.pos for v in variants for m in dms_mod.parse_variant(v)})
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return {
        "method": method,
        "seconds": round(dt, 3),
        "n_variants": len(variants),
        "n_masked_positions": n_pos,
        "variants_per_s": round(len(variants) / dt, 1),
        "peak_mem_gb": round(peak / 1024**3, 3),
        "_scores": scores,
    }


def free(model) -> None:
    """Reclaim device memory between configs.

    NOTE: `del model` here only drops this function's local name. The CALLER
    must also drop its own reference (see run_matrix, which rebinds to None)
    or the model stays alive regardless of what this function does. That is
    exactly the bug _reset_peak documents.
    """
    del model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
