"""Find out why the DMS throughput column is inverted.

Symptom: fp32 scores 'fastest' and the quantized configs 'slowest', exactly
inverting the bulk-throughput column. Two warmup fixes did not change it, so the
cause is not compilation leaking into the timed region.

This times the pieces separately so the real cost centre is visible:
  - one raw forward at the DMS shape (n_positions x L)
  - a full masked_marginals pass, compiled and uncompiled
  - the same with the .cpu() transfer per position removed
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import dms as dms_mod
import models

MODEL = os.environ.get("DIAG_MODEL", "650M")
WT_LEN = int(os.environ.get("DIAG_WT_LEN", "40"))

aa = "ACDEFGHIKLMNPQRSTVWY"
wt = (aa * 40)[:WT_LEN]
variants = [f"{wt[p-1]}{p}{m}" for p in range(1, WT_LEN + 1)
            for m in aa if m != wt[p-1]][:200]
n_pos = len({m.pos for v in variants for m in dms_mod.parse_variant(v)})
print(f"model={MODEL} WT_len={WT_LEN} variants={len(variants)} positions={n_pos}")


def timeit(fn, n=3):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


for compiled in (False, True):
    torch._dynamo.reset()
    net, tok, info = models.load(MODEL, "bf16", compile_model=compiled)
    tag = "compiled" if compiled else "eager"

    enc = tok([wt] * n_pos, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        fwd = timeit(lambda: net(**enc), n=5)
    print(f"[{tag}] one forward at ({n_pos}, {enc['input_ids'].shape[1]}): "
          f"{fwd*1000:.1f} ms")

    with torch.no_grad():
        full = timeit(lambda: dms_mod.masked_marginals(
            net, tok, wt, variants, batch_size=16, device="cuda"), n=2)
    print(f"[{tag}] full masked_marginals pass:            {full:.2f} s")
    print(f"[{tag}] implied forwards' share:               "
          f"{fwd / full * 100:.2f}%  <- if tiny, the cost is NOT inference")

    # How much is the per-position .cpu() sync + log_softmax over the 33-token
    # vocab, versus the model itself?
    ids = enc["input_ids"].clone()
    with torch.no_grad():
        out = net(input_ids=ids, attention_mask=enc["attention_mask"]).logits

        def transfer():
            for p in range(n_pos):
                torch.log_softmax(out[p, p + 1].float(), dim=-1).cpu()
        tr = timeit(transfer, n=5)
    print(f"[{tag}] per-position .cpu() transfers only:    {tr*1000:.1f} ms")

    del net
    import gc; gc.collect(); torch.cuda.empty_cache()
    print()
