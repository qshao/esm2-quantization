"""Isolate why torchao's quantized paths show identical eager/compiled timings.

In the config matrix, int8_dyn and int4_wo were bit-for-bit the same speed with
and without torch.compile, while bf16 and int8_wo both sped up. That pattern
means Dynamo is not compiling those modules at all -- it is graph-breaking on
torchao's tensor subclasses and silently falling back to eager.

This isolates a single FFN block (ESM-2 3B shapes: 2560 -> 10240 -> 2560) and
compares compile modes, then counts graph breaks directly.
"""

import time

import torch
import torch.nn as nn
from torchao.quantization import (
    quantize_,
    int8_dynamic_activation_int8_weight,
    int8_weight_only,
)

D, F, B, L = 2560, 10240, 32, 512
x = torch.randn(B, L, D, device="cuda", dtype=torch.bfloat16)


def mk():
    return nn.Sequential(
        nn.Linear(D, F, bias=True), nn.GELU(), nn.Linear(F, D, bias=True)
    ).cuda().bfloat16()


def mk_q(factory):
    m = mk()
    quantize_(m, factory())
    return m


def timeit(m, n=20):
    for _ in range(5):
        m(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        m(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


print(f"{'variant':34s} {'ms':>8s}")
print("-" * 44)
print(f"{'bf16 eager':34s} {timeit(mk()):8.2f}")
print(f"{'bf16 compiled (dynamic=True)':34s} {timeit(torch.compile(mk(), dynamic=True)):8.2f}")
print(f"{'bf16 compiled (static)':34s} {timeit(torch.compile(mk())):8.2f}")

print(f"{'int8_wo eager':34s} {timeit(mk_q(int8_weight_only)):8.2f}")
print(f"{'int8_wo compiled (static)':34s} {timeit(torch.compile(mk_q(int8_weight_only))):8.2f}")

f = int8_dynamic_activation_int8_weight
print(f"{'int8_dyn eager':34s} {timeit(mk_q(f)):8.2f}")
print(f"{'int8_dyn compiled (dynamic=True)':34s} {timeit(torch.compile(mk_q(f), dynamic=True)):8.2f}")
print(f"{'int8_dyn compiled (static)':34s} {timeit(torch.compile(mk_q(f))):8.2f}")
mode = "max-autotune"
print(f"{'int8_dyn ' + mode:34s} {timeit(torch.compile(mk_q(f), mode=mode)):8.2f}")

# Count graph breaks explicitly rather than inferring them from timings.
print("\n=== dynamo explain: int8_dyn ===")
torch._dynamo.reset()
try:
    exp = torch._dynamo.explain(mk_q(f))(x)
    print("graph breaks:", exp.graph_break_count)
    for r in (exp.break_reasons or [])[:5]:
        print("  -", str(r)[:200])
except Exception as e:
    print("explain failed:", type(e).__name__, e)

print("\n=== dynamo explain: bf16 (control) ===")
torch._dynamo.reset()
try:
    exp = torch._dynamo.explain(mk())(x)
    print("graph breaks:", exp.graph_break_count)
except Exception as e:
    print("explain failed:", type(e).__name__, e)

# max-autotune enables CUDA graphs, whose outputs are invalidated by the next
# run -- fatal for a harness that retains logits/embeddings. Check that the
# -no-cudagraphs variant keeps the autotuning win.
m2 = "max-autotune-no-cudagraphs"
print(f"\n{'int8_dyn ' + m2:34s} {timeit(torch.compile(mk_q(f), mode=m2)):8.2f}")
print(f"{'bf16 ' + m2:34s} {timeit(torch.compile(mk(), mode=m2)):8.2f}")
