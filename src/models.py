"""ESM-2 loading with a pluggable quantization config.

Parameterized by model id so the identical config matrix runs on 650M (fast dev
loop, weights already cached locally) and 3B (the target).
"""

from __future__ import annotations

import os
import time

import torch

import quant

# Keep every download/read pointed at the project cache -- $HOME is small and
# the 650M weights are already there.
os.environ.setdefault("HF_HOME", "/project/qsh226_uksr/qsh226/hf_cache")

MODELS = {
    "650M": "facebook/esm2_t33_650M_UR50D",
    "3B": "facebook/esm2_t36_3B_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",  # smoke-test size
    # 48 layers x 5120 hidden = 5.33x the per-token compute of 3B, and a 60.5 GB
    # download. fp32 peaks near 54 GB, so it needs an H200; on an 80 GB A100 the
    # fp32 reference will not fit alongside its activations.
    "15B": "facebook/esm2_t48_15B_UR50D",
}


def resolve(model: str) -> str:
    return MODELS.get(model, model)


def _dtype_kwarg(dtype):
    """transformers renamed torch_dtype -> dtype in v5; support both."""
    import transformers
    major = int(transformers.__version__.split(".")[0])
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}


def load(
    model: str = "650M",
    quant_name: str = "bf16",
    attn: str = "sdpa",
    device: str = "cuda",
    compile_model: bool = False,
    compile_mode: str = "max-autotune-no-cudagraphs",
):
    """Load ESM-2 under a named quantization config.

    Returns (model, tokenizer, info-dict).
    """
    from transformers import AutoTokenizer, EsmForMaskedLM

    cfg = quant.get(quant_name)
    ok, why = cfg.check_supported()
    if not ok:
        raise RuntimeError(f"quant config {quant_name!r} unsupported here: {why}")

    model_id = resolve(model)
    tok = AutoTokenizer.from_pretrained(model_id)

    kwargs = dict(attn_implementation=attn, **_dtype_kwarg(cfg.dtype))

    if cfg.bnb:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(**cfg.bnb_kwargs)
        # bitsandbytes places the model itself.
        kwargs["device_map"] = {"": 0} if device == "cuda" else device

    t0 = time.time()
    # Measure the real resident weight cost as an allocator delta. Summing
    # element_size() over parameters lies for torchao configs, because the
    # quantized data lives inside tensor subclasses whose outer tensor still
    # reports the original dtype.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    net = EsmForMaskedLM.from_pretrained(model_id, **kwargs)
    if not cfg.bnb:
        net = net.to(device)
    net.eval()

    # torchao must run after the model is on-device: its INT8/FP8 paths swap in
    # tensor subclasses that expect CUDA tensors.
    net = quant.apply_torchao(net, cfg)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    weight_bytes = (
        torch.cuda.memory_allocated() - mem_before if torch.cuda.is_available() else 0
    )

    if compile_model:
        # mode matters enormously for the quantized configs, and the default is
        # NOT enough. Measured on one ESM-2 3B-shaped FFN block (A100):
        #   bf16 eager                   7.91 ms
        #   int8_dyn eager              33.35 ms
        #   int8_dyn compile dynamic=True 11.73 ms
        #   int8_dyn compile default     11.48 ms
        #   int8_dyn max-autotune         5.41 ms
        #   int8_dyn max-autotune-no-cudagraphs 5.28 ms <- 1.49x faster than bf16
        # -no-cudagraphs is the default: plain max-autotune captures CUDA graphs,
        # whose output tensors are invalidated by the next run. This harness
        # retains logits/embeddings across calls for fidelity scoring, so graph
        # capture fails with "accessing tensor output of CUDAGraphs that has been
        # overwritten". The no-cudagraphs variant keeps the autotuning win (it
        # measured marginally faster) without that constraint.
        # Only max-autotune lets Inductor autotune the INT8 Triton matmuls, which
        # is where the W8A8 win actually comes from. dynamic=True is also avoided:
        # it produces shape-generic kernels that give up much of the gain.
        # Static shapes are kept manageable by padding to a fixed multiple in the
        # benchmark loop; raise the recompile limit to accommodate the buckets.
        torch._dynamo.config.cache_size_limit = max(
            64, torch._dynamo.config.cache_size_limit
        )
        net = torch.compile(net, mode=compile_mode)

    load_s = time.time() - t0

    info = dict(
        model=model,
        model_id=model_id,
        quant=quant_name,
        attn=attn,
        compiled=compile_model,
        compile_mode=compile_mode if compile_model else None,
        dtype=str(cfg.dtype),
        load_seconds=round(load_s, 1),
        weight_bytes=weight_bytes,
        notes=cfg.notes,
    )
    return net, tok, info


def weight_footprint(net: torch.nn.Module) -> int:
    """Actual on-device bytes held by parameters and buffers.

    Uses element_size() * nelement() where available, and falls back to
    untyped_storage for quantized tensor subclasses that report odd dtypes.
    """
    total = 0
    seen = set()
    for t in list(net.parameters()) + list(net.buffers()):
        if id(t) in seen:
            continue
        seen.add(id(t))
        try:
            total += t.untyped_storage().nbytes()
        except Exception:
            try:
                total += t.nelement() * t.element_size()
            except Exception:
                pass
    return total


def describe(net: torch.nn.Module) -> None:
    """Print which Linears actually got quantized -- a cheap sanity check that
    the filter matched what we intended."""
    hit, miss = [], []
    for fqn, m in net.named_modules():
        if isinstance(m, torch.nn.Linear) or "Linear" in type(m).__name__:
            (hit if quant.encoder_linear_filter(m, fqn) else miss).append(fqn)
    print(f"[describe] {len(hit)} encoder Linears targeted, {len(miss)} left alone")
    for fqn in miss[:12]:
        print(f"           untouched: {fqn}")
