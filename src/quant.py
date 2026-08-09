"""Quantization config registry for ESM-2.

Design rule: quantize ONLY the transformer-block Linears (QKV, attention output,
FFN in/out). Leave the LM head, contact head, embeddings, and every LayerNorm /
rotary / softmax in high precision.

Rationale specific to ESM-2:
  * The embedding table is 33 x d_model -- quantizing it saves nothing and costs
    accuracy on a lookup that is already exact.
  * The LM head feeds the log-ratio used by variant-effect scoring, which is a
    difference of two near-equal numbers. Quantization noise there does not
    cancel, it gets amplified relative to the signal.
  * ~99% of the parameters live in the encoder blocks anyway, so the memory we
    give up by excluding these is negligible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

# Linear modules inside an encoder block, by fully-qualified-name suffix.
_BLOCK_LINEAR = re.compile(
    r"encoder\.layer\.\d+\.("
    r"attention\.self\.(query|key|value)"
    r"|attention\.output\.dense"
    r"|intermediate\.dense"
    r"|output\.dense"
    r")$"
)

# Never quantize these, regardless of match above.
_EXCLUDE = ("lm_head", "contact_head", "embeddings")


def encoder_linear_filter(module: torch.nn.Module, fqn: str) -> bool:
    """torchao filter_fn: True for encoder-block Linears only."""
    if not isinstance(module, torch.nn.Linear):
        return False
    if any(tok in fqn for tok in _EXCLUDE):
        return False
    return bool(_BLOCK_LINEAR.search(fqn))


@dataclass
class QuantConfig:
    name: str
    # Compute/storage dtype the model is loaded in before any quantization.
    dtype: torch.dtype = torch.bfloat16
    # torchao quantization applied post-load; None means no torchao step.
    torchao_factory: Optional[Callable[[], object]] = None
    # bitsandbytes path is applied at from_pretrained time instead.
    bnb: bool = False
    bnb_kwargs: dict = field(default_factory=dict)
    # Minimum CUDA compute capability required, as (major, minor).
    min_cc: tuple[int, int] = (0, 0)
    notes: str = ""

    def check_supported(self) -> tuple[bool, str]:
        if not torch.cuda.is_available():
            return (self.min_cc == (0, 0), "no CUDA device")
        cc = torch.cuda.get_device_capability()
        if cc < self.min_cc:
            return (
                False,
                f"needs compute capability >= {self.min_cc}, device is {cc}",
            )
        return True, ""


def _ao():
    """Import torchao quantization factories, tolerating the API rename.

    torchao renamed the functional factories (int8_weight_only) to config
    classes (Int8WeightOnlyConfig) around 0.10. Support both.
    """
    import torchao.quantization as q

    def pick(*names):
        for n in names:
            if hasattr(q, n):
                return getattr(q, n)
        raise AttributeError(f"torchao.quantization has none of {names}")

    return {
        "int8_wo": pick("Int8WeightOnlyConfig", "int8_weight_only"),
        "int4_wo": pick("Int4WeightOnlyConfig", "int4_weight_only"),
        "int8_dyn": pick(
            "Int8DynamicActivationInt8WeightConfig",
            "int8_dynamic_activation_int8_weight",
        ),
        # Same factory, asymmetric activation mapping. Bound as a lambda so the
        # keyword survives the config-class/function rename above.
        "int8_dyn_asym": lambda: pick(
            "Int8DynamicActivationInt8WeightConfig",
            "int8_dynamic_activation_int8_weight",
        )(act_mapping_type=__import__(
            "torchao.quantization.quant_primitives", fromlist=["MappingType"]
        ).MappingType.ASYMMETRIC),
        "fp8_dyn": pick(
            "Float8DynamicActivationFloat8WeightConfig",
            "float8_dynamic_activation_float8_weight",
        ),
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CONFIGS: dict[str, QuantConfig] = {
    # ---- references ----
    "fp32": QuantConfig(
        name="fp32",
        dtype=torch.float32,
        notes="Ground-truth reference. All fidelity metrics are measured against this.",
    ),
    "bf16": QuantConfig(
        name="bf16",
        dtype=torch.bfloat16,
        min_cc=(8, 0),
        notes="Practical baseline on A100/H200. Preferred over fp16 -- ESM-2 has "
        "known overflow/NaN issues in fp16 attention on some sequences.",
    ),
    "fp16": QuantConfig(
        name="fp16",
        dtype=torch.float16,
        notes="Only baseline available on V100 (sm_70 has no bf16). Watch for NaNs.",
    ),
    # ---- memory track (weight-only; expect SLOWER than bf16 on encoders) ----
    "int8_wo": QuantConfig(
        name="int8_wo",
        dtype=torch.bfloat16,
        torchao_factory=lambda: _ao()["int8_wo"](),
        min_cc=(8, 0),
        notes="Weight-only INT8. Halves weight memory. Dequantizes to bf16 before "
        "the GEMM, so it does NOT speed up compute-bound encoder inference.",
    ),
    # NOTE: bitsandbytes is NOT usable on this cluster. Its GPU-enabled wheels
    # require a newer glibc than CentOS 7.6 provides, so the only installable
    # build (0.42.0) is CPU-only and its NF4 path is dead. The 4-bit track
    # therefore goes through torchao instead, which reaches the same place:
    # ~4x weight compression, and a viable QLoRA base via peft.
    "int4_wo": QuantConfig(
        name="int4_wo",
        dtype=torch.bfloat16,
        torchao_factory=lambda: _ao()["int4_wo"](),
        min_cc=(8, 0),
        notes="torchao weight-only INT4 (tinygemm). ~4x weight memory reduction; "
        "the QLoRA base and the 'fit on a smaller GPU' config. Expect a THROUGHPUT "
        "REGRESSION at inference -- this is for the fine-tuning memory wall, not "
        "for speed. Requires bf16 compute dtype.",
    ),
    # ---- speed track (W8A8: activations quantized too, real low-precision GEMMs) ----
    "int8_dyn": QuantConfig(
        name="int8_dyn",
        dtype=torch.bfloat16,
        torchao_factory=lambda: _ao()["int8_dyn"](),
        min_cc=(8, 0),
        notes="W8A8 INT8, per-channel weight + dynamic per-token activation scales. "
        "Real INT8 tensor-core GEMMs on A100/H200. This is the A100 speed config. "
        "If accuracy drops, the cause is ESM-2's activation outliers -- try SmoothQuant.",
    ),
    # Two probes for WHY int8_dyn collapses on some assays. int8_dyn already
    # uses per-token activation and per-channel weight scales, so the usual
    # "quantize at finer granularity" fix is already applied and cannot be the
    # explanation. What per-token scaling still cannot absorb is a single
    # channel that is huge WITHIN a token, which is the activation-outlier
    # regime SmoothQuant was built for.
    "int8_dyn_asym": QuantConfig(
        name="int8_dyn_asym",
        dtype=torch.bfloat16,
        torchao_factory=lambda: _ao()["int8_dyn_asym"](),
        min_cc=(8, 0),
        notes="int8_dyn with ASYMMETRIC activation quantization (zero-point). "
        "Symmetric scaling wastes half the INT8 range on one-sided activations, "
        "which post-GELU tensors are. One-line change, no calibration.",
    ),
    "int8_dyn_sq": QuantConfig(
        name="int8_dyn_sq",
        dtype=torch.bfloat16,
        torchao_factory=None,   # needs calibration; applied by models.load
        min_cc=(8, 0),
        notes="int8_dyn preceded by SmoothQuant: a per-channel smoothing factor "
        "migrates activation outliers into the weights, which are quantized "
        "per-channel and can absorb them. Requires a calibration pass, so it is "
        "applied separately rather than through torchao_factory.",
    ),
    "fp8_dyn": QuantConfig(
        name="fp8_dyn",
        dtype=torch.bfloat16,
        torchao_factory=lambda: _ao()["fp8_dyn"](),
        min_cc=(8, 9),
        notes="W8A8 FP8 (E4M3). Native tensor-core support on H200 only. Best "
        "speed/accuracy/effort tradeoff for the embedding-extraction workload.",
    ),
}


def get(name: str) -> QuantConfig:
    if name not in CONFIGS:
        raise KeyError(f"unknown quant config {name!r}; have {sorted(CONFIGS)}")
    return CONFIGS[name]


def apply_torchao(model: torch.nn.Module, cfg: QuantConfig) -> torch.nn.Module:
    """Apply the torchao quantization step in-place, if this config has one."""
    if cfg.torchao_factory is None:
        return model
    from torchao.quantization import quantize_

    n_before = sum(
        1
        for fqn, m in model.named_modules()
        if encoder_linear_filter(m, fqn)
    )
    quantize_(model, cfg.torchao_factory(), filter_fn=encoder_linear_filter)
    print(f"[quant] {cfg.name}: applied torchao to {n_before} encoder Linears")
    return model


def apply_smoothquant(model, tok, calib_seqs, device="cuda", alpha=0.5,
                      batch_size=8):
    """SmoothQuant + int8 dynamic, applied to encoder Linears only.

    Unlike every other config here this one is NOT data-free: a per-channel
    smoothing factor s is chosen from observed activation ranges, the
    activation is divided by s and the weight multiplied by it. The product is
    unchanged, but the activation outliers move into the weights, which are
    quantized per-channel and can absorb them.

    Calibration uses masked copies of the target sequences themselves, i.e. the
    exact input distribution that will be scored. That is the favourable case
    for SmoothQuant; a general-purpose recipe calibrated on unrelated proteins
    would be a different, harder experiment.

    Note that insert_smooth_quant_observer_ has no filter argument and observes
    every Linear including the LM head. Only encoder Linears are converted by
    the quantize_ call below; the rest keep their observer wrapper, which
    forwards to the original module and leaves their numerics untouched.
    """
    import torch
    from torchao.prototype.smoothquant import (insert_smooth_quant_observer_,
                                               smooth_quant)
    from torchao.quantization import quantize_

    insert_smooth_quant_observer_(model, alpha=alpha, quant_mode="dynamic")
    with torch.no_grad():
        for seq in calib_seqs:
            enc = tok(seq, return_tensors="pt", add_special_tokens=True)
            ids = enc["input_ids"][0].to(device)
            attn = enc["attention_mask"][0].to(device)
            n = min(batch_size, ids.numel() - 2)
            X = ids.unsqueeze(0).repeat(n, 1)
            for r in range(n):
                X[r, r + 1] = tok.mask_token_id
            model(input_ids=X, attention_mask=attn.unsqueeze(0).expand(n, -1))
    quantize_(model, smooth_quant(), filter_fn=encoder_linear_filter)

    # Any Linear the filter did NOT convert still carries its observer, whose
    # forward keeps calling obs(input). That is wasted work on every token and
    # it eventually raises "Can't update existing min_val" once ranges shift
    # -- which killed one assay in the first smoke run. Unwrap them back to
    # plain Linears carrying the same parameters.
    import torch.nn as nn
    from torchao.prototype.smoothquant import SmoothQuantObservedLinear

    n_unwrapped = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, SmoothQuantObservedLinear):
                lin = nn.Linear(child.in_features, child.out_features,
                                bias=child.bias is not None,
                                device=child.weight.device,
                                dtype=child.weight.dtype)
                with torch.no_grad():
                    lin.weight.copy_(child.weight)
                    if child.bias is not None:
                        lin.bias.copy_(child.bias)
                setattr(parent, child_name, lin)
                n_unwrapped += 1
    if n_unwrapped:
        print(f"[smoothquant] unwrapped {n_unwrapped} unconverted observed Linears")
    return model
