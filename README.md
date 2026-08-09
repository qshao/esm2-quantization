# ESM-2 quantization: speed + memory harness

Measures quantization configs for ESM-2 on speed **and** fidelity together, on
the LCC cluster. Parameterized by model size, so the identical config matrix
runs on 650M (fast dev loop) and 3B (the target).

## The framing that drives the design

ESM-2 is a BERT-style **encoder**: one forward pass, no KV cache, no
autoregressive decode. That puts it in the **compute-bound** regime, not the
memory-bandwidth-bound regime that decode-time LLMs live in.

Consequence: the popular quantization toolchains (GPTQ, AWQ, bitsandbytes NF4,
GGUF) are *weight-only*. They store INT4/INT8 and dequantize to bf16 right
before an ordinary bf16 GEMM. That is a large win at batch-1 LLM decode because
you are starving on bandwidth. Here it adds work to an already compute-saturated
kernel.

So the two goals separate, and the harness reports both:

| Goal | Config |
|---|---|
| Memory (fit smaller GPU, QLoRA base) | `int8_wo`, `int4_wo` -- weight-only |
| Speed | `int8_dyn` (A100), `fp8_dyn` (H200) -- W8A8, activations quantized too |

## torch.compile mode is not optional, and the default is not enough

The single most important measurement so far. On one ESM-2 3B-shaped FFN block
(2560 -> 10240 -> 2560, batch 32 x 512, A100):

| variant | ms |
|---|---|
| bf16 eager | 7.91 |
| bf16 compiled | 7.91 |
| int8_dyn eager | 33.35 |
| int8_dyn compiled, `dynamic=True` | 11.73 |
| int8_dyn compiled, default mode | 11.48 |
| **int8_dyn `mode="max-autotune"`** | **5.43 (1.46x faster than bf16)** |

Only `max-autotune` lets Inductor autotune the INT8 Triton matmuls, which is
where the entire W8A8 win lives. Measured with default mode, W8A8 looks 3-4x
*slower* than bf16 and would be wrongly discarded. `dynamic=True` is also
avoided -- it emits shape-generic kernels that give up most of the gain; static
shapes are kept manageable by padding to a fixed multiple (`pad_multiple=128`)
and raising Dynamo's recompile limit.

Verified with `torch._dynamo.explain`: **0 graph breaks** for both bf16 and
int8_dyn, so this is a kernel-selection effect, not a Dynamo fallback.

Full-model measurement on 650M (A100, eager vs compiled-default) also showed
`int8_wo` going from 0.80x to **0.99x of bf16 while halving weight memory
(1.21 -> 0.61 GB) and cutting peak memory 28% (2.11 -> 1.52 GB)** -- i.e. with
compile, weight-only INT8 is close to free on throughput, which is the opposite
of the eager-mode conclusion.

Reproduce with `python src/diag_compile.py`.

## Fidelity: three metrics, increasing sensitivity

A config that looks perfect on embeddings can be unusable for variant effect
scoring. The harness always reports all three:

1. **embedding cosine** -- mean-pooled. Very forgiving; error averages out over
   the sequence.
2. **logit KL** -- per-residue distribution drift. Moderately sensitive.
3. **DMS spearman vs fp32** -- `log p(mut) - log p(wt)`. **Most sensitive**,
   because it is a difference of two near-equal numbers, so quantization noise
   does not cancel and is amplified relative to the signal.

Already visible at 650M: `int4_wo` holds embedding cosine at 0.997 while DMS
rank correlation drops to 0.965. Validating on embeddings alone is how a broken
variant-effect pipeline ships.

All three are drift-from-fp32 metrics, so they bound how far a config moves your
answer but cannot say whether it moved toward or away from the truth. Only a
real assay does that -- see the ProteinGym section, where the config with the
worst drift turned out to *gain* rank correlation against experiment on one
assay and lose it on another.

## Cluster constraints (the non-obvious part)

Every node, **including the H200 nodes**, runs CentOS 7.6 / glibc 2.17. This
drove several forced choices:

| Constraint | Consequence |
|---|---|
| glibc 2.17; torch went manylinux_2_28 at 2.7 | **torch 2.6.0 is the newest installable release.** Bundles cu124; driver is 550.54.14, an exact match. |
| bitsandbytes GPU wheels need newer glibc | Only 0.42.0 installs, and it is **CPU-only** -- its NF4 path is dead. 4-bit goes through torchao instead. |
| `/usr/bin/gcc` is 4.8.5 | Triton cannot build its helper module, so **torch.compile fails**. Fixed by pointing CC/CXX at GCC 12.1.0 (OpenHPC, fixed path -- `module load` is unreliable inside non-interactive srun). |
| `$HOME` and `/project` are at quota | Triton/Inductor/HF/pip caches all default there and die with "Disk quota exceeded". All relocated to `/scratch`. |

`env/activate.sh` handles all of the above; source it before running anything.

torchao still works because its INT8/FP8 paths dispatch to `torch._int_mm` /
`torch._scaled_mm`, which live inside torch and support sm_90. So FP8 on the
H200s is reachable despite the version ceiling.

**Escape hatch** if a newer stack is ever needed (TransformerEngine,
FlashAttention-3): `ccs/singularity-3.8.2` module + an NGC PyTorch image.

## Hardware

| Partition | GPU | Notes |
|---|---|---|
| `H8V141_SAP112M2000_L` | 8x H200 143 GB, sm_90 | FP8 tensor cores. Target for `fp8_dyn`. |
| `A2V80_ICE56M256_L` | 2x A100 80 GB, sm_80 | bf16 + INT8 tensor cores, no FP8. |
| `V4V32_*` | V100 32 GB, sm_70 | No bf16, no INT8 tensor cores. Quantization buys memory only, never speed. |

## Layout

```
env/activate.sh       cluster env fixes -- source this first
env/setup_env.sh      builds the conda env with glibc-2.17-safe pins
src/quant.py          config registry; quantizes encoder Linears ONLY
src/models.py         ESM-2 loading + weight-footprint measurement
src/data.py           FASTA, length-bucketed batching
src/dms.py            masked-marginals + wt-marginals variant scoring
src/validate.py       the three fidelity metrics (+ ground-truth correlation)
src/analyze_dms.py    paired bootstrap: is a ground-truth shift real or noise?
src/bench.py          throughput / latency / peak memory
src/run_matrix.py     driver: fp32 reference, then every config against it
src/fetch_proteingym.py     3-assay fetch (v0.1 HF CSVs + GitHub reference WT)
src/fetch_proteingym_v1.py  full v1 benchmark from the HF parquet (217 assays)
src/run_proteingym.py       full-benchmark driver: ONE config, one load, all assays
src/aggregate_proteingym.py benchmark-wide accuracy/speed/RAM + assay bootstrap
slurm/run_matrix.sbatch
slurm/run_dms.sbatch        real ProteinGym assays, one job per model
slurm/run_proteingym.sbatch full benchmark, array job -- one task per config
data/proteingym/      3-assay subset (see data/proteingym/README.md)
data/proteingym_v1/   full benchmark: 217 assays + index.json
```

## Running

```bash
sbatch -p A2V80_ICE56M256_L --mem=120G slurm/run_matrix.sbatch 3B fp32,bf16,int8_wo,int8_dyn
sbatch -p H8V141_SAP112M2000_L --mem=200G slurm/run_matrix.sbatch 3B fp32,bf16,int8_dyn,fp8_dyn
```

`fp8_dyn` self-skips with a clear reason on A100 rather than failing the run.

## Reproducing a quantized model

**No quantized checkpoints are published, deliberately.** Every config here is a
data-free, deterministic function of the public MIT-licensed ESM-2 weights --
`quantize_` with no calibration set, no search, no randomness. A 2.66 GB
`int8_wo` checkpoint would carry no information these ten lines don't, and
torchao serialises as tensor subclasses whose layout changes between releases,
so a checkpoint pins you to one torchao version. The recipe does not.

```python
import torch, sys
from transformers import EsmForMaskedLM
from torchao.quantization import quantize_, int8_weight_only
sys.path.insert(0, "src")
from quant import encoder_linear_filter        # encoder-block Linears ONLY

model = EsmForMaskedLM.from_pretrained(
    "facebook/esm2_t36_3B_UR50D", dtype=torch.bfloat16).cuda().eval()
quantize_(model, int8_weight_only(), filter_fn=encoder_linear_filter)
```

Swap `int8_weight_only()` for `int4_weight_only()`,
`int8_dynamic_activation_int8_weight()` or
`float8_dynamic_activation_float8_weight()`; `src/quant.py` holds all six as a
registry. Verified on 150M: 180 encoder Linears converted, `lm_head` untouched,
real weight storage 296 -> 151 MB.

**`filter_fn` is the part that matters.** The conventional route --
`TorchAoConfig("int8_weight_only")` passed to `from_pretrained` -- quantizes
*every* Linear including the LM head, and **will not reproduce these results**.
The variant score is a difference of two near-equal log-probabilities, so noise
in the LM head is amplified rather than averaged away. If you publish a
checkpoint, set `modules_to_not_convert` to match `encoder_linear_filter` and
confirm a known rho before releasing it.

`int8_dyn` and `fp8_dyn` are not really checkpoints at all: they quantize
activations at *runtime*, so the weights are only half of what they are.

**Measurement gotcha.** `sum(p.numel() * p.element_size())` does not see this
quantization -- a tensor subclass reports its *logical* dtype (bf16), not its
packed int8 storage, so the naive sum returns the unquantized figure and looks
like the quantization silently failed. Read `weight.tensor_impl.int_data`, or
measure `torch.cuda.max_memory_allocated` as `src/bench.py` does.

## Results: ESM2-3B on H200 (SDPA, max-autotune-no-cudagraphs)

Reproduced across three independent jobs; bf16 throughput varied 0.4%.

| config | RAM: weights | RAM: peak | res/s | vs bf16 | emb_cos | logit KL | DMS rho |
|---|---|---|---|---|---|---|---|
| fp32 (ref) | 10.72 GB | 25.37 GB | 6,750 | 0.12x | - | - | - |
| bf16 | 5.29 GB | 12.63 GB | 57,056 | 1.00x | 0.999949 | 1.17e-04 | 0.9998 |
| int8_wo | 2.66 GB | 8.48 GB | 54,816 | 0.96x | 0.999959 | 1.05e-04 | 0.9998 |
| int8_dyn | 2.65 GB | **7.07 GB** | 59,996 | 1.05x | 0.989592 | 1.40e-02 | 0.9928 |
| **fp8_dyn** | 2.65 GB | 7.24 GB | **65,376** | **1.15x** | 0.999920 | 4.60e-04 | 0.9988 |
| int4_wo | **1.60 GB** | 7.69 GB | 3,813 | 0.07x | 0.998904 | 1.89e-03 | 0.9975 |

DMS scoring throughput (uncompiled, 200 variants over 11 masked positions):

| config | variants/s |
|---|---|
| **bf16** | **8,502** |
| int8_wo | 5,815 |
| fp8_dyn | 2,277 |
| fp32 | 2,077 |
| int8_dyn | 530 |

## The two workloads want OPPOSITE configurations

This is the main practical conclusion.

**Bulk embedding extraction** is compute-bound: large batches, big GEMMs.
Quantization pays. `fp8_dyn` + `max-autotune` wins on all three axes -- 1.15x
faster than bf16, weights 5.29 -> 2.65 GB, peak 12.63 -> 7.24 GB, with accuracy
indistinguishable from bf16 (emb_cos 0.999920).

**DMS / variant-effect scoring** is latency-bound: the masked batch is
n_positions x L, which is tiny. There is no GEMM large enough for low precision
to help, so the per-linear quantize/dequantize overhead is pure loss. **bf16
eager is 16x faster than int8_dyn here** (8,502 vs 530 variants/s), and
`max-autotune` is actively harmful -- autotuning a new shape costs ~15 s and
recurs rather than amortising, against a pass that does 0.03 s of real work.

So: quantize + compile for embeddings, plain bf16 eager for DMS scoring.
Applying the embedding-optimised config to DMS makes it an order of magnitude
slower.

The crossover depends on the number of masked positions. With hundreds of
positions (a real assay on a 300-residue protein) compilation would amortise and
the balance shifts; with 11 positions it cannot.

## Real assays: ProteinGym (the accuracy column that actually decides things)

> **Superseded by the full-benchmark section below.** Two of the three
> conclusions in this section did not survive going from 3 assays to 201. Kept
> because the failure mode is the point: every number here is correct, and the
> generalisation drawn from them was not.

Everything above measures fidelity **against fp32**. That answers "did
quantization change the model's answer", not "did it change the right answer".
Three ProteinGym substitution assays, chosen to span the length range:

| assay | L | singles | masked positions |
|---|---|---|---|
| `IF1_ECOLI_Kelsic_2016` | 72 | 1,367 | 72 |
| `BLAT_ECOLX_Stiffler_2015` | 286 | 4,996 | 263 |
| `HSP82_YEAST_Flynn_2019` | 709 | 13,194 | 707 |

ESM2-3B, H200, uncompiled. `rho_expt` is Spearman against the wet-lab
measurements; `shift` is the change from fp32, with a 2000-sample **paired**
bootstrap over variants (both configs scored on the same resample, so assay
noise cancels).

| config | IF1 shift | BLAT shift | HSP82 shift |
|---|---|---|---|
| fp32 | (ref) rho 0.5561 | (ref) rho 0.5893 | (ref) rho 0.2931 |
| bf16 | -0.0018 *real* | +0.0003 noise | +0.0002 noise |
| int8_wo | +0.0009 noise | +0.0019 *real* | +0.0002 noise |
| int8_dyn | -0.0057 noise | **+0.0266 real** | +0.0006 noise |
| fp8_dyn | +0.0023 noise | **+0.0194 real** | +0.0012 noise |
| int4_wo | **-0.0167 real** | **+0.0495 real** | -0.0027 noise |

Two results worth keeping:

**1. Quantization does not systematically degrade variant-effect accuracy.**
Of 15 (assay, config) pairs, 11 shifts are upward. `int4_wo` -- the worst config
on every fidelity metric -- gives the *best* agreement with experiment on BLAT
(+0.0495, CI [+0.0433, +0.0561]). That is not resampling noise; it survives the
bootstrap. It is also not a reason to quantize for accuracy, because:

**2. Fidelity-vs-fp32 predicts the SIZE of the shift, not its SIGN.** Across all
15 pairs, `r(1 - rho_fp32, |shift|) = 0.87`. But `int4_wo` moves +0.0495 on BLAT
and -0.0167 on IF1. So the cheap metric is still the right screen -- it bounds
how far your answer can move -- but it cannot tell you which way, and the
direction is a property of the assay, not the config.

Practical reading: `bf16` and `int8_wo` shift ground-truth agreement by at most
0.002 on any assay tested, which is below anything an experiment would resolve.
Those are the safe choices. `int4_wo` is a +-0.05 gamble on an unknown sign.

This supersedes the earlier synthetic-scan verdict, which ranked `int8_dyn`
(0.9928 vs fp32) as risky and `fp8_dyn` (0.9988) as clearly safer. Both numbers
were correct and neither predicted experimental accuracy: on BLAT the "risky"
config gained +0.027. A 40-residue synthetic scan over 11 positions was
measuring drift that no assay could resolve.

### DMS speed and memory on real assays (3B, uncompiled)

| config | weights | peak @L=72 | peak @L=709 | var/s @L=72 | var/s @L=709 |
|---|---|---|---|---|---|
| fp32 | 10.72 | 10.87 | 11.85 | 1,459 | 187 |
| **bf16** | 5.29 | 5.38 | 5.88 | **6,824** | **1,373** |
| int8_wo | 2.66 | 2.77 | 3.25 | 4,510 | 1,068 |
| int8_dyn | 2.65 | 2.87 | 3.36 | 390 | 298 |
| fp8_dyn | 2.65 | 2.76 | 3.35 | 1,651 | 1,100 |
| int4_wo | **1.61** | **1.70** | **2.20** | 1,045 | 112 |

Weights are ~90% of DMS peak memory even at L=709 (bf16: 5.29 GB weights, 0.59
GB activations), because masked-marginals keeps the batch narrow and never lets
the O(L^2) term take over the way bulk extraction does. So quantization
addresses nearly all of DMS peak memory -- the opposite of the long-sequence
embedding case.

`fp8_dyn` closes on bf16 as the assay grows -- 0.24x, 0.50x, 0.80x of bf16 speed
at L=72/286/709 -- because its overhead is fixed per call while real work scales.
At L=709 it costs 20% speed for 43% less memory.

**DMS recommendation: bf16 eager.** Fastest at every assay size, and ground-truth
accuracy indistinguishable from fp32. Drop to `int8_wo` if memory is tight (3.25
vs 5.88 GB, 78% of bf16 speed, accuracy still within 0.002).

## Full ProteinGym benchmark: 201 assays, 2.41M variants

`sbatch --array=0-5 -p H8V141_SAP112M2000_L --mem=200G slurm/run_proteingym.sbatch 3B`

All 217 substitution assays pass wild-type validation; 201 fit ESM-2's 1022-residue
context and are used. 40,775 masked positions -- that, not the variant count, is
the size of the job. One array task per config, each loading its model once.
3B: 3.1 GPU-hours total, longest config 74 min.

Significance uses a **paired bootstrap resampling proteins**, not assays: the 201
assays cover only 173 proteins (BLAT_ECOLX appears 4 times), so resampling assays
would count correlated replicates as independent evidence. In practice it barely
mattered -- intervals moved in the 4th decimal -- but that is now measured rather
than assumed. Both intervals are reported side by side.

### Benchmark-mean accuracy (Spearman vs experiment, mean over 201 assays)

| config | 3B: mean rho | delta | 95% CI | | 650M: mean rho | delta | 95% CI |
|---|---|---|---|---|---|---|---|
| fp32 | 0.4398 | (ref) | | | 0.4459 | (ref) | |
| bf16 | 0.4399 | +0.0002 | [-0.0001, +0.0004] | | 0.4460 | +0.0000 | [-0.0003, +0.0004] |
| int8_wo | 0.4399 | +0.0001 | [-0.0002, +0.0005] | | 0.4457 | -0.0002 | [-0.0006, +0.0002] |
| fp8_dyn | 0.4403 | +0.0005 | [-0.0007, +0.0017] | | 0.4452 | -0.0008 | [-0.0020, +0.0005] |
| int8_dyn | 0.4376 | -0.0021 | [-0.0072, +0.0017] | | 0.4440 | **-0.0020** | [-0.0032, -0.0009] |
| int4_wo | 0.4435 | **+0.0037** | [+0.0007, +0.0068] | | 0.4395 | **-0.0064** | [-0.0105, -0.0018] |

Bold = significant under the protein bootstrap.

### The mean is the wrong safety criterion

Every benchmark mean above is smaller than 0.007. The per-assay tail is not:

| config | worst assay delta (3B) | assays \|d\|>0.05 | worst assay delta (650M) | assays \|d\|>0.05 |
|---|---|---|---|---|
| bf16 | -0.0072 | 0 | -0.0040 | 0 |
| int8_wo | -0.0106 | 0 | -0.0120 | 0 |
| fp8_dyn | -0.0344 | 0 | -0.0316 | 0 |
| int8_dyn | **-0.3688** | 6 | -0.0411 | 0 |
| int4_wo | -0.0556 | 5 | **-0.2024** | 6 |

`int8_dyn` at 3B is "noise" on the benchmark mean (-0.0021, p=0.34) and destroys
`UBR5_HUMAN_Tsuboyama_2023_1I2T`: rho against experiment falls **0.591 -> 0.223**,
with fidelity-vs-fp32 at 0.393. A config can pass the benchmark average and still
be unusable on your protein. Averages hide exactly the failure you care about.

### Three findings that only appear at benchmark scale

**1. The 3-assay verdict was backwards, and the direction is model-dependent.**
The 3-assay run found 11 of 15 shifts upward and `int4_wo` gaining +0.0495 on
BLAT. At 201 assays `int4_wo` is significant in **opposite directions on the two
models** -- +0.0037 at 3B, -0.0064 at 650M. BLAT was a real effect on an
unrepresentative assay. There is no stable sign to exploit.

**2. Fidelity predicts magnitude, not sign -- now with the sign part nailed
down.** Across 1005 assay/config pairs, `r(1 - rho_fp32, |delta|)` = 0.81 at 3B
and 0.74 at 650M. The *signed* correlation is -0.58 at 3B and +0.10 at 650M:
inconsistent in sign between models, so it predicts nothing directional.
`rho_fp32` remains the right cheap screen -- it bounds how far you can move --
and it still cannot tell you which way.

**3. Damage concentrates where the model is already weakest.** At 650M,
`int4_wo` costs 0.0174 on low-MSA-depth assays and 0.0042 on high-depth ones.
Quantization hurts most where there is least signal to lose.

### Speed and RAM over the whole benchmark (3B, H200, uncompiled)

| config | wall-clock | positions/s | peak RAM median | peak max |
|---|---|---|---|---|
| fp32 | 45.0 min | 25.6 | 11.10 GB | 12.20 GB |
| **bf16** | **7.3 min** | **200.6** | 5.50 GB | 6.05 GB |
| int8_wo | 9.2 min | 139.3 | 2.87 GB | 3.42 GB |
| fp8_dyn | 10.2 min | 89.2 | 2.90 GB | 3.56 GB |
| int8_dyn | 40.0 min | 20.1 | 2.94 GB | 3.51 GB |
| int4_wo | 73.8 min | 17.3 | **1.82 GB** | **2.37 GB** |

**Recommendation, unchanged in shape and now properly evidenced: `bf16` eager for
DMS.** 6.2x faster than fp32 end-to-end, and across 402 assay/model pairs its
ground-truth agreement never moves by more than 0.018. `int8_wo` is the memory
fallback: 2.87 vs 5.50 GB, 69% of bf16 speed, max drift 0.016. `int8_dyn` and
`int4_wo` are both slower *and* carry a catastrophic tail -- they are not
trade-offs here, they are strictly worse.

### Model choice does NOT dominate precision choice

An earlier 3-assay reading -- 650M beating 3B by 0.14 Spearman on BLAT -- suggested
model selection mattered more than precision selection. Over 201 assays that does
not hold: 650M 0.4459 vs 3B 0.4398, delta +0.0062, 95% CI [-0.0061, +0.0185],
p = 0.35, with 650M ahead on 110 of 201 assays. **The two models are
statistically indistinguishable benchmark-wide.** The BLAT gap was real for BLAT
and did not generalise -- the same trap as the `int4_wo`/BLAT result, found the
same way.

Worth keeping anyway: the *uncertainty* on model choice (+-0.012) is wider than
any quantization effect measured here (<=0.007). Precision is the settled
variable; which model to use is not.

## What quantization does not fix

Two effects dominate real ESM-2 pipelines and neither is addressed by
quantization:

* **Padding waste.** A naive "loop over FASTA, pad to longest" pipeline commonly
  wastes 50-80% of its FLOPs. `data.bucket_batches` sorts by length and fills
  under a token budget. Free, and no accuracy cost.
* **DMS masked-marginals cost.** Scoring is one forward per *mutated position*,
  so runtime scales with the number of masked positions, not the per-forward
  cost. `src/dms.py` masks only positions carrying variants and batches the
  masked copies. Reference implementations typically do neither.
