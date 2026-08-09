#!/usr/bin/env bash
# Build the esmquant environment.
#
# HARD CONSTRAINT discovered on this cluster: every node, including the H200
# nodes, runs CentOS 7.6 / glibc 2.17. PyTorch moved to manylinux_2_28 wheels
# (glibc >= 2.28) at torch 2.7, so:
#
#   torch 2.6.0 is the newest installable release here.
#
# That is still enough for the plan: torchao's INT8/FP8 paths dispatch to
# torch._int_mm / torch._scaled_mm, which live inside torch and support sm_90,
# so FP8 W8A8 on the H200s is reachable. Driver is 550.54.14 (CUDA 12.4), and
# torch 2.6.0's default PyPI wheel bundles cu124 -- an exact match.
#
# --only-binary=:all: is deliberate: the system compiler is gcc 4.8.5, so any
# source fallback produces a confusing build failure instead of a clear
# "no compatible wheel" error.
#
# If we later need a newer stack (TransformerEngine, FlashAttention-3), the
# escape hatch is the `ccs/singularity-3.8.2` module + an NGC PyTorch image.
set -euo pipefail

ENV_PREFIX=/scratch/qsh226/envs/esmquant
CONDA_BASE=/project/qsh226_uksr/qsh226/miniconda3

source "$CONDA_BASE/etc/profile.d/conda.sh"

if [ ! -d "$ENV_PREFIX" ]; then
    conda create -p "$ENV_PREFIX" python=3.11 -y
fi
conda activate "$ENV_PREFIX"

# /project is over quota, so keep every cache and temp dir on /scratch.
export PIP_CACHE_DIR=/scratch/qsh226/pip-cache
export TMPDIR=/scratch/qsh226/tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

PIP="pip install --no-cache-dir --only-binary=:all:"

# Pinned to the newest releases that still publish manylinux_2_17 wheels.
$PIP "numpy==2.2.6"
$PIP "torch==2.6.0"

# torchao 0.9.0 is the release aligned with the torch 2.6 cycle.
$PIP "torchao==0.9.0"

# Pure-python, so glibc is irrelevant. transformers held at 4.x for torch 2.6.
$PIP "transformers==4.57.6" "peft==0.15.2" "accelerate==1.6.0" "safetensors" "tqdm"

# Versions below still ship manylinux_2_17 wheels.
$PIP "scipy==1.15.3" "pandas==2.2.3" "biopython==1.85"

# bitsandbytes is OPTIONAL and expected to be marginal here: the newest build
# that works on glibc 2.17 is 0.42.0. If it fails or misbehaves we use torchao's
# native NF4 for the QLoRA track instead, so do not fail the whole setup.
$PIP "bitsandbytes==0.42.0" || echo "[warn] bitsandbytes unavailable -- will use torchao NF4 for the QLoRA track"

python - <<'PY'
import torch, transformers, torchao
print("torch       ", torch.__version__, "| bundled cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("torchao     ", torchao.__version__)
try:
    import bitsandbytes; print("bitsandbytes", bitsandbytes.__version__)
except Exception as e:
    print("bitsandbytes UNAVAILABLE:", type(e).__name__)
from transformers import EsmForMaskedLM
print("EsmForMaskedLM import OK")
PY
