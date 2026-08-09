# Source this before running anything. Handles three cluster-specific problems:
#
# 1. torch.compile / Triton shell out to a C compiler, and /usr/bin/gcc here is
#    4.8.5, which cannot build Triton's helper module. GCC 12.1.0 is available
#    at a fixed path via OpenHPC. `module load` is unreliable inside
#    non-interactive srun, so CC/CXX are set explicitly.
#
# 2. $HOME and /project are both at their quota. Every cache these tools create
#    by default (Triton, Inductor, HuggingFace, pip, TMPDIR) has to be moved to
#    /scratch or compilation dies with "Disk quota exceeded".
#
# 3. torch.compile is not optional for the speed track: torchao's INT8/FP8
#    paths only pay off once Inductor fuses the quant/dequant around the GEMM.
#    In eager mode they are slower than bf16.

GCCROOT=/opt/ohpc/pub/compiler/gcc/12.1.0
export CC="$GCCROOT/bin/gcc"
export CXX="$GCCROOT/bin/g++"
export LD_LIBRARY_PATH="$GCCROOT/lib64:${LD_LIBRARY_PATH:-}"

export SCRATCH_BASE=/scratch/qsh226
export TMPDIR="$SCRATCH_BASE/tmp"
export HF_HOME="$SCRATCH_BASE/hf_cache"
export TRITON_CACHE_DIR="$SCRATCH_BASE/tmp/triton"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH_BASE/tmp/inductor"
export PIP_CACHE_DIR="$SCRATCH_BASE/pip-cache"
export XDG_CACHE_HOME="$SCRATCH_BASE/tmp/xdg"

mkdir -p "$TMPDIR" "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"

source /project/qsh226_uksr/qsh226/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/qsh226/envs/esmquant
