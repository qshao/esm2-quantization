#!/bin/bash
#SBATCH --job-name=esmquant
#SBATCH --account=gol_qsh226_uksr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.out
#
# Usage:
#   sbatch -p H8V141_SAP112M2000_L --mem=120G slurm/run_matrix.sbatch 3B fp32,bf16,int8_wo,int8_dyn,fp8_dyn
#   sbatch -p A2V80_ICE56M256_L    --mem=120G slurm/run_matrix.sbatch 3B fp32,bf16,int8_wo,int8_dyn
#
# fp8_dyn is H200-only (needs compute capability >= 8.9); on A100 the harness
# skips it with a clear reason rather than failing the run.

set -euo pipefail

MODEL="${1:-650M}"
CONFIGS="${2:-fp32,bf16,int8_wo,int8_dyn}"
# See run_dms.sbatch: `shift N || true` silently leaks positional args into "$@"
# when fewer than N were passed.
shift $(( $# > 2 ? 2 : $# ))

cd /scratch/qsh226/ESM2-3B
# Sets CC/CXX to GCC 12 (torch.compile needs it) and relocates every cache off
# the quota-exhausted $HOME and /project. See env/activate.sh.
source env/activate.sh
mkdir -p results

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, '| cc', torch.cuda.get_device_capability())"
echo

python -u src/run_matrix.py \
    --model "$MODEL" \
    --configs "$CONFIGS" \
    --out "results/matrix_${MODEL}_${SLURM_JOB_ID}.json" \
    "$@"
