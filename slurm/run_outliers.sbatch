#!/bin/bash
#SBATCH --job-name=esmdiag
#SBATCH --account=gol_qsh226_uksr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.out
set -euo pipefail
cd /scratch/qsh226/ESM2-3B
source env/activate.sh
python -u src/diag_outliers.py --model "${1:-3B}"
