#!/bin/bash
#SBATCH --job-name=esmdms
#SBATCH --account=gol_qsh226_uksr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.out
#
# Real ProteinGym assays, one job, three assays spanning the length range.
#
#   sbatch -p H8V141_SAP112M2000_L --mem=200G slurm/run_dms.sbatch 3B fp32,bf16,int8_wo,int8_dyn,fp8_dyn,int4_wo
#   sbatch -p A2V80_ICE56M256_L    --mem=120G slurm/run_dms.sbatch 650M fp32,bf16,int8_wo,int8_dyn
#
# Throughput is skipped on purpose: DMS scoring runs uncompiled (see
# bench.bench_dms), and the compiled throughput matrix is already established in
# results/matrix_3B_*.json. Skipping it avoids paying max-autotune cost for
# numbers this run does not use.
#
# Length span is the point. Peak memory during masked-marginals scoring is
# O(n_positions x L^2), so a 72-residue assay and a 709-residue one probe
# genuinely different memory regimes -- which is exactly where the claim "int8
# halves your RAM" either holds or stops holding.

set -euo pipefail

MODEL="${1:-650M}"
CONFIGS="${2:-fp32,bf16,int8_wo,int8_dyn}"
ASSAYS="${3:-IF1_ECOLI_Kelsic_2016,BLAT_ECOLX_Stiffler_2015,HSP82_YEAST_Flynn_2019}"
# Shift only what was actually passed. `shift 3` FAILS when fewer than 3 args
# were given, and `|| true` then swallows the failure while leaving the
# positional args in "$@" -- where they get appended to the python command as
# stray arguments. Relying on a default for $3 was enough to trigger it.
shift $(( $# > 3 ? 3 : $# ))

cd /scratch/qsh226/ESM2-3B
source env/activate.sh
mkdir -p results

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, '| cc', torch.cuda.get_device_capability())"
echo

D=data/proteingym
for A in ${ASSAYS//,/ }; do
    echo
    echo "############################################################"
    echo "### ASSAY: $A"
    echo "############################################################"
    # dms_batch 8 keeps the long assay's masked batch from dominating peak
    # memory; it does not change any score, only how many masked copies are
    # resident at once.
    python -u src/run_matrix.py \
        --model "$MODEL" \
        --configs "$CONFIGS" \
        --skip_throughput \
        --dms_csv "$D/$A.csv" \
        --dms_wt "$D/$A.wt.txt" \
        --dms_variant_col mutant \
        --dms_score_col DMS_score \
        --n_variants 100000 \
        --dms_batch 8 \
        --fidelity_n 4 \
        --n_seqs 4 \
        --out "results/dms_${MODEL}_${A}_${SLURM_JOB_ID}.json" \
        "$@"
done

echo
echo "=== all assays done: job $SLURM_JOB_ID ==="
