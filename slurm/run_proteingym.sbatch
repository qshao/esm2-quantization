#!/bin/bash
#SBATCH --job-name=pgym
#SBATCH --account=gol_qsh226_uksr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm-%A_%a.out
#SBATCH --error=results/slurm-%A_%a.out
#
# The full ProteinGym substitution benchmark: 201 assays (of 217; the rest
# exceed ESM-2's 1022-residue context), 2.41M variants, 40,775 masked positions.
#
#   sbatch --array=0-5 -p H8V141_SAP112M2000_L --mem=200G slurm/run_proteingym.sbatch 3B
#   sbatch --array=0-5 -p A2V80_ICE56M256_L    --mem=120G slurm/run_proteingym.sbatch 650M
#
# An optional second argument overrides the config list, for the W8A8 repairs
# that are not part of the six-config sweep. Size the array to match it:
#
#   sbatch --array=0-1 -p H8V141_SAP112M2000_L --mem=200G \
#       slurm/run_proteingym.sbatch 3B int8_dyn_asym,int8_dyn_sq
#
# One array task per quantization config, and the tasks are genuinely
# independent: each writes only its own per-variant scores, and every
# cross-config comparison (rho vs fp32, paired bootstrap) is deferred to
# aggregate_proteingym.py. So no task waits on fp32, a single config can be
# re-run alone, and one config dying costs only that config.
#
# Resumable by design -- output appends to results/proteingym/{model}_{config}
# .jsonl.gz and a restart skips assays already present. Resubmitting the same
# array after a timeout picks up where it stopped rather than starting over.
#
# Not a --dms_compiled run: see bench.bench_dms. Assays here span 37-934
# residues and every one is a distinct input shape, so max-autotune would pay
# its per-shape cost 201 times over and measure compilation, not inference.

set -euo pipefail

MODEL="${1:-650M}"
CONFIGS=(fp32 bf16 int8_wo int8_dyn fp8_dyn int4_wo)
# `shift 1` fails under `set -e` when no args were passed, and `|| true` would
# leave the arg in "$@" to be appended to the python command as a stray
# positional. Shift only what is actually there.
shift $(( $# > 1 ? 1 : $# ))

# A second positional argument replaces the config list. Anything starting with
# `-` is a flag for run_proteingym.py, not a config list, so it is left in "$@"
# -- otherwise `... 3B --dms_batch 4` would silently be read as one config named
# "--dms_batch" and score nothing.
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
    IFS=',' read -r -a CONFIGS <<< "$1"
    shift
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "${#CONFIGS[@]}" ]; then
    echo "array index $IDX exceeds ${#CONFIGS[@]} configs"; exit 1
fi
CFG="${CONFIGS[$IDX]}"

cd /scratch/qsh226/ESM2-3B
source env/activate.sh
mkdir -p results/proteingym

echo "=== node $(hostname) | model $MODEL | config $CFG | array task $IDX ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, '| cc', torch.cuda.get_device_capability())"
echo

# dms_batch 8: caps how many masked copies are resident at once, so peak memory
# stays comparable across the 37-934 residue range instead of being set by
# whichever assay happens to be longest. It changes no score.
python -u src/run_proteingym.py \
    --model "$MODEL" \
    --config "$CFG" \
    --max_len 1022 \
    --dms_batch 8 \
    --out_dir results/proteingym \
    "$@"

echo "=== done: $MODEL/$CFG (job $SLURM_JOB_ID) ==="
