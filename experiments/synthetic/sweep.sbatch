#!/bin/bash
# The synthetic benchmark with learned encoders: latent dimensions N = 4, 6, ..., 14
# times seeds 0..19, one GPU per run. Submit from the repository root, naming a GPU
# partition of your cluster:
#
#   sbatch --partition=<gpu-partition> experiments/synthetic/sweep.sbatch
#
# Environment (optional): PYTHON, the interpreter with requirements.txt installed
# (default: python3); OUT_DIR, where the JSON records go (default: results/synthetic).
# When all tasks are done, from the repository root (add --results "$OUT_DIR" to
# aggregate.py if you changed OUT_DIR):
#
#   python experiments/synthetic/aggregate.py && python experiments/synthetic/plot.py
#
#SBATCH --job-name=synthetic
#SBATCH --array=0-119
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-synthetic-%A_%a.out

set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUT_DIR="${OUT_DIR:-results/synthetic}"
DIMS=(4 6 8 10 12 14)
N_SEEDS=20

TASK="${SLURM_ARRAY_TASK_ID}"
N="${DIMS[$((TASK / N_SEEDS))]}"
SEED=$((TASK % N_SEEDS))

echo "task=${TASK} N=${N} seed=${SEED}"
"${PYTHON}" -u experiments/synthetic/run.py --N "${N}" --seed "${SEED}" --out "${OUT_DIR}"
