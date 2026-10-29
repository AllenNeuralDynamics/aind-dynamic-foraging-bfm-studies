#!/bin/bash
# r2 freeze as a SLURM array: one K-grid run per task, each written to $R2_PARTS as a part.
# Then merge on a compute node (writes analysis/frozen_r2.json and the weight files):
#   srun --ntasks=1 -p aind --cpus-per-task=2 --mem=16G env R2_MERGE=1 R2_PARTS=... \
#       WRAPPER_ROOT=... DISPATCHER_ROOT=... make -C studies/14-glmhmm-vs-gru-heldout freeze-r2
#
#   sbatch --chdir=studies/14-glmhmm-vs-gru-heldout \
#       --export=ALL,R2_PARTS=<dir>,WRAPPER_ROOT=<wrapper>,DISPATCHER_ROOT=<dispatcher>,PY=<python> \
#       studies/14-glmhmm-vs-gru-heldout/analysis/freeze_r2.sbatch
#SBATCH --job-name=s14-freeze-r2
#SBATCH --partition=aind
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%A_%a.log
set -euo pipefail
EMISSIONS=(hurdle reference)
export R2_RUNS="${EMISSIONS[$((SLURM_ARRAY_TASK_ID / 10))]}:$((SLURM_ARRAY_TASK_ID % 10 + 1))"
echo "task $SLURM_ARRAY_TASK_ID: $R2_RUNS on $(hostname)"
"${PY:-python}" analysis/freeze_r2.py
