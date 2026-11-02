#!/bin/bash
# r3 freeze as a SLURM array: one K-grid run per task. Each task fits its run into $R3_PARTS
# (refits, both labellings, every cut) and then scores that part (R3_SCORE=1).
# Then merge on a compute node (writes analysis/frozen_r3.json and the npz files):
#   srun --ntasks=1 -p aind --cpus-per-task=2 --mem=32G env R3_MERGE=1 R3_PARTS=... \
#       WRAPPER_ROOT=... DISPATCHER_ROOT=... make -C studies/14-glmhmm-vs-gru-heldout freeze-r3
#
#   sbatch --chdir=studies/14-glmhmm-vs-gru-heldout \
#       --export=ALL,R3_PARTS=<dir>,WRAPPER_ROOT=<wrapper>,DISPATCHER_ROOT=<dispatcher>,PY=<python> \
#       studies/14-glmhmm-vs-gru-heldout/analysis/freeze_r3.sbatch
#SBATCH --job-name=s14-freeze-r3
#SBATCH --partition=aind
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%A_%a.log
set -euo pipefail
EMISSIONS=(hurdle reference)
export R3_RUNS="${EMISSIONS[$((SLURM_ARRAY_TASK_ID / 10))]}:$((SLURM_ARRAY_TASK_ID % 10 + 1))"
echo "task $SLURM_ARRAY_TASK_ID: $R3_RUNS on $(hostname)"
"${PY:-python}" analysis/freeze_r3.py
R3_SCORE=1 "${PY:-python}" analysis/freeze_r3.py
