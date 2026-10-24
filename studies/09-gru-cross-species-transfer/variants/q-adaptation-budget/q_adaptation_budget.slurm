#!/bin/bash
#SBATCH -J study09_qk
#SBATCH -p aind
#SBATCH -t 12:00:00
#SBATCH -c 24
#SBATCH --mem=64G
#SBATCH --array=0-7
#SBATCH --mail-type=FAIL,END

set -euo pipefail

source "${CONDA_SH:?CONDA_SH must point to conda.sh}"
conda activate dynamic-foraging-bfm-cpu 2>/dev/null || conda activate disrnn-cpu

DISPATCHER_ROOT="${DISPATCHER_ROOT:?DISPATCHER_ROOT is required}"
STUDIES_ROOT="${STUDIES_ROOT:?STUDIES_ROOT is required (an aind-dynamic-foraging-bfm-studies checkout)}"
WRAPPER_ROOT="${WRAPPER_ROOT:?WRAPPER_ROOT is required}"
MODELS_ROOT="${MODELS_ROOT:?MODELS_ROOT is required}"
RUN_ROOT="${BFM_OUTPUT_DIR:?BFM_OUTPUT_DIR is required}/study09-qk-${SLURM_ARRAY_JOB_ID}"
DATA_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/study09-external-${SLURM_ARRAY_TASK_ID}"
# dispatcher #166: one task per (dataset, budget). Q has no K=0 (zero-shot) condition, and
# every Chen (mouse) subject has exactly 4 adaptation sessions, so Chen has no K=8 and its
# K=4 is the full half (analysis/adaptation_budget_cohort.json).
CELLS=(grossman:1 grossman:2 grossman:4 grossman:8 grossman:half chen:1 chen:2 chen:half)
DATASET="${CELLS[$SLURM_ARRAY_TASK_ID]%%:*}"
BUDGET="${CELLS[$SLURM_ARRAY_TASK_ID]##*:}"
BUDGET_ARGS=()
if [[ "${BUDGET}" != half ]]; then BUDGET_ARGS=(--adapt-sessions-per-subject "${BUDGET}"); fi

export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-q-adaptation-budget@slurm-${SLURM_ARRAY_JOB_ID}}"
export BFM_META_VARIANT=q-adaptation-budget
export PYTHONPATH="${MODELS_ROOT}/src:${DISPATCHER_ROOT}/code:${WRAPPER_ROOT}/code:${PYTHONPATH:-}"
export BFM_WRAPPER_CODE="${WRAPPER_ROOT}/code"
export DISPATCHER_COMMIT
export STUDIES_COMMIT
export WRAPPER_COMMIT
export FORAGING_MODELS_COMMIT
DISPATCHER_COMMIT="$(git -C "${DISPATCHER_ROOT}" rev-parse HEAD)"
STUDIES_COMMIT="$(git -C "${STUDIES_ROOT}" rev-parse HEAD)"
WRAPPER_COMMIT="$(git -C "${WRAPPER_ROOT}" rev-parse HEAD)"
FORAGING_MODELS_COMMIT="$(git -C "${MODELS_ROOT}" rev-parse HEAD)"

python -m external_bandit_datasets.cli "${DATASET}" --cache-root "${DATA_ROOT}"
python "${STUDIES_ROOT}/studies/09-gru-cross-species-transfer/run_q_baseline.py" \
  --dataset "${DATASET}" \
  --data-root "${DATA_ROOT}/canonical" \
  --output-root "${RUN_ROOT}" \
  --subject-workers 12 \
  "${BUDGET_ARGS[@]}"
