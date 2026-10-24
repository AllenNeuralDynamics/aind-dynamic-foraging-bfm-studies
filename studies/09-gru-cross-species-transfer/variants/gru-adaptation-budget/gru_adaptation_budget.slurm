#!/bin/bash
#SBATCH -J study09_gruk
#SBATCH -p aind
#SBATCH -t 02:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-29
#SBATCH --mail-type=FAIL,END

# HPC route for the gru-adaptation-budget grid (dispatcher #166), used when every Beaker
# hub pool is saturated. One task per (dataset, K, source cell); the same runner, source
# manifest and budgets as the sweep_*.yaml grids. All tasks on A100 so numerics are uniform.
set -euo pipefail

source "${CONDA_SH:?CONDA_SH must point to conda.sh}"
conda activate dynamic-foraging-bfm-gpu

DISPATCHER_ROOT="${DISPATCHER_ROOT:?DISPATCHER_ROOT is required}"
STUDIES_ROOT="${STUDIES_ROOT:?STUDIES_ROOT is required (an aind-dynamic-foraging-bfm-studies checkout)}"
WRAPPER_ROOT="${WRAPPER_ROOT:?WRAPPER_ROOT is required}"
MODELS_ROOT="${MODELS_ROOT:?MODELS_ROOT is required}"
# a --target install of the pinned pyarrow-21.0.0 wheel (gru_transfer checks the version;
# the env itself is not modified)
PYARROW_DIR="${PYARROW_DIR:?PYARROW_DIR is required}"
RUN_ROOT="${BFM_OUTPUT_DIR:?BFM_OUTPUT_DIR is required}/study09-gruk-${SLURM_ARRAY_JOB_ID}"
DATA_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/study09-external-gruk-${SLURM_ARRAY_TASK_ID}"

CELLS=()
for k in 0 1 2 4 8; do for s in 0 1 2; do CELLS+=("grossman:${k}:d614-s${s}"); done; done
for k in 0 1 2; do for s in 0 1 2; do CELLS+=("chen:${k}:d614-s${s}"); done; done
for d in grossman chen; do for s in 0 1 2; do CELLS+=("${d}:half:d614-s${s}"); done; done
IFS=: read -r DATASET BUDGET SOURCE_KEY <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
BUDGET_ARGS=()
if [[ "${BUDGET}" != half ]]; then BUDGET_ARGS=(--adapt-sessions-per-subject "${BUDGET}"); fi

export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:?WANDB_RUN_GROUP is required}"
# the runner reads W&B over GraphQL with WANDB_API_KEY (a Beaker secret there); on HPC the
# key lives in ~/.netrc, read here rather than passed on the sbatch command line
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY="$(awk '/api.wandb.ai/{f=1} f&&/password/{print $2; exit}' ~/.netrc)"
  export WANDB_API_KEY
fi
export BFM_META_STUDY=09-gru-cross-species-transfer
export BFM_META_VARIANT=gru-adaptation-budget
export BFM_META_LAUNCH_ID="${WANDB_RUN_GROUP#*@}"
export PYTHONPATH="${PYARROW_DIR}:${MODELS_ROOT}/src:${DISPATCHER_ROOT}/code:${WRAPPER_ROOT}/code:${PYTHONPATH:-}"
export DISPATCHER_COMMIT
export STUDIES_COMMIT
export WRAPPER_COMMIT
export FORAGING_MODELS_COMMIT
DISPATCHER_COMMIT="$(git -C "${DISPATCHER_ROOT}" rev-parse HEAD)"
STUDIES_COMMIT="$(git -C "${STUDIES_ROOT}" rev-parse HEAD)"
WRAPPER_COMMIT="$(git -C "${WRAPPER_ROOT}" rev-parse HEAD)"
FORAGING_MODELS_COMMIT="$(git -C "${MODELS_ROOT}" rev-parse HEAD)"

# CANONICAL_ROOT: a canonical/ directory built once by the same CLI, read-only here. Per-task
# downloads of the release hit the host's rate limit (HTTP 429) when many tasks start at once.
if [[ -n "${CANONICAL_ROOT:-}" ]]; then
  CANONICAL="${CANONICAL_ROOT}"
else
  python -m external_bandit_datasets.cli "${DATASET}" --cache-root "${DATA_ROOT}"
  CANONICAL="${DATA_ROOT}/canonical"
fi
cd "${WRAPPER_ROOT}/code"
python -m external_transfer.gru_transfer \
  --dataset "${DATASET}" \
  --source-key "${SOURCE_KEY}" \
  --source-manifest "${STUDIES_ROOT}/studies/09-gru-cross-species-transfer/source_runs.json" \
  --data-root "${CANONICAL}" \
  --output-root "${RUN_ROOT}/${DATASET}-${BUDGET}-${SOURCE_KEY}" \
  "${BUDGET_ARGS[@]}"
