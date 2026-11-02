#!/bin/bash
# Study 12 / two-level-em: literature models as 2-level EM (session -> subject, Laplace-EM),
# scored by MEvolve's `mevolve.score` CLI at its reported settings (ADR 0004 defaults:
# --max-em 100, --n-draws 2048, --seed 0), on all 763 subjects (--pool all).
# One whole node; 56 workers = one per physical core (MEvolve's throughput test).
#
#   sbatch --export=ALL,MEVOLVE_ROOT=<clean MEvolve checkout at the pinned SHA>,OUT=<dir>,MODELS="<m1> <m2>" \
#          -J s12-2level -o <dir>/slurm-%j.out run_2level.sbatch
# As an array (one model per task, MODELS indexed by SLURM_ARRAY_TASK_ID):
#   sbatch --array=0-<n-1>%<k> ... -o <dir>/slurm-%A_%a.out run_2level.sbatch
#SBATCH -p aind
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --mem=200G
#SBATCH -t 24:00:00
#SBATCH --constraint=6238R
set -euo pipefail
: "${MEVOLVE_ROOT:?}" "${OUT:?}" "${MODELS:?}"
FREEZE=/allen/aind/scratch/han.hou/mevolve/freeze/20260603
PY=/allen/aind/scratch/han.hou/miniforge3/envs/mevolve/bin/python
mkdir -p "$OUT"
cd "$MEVOLVE_ROOT"
echo "mevolve $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) host=$(hostname) start=$(TZ=America/Los_Angeles date '+%F %T %Z')"
read -r -a all_models <<< "$MODELS"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then all_models=("${all_models[$SLURM_ARRAY_TASK_ID]}"); fi   # array: one model per task
args=(); for m in "${all_models[@]}"; do args+=(--model "$m"); done
echo "models: ${all_models[*]}"
PYTHONPATH=src "$PY" -m mevolve.score "${args[@]}" --pool all --freeze "$FREEZE" --out "$OUT" --workers 56
echo "end=$(TZ=America/Los_Angeles date '+%F %T %Z')"
