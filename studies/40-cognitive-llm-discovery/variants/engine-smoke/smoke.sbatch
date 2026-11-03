#!/bin/bash
#SBATCH --job-name=mevolve-engine-smoke
#SBATCH --partition=aind
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=200G
#SBATCH --time=06:00:00
# Gate 3 engine smoke (notes.md): two 10-proposal ShinkaEvolve runs, effort medium then high.
# Submitted from clean checkouts at the launch commits, with these exported:
#   VARIANT_DIR   this directory in the studies checkout at STUDIES_SHA
#   MEVOLVE_SRC   src/ of a MEvolve checkout at MEVOLVE_SHA
#   LAUNCH_ID, LAUNCH_LABEL, STUDIES_SHA, MEVOLVE_SHA
# The API keys come from ~/.config/mevolve/ and are never printed; the evaluation script removes
# them before any candidate program runs.
set -euo pipefail
: "${VARIANT_DIR:?}" "${MEVOLVE_SRC:?}" "${LAUNCH_ID:?}" "${LAUNCH_LABEL:?}" "${STUDIES_SHA:?}" "${MEVOLVE_SHA:?}"
PY=/allen/aind/scratch/han.hou/miniforge3/envs/mevolve-search/bin/python
export PYTHONPATH="$MEVOLVE_SRC"                 # the launch commit, ahead of the env's editable install
export WANDB_GROUP="engine-smoke@${LAUNCH_ID}"
export MEVOLVE_LAUNCH_DIR="/allen/aind/scratch/han.hou/mevolve/runs/engine-smoke/${LAUNCH_ID}"
export ANTHROPIC_API_KEY="$(cat ~/.config/mevolve/anthropic_api_key)"
export WANDB_API_KEY="$(cat ~/.config/mevolve/wandb_api_key)"
mkdir -p "$MEVOLVE_LAUNCH_DIR"
trap 'for e in medium high; do [ -d "$TMPDIR/$e" ] && cp -r "$TMPDIR/$e" "$MEVOLVE_LAUNCH_DIR/" || true; done' EXIT
cd "$TMPDIR"                                     # no .env here for Shinka to load
"$PY" -c "import mevolve, sys; assert mevolve.__file__.startswith('$MEVOLVE_SRC'), mevolve.__file__"
echo "node $(hostname) cpus $(nproc) launch $LAUNCH_ID studies $STUDIES_SHA mevolve $MEVOLVE_SHA"
for effort in medium high; do
  "$PY" -m mevolve.search.run --config "$VARIANT_DIR/run_${effort}.json" || echo "run $effort failed"
done
"$PY" -m mevolve.search.report "$MEVOLVE_LAUNCH_DIR/medium" "$MEVOLVE_LAUNCH_DIR/high" \
  --out "$MEVOLVE_LAUNCH_DIR/comparison.md"
