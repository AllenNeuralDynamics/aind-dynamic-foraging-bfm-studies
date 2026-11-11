#!/bin/bash
#SBATCH --job-name=mevolve-gate4-noseed
#SBATCH --partition=aind
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=200G
#SBATCH --time=12:00:00
# Gate 4 control (notes.md): 100 proposals on gate4-recovery's simulated mice with no seed programs,
# then the winner and the generating program scored once on the 24 held-out simulated mice.
# Submitted from clean checkouts at pushed commits, with these exported:
#   VARIANT_DIR   this directory in the studies checkout at STUDIES_SHA
#   MEVOLVE_SRC   src/ of a MEvolve checkout at MEVOLVE_SHA
#   LAUNCH_ID, LAUNCH_LABEL, STUDIES_SHA, MEVOLVE_SHA
# SIM_DIR is fixed below: gate4-recovery's simulated mice (setting a).
# The API keys come from ~/.config/mevolve/ and are never printed; the evaluation script removes
# them before any candidate program runs.
set -euo pipefail
: "${VARIANT_DIR:?}" "${MEVOLVE_SRC:?}" "${LAUNCH_ID:?}" "${LAUNCH_LABEL:?}" "${STUDIES_SHA:?}" "${MEVOLVE_SHA:?}"
PY=/allen/aind/scratch/han.hou/miniforge3/envs/mevolve-search/bin/python
export SIM_DIR=/allen/aind/scratch/han.hou/mevolve/gate4/ttck_a_1100041   # setting a: the first whose pre-check passed
HELDOUT="$SIM_DIR/simulated_two_timescale_ck_heldout.json.gz"
export PYTHONPATH="$MEVOLVE_SRC" VARIANT_DIR
export WANDB_GROUP="gate4-noseed@${LAUNCH_ID}"
export MEVOLVE_LAUNCH_DIR="/allen/aind/scratch/han.hou/mevolve/runs/gate4-noseed/${LAUNCH_ID}"
export ANTHROPIC_API_KEY="$(cat ~/.config/mevolve/anthropic_api_key)"
export WANDB_API_KEY="$(cat ~/.config/mevolve/wandb_api_key)"
mkdir -p "$MEVOLVE_LAUNCH_DIR"
trap '[ -d "$TMPDIR/search" ] && cp -r "$TMPDIR/search" "$MEVOLVE_LAUNCH_DIR/" || true' EXIT
cd "$TMPDIR"                                     # no .env here for Shinka to load
"$PY" -c "import mevolve, sys; assert mevolve.__file__.startswith('$MEVOLVE_SRC'), mevolve.__file__"
echo "node $(hostname) cpus $(nproc) launch $LAUNCH_ID studies $STUDIES_SHA mevolve $MEVOLVE_SHA"
status=0                                         # try every step, then fail if any did
search_ok=1
"$PY" -m mevolve.search.run --config "$VARIANT_DIR/run.json" || { echo "search failed"; status=1; search_ok=0; }
"$PY" -m mevolve.search.report "$MEVOLVE_LAUNCH_DIR/search" || { echo "report failed"; status=1; }
if [ "$search_ok" = 1 ]; then                    # the held-out mice are used once, on the finished search only
  "$PY" "$VARIANT_DIR/heldout.py" --run "$MEVOLVE_LAUNCH_DIR/search" --config "$VARIANT_DIR/run.json" \
    --heldout "$HELDOUT" --target "$VARIANT_DIR/generator/two_timescale_ck.py" --out "$MEVOLVE_LAUNCH_DIR/heldout" || { echo "held-out check failed"; status=1; }
fi
exit "$status"
