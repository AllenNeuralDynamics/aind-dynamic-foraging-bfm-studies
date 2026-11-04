#!/bin/bash
#SBATCH --job-name=mevolve-gate4-search
#SBATCH --partition=aind
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=200G
#SBATCH --time=12:00:00
# Gate 4, second half, step 2 (notes.md): 100 proposals on the simulated mice, then the winner and
# Hattori2019 scored once on the 24 held-out simulated mice. Run only after the pre-check passed and
# its exclusions are in run.json. Submitted from clean checkouts at pushed commits, with these exported:
#   VARIANT_DIR   this directory in the studies checkout at STUDIES_SHA
#   MEVOLVE_SRC   src/ of a MEvolve checkout at MEVOLVE_SHA
#   LAUNCH_ID, LAUNCH_LABEL, STUDIES_SHA, MEVOLVE_SHA
# The API keys come from ~/.config/mevolve/ and are never printed; the evaluation script removes
# them before any candidate program runs.
set -euo pipefail
: "${VARIANT_DIR:?}" "${MEVOLVE_SRC:?}" "${LAUNCH_ID:?}" "${LAUNCH_LABEL:?}" "${STUDIES_SHA:?}" "${MEVOLVE_SHA:?}"
PY=/allen/aind/scratch/han.hou/miniforge3/envs/mevolve-search/bin/python
HELDOUT=/allen/aind/scratch/han.hou/mevolve/gate4/second_half_cee79da/simulated_Hattori2019_heldout.json.gz
export PYTHONPATH="$MEVOLVE_SRC"
export WANDB_GROUP="gate4-recovery@${LAUNCH_ID}"
export MEVOLVE_LAUNCH_DIR="/allen/aind/scratch/han.hou/mevolve/runs/gate4-recovery/${LAUNCH_ID}"
export ANTHROPIC_API_KEY="$(cat ~/.config/mevolve/anthropic_api_key)"
export WANDB_API_KEY="$(cat ~/.config/mevolve/wandb_api_key)"
mkdir -p "$MEVOLVE_LAUNCH_DIR"
trap '[ -d "$TMPDIR/search" ] && cp -r "$TMPDIR/search" "$MEVOLVE_LAUNCH_DIR/" || true' EXIT
cd "$TMPDIR"                                     # no .env here for Shinka to load
"$PY" -c "import mevolve, sys; assert mevolve.__file__.startswith('$MEVOLVE_SRC'), mevolve.__file__"
echo "node $(hostname) cpus $(nproc) launch $LAUNCH_ID studies $STUDIES_SHA mevolve $MEVOLVE_SHA"
status=0                                         # try every step, then fail if any did
"$PY" -m mevolve.search.run --config "$VARIANT_DIR/run.json" || { echo "search failed"; status=1; }
"$PY" -m mevolve.search.report "$MEVOLVE_LAUNCH_DIR/search" || { echo "report failed"; status=1; }
"$PY" "$VARIANT_DIR/heldout.py" --run "$MEVOLVE_LAUNCH_DIR/search" --config "$VARIANT_DIR/run.json" \
  --heldout "$HELDOUT" --target Hattori2019 --out "$MEVOLVE_LAUNCH_DIR/heldout" || { echo "held-out check failed"; status=1; }
exit "$status"
