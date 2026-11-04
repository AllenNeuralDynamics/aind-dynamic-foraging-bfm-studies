#!/bin/bash
#SBATCH --job-name=mevolve-gate4-precheck
#SBATCH --partition=aind
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=200G
#SBATCH --time=03:00:00
# Gate 4, second half, step 2 (notes.md): before any Claude call, score the template and all 19
# library programs and the target on the simulated mice with the search's settings, and apply the pre-registered
# rule. The scores land in the seed cache the search reuses. No API key is exported.
# Submitted from clean checkouts at pushed commits, with these exported:
#   VARIANT_DIR   this directory in the studies checkout at STUDIES_SHA
#   MEVOLVE_SRC   src/ of a MEvolve checkout at MEVOLVE_SHA
#   OUT           where precheck.json goes
#   STUDIES_SHA, MEVOLVE_SHA
set -euo pipefail
: "${VARIANT_DIR:?}" "${MEVOLVE_SRC:?}" "${OUT:?}" "${STUDIES_SHA:?}" "${MEVOLVE_SHA:?}"
PY=/allen/aind/scratch/han.hou/miniforge3/envs/mevolve-search/bin/python
export PYTHONPATH="$MEVOLVE_SRC"
export MEVOLVE_LAUNCH_DIR="$OUT" LAUNCH_ID=precheck LAUNCH_LABEL=gate4-precheck WANDB_GROUP=none   # run.json's variables; nothing logs to W&B
mkdir -p "$OUT"
cd "$TMPDIR"
"$PY" -c "import mevolve, sys; assert mevolve.__file__.startswith('$MEVOLVE_SRC'), mevolve.__file__"
echo "node $(hostname) cpus $(nproc) studies $STUDIES_SHA mevolve $MEVOLVE_SHA"
"$PY" -m mevolve.search.precheck --config "$VARIANT_DIR/run.json" --target "$VARIANT_DIR/generator/hattori_satiety.py" --out "$OUT/precheck.json"
