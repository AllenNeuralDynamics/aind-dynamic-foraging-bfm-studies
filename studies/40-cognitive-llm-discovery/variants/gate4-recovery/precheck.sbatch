#!/bin/bash
#SBATCH --job-name=mevolve-gate4-precheck
#SBATCH --partition=aind
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=200G
#SBATCH --time=02:00:00
# Gate 4, second half, step 2 (notes.md): before any Claude call, score everything Claude is shown
# (the template and the three seeds; the library is off), the generating program without its slow
# trace, and the generating program, on the simulated mice with the search's settings, and apply
# the pre-registered rule. The scores land in the seed cache the search reuses. No API key is
# exported. Submitted from clean checkouts at pushed commits, with these exported:
#   VARIANT_DIR   this directory in the studies checkout at STUDIES_SHA
#   MEVOLVE_SRC   src/ of a MEvolve checkout at MEVOLVE_SHA
#   SIM_DIR       truths.sbatch's OUT: the simulated mice
#   STUDIES_SHA, MEVOLVE_SHA
set -euo pipefail
: "${VARIANT_DIR:?}" "${MEVOLVE_SRC:?}" "${SIM_DIR:?}" "${STUDIES_SHA:?}" "${MEVOLVE_SHA:?}"
PY=/allen/aind/scratch/han.hou/miniforge3/envs/mevolve-search/bin/python
export PYTHONPATH="$MEVOLVE_SRC" VARIANT_DIR SIM_DIR
export MEVOLVE_LAUNCH_DIR="$SIM_DIR" LAUNCH_ID=precheck LAUNCH_LABEL=gate4-precheck WANDB_GROUP=none   # run.json's variables; nothing logs to W&B
cd "$TMPDIR"
"$PY" -c "import mevolve, sys; assert mevolve.__file__.startswith('$MEVOLVE_SRC'), mevolve.__file__"
echo "node $(hostname) cpus $(nproc) studies $STUDIES_SHA mevolve $MEVOLVE_SHA sim $SIM_DIR"
"$PY" -m mevolve.search.precheck --config "$VARIANT_DIR/run.json" --target "$VARIANT_DIR/generator/two_timescale_ck.py" \
  --reference "$VARIANT_DIR/generator/one_timescale_ck.py" --out "$SIM_DIR/precheck.json"
