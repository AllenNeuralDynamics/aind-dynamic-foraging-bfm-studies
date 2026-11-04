#!/bin/bash
#SBATCH --job-name=mevolve-gate4-truths
#SBATCH --partition=aind
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=200G
#SBATCH --time=06:00:00
# Gate 4, second half, step 1 (notes.md): fit the generating program to the real template mice
# (2-level EM, at most 100 iterations), then simulate the generation, validation and held-out mice
# from those fits. No API key is exported. Submitted from clean checkouts at pushed commits, with:
#   VARIANT_DIR   this directory in the studies checkout at STUDIES_SHA
#   STUDIES_ROOT  that checkout's root (recorded by mevolve.gate4)
#   MEVOLVE_SRC   src/ of a MEvolve checkout at MEVOLVE_SHA
#   OUT           where the truths and simulated files go
#   STUDIES_SHA, MEVOLVE_SHA
set -euo pipefail
: "${VARIANT_DIR:?}" "${STUDIES_ROOT:?}" "${MEVOLVE_SRC:?}" "${OUT:?}" "${STUDIES_SHA:?}" "${MEVOLVE_SHA:?}"
PY=/allen/aind/scratch/han.hou/miniforge3/envs/mevolve-search/bin/python
export PYTHONPATH="$MEVOLVE_SRC"
G="$VARIANT_DIR/generator/hattori_satiety.py"
COMMON=(--freeze /allen/aind/scratch/han.hou/mevolve/freeze/20260603
        --cohorts /allen/aind/scratch/han.hou/mevolve/freeze/cohorts_draft_seed0.json   # = reference/cohorts_seed0.json (#104)
        --studies "$STUDIES_ROOT" --out "$OUT" --program-path "$G" --workers 56)
mkdir -p "$OUT"
cd "$TMPDIR"
"$PY" -c "import mevolve, sys; assert mevolve.__file__.startswith('$MEVOLVE_SRC'), mevolve.__file__"
echo "node $(hostname) cpus $(nproc) studies $STUDIES_SHA mevolve $MEVOLVE_SHA"
for spec in "generation 0 generation" "validation 0 validation" "validation 24 heldout"; do
  set -- $spec
  "$PY" -m mevolve.gate4 "${COMMON[@]}" --cohort "$1" --offset "$2" --tag "$3" --fit-truths
  "$PY" -m mevolve.gate4 "${COMMON[@]}" --cohort "$1" --offset "$2" --tag "$3" --simulate-only \
    --truths "$OUT/truths_hattori_satiety_$3.json"
done
