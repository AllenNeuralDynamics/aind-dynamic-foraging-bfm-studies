#!/bin/bash
#SBATCH --job-name=mevolve-gate4-truths
#SBATCH --partition=aind
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=200G
#SBATCH --time=01:00:00
# Gate 4, second half, step 1 (notes.md): write the hand-set generating parameters of SETTING for
# every template mouse, then simulate the generation, validation and held-out mice from them. Nothing
# is fitted. No API key is exported. Submitted from clean checkouts at pushed commits, with:
#   SETTING       a hand-set setting of make_truths.py (a, b or c)
#   VARIANT_DIR   this directory in the studies checkout at STUDIES_SHA
#   STUDIES_ROOT  that checkout's root (recorded by mevolve.gate4)
#   MEVOLVE_SRC   src/ of a MEvolve checkout at MEVOLVE_SHA
#   OUT           where the truths and simulated files go
#   STUDIES_SHA, MEVOLVE_SHA
set -euo pipefail
: "${SETTING:?}" "${VARIANT_DIR:?}" "${STUDIES_ROOT:?}" "${MEVOLVE_SRC:?}" "${OUT:?}" "${STUDIES_SHA:?}" "${MEVOLVE_SHA:?}"
PY=/allen/aind/scratch/han.hou/miniforge3/envs/mevolve-search/bin/python
export PYTHONPATH="$MEVOLVE_SRC"
G="$VARIANT_DIR/generator/two_timescale_ck.py"
COHORTS=/allen/aind/scratch/han.hou/mevolve/freeze/cohorts_draft_seed0.json   # = reference/cohorts_seed0.json (#104)
COMMON=(--freeze /allen/aind/scratch/han.hou/mevolve/freeze/20260603 --cohorts "$COHORTS"
        --studies "$STUDIES_ROOT" --out "$OUT" --program-path "$G" --truths "$OUT/truths.json" --simulate-only)
mkdir -p "$OUT"
cd "$TMPDIR"
"$PY" -c "import mevolve, sys; assert mevolve.__file__.startswith('$MEVOLVE_SRC'), mevolve.__file__"
echo "node $(hostname) cpus $(nproc) setting $SETTING studies $STUDIES_SHA mevolve $MEVOLVE_SHA"
"$PY" "$VARIANT_DIR/make_truths.py" --setting "$SETTING" --cohorts "$COHORTS" --out "$OUT/truths.json"
for spec in "generation 0 generation" "validation 0 validation" "validation 24 heldout"; do
  set -- $spec
  "$PY" -m mevolve.gate4 "${COMMON[@]}" --cohort "$1" --offset "$2" --tag "$3"
done
