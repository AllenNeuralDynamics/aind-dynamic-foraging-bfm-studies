#!/bin/bash
#SBATCH --job-name=mevolve-simplify
#SBATCH --partition=aind
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@120
# One simplification run of study 40 (notes.md; MEvolve ADR 0006): the driver job, as for the search
# (variants/seeded-islands/search.sbatch). Each evaluation is its own Slurm job of 24 CPUs. Before
# starting, run.json's floor and seed programs must equal analysis/simplify/floors.json. A resume is a
# new launch with RESUME_OF=<the first launch's LAUNCH_ID>; DRY=1 is the $0 dry run, REHEARSAL=1 runs 6
# real proposals (new calls stop at $2, no W&B) in a launch directory of its own.
# Submitted from clean checkouts at pushed commits, with these exported:
#   VARIANT_DIR   this directory in the studies checkout at STUDIES_SHA
#   MEVOLVE_SRC   src/ of a MEvolve checkout at MEVOLVE_SHA
#   LAUNCH_ID, LAUNCH_LABEL, STUDIES_SHA, MEVOLVE_SHA [, RESUME_OF]
# The API keys come from ~/.config/mevolve/ and are never printed; evaluation jobs never receive them.
set -euo pipefail
: "${VARIANT_DIR:?}" "${MEVOLVE_SRC:?}" "${LAUNCH_ID:?}" "${LAUNCH_LABEL:?}" "${STUDIES_SHA:?}" "${MEVOLVE_SHA:?}"
PY=/allen/aind/scratch/han.hou/miniforge3/envs/mevolve-search/bin/python
RUN_ID="${RESUME_OF:-$LAUNCH_ID}"
VARIANT="$(basename "$VARIANT_DIR")"
export STUDY_DIR="$(cd "$VARIANT_DIR/../.." && pwd)"
export PYTHONPATH="$MEVOLVE_SRC"
export HF_HOME=/allen/aind/scratch/han.hou/.cache/huggingface
export WANDB_GROUP="${VARIANT}@${RUN_ID}"
KIND="$VARIANT"; [ "${REHEARSAL:-0}" = 1 ] && KIND="${VARIANT}-rehearsal"
export MEVOLVE_LAUNCH_DIR="/allen/aind/scratch/han.hou/mevolve/runs/${KIND}/${RUN_ID}"
export ANTHROPIC_API_KEY="$(cat ~/.config/mevolve/anthropic_api_key)"
export WANDB_API_KEY="$(cat ~/.config/mevolve/wandb_api_key)"
mkdir -p "$MEVOLVE_LAUNCH_DIR"
trap '[ -d "$TMPDIR/search" ] && cp -rn "$TMPDIR/search" "$MEVOLVE_LAUNCH_DIR/" || true' EXIT
cd "$TMPDIR"                                     # no .env here for Shinka to load
"$PY" -c "import mevolve, sys; assert mevolve.__file__.startswith('$MEVOLVE_SRC'), mevolve.__file__"
CONFIG="$TMPDIR/run.json"
"$PY" - "$VARIANT_DIR/run.json" "$STUDY_DIR/analysis/simplify/floors.json" "$VARIANT" "$CONFIG" "${REHEARSAL:-0}" <<'PYEOF'
import json, sys
run, floors = (json.load(open(p)) for p in sys.argv[1:3])
variant, out, rehearsal = sys.argv[3], sys.argv[4], sys.argv[5] == "1"
want = floors["runs"][variant]
assert run["mode"] == "simplify" and run["floor"] == round(want["floor"], 6), (run["floor"], want["floor"])
assert run["seed_programs"] == want["seed_programs"], "seed programs differ from floors.json"
if rehearsal:
    run.update({"n_proposals": 6, "max_usd": 2.0, "wandb": None})
json.dump(run, open(out, "w"), indent=1)
PYEOF
echo "node $(hostname) variant $VARIANT launch $LAUNCH_ID run $RUN_ID studies $STUDIES_SHA mevolve $MEVOLVE_SHA"
if [ "${DRY:-0}" = 1 ]; then
  "$PY" -m mevolve.search.run --config "$CONFIG" --dry
  exit 0
fi
"$PY" -m mevolve.search.run --config "$CONFIG" ${RESUME_OF:+--resume} &
SEARCH=$!
trap 'kill -TERM $SEARCH 2>/dev/null; wait $SEARCH' TERM
wait $SEARCH
