#!/bin/bash
#SBATCH --job-name=mevolve-seeded-islands-cap12
#SBATCH --partition=aind
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=56
#SBATCH --mem-per-cpu=3500M
#SBATCH --time=06:00:00
# Study 40's cap test (notes.md): seeded-islands-smoke with program limits of 12 parameters and 8 latent
# variables. 16 slots of 56 CPUs (the CPUs of 8 nodes, placed wherever 56 are free): each evaluation is an
# srun step of 56 CPUs (28 workers, one per core), 16 at a time (MEvolve's eval_launch "srun"); the
# search, the embedding server and Claude run in the batch step on the first node. --mem-per-cpu lets two
# steps share a node. Pass DRY=1 for the $0 dry run (embedding server, seeds scored into seed_cache,
# runner built; no Claude call, no W&B).
# Submitted from clean checkouts at pushed commits, with these exported:
#   VARIANT_DIR   this directory in the studies checkout at STUDIES_SHA
#   MEVOLVE_SRC   src/ of a MEvolve checkout at MEVOLVE_SHA
#   LAUNCH_ID, LAUNCH_LABEL, STUDIES_SHA, MEVOLVE_SHA
# The API keys come from ~/.config/mevolve/ and are never printed; the evaluation script removes
# them before any candidate program runs.
set -euo pipefail
: "${VARIANT_DIR:?}" "${MEVOLVE_SRC:?}" "${LAUNCH_ID:?}" "${LAUNCH_LABEL:?}" "${STUDIES_SHA:?}" "${MEVOLVE_SHA:?}"
PY=/allen/aind/scratch/han.hou/miniforge3/envs/mevolve-search/bin/python
export STUDY_DIR="$(cd "$VARIANT_DIR/../.." && pwd)"
export PYTHONPATH="$MEVOLVE_SRC"
export HF_HOME=/allen/aind/scratch/han.hou/.cache/huggingface        # the embedding model, pinned in MEvolve
export WANDB_GROUP="seeded-islands-cap12@${LAUNCH_ID}"
export MEVOLVE_LAUNCH_DIR="/allen/aind/scratch/han.hou/mevolve/runs/seeded-islands-cap12/${LAUNCH_ID}"
export ANTHROPIC_API_KEY="$(cat ~/.config/mevolve/anthropic_api_key)"
export WANDB_API_KEY="$(cat ~/.config/mevolve/wandb_api_key)"
mkdir -p "$MEVOLVE_LAUNCH_DIR"
# after a crash or a time-out, keep what run.main did not copy; never overwrite the archived files
trap '[ -d "$TMPDIR/search" ] && cp -rn "$TMPDIR/search" "$MEVOLVE_LAUNCH_DIR/" || true' EXIT
cd "$TMPDIR"                                     # no .env here for Shinka to load
"$PY" -c "import mevolve, sys; assert mevolve.__file__.startswith('$MEVOLVE_SRC'), mevolve.__file__"
"$PY" - "$VARIANT_DIR/run.json" "$STUDY_DIR/analysis/seed_choice.json" <<'PYEOF'
import json, sys
run, choice = (json.load(open(p)) for p in sys.argv[1:])
assert run["seeds"] == choice["seeds"], f"run.json seeds {run['seeds']} != seed_choice.json {choice['seeds']}"
PYEOF
echo "node $(hostname) cpus $(nproc) launch $LAUNCH_ID studies $STUDIES_SHA mevolve $MEVOLVE_SHA"
if [ "${DRY:-0}" = 1 ]; then
  "$PY" -m mevolve.search.run --config "$VARIANT_DIR/run.json" --dry
  exit 0
fi
"$PY" -m mevolve.search.run --config "$VARIANT_DIR/run.json"     # writes the report and the archive itself
