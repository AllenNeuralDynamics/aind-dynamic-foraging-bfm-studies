#!/bin/bash
#SBATCH --job-name=mevolve-seeded-islands
#SBATCH --partition=aind
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=1500M
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:TERM@120
# Study 40's main search (notes.md). Slots of 24 CPUs placed wherever 24 are free on CPU nodes: each
# evaluation is an srun step of 24 CPUs (12 workers, one per core); the search, the embedding server and
# Claude run in the batch step. The number of slots is chosen at launch from the free-CPU landscape so the
# job ends earliest (sbatch --ntasks=N overrides the 16 above); this script runs N evaluations and N/2
# proposals at a time. 1500M per CPU covers a 12-worker evaluation.
# The search snapshots its database to $MEVOLVE_LAUNCH_DIR/snapshot every 10 minutes; at the time limit
# Slurm signals this script 120 s early and the search takes a last snapshot. A resume is a new launch
# (its own LAUNCH_ID and label) with RESUME_OF=<the first launch's LAUNCH_ID>: same launch directory, same
# W&B group, `--resume`. Pass DRY=1 for the $0 dry run, or FAKE=1 for a $0 rehearsal of this script on real
# mice: canned programs instead of Claude, 6 proposals, no W&B, its own launch directory.
# Submitted from clean checkouts at pushed commits, with these exported:
#   VARIANT_DIR   this directory in the studies checkout at STUDIES_SHA
#   MEVOLVE_SRC   src/ of a MEvolve checkout at MEVOLVE_SHA
#   LAUNCH_ID, LAUNCH_LABEL, STUDIES_SHA, MEVOLVE_SHA [, RESUME_OF]
# The API keys come from ~/.config/mevolve/ and are never printed; the evaluation script removes them
# before any candidate program runs, and evaluation workers cannot read files outside Python's paths.
set -euo pipefail
: "${VARIANT_DIR:?}" "${MEVOLVE_SRC:?}" "${LAUNCH_ID:?}" "${LAUNCH_LABEL:?}" "${STUDIES_SHA:?}" "${MEVOLVE_SHA:?}"
PY=/allen/aind/scratch/han.hou/miniforge3/envs/mevolve-search/bin/python
RUN_ID="${RESUME_OF:-$LAUNCH_ID}"
export STUDY_DIR="$(cd "$VARIANT_DIR/../.." && pwd)"
export PYTHONPATH="$MEVOLVE_SRC"
export HF_HOME=/allen/aind/scratch/han.hou/.cache/huggingface        # the embedding model, pinned in MEvolve
export WANDB_GROUP="seeded-islands@${RUN_ID}"
KIND=seeded-islands; [ "${FAKE:-0}" = 1 ] && KIND=seeded-islands-rehearsal
export MEVOLVE_LAUNCH_DIR="/allen/aind/scratch/han.hou/mevolve/runs/${KIND}/${RUN_ID}"
[ "${FAKE:-0}" = 1 ] || export ANTHROPIC_API_KEY="$(cat ~/.config/mevolve/anthropic_api_key)"
export WANDB_API_KEY="$(cat ~/.config/mevolve/wandb_api_key)"
mkdir -p "$MEVOLVE_LAUNCH_DIR"
# after a crash or a time-out, keep what run.main did not copy; never overwrite the archived files
trap '[ -d "$TMPDIR/search" ] && cp -rn "$TMPDIR/search" "$MEVOLVE_LAUNCH_DIR/" || true' EXIT
cd "$TMPDIR"                                     # no .env here for Shinka to load
"$PY" -c "import mevolve, sys; assert mevolve.__file__.startswith('$MEVOLVE_SRC'), mevolve.__file__"
CONFIG="$TMPDIR/run.json"                        # run.json with this job's slot count
"$PY" - "$VARIANT_DIR/run.json" "$STUDY_DIR/analysis/seed_choice.json" "$SLURM_NTASKS" "$CONFIG" "${FAKE:-0}" <<'PYEOF'
import json, sys
run, choice = (json.load(open(p)) for p in sys.argv[1:3])
slots, out, fake = int(sys.argv[3]), sys.argv[4], sys.argv[5] == "1"
assert run["seeds"] == choice["seeds"], f"run.json seeds {run['seeds']} != seed_choice.json {choice['seeds']}"
run.update({"max_evaluation_jobs": slots, "max_proposal_jobs": max(1, slots // 2)})
if fake:        # the $0 rehearsal: the same config, a few canned proposals, nothing logged
    run.update({"n_proposals": 6, "max_usd": 0.1, "patch_types": ["full"], "wandb": None})
json.dump(run, open(out, "w"), indent=1)
PYEOF
echo "node $(hostname) slots $SLURM_NTASKS launch $LAUNCH_ID run $RUN_ID studies $STUDIES_SHA mevolve $MEVOLVE_SHA"
if [ "${DRY:-0}" = 1 ]; then
  "$PY" -m mevolve.search.run --config "$CONFIG" --dry
  exit 0
fi
# the search runs in the background so the TERM before the time limit reaches it (a last snapshot)
"$PY" -m mevolve.search.run --config "$CONFIG" ${RESUME_OF:+--resume} $([ "${FAKE:-0}" = 1 ] && echo --fake-llm) &
SEARCH=$!
trap 'kill -TERM $SEARCH 2>/dev/null; wait $SEARCH' TERM
wait $SEARCH
