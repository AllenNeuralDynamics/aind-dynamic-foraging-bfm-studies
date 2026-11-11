#!/bin/bash
#SBATCH --job-name=mevolve-seeded-islands
#SBATCH --partition=aind
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:TERM@120
# Study 40's main search (notes.md). This job is the driver: the search, the embedding server and Claude
# (about 10 GB). Each evaluation is its own Slurm job of 24 CPUs (12 workers, one per core; eval_launch
# "job" in run.json), so the search grows and shrinks with the CPUs free on the partition: at most
# max_evaluation_jobs evaluations and max_proposal_jobs proposals at a time.
# The search snapshots its database to $MEVOLVE_LAUNCH_DIR/snapshot every 10 minutes; at the time limit
# Slurm signals this script 120 s early and the search takes a last snapshot. A resume is a new launch
# (its own LAUNCH_ID and label) with RESUME_OF=<the first launch's LAUNCH_ID>: same launch directory, same
# W&B group, `--resume`. Pass DRY=1 for the $0 dry run, or REHEARSAL=1 for a rehearsal of this script on
# real mice with real Claude calls: 6 proposals (about $1; no call starts once $2 is spent), no W&B, its own
# launch directory.
# Submitted from clean checkouts at pushed commits, with these exported:
#   VARIANT_DIR   this directory in the studies checkout at STUDIES_SHA
#   MEVOLVE_SRC   src/ of a MEvolve checkout at MEVOLVE_SHA
#   LAUNCH_ID, LAUNCH_LABEL, STUDIES_SHA, MEVOLVE_SHA [, RESUME_OF]
# The API keys come from ~/.config/mevolve/ and are never printed; evaluation jobs never receive them, and
# evaluation workers cannot read files outside Python's paths.
set -euo pipefail
: "${VARIANT_DIR:?}" "${MEVOLVE_SRC:?}" "${LAUNCH_ID:?}" "${LAUNCH_LABEL:?}" "${STUDIES_SHA:?}" "${MEVOLVE_SHA:?}"
PY=/allen/aind/scratch/han.hou/miniforge3/envs/mevolve-search/bin/python
RUN_ID="${RESUME_OF:-$LAUNCH_ID}"
export STUDY_DIR="$(cd "$VARIANT_DIR/../.." && pwd)"
export PYTHONPATH="$MEVOLVE_SRC"
export HF_HOME=/allen/aind/scratch/han.hou/.cache/huggingface        # the embedding model, pinned in MEvolve
export WANDB_GROUP="seeded-islands@${RUN_ID}"
KIND=seeded-islands; [ "${REHEARSAL:-0}" = 1 ] && KIND=seeded-islands-rehearsal
export MEVOLVE_LAUNCH_DIR="/allen/aind/scratch/han.hou/mevolve/runs/${KIND}/${RUN_ID}"
export ANTHROPIC_API_KEY="$(cat ~/.config/mevolve/anthropic_api_key)"
export WANDB_API_KEY="$(cat ~/.config/mevolve/wandb_api_key)"
mkdir -p "$MEVOLVE_LAUNCH_DIR"
# after a crash or a time-out, keep what run.main did not copy; never overwrite the archived files
trap '[ -d "$TMPDIR/search" ] && cp -rn "$TMPDIR/search" "$MEVOLVE_LAUNCH_DIR/" || true' EXIT
cd "$TMPDIR"                                     # no .env here for Shinka to load
"$PY" -c "import mevolve, sys; assert mevolve.__file__.startswith('$MEVOLVE_SRC'), mevolve.__file__"
CONFIG="$TMPDIR/run.json"                        # run.json, or the rehearsal's version of it
"$PY" - "$VARIANT_DIR/run.json" "$STUDY_DIR/analysis/seed_choice.json" "$CONFIG" "${REHEARSAL:-0}" <<'PYEOF'
import json, sys
run, choice = (json.load(open(p)) for p in sys.argv[1:3])
out, rehearsal = sys.argv[3], sys.argv[4] == "1"
assert run["seeds"] == choice["seeds"], f"run.json seeds {run['seeds']} != seed_choice.json {choice['seeds']}"
if rehearsal:   # the same config with real Claude calls: a few proposals, new calls stop at $2, nothing logged
    run.update({"n_proposals": 6, "max_usd": 2.0, "wandb": None})
json.dump(run, open(out, "w"), indent=1)
PYEOF
echo "node $(hostname) launch $LAUNCH_ID run $RUN_ID studies $STUDIES_SHA mevolve $MEVOLVE_SHA"
if [ "${DRY:-0}" = 1 ]; then
  "$PY" -m mevolve.search.run --config "$CONFIG" --dry
  exit 0
fi
# the search runs in the background so the TERM before the time limit reaches it (a last snapshot)
"$PY" -m mevolve.search.run --config "$CONFIG" ${RESUME_OF:+--resume} &
SEARCH=$!
trap 'kill -TERM $SEARCH 2>/dev/null; wait $SEARCH' TERM
wait $SEARCH
