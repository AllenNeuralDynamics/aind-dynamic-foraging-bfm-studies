#!/bin/bash
# Variant scoring: one array task = one row of jobs.csv (model x pass x session-lambda mode).
# Runs the wrapper's held-out re-scorer (post_training_analysis.heldout_rescore, wrapper PR #128) on
# CPU: model 0's check run took 56 CPUs, ~65 GB and ~3.5 h. Submit from the studies checkout:
#   sbatch --array=0-<n-1>%39 --export=ALL,WRAPPER_ROOT=<wrapper@sha>,S17_GROUP=scoring@<launch_id> \
#     studies/17-gru-crossfit-predictability/variants/scoring/score.slurm
# (%39: the aind QOS allows 2200 CPUs per user.)
#SBATCH -J s17score
#SBATCH -p aind
#SBATCH --constraint=cpu
#SBATCH -c 56
#SBATCH --mem=120G
#SBATCH -t 10:00:00
#SBATCH -o /home/han.hou/logfile/s17score_%A_%a.out
#SBATCH -e /home/han.hou/logfile/s17score_%A_%a.err
set -euo pipefail
: "${WRAPPER_ROOT:?}" "${S17_GROUP:?}" "${S17_SCORING:?}" "${S17_JOBS:?}" "${S17_ROLES:?}"
PY=/allen/aind/scratch/han.hou/miniforge3/envs/dynamic-foraging-bfm-cpu/bin/python
row=$($PY - "$S17_JOBS" "$SLURM_ARRAY_TASK_ID" <<'EOF'
import csv, sys
r = [x for x in csv.DictReader(open(sys.argv[1])) if x["job"] == sys.argv[2]][0]
print(r["name"], r["pass"], r["session_lambda"], r["model_dir"], r["subjects"], r["out"])
EOF
)
read -r NAME PASS LAM MODEL_DIR SUBJECTS OUT <<<"$row"
SWAP=""; [ "$PASS" = "B" ] && SWAP="--swap"
export WANDB_API_KEY=$(cat ~/.config/wandb/api_key.NeuralDynamics)
export WANDB_DIR=$S17_SCORING/wandb; mkdir -p "$WANDB_DIR"
cd "$WRAPPER_ROOT/code"
export PYTHONPATH="$WRAPPER_ROOT/code:${PYTHONPATH:-}"
echo "[s17score] task $SLURM_ARRAY_TASK_ID: $NAME pass $PASS lambda $LAM -> $OUT"
$PY -m post_training_analysis.heldout_rescore \
  --model-dir "$MODEL_DIR" --subjects "$SUBJECTS" --roles "$S17_ROLES" $SWAP \
  --checkpoint best_eval --session-lambda "$LAM" \
  --output-dir "$S17_SCORING/out/$OUT" \
  --wandb-entity NeuralDynamics --wandb-project dfbfm_17-gru-crossfit-predictability \
  --wandb-group "$S17_GROUP" --wandb-name "$NAME-pass$PASS-lam$LAM"
