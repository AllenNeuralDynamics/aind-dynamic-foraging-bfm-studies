#!/bin/bash
# The HPC twin of launch.py's Beaker task: study 01's offline held-out fine-tune of the v1 D=614
# GRU for one seed (array index), from the source checkpoints fetched to $PRIOR_ROOT/s<seed>/run.
# Submitted by launch_hpc.py, which passes WRAPPER, MODELS, CONFIG, PRIOR_ROOT and OUT_ROOT.
#SBATCH --partition=aind
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
set -euo pipefail
SEED=$SLURM_ARRAY_TASK_ID
OUT="$OUT_ROOT/s$SEED"
mkdir -p "$OUT"
# study 01's config, with this node's paths for the source run and the outputs
python - "$CONFIG" "$PRIOR_ROOT/s$SEED/run" "$OUT/heldout" "$OUT/heldout.yaml" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg["source_run"]["model_dir"] = sys.argv[2]
cfg["output"]["output_root"] = sys.argv[3]
open(sys.argv[4], "w").write(yaml.safe_dump(cfg))
PY
cd "$WRAPPER/code"
export PYTHONPATH="$WRAPPER/code:$MODELS/src"
python -m run_analysis finetune --config "$OUT/heldout.yaml"
