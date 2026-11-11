#!/bin/bash
# The held-out fine-tune of study 02's H256 / D614 / seed 0 3-way GRU, redone offline (notes.md).
# Step 1 assembles the source run's model_dir from its W&B artifact (read only, pinned version),
# step 2 runs the wrapper's fine-tune with heldout_offline.yaml. Submitted by launch_hpc.py, which
# passes WRAPPER, MODELS, CONFIG, OUT_ROOT, CPU_ENV and WANDB_ENV.
#SBATCH --partition=aind
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=16:00:00
set -euo pipefail
SOURCE_RUN=nxd-3way-20260707-075401-80a6d102
SOURCE_ARTIFACT_VERSION=v0          # the version r1 read (digest 585e7419016aa3399e3b58406b1a94db)
mkdir -p "$OUT_ROOT"

# 1) model_dir = <OUT_ROOT>/prior/run: checkpoints from the artifact, inputs.yaml from the run config.
#    The shared cpu env's wandb cannot read artifacts, so this step uses WANDB_ENV.
"$WANDB_ENV/bin/python" - "$WRAPPER/code/post_training_analysis" "$SOURCE_RUN" "$SOURCE_ARTIFACT_VERSION" \
    "$OUT_ROOT/prior" <<'PY'
import hashlib, json, netrc, os, sys
os.environ["WANDB_API_KEY"] = netrc.netrc().authenticators("api.wandb.ai")[2]   # read only
sys.path.insert(0, sys.argv[1])
from wandb_model_dir import hydrate_model_dir
run_dir = hydrate_model_dir(sys.argv[2], project="mice_ignore_scaling", entity="AIND-disRNN",
                            dest=sys.argv[4], version=sys.argv[3])
params = run_dir / "outputs" / "checkpoints" / "step_90000" / "params.json"
json.dump({"model_dir": str(run_dir), "artifact_version": sys.argv[3],
           "step_90000_params_sha256": hashlib.sha256(params.read_bytes()).hexdigest()},
          open(f"{sys.argv[4]}/prior.json", "w"), indent=1)
PY

# 2) the fine-tune, with this launch's paths for the source run and the outputs
"$CPU_ENV/bin/python" - "$CONFIG" "$OUT_ROOT/prior/run" "$OUT_ROOT/heldout" "$OUT_ROOT/heldout.yaml" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg["source_run"]["model_dir"] = sys.argv[2]
cfg["output"]["output_root"] = sys.argv[3]
open(sys.argv[4], "w").write(yaml.safe_dump(cfg))
PY
cd "$WRAPPER/code"
export PYTHONPATH="$WRAPPER/code:$MODELS/src"
"$CPU_ENV/bin/python" -m run_analysis finetune --config "$OUT_ROOT/heldout.yaml"
