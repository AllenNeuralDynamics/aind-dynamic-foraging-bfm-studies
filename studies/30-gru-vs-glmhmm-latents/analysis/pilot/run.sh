#!/bin/bash
# Usage: sbatch [opts] run.sh <script.py> [args]; runs a pilot stage on an aind compute node.
set -euo pipefail
export WRAPPER_ROOT=/home/han.hou/code/wt/wrapper-study30-pilot
export DISPATCHER_ROOT=/home/han.hou/code/wt/dispatcher-study30-pilot
export PYTHONPATH=$WRAPPER_ROOT/code
export JAX_PLATFORMS=cpu
export WANDB_MODE=disabled
cd /allen/aind/scratch/han.hou/pilots/study30-probe
echo "host $(hostname) start $(date)"
/allen/aind/scratch/han.hou/miniforge3/envs/dynamic-foraging-bfm-cpu/bin/python "$@"
echo "end $(date)"
