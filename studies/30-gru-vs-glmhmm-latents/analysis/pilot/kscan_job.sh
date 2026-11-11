#!/bin/bash
# Usage: sbatch kscan_job.sh "<fset> <emissions> <Ks>" ["<fset> <emissions> <Ks>" ...]
# Runs kscan_probes.py once per argument, in order; restart-safe (done units are skipped).
set -euo pipefail
for spec in "$@"; do
  /allen/aind/scratch/han.hou/pilots/study30-probe/scripts/run.sh /allen/aind/scratch/han.hou/pilots/study30-probe/scripts/kscan_probes.py $spec
done
