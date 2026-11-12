#!/bin/bash
# Usage: sbatch shuffle_chain.sh "<fset> <null> <K> <seeds>" [...]; runs 99_shuffle_null.py per spec, skipping done units.
set -euo pipefail
for spec in "$@"; do
  /allen/aind/scratch/han.hou/pilots/study30-probe/scripts/run.sh /allen/aind/scratch/han.hou/pilots/study30-probe/scripts/99_shuffle_null.py $spec
done
