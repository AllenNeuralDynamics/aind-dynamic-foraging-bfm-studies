#!/bin/bash
#SBATCH -J s13_rescue
#SBATCH -p aind
#SBATCH -t 05:00:00
#SBATCH -c 2
#SBATCH --mem=200G
#SBATCH --array=0-15
# Rescue of study 13 hattori production tasks that ran out of memory at 64G during scoring.
# Same command as launch_record/production/sweep_submitted.yaml; only memory differs, and the
# command runs directly (no sweep agent), one subject per task.
set -euo pipefail
SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" /home/han.hou/code/wt/hpc13b/rescue/subjects.txt)
export WANDB_RUN_GROUP=hattori@20261026-002131
# Runtime setup copied from the launch's dispatcher code/hpc/slurm/wandb_sweep_cpu.slurm (5936f5d).
source /home/han.hou/code/wt/hpc13b/aind-dynamic-foraging-bfm-dispatcher/code/hpc/slurm/user.env
set +u; source "$CONDA_SH"; conda activate dynamic-foraging-bfm-cpu; set -u
if [ -f /usr/share/Modules/init/bash ]; then
    set +u; source /usr/share/Modules/init/bash; module load gcc/14-rocky9 2>/dev/null || true; set -u
    command -v g++ >/dev/null && export CXX="$(command -v g++)" CC="$(command -v gcc)"
fi
WRAPPER_ROOT=/home/han.hou/code/wt/hpc13b/aind-dynamic-foraging-bfm-wrapper
cd "$WRAPPER_ROOT/code"
export PYTHONPATH="$WRAPPER_ROOT/code:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export BFM_OUTPUT_DIR="${BFM_OUTPUT_DIR:-$HOME/outputs/disrnn}"
python -m run_hpc data=mice_snapshot_scaling model=hb_hattori model.estimator=heldout_two_level model.num_warmup=2000 model.num_samples=1500 model.num_chains=4 +model.session_draws_per_posterior_draw=24 data.subject_ratio=0.016 seed=0 wandb.entity=AIND-disRNN wandb.project=mice_data_scaling +meta.study=13-hb-subject-vs-gru-heldout +meta.variant=hattori +meta.launch_id=20261026-002131 +meta.config_hash=a3a3cb4a +meta.dispatcher_git_commit=5936f5d0b94b1fb5ec9d95058067716f261e5555 +meta.dispatcher_git_branch=HEAD +meta.dispatcher_git_dirty=no +meta.wrapper_git_commit=04878a6cdb081b0156b96ac1e61dde5de58d492e +meta.wrapper_git_branch=HEAD +meta.wrapper_git_dirty=no +meta.studies_commit=32bcf415c5f18d16a07dda5fb4dee914feabd177 +meta.studies_git_branch=feat/27-study-13-hb-subject +meta.studies_git_dirty=no +meta.sweep_yaml=/home/han.hou/code/wt/studies-27/studies/13-hb-subject-vs-gru-heldout/variants/hattori/sweep_hpc_per_subject.yaml +meta.owner=han.hou +meta.launcher_cmd=/home/han.hou/code/wt/hpc13b/aind-dynamic-foraging-bfm-di... +meta.mode=cpu +meta.label=production '+meta.note='"'"'Study 13 hattori production: each of the 149 held-out sub...'"'"'' +wandb.group=hattori@20261026-002131 '+meta.slurm_job_id='"'"'${oc.env:SLURM_JOB_ID,unknown}'"'"'' '+meta.slurm_array_job_id='"'"'${oc.env:SLURM_ARRAY_JOB_ID,unknown}'"'"'' '+meta.slurm_array_task_id='"'"'${oc.env:SLURM_ARRAY_TASK_ID,unknown}'"'"'' "data.test_subject_ids=[$SUBJECT]" +meta.intervention=rescue-oom
