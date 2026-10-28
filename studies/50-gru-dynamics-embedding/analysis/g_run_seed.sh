#!/usr/bin/env bash
# G1: rerun the core Study 50 producers for another seed of model C.
#   analysis/g_run_seed.sh C-s1 [script ...]
# Outputs go to analysis/seeds/<key>/ (see _pk01.MODEL_KEY); logs to _cache/seeds/<key>/.
# Steps run in Makefile dependency order; a failing step is logged and the chain continues.
set -u
KEY=$1; shift
cd "$(dirname "$0")/.."
export S12_MODEL_KEY=$KEY PYTHONPATH=../../../aind-dynamic-foraging-bfm-wrapper/code
LOGS=analysis/_cache/seeds/$KEY; mkdir -p "$LOGS"
STEPS=("$@")
if [ ${#STEPS[@]} -eq 0 ]; then
  STEPS=(b3_pca b3_state_space b3c_history b7_decoders
         c2_rl_rollouts c1_impulse c1_kernels c1_figure c2_compare c4_behaviour c4e_signatures
         f3_embedding_features f1_embedding_extremes d_dynamics
         i1_cluster_axes i2_tdr i3_tdr_dynamics e2_ablations e2b_conditional_ablations e1_subspaces
         i4_distilled_model i5_mouse_dynamics i7_second_variable i8_v_tests i9_v_kernel i9d_v_kernel_long i10_gates
         h2_learning_rate h3_q_phase_portraits h4_scenarios
         f2_mouse_dynamics f5_embedding_atlas f6_embedding_types f7_feature_library f10_embedding_behaviour_maps)
fi
for s in "${STEPS[@]}"; do
  t0=$(date +%s)
  if python3 "analysis/$s.py" > "$LOGS/$s.log" 2>&1; then status=ok; else status=FAIL; fi
  echo "$(date +%H:%M:%S) $KEY $s $status $(( $(date +%s) - t0 ))s" | tee -a "$LOGS/_status.txt"
done
