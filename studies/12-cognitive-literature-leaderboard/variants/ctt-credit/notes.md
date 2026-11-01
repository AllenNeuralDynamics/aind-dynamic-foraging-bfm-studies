# Variant `ctt-credit`

**What.** CompareToThreshold rewritten in stay credit c = β(V − θ), the logit of staying
(`ForagerCompareThresholdCredit`, [models PR 95](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/pull/95);
config `baseline_rl_lit_compare_to_threshold_credit`). c moves toward `stay_credit_win` after a win
and `stay_credit_loss` after a loss, resets to 0 on a switch, and P(stay) = σ(c + biasL on Left).
Same likelihood as CompareToThreshold (c_win = β(1 − θ), c_loss = −βθ; tested to 1e-12); only the
parameters and their bounds differ: both credits in [−20, 20], where CompareToThreshold bounds
θ to [−1, 1], that is c_loss ≤ c_win / 2. Fit and data as per-subject-de. Sweep: [`sweep_hpc.yaml`](sweep_hpc.yaml).

**Why.** Galen reported that the credit form fits better. With the same likelihood it can only do
so through its bounds or an easier search: 72 of the 763 subjects sit at CompareToThreshold's
θ = −1 bound (study 01 r8's fits). Report r2 tests whether they gain, and whether anyone else does.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `ctt-credit@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Not launched.
