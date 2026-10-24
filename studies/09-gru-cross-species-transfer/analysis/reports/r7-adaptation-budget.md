---
id: r7
slug: adaptation-budget
status: draft
authors: [han, claude]
issue: 166
wandb_groups:
  - gru-adaptation-budget@20261024-060709-hpc
  - q-adaptation-budget@20261024-035709
inputs:
  - analysis/adaptation_budget_cohort.json
  - analysis/adaptation_budget_results.json
reproduce: make -C studies/09-gru-cross-species-transfer r7
---

# r7 -- How transfer depends on the target adaptation budget

**Question.** The matched-half results (r1) adapt each new subject's embedding on its whole
adaptation half. How much of the GRU's transfer survives when it sees only K adaptation
sessions, and where does it cross the subject-level Bari2019 Q fit on the same sessions?

**Design.** Grossman (mouse) and Chen (mouse); frozen-core H=128 E4 GRU from the Study 01
D=614 source cells (seeds 0--2), embedding-only fine-tune (500 steps), at K in
{0, 1, 2, 4, 8, half} adaptation sessions -- the first K of each subject's chronological
odd-position sessions. Bari2019 common Q is refit per subject at every K >= 1 (a subject-level
Q fit has no zero-shot condition). Test sessions, and therefore the scored trials, are
identical at every K; the freeze asserts exact ordered trial-key equality across every cell.

**Cohorts, fixed before any result** (`analysis/adaptation_budget_cohort.json`): every Chen
subject has exactly 4 adaptation sessions, so the shared curve is K <= 4 over subjects with
>= 4 adaptation sessions (Grossman 46/48, Chen 32/32; Chen's K=4 is its half), and Grossman's
16 subjects with >= 8 sessions carry K=8 and half. Each cohort's numbers use the same
subjects at every K.

## Results

<!-- BEGIN result-1 -->
<!-- END result-1 -->
