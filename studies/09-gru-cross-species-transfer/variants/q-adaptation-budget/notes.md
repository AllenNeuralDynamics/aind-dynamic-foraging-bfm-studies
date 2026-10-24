# Bari2019 common-Q adaptation budget (dispatcher #166)

The `q-matched-half` model (one learning rate, one unchosen-value forgetting rate, a
one-step choice kernel, softmax) fitted per subject on the first K adaptation sessions and
scored on the same held-out test sessions as `../gru-adaptation-budget/`
(`run_q_baseline.py --adapt-sessions-per-subject K`).

K in {1, 2, 4, 8, half} for Grossman (mouse) and {1, 2, half} for Chen (mouse): a subject-level
Q fit needs at least one session, so there is no K=0, and Chen has exactly 4 adaptation
sessions per subject. Cohorts: `../../analysis/adaptation_budget_cohort.json`.

CPU only: an eight-task SLURM array on Allen HPC (`q_adaptation_budget.slurm`); each task
regenerates the checksum-verified canonical table and split, then fits. Never Beaker.
