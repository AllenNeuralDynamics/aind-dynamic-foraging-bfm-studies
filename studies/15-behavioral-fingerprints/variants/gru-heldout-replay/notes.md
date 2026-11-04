# Variant `gru-heldout-replay`

**What.** Study 01's v1 GRU at D = 614 (H128, three seeds, training experiment
[`01KVQ7EJ3C5YJ8FJVNJB8C8N36`](https://beaker.org/ex/01KVQ7EJ3C5YJ8FJVNJB8C8N36)), re-fine-tuned
offline on the held-out subjects with study 01's own held-out config (`heldout_offline.yaml`: 500
steps fit only the held-out subjects' embeddings on their even sessions, then the odd sessions are
scored), on current wrapper main, which also writes each scored trial's p(right)
(`test_trial_predictions.csv`). `launch.py` submits one Beaker task per seed.

**Why.** Han, 2026-11-04: the GRU should appear beside the literature models in the replay (P1)
and rollout (P2) plots, not wait for P3. Replay needs the GRU's per-trial predictions, and no run
saved them. The held-out split is the one study 15 uses: within each subject, odd sessions are
scored, ignored trials removed; the scored trial counts equal the MEvolve freeze's on all 146
shared subjects. Before its tallies count, each subject's replayed likelihood must equal study 01
r2's stored value (`cross_study_literature.gru_per_subject`).

**Launches.** Each launch writes its record to `launch_record/<label>/` and mints one W&B group
`gru-heldout-replay@<launch_id>` (docs/adr/0001), here only a provenance key: the config keeps
`wandb: {}`, so no W&B run is logged, and the outputs are the tasks' Beaker result datasets.

**Status.** Launched 2026-11-04 12:01 PT as `gru-heldout-replay@20261104-120131`, Beaker
experiment [`01M6M2E1FPKJ52128XK2K52B29`](https://beaker.org/ex/01M6M2E1FPKJ52128XK2K52B29)
(3 tasks); record in `launch_record/gru-replay/`.
