# Variant `gru-heldout-replay`

**What.** Study 01's v1 GRU at D = 614 (H128, three seeds, training experiment
[`01KVQ7EJ3C5YJ8FJVNJB8C8N36`](https://beaker.org/ex/01KVQ7EJ3C5YJ8FJVNJB8C8N36)), re-fine-tuned
offline on the held-out subjects with study 01's own held-out config (`heldout_offline.yaml`: 500
steps fit only the held-out subjects' embeddings on their even sessions, then the odd sessions are
scored), on current wrapper main, which also writes each scored trial's p(right)
(`test_trial_predictions.csv`). `launch.py` submits one Beaker task per seed.

**Why.** Han, 2026-11-04: the GRU should appear beside the literature models in the replay (P1)
and rollout (P2) plots, not wait for P3. Replay needs the GRU's per-trial predictions, and no run
saved them. The held-out split must be the one study 15 uses: the MEvolve freeze's 149 subjects
(snapshot 20260603), odd sessions scored, ignored trials removed. Before its tallies count, each subject's replayed likelihood must equal study 01
r2's stored value (`cross_study_literature.gru_per_subject`).

**Launches.** Each launch writes its record to `launch_record/<label>/` and mints one W&B group
`gru-heldout-replay@<launch_id>` (docs/adr/0001), here only a provenance key: the config keeps
`wandb: {}`, so no W&B run is logged, and the outputs are the tasks' Beaker result datasets.

**Status.** Launched 2026-11-04 12:01 PT as `gru-heldout-replay@20261104-120131`, Beaker
experiment [`01M6M2E1FPKJ52128XK2K52B29`](https://beaker.org/ex/01M6M2E1FPKJ52128XK2K52B29)
(3 tasks); record in `launch_record/gru-replay/`. The Beaker hubs were full (0 schedulable GPUs,
37 queued on onprem-h200), so the same job also runs on the Allen HPC (`launch_hpc.py`,
`run_hpc.sbatch`; Han, 2026-11-04): whichever finishes first feeds P1.

**Correction (2026-11-04).** The first HPC launch (`gru-heldout-replay@20261104-124134`, array
27614186) inherited the source run's selectors, which name no snapshot, so the wrapper re-selected
the held-out cohort from the live database: 162 subjects, only 67 of them in the freeze's 149. On
the 66 of those whose sessions match the freeze, all three seeds reproduce study 01 r2 per subject
to 1e-5; the one whose sessions differ (841363, with newer sessions) misses by 0.049. Its outputs
are not used. `heldout_offline.yaml` now pins the selectors to the freeze's (snapshot 20260603), and
a dry run of the selection with them gives exactly the freeze's 149. Relaunched on the HPC as
`gru-heldout-replay@20261104-150009` (array 27621304, record in `launch_record/gru-replay-hpc-snap/`).
The Beaker launch (`01M6M2E1FPKJ52128XK2K52B29`) was submitted with the old config and has the same
flaw.

**Beaker launch cancelled (2026-11-04 19:40 PT).** Its three tasks had not been scheduled. It was
submitted with the first config, which re-selects the held-out cohort from the live database, so it
would have repeated the first HPC launch's flaw (Han asked for the cancellation). No results.
