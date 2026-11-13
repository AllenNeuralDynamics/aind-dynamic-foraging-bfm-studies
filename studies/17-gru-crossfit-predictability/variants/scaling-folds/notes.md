# Variant `scaling-folds`

**What.** The cross-validated scaling curve (PLAN.md, "Scaling curve"). For each pool fold k,
fold model k's training mice are subsampled to D ~ 10, 30 and 100 (studies 01/02 subject ratios
0.016, 0.049, 0.163 per curriculum, nested, seed 0; `reference/scaling_subsets.json`), and a
3-way GRU H256 (the `pool-folds` recipe) is trained on each: 4 folds x 3 D = 12 models. The
full-size point is the `pool-folds` model of the same fold; the test fold's curve is study 02's
`nxd-3way` grid (same recipe, D = 10, 30, 100, 614, seeds 0-2).

**Why.** Studies 01/02 measured scaling on the 149 test mice only. Scoring every mouse with a
model that never saw it gives the curve over all 763 mice, with fold-to-fold error bars, and
shows whether H256 is still climbing at full size (study 02: not plateaued).

**Launches.** Two launches, one W&B group each (docs/adr/0001): `sweep_d100.yaml` +
`experiment_beaker_d100.yaml` on allocated on-prem H200 seats (Han: the 8 allocated seats take
the longer runs), and `sweep_d10_d30.yaml` + `experiment_beaker_d10_d30.yaml` preemptible on AWS
H200. Records go to `launch_record/<label>/`.

**Status.** Not launched.
