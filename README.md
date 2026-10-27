# aind-dynamic-foraging-bfm-studies

The scientific studies of the `dynamic-foraging-bfm` stack: a behavioural foundation model
for rodent dynamic foraging. Each study is one scientific question, with its launch
manifests, immutable launch records, frozen inputs, analysis code and reports in one folder.

**Browse it:** the [atlas](https://allenneuraldynamics.github.io/aind-dynamic-foraging-bfm-atlas/)
shows every study, report and number here, traced back to its producer, inputs, W&B group and
commits ([source](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-atlas)).

## The stack

[![Behavioral foundation model stack](https://raw.githubusercontent.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/main/docs/diagrams/bfm-stack.png)](https://raw.githack.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/main/docs/diagrams/bfm-stack.html)

**[Open the interactive version](https://raw.githack.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/main/docs/diagrams/bfm-stack.html)**,
also on the atlas's [About page](https://allenneuraldynamics.github.io/aind-dynamic-foraging-bfm-atlas/#/about);
its source is the dispatcher's `docs/diagrams/bfm-stack.architecture.json`.

This repository is the science record in that loop. A variant's sweep and experiment
manifests go to the dispatcher's launchers, which refuse a dirty study directory and write a
launch record the variant keeps under `launch_record/`; the wrapper trains and logs to W&B; a freeze pulls what
a study needs into committed files, its producers compute the reports from those, and the
atlas publishes them.

| Repository | Role |
|---|---|
| **`aind-dynamic-foraging-bfm-studies`** (this repo) | What to run and why: studies, variants, launch records, frozen inputs, analysis, reports, and the schemas and checker that hold them to the rules |
| [`aind-dynamic-foraging-bfm-dispatcher`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher) | Control plane: shared Hydra configs, Beaker / SLURM launchers, launch provenance, the agent skills pack |
| [`aind-dynamic-foraging-bfm-wrapper`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper) | Training and analysis payload: data loaders, the four model families, post-training analysis |
| [`aind-dynamic-foraging-bfm-atlas`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-atlas) | The browsable site, built from this repo's `main` |

Launches and some study scripts expect the dispatcher and the wrapper side by side with this
repository; the atlas needs no checkout of its own to read this one:

```text
parent/
  aind-dynamic-foraging-bfm-dispatcher/   # control plane: Hydra configs, Beaker / SLURM launchers
  aind-dynamic-foraging-bfm-studies/      # this repository
  aind-dynamic-foraging-bfm-wrapper/      # training and analysis payload
```

```bash
git clone https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher.git
git clone https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies.git
git clone https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper.git
```

This repository was extracted from the dispatcher's `studies/` tree with its full history;
`MIGRATION.md` records how, and the tag `studies-split-v1` marks the first published state.
Launch records older than the split name dispatcher paths (`code/launch_...`) and dispatcher
commits; that is what actually ran and is left as written.

## Studies

| study | question |
|---|---|
| [`01-gru-scaling-law`](studies/01-gru-scaling-law/) | How does GRU held-out likelihood scale with the number of mice (D) and model size? |
| [`02-gru-scaling-law-ignore`](studies/02-gru-scaling-law-ignore/) | Does including ignore trials as a third output change the scaling law? |
| [`03-disrnn-beta-scan`](studies/03-disrnn-beta-scan/) | How does the disRNN bottleneck penalty (β, multiplier) trade likelihood for sparsity at D≈100? |
| [`04-gru-vs-disrnn-embedding-recovery`](studies/04-gru-vs-disrnn-embedding-recovery/) | Do GRU and disRNN subject embeddings recover the generative parameters of synthetic cohorts? |
| [`05-disrnn-scaling-law`](studies/05-disrnn-scaling-law/) | How does the disRNN scale with D at a fixed penalty? |
| [`06-disrnn-operating-point-at-scale`](studies/06-disrnn-operating-point-at-scale/) | Where on the penalty × D surface is the disRNN's best operating point? |
| [`07-gru-timing-inputs`](studies/07-gru-timing-inputs/) | Do previous-trial reaction time and lick counts help the GRU, and why? |
| [`08-hb-vs-gru-heldout`](studies/08-hb-vs-gru-heldout/) | Where does a hierarchical Bayesian cognitive baseline rank against the GRU on held-out mice? |
| [`09-gru-cross-species-transfer`](studies/09-gru-cross-species-transfer/) | Does the mouse-trained GRU transfer to external two-arm-bandit datasets across species? |
| [`10-gru-reservoir-ablation`](studies/10-gru-reservoir-ablation/) | How much of the GRU's performance survives freezing its recurrent core at random? |
| [`11-disrnn-near-gru-ceiling`](studies/11-disrnn-near-gru-ceiling/) | How much of the residual disRNN–GRU gap closes when the disRNN is relaxed toward a GRU? |
| [`13-hb-subject-vs-gru-heldout`](studies/13-hb-subject-vs-gru-heldout/) | Does a two-level hierarchical Bayesian baseline, conditioned only on each held-out mouse's context sessions, match the GRU's held-out likelihood? |

New studies, variants and reports start from the scaffolder (`make new-study`, `make
new-variant`, `make new-report`). Each study's `README.md` carries its verdict; `analysis/reports/` the numbered reports;
`variants/<variant>/notes.md` what each variant differs in and where it stands;
`variants/<variant>/launch_record*/` what was launched, when, with which commits.

## Regenerating a study's reports

Reports are code. Every report is produced from committed frozen inputs by its study's
`Makefile`, offline, with no W&B or cluster access:

```bash
python3 -m pip install -r requirements-dev.txt
make -C studies/08-hb-vs-gru-heldout all      # one study
python3 scripts/check_regeneration.py        # every study, as CI runs it
```

Two consecutive runs differ only in each artifact's `_meta` block (timestamps, checkout
SHA); any other difference is a result change and belongs in a reviewed commit. Figures are
paper-style SVGs drawn with `studies/util/figure_style.py`; `make validate` runs the
provenance checker and the figure check that CI gates on. Pulling new
numbers from W&B is a separate, credentialed producer target in each `Makefile`
(`freeze-*`, `pull-*`) and is never a CI prerequisite.

## Launching a variant

Launch through the dispatcher's launchers, from this repository's root, after loading the
`beaker-launch` or `hpc-launch` skill:

```bash
python ../aind-dynamic-foraging-bfm-dispatcher/code/launch_beaker_resumable.py \
  --sweep      studies/<study>/variants/<variant>/sweep_beaker.yaml \
  --experiment studies/<study>/variants/<variant>/experiment_beaker.yaml \
  --workspace  ai1/aind-dynamic-foraging-foundation-model \
  --label <label> --note "why this run exists and what it should teach us"
```

The launcher refuses to launch from a `studies/<study>/` directory with uncommitted changes,
or on the live foraging database rather than a pinned snapshot. Its launch record
(`schemas/launch-record.v2.schema.json`) names the studies commit holding the sweep, the
dispatcher commit running the launcher, the wrapper and foraging-models commits the job will
run, the data snapshot and the image. Copy the record it writes into the variant's
`launch_record/`.

## Conventions

`AGENTS.md` here states the rules an agent must follow in this repository. The detailed
conventions are skills in the dispatcher's
[`aind-behavior-foundation-model-skills/`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/tree/main/aind-behavior-foundation-model-skills):
`study-conventions` (layout, naming, W&B groups, launch provenance),
`posthoc-reporting` (report, JSON and `results.md` contracts), `issue-tracking`.
A study branch started in the dispatcher before the split moves here with
[`docs/migrate-a-live-study-branch.md`](docs/migrate-a-live-study-branch.md).

Work is tracked on the
[AIND-behavior-fm project board](https://github.com/orgs/AllenNeuralDynamics/projects/184/).

Everything is plain git; there is no git-LFS (declined in dispatcher #180). CI rejects any
tracked file over 10 MiB outside the allowlist in `scripts/check_file_sizes.py`, because a
large file merged into `main` stays in history for good. Raw data, caches and checkpoints
belong in S3, a Beaker dataset or a W&B artifact, committed here only as an id plus digest.
