# AGENTS.md — aind-dynamic-foraging-bfm-studies

Scientific study definitions, launch manifests, frozen inputs, analysis, and reports for
`dynamic-foraging-bfm`, a behavioural foundation model for rodent dynamic foraging. The
sibling dispatcher owns the launch and runtime control plane; the wrapper owns training.
This file is self-contained for the rules an agent must obey from a studies-only checkout;
the links give detail, they do not replace the rules.

## Required workflow

- Every tracked change starts with an assigned GitHub issue on project 184
  (`AIND-behavior-fm`). File first, then work.
- One scientific question per `studies/NN-{model}-{purpose}/`; conditions are
  self-contained variants under `variants/<variant>/`, not new studies. `NN` is an
  accession number: never renumber or reuse one.
- Treat `launch_record/**` as immutable history. Do not rewrite old paths, refs, names, or
  results to match the current tree; a stale path in a frozen record is part of the record.
- Scientific launches use clean checkouts and immutable full SHAs for studies, dispatcher,
  wrapper, and foraging-models. The launchers refuse a sweep whose `studies/<study>/`
  directory has uncommitted changes. Always pass `--label` and a `--note` saying why the
  run exists and what it should teach us; check schedulable capacity before any launch of
  more than 4 GPUs or 4 concurrent tasks; submit only to `hub` clusters on Beaker.
- **Never run Python on the HPC login node.** Analyses, freezes and smoke tests go through
  `srun`/`sbatch` onto a compute node; the only exceptions are the dispatcher's
  submit-only launchers. Report producers that read committed files run locally.
- Reports are code: regenerate them from committed frozen inputs with the study's
  `Makefile`, keep W&B pull caches ignored, pin the W&B groups a producer reads, key every
  frozen number by an immutable digest, and inspect numeric and artifact diffs before
  claiming a result. New analysis JSON carries `_meta.studies_git_sha` and the
  `source_*_git_shas` arrays; never relabel this repository's HEAD as dispatcher provenance.
- **Figures are paper-style SVGs.** Draw every figure with `studies/util/figure_style.py`
  (`apply_paper_style()`, 5–7 pt Arial) at the size it is printed, sized by its content, not the
  column: ≈15 mm per bar or violin, 35–45 mm per scatter panel, + 18 mm per y axis
  (`fs.fit_width()`), at most 183 mm wide, any height. Save it only with
  `fs.save_figure(fig, ...svg)`, which refuses text that Illustrator cannot edit. No PNG/PDF
  figures. Reports embed the paper style; a slide version (`fs.apply_slide_style()`,
  `style="slide"`) goes under `analysis/slides/`. `make validate` and CI run `scripts/check_figures.py`; studies 01–11 are exempt until redrawn (#44).
- **Never retype another study's number.** Read it from that study's committed JSON (readers for
  the shared references are in `studies/util/cross_study.py`), pass that source JSON (the
  helper's `*_SOURCE` path, not the helper) in `build_meta(inputs=...)`, and
  name the source in the report's `upstream` (`{study, report}` for a result, `{study,
  wandb_group}` for raw data). The checker then flags the report when that source is corrected.
- **Start from the scaffolder**: `make new-study`, `make new-variant`, `make new-report` (usage
  in the root `Makefile`). Their output passes `make validate` as written.
- **Layout and provenance are schema-checked.** The rules are `schemas/` (fields, layout) and
  `CONTEXT.md` (vocabulary: study, variant, launch, intervention, report, freeze, ...); read
  them rather than copying an old study, whose files may predate a rule. Write analysis JSON
  with `build_meta(..., code=[__file__], inputs=[...])` + `write_artifact()`. Run
  `make validate` before opening a PR: every new finding prints a fix, and CI's `provenance`
  check fails on it. Fix the study; never loosen a schema or add to
  `schemas/provenance-baseline.json` to get green.
- Use Conventional Commits. Open a PR for every change; **never squash-merge**, and never
  merge any PR without explicit user confirmation. Resolve every review thread.
- Human-facing times are Seattle time (`TZ=America/Los_Angeles`, e.g. `10:48 PT`), and any
  run you mention links its W&B project or run.
- Never global-replace `disrnn`: in code it names a model family, and under
  `launch_record/` and `analysis/` it is frozen history.

## Repository relationship

Launch commands assume this side-by-side layout:

    aind-dynamic-foraging-bfm-dispatcher/
    aind-dynamic-foraging-bfm-studies/        # this repository
    aind-dynamic-foraging-bfm-wrapper/        # training payload; some study scripts import it

From this repository, invoke launchers through
`../aind-dynamic-foraging-bfm-dispatcher/code/`, e.g.

    python ../aind-dynamic-foraging-bfm-dispatcher/code/launch_beaker_resumable.py \
      --sweep studies/<study>/variants/<variant>/sweep_beaker.yaml \
      --experiment studies/<study>/variants/<variant>/experiment_beaker.yaml \
      --label <label> --note "..."

Shared Hydra configs stay in the dispatcher (`code/config/`); a study's exact sweep and
experiment manifests stay under its variant. The detailed operational rules and the skills
live in the dispatcher:

- https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/blob/main/AGENTS.md
- https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/tree/main/aind-behavior-foundation-model-skills
- https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/blob/main/docs/repo-split-plan.md

Load `study-conventions`, `posthoc-reporting`, `issue-tracking`, and the relevant launch
skill (`beaker-launch` or `hpc-launch`) before creating a study, producing a report, or
submitting work. If a skill and this file disagree, follow the more restrictive one and fix
both.
