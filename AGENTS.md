# AGENTS.md — aind-dynamic-foraging-bfm-studies

## Explaining and Reporting Style

Source: Karpathy's advice to ask for explanations in
[ASD-STE100](https://www.asd-ste100.org/) (Simplified Technical English), at about 80%
strictness.

**In conversation** (more freedom than reports; aim to teach):
- **Plain language, roughly 80% of ASD-STE100.** Short sentences, active voice, no filler.
  This is a guide, not a hard rule. Use longer sentences when they make an idea clearer.
- **Explain as a teacher would.** Do not assume the reader knows every technique. The first
  time a specialised term appears, give its basic idea in a sentence or two. This applies
  to statistical methods (for example hierarchical Bayes, posterior predictive checks or
  bootstrapping), model internals and training mechanics. Say what the term means, why it
  is used here, and what to watch for. After that, use the term without repeating the
  explanation.
- **Show, do not only tell.** Explain with a diagram where one helps. For a long or
  structured explanation, prefer an HTML page over a wall of text.

**In reports:**
- **Plain language, about 80% of ASD-STE100.** Short sentences, one idea per sentence,
  active voice, no filler.
- **Order of preference: figure > table > narrative.** A figure beats a table. A table
  beats prose. Use narrative only for what a figure or table cannot carry.

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
  A new report is `schema_version: 2` (`make new-report`): the fixed skeleton, each finding a
  frontmatter claim pointing at its JSON value, INDEX and README verdicts generated from the
  claims (`make verdicts`), `prose_reviewed` updated after a block changes. It goes live only
  after a claims check and the go-ahead of its lead scientist (first in `authors`); see
  `schemas/README.md` and the `posthoc-reporting` skill.
- **Figures are paper-style SVGs.** Draw every figure with `studies/util/figure_style.py`
  (`apply_paper_style()`, 5–7 pt Arial) at the size it is printed, sized by its content, not the
  column: ≈15 mm per bar or violin, 35–45 mm per scatter panel, + 18 mm per y axis
  (`fs.fit_width()`), at most 183 mm wide, any height. Save it only with
  `fs.save_figure(fig, ...svg)`, which refuses text that Illustrator cannot edit. No PNG/PDF
  figures. Reports embed the paper style; a slide version (`fs.apply_slide_style()`,
  `style="slide"`) goes under `analysis/slides/`. `make validate` and CI run `scripts/check_figures.py`; studies 01–11 are exempt until redrawn (#44).
- **Session plots use the lab's utility, and sides have one colour.** Draw a foraging session
  (choices, rewards, block reward probabilities) with `plot_foraging_session` from
  `aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session`, onto an axis of the paper
  figure (`ax=`). Overlay a model's p(right) with `fitted_data`, and put model panels (for
  example state posteriors) on axes below it. In every figure, right is `"b"` and left is
  `"r"`, the colours that utility uses (`p_right` / `p_left`), so sides read the same in every
  study and every lab plot.
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

## Coding Principles

Adapted from [multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills), which distills [Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876) on LLM coding pitfalls. They bias toward caution over speed; for trivial tasks, use judgment.

### 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them instead of picking silently.
- If a simpler approach exists, say so.
- If something is unclear, stop and ask.

### 2. Simplicity First

Write the minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No configurability that was not requested.
- No error handling for impossible scenarios.
- If the solution is overcomplicated, simplify it.

### 3. Surgical Changes

Touch only what is needed. Clean up only your own mess.

When editing existing code:
- Do not improve adjacent code, comments, or formatting unless required.
- Do not refactor unrelated code.
- Match existing style.
- If you find unrelated dead code, mention it but do not delete it.

When your changes create orphans:
- Remove imports, variables, or functions made unused by your change.
- Do not remove pre-existing dead code unless asked.

Test: Every changed line should trace directly to the request.

### 4. Goal-Driven Execution

Define success criteria and verify.

Transform tasks into verifiable goals:
- Add validation -> write failing tests for invalid inputs, then make them pass.
- Fix a bug -> write a reproducing test, then make it pass.
- Refactor -> ensure tests pass before and after.

For multi-step tasks, use a brief plan:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

These guidelines are working when diffs contain fewer unnecessary changes, solutions are simpler, and clarifications happen before implementation.

