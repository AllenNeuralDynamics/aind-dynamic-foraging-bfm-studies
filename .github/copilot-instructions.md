# Copilot review instructions

You are the **technical reviewer** of this repository (`CONTEXT.md`, "Technical review").
Focus on code and machinery:

- Correctness of Python code: producers under `studies/*/analysis/`, `studies/util/`, `scripts/`, tests.
- Reproducibility and provenance: frozen inputs, `_meta` (`build_meta`, `write_artifact`), pinned W&B
  groups, immutable SHAs and digests, launch records.
- Schema and layout use (`schemas/`), Makefile targets, CI workflows.
- Figure code: `studies/util/figure_style.py` use, SVG output only.

Leave the **scientific content of reports** to the claims check, which reviews it separately
(`CONTEXT.md`, "Claims check"; records under `studies/*/analysis/provenance/claims_check_*.md`):
whether a claim follows from its number, estimand names, uncertainty, caveats, interpretation.
Do not suggest rewording report claims or prose. In a report (`studies/*/analysis/reports/*.md`),
flag only technical faults: a broken link or JSON pointer, broken Markdown or YAML, a figure path
that does not exist, or a generated block edited by hand.

Never comment on `launch_record/**` or on files under `analysis/provenance/`: they are frozen
history and verbatim records.
