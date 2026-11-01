# Study 40 explainers

Two pages for readers new to the study, written before any implementation (step −1 of `PLAN.md`).

| file | what it is | rebuild |
|---|---|---|
| `study40_animated_explainer.html` | six animated scenes, about 3 minutes; toy data except the reference numbers | `make explainer` refills the numbers from committed JSON (`build_explainer.py`) |
| `study40_schematic.html` | static map of the study's steps and decision points | archify ≥ 3.2: `archify deliver workflow reference/study40_schematic.workflow.json analysis/explainers/study40_schematic.html --quality showcase` |

`build_explainer.py` reads study 01 `rl_baseline.json` (through `studies/util/cross_study.py`) and
`report_data.json`, study 13 `r5_ctt_pooled.json`, and study 10's split list. It rewrites only the
block between `// BEGIN REFS` and `// END REFS`, and writes the same numbers to `explainer_refs.json`
with a v2 `_meta`, so `make validate` flags the page when a source result is corrected. The page's
"Text version" gives every scene in words, with the same numbers. A URL hash such as `#s=3&t=12` freezes scene 3 at
12 s, for render checks.
