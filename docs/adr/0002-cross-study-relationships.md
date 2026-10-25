# Cross-study relationships are motivation, upstream and correction; citations are not

Three declared relationships connect studies and reports, and only these are drawn or checked:

- **Motivation**: why a study exists. A required `motivated_by` list in the study frontmatter
  points at earlier results (`{study, report, note}`, `{study, note}` or `{issue, note}`),
  each with a one-line note on what that result showed. `[]` marks a root study.
- **Upstream**: a report uses another study's report or launch as input or comparison,
  declared in the report's `upstream` and checked: the checker warns when the cited report has
  been corrected or replaced, and when a producer reads another study's files undeclared.
- **Correction** and supersession: a report amends part of an earlier report (`corrects`) or
  replaces it (`superseded_by`).

A **citation**, a markdown link to another study or report, is not a relationship.

We chose this because the links in our prose did not mean one thing. Of the nine linked study
pairs on 2026-10-25, three were real use, two pointed forward to a later study ("see study 06
for the fix"), and four were background mentions; meanwhile 02, 07, 08, 09, 10 and 11 depended
on earlier studies without linking to them at all. A graph built from
links would draw dependencies that do not exist and miss the ones that do, and could say
neither why a study was started nor what it uses.

## Considered options

- **Keep citations as the relationship, and infer the rest.** Rejected: links fire on forward
  pointers and miss plain-text mentions, and carry no reason. Citations that point back in
  time catch 4 of the 11 motivations, with no report and no note.
- **Upstream only.** Rejected: it records what a report *uses*, which is checked and needed
  for staleness, but not why a study exists. Study 11 uses study 01's GRU numbers; it exists
  because of study 06's result.
- **Motivation at report level.** Deferred: within a study, report order mostly carries it.

## Consequences

- `motivated_by` is required for every study; the scaffolder asks for it
  (`make new-study ... MOTIVATED_BY=06/r2` or `none`), and the 11 existing studies were
  backfilled from their READMEs.
- The checker refuses an unresolved motivation and one that points at a later study, and notes
  (info) more than three, which usually mixes in what the study uses.
- The atlas study map draws motivation and correction by default and upstream as a layer;
  citations stay clickable links in the prose and are not drawn.
