# A report's verdict is its frontmatter claims, and a fresh reader checks them before it goes live

Every report written to `schema_version: 2` has the same skeleton: Title (the finding),
Question, Answer (numbered findings F1…), Methods, Results, What it means, Caveats, Next,
Provenance. Each finding is a frontmatter **claim** tied by a JSON Pointer to the number
that supports it. Those claims are the only home of the verdict: INDEX.md rows and the
README `## Verdict` are generated from them, prose cites findings by ID and adds no new
claims, and a producer never prints a fixed conclusion sentence. Before a report goes
`live`, and whenever a live report's claims change, a **claims check** runs: a new,
read-only agent that did not write the report compares every claim with the numbers.
Then the report's lead scientist, the first name in its `authors`, gives the go-ahead.

We chose this because the text of our reports did not hold still while the numbers did.
A survey of all 46 reports on 2026-10-28 found every script-written result block matched
its JSON, but:

| Failure | Example |
|---|---|
| Prose went stale after a block was regenerated | 13 r2 kept the pre-rescue "12% fail k̂"; 13 r3 said "draft, same reasons as r1" after r1 went live |
| Claims swung on unchanged numbers | 13 r1's account of the HB gain reversed about five times in 40 hours; 04 r1 had four "overstated claim" fixes in one day |
| A producer printed a fixed conclusion | 01 r7 always prints "support for an N×D synergy", even when the comparison is `n/a` |
| Prose numbers had no source | 13 r1's "~0.0004 GRU seed spread"; other reports give 0.00002–0.00006 |
| The verdict lived in three places | report, INDEX.md and README disagreed for studies 10, 11 and 13 |
| One label, two estimands | "held-out likelihood" for both a per-subject median (13 r2) and a trial-pooled value (13 r1) |
| No shared form | five title styles, two templates that disagreed, seven ways to write uncertainty |

The rules live in the dispatcher `posthoc-reporting` skill (dispatcher #211); this record
keeps the reasons.

## Decisions

- **Readers, in layers.** The lead scientist deciding what to run next reads the Question and Answer; lab
  members read Methods and What it means, with each specialised term explained at first
  use and linked to the `CONTEXT.md` glossary; a paper and later sessions read Methods,
  Results and Provenance.
- **Findings ordered as an argument**, headline first, then support, then limits; about
  five at most.
- **Methods** are a script-written facts block (mice, sessions, split, seeds, versions,
  interval type) plus prose; an approach the study's reports share lives once in the
  study README.
- **Uncertainty**: a 95% CI (credible interval for Bayesian fits) or the SEM, always named,
  with n and what n counts.
- **Estimands** carry their full glossary name, including the aggregation, in tables,
  claims and the JSON.
- **Cross-study numbers** come from a registry of named versions (`studies/util/references.py`)
  (source and conditions, never a stored value), one default per comparison; a report
  prints the version name next to the number.
- **Changing a report**: after a block changes, the prose is re-read and `prose_reviewed`
  bumped; a live report's title and claims change only with new numbers, or a CHANGELOG
  entry plus the lead scientist's review; a draft states its `draft_reason`.
- **Draft → live** needs a seven-item gate, a claims check and the lead scientist's explicit
  go-ahead.
- **Old reports** stay on `schema_version: 1` until next changed; the retrofit is the bump.

## Considered options

- **Rules in the skill only.** Rejected: sessions read the skill unevenly; the
  machine-checkable parts (skeleton, claim pointers, `draft_reason`, generated verdicts)
  go in the scaffolder and the checker too.
- **Producers write the claims from thresholds.** Rejected: very stable but rigid, and the
  thresholds are hard to choose. Claims are hand-written but each must point at a number;
  only verdict words are chosen by code.
- **Placeholders for prose numbers** (`{{r1:fit.delta|.4f}}` filled by a renderer).
  Rejected as too heavy. Instead prose cites findings by ID, an advisory note flags prose
  numbers that match nothing in the JSON, and the claims check covers the rest.
- **Rely on the claims check alone.** Rejected: an LLM reviewer misses some cases and varies
  between runs. Properties that must always hold are checked by machine; judgement goes to
  the reviewer.
- **A canonical value per comparison.** Rejected: the four GRU ceilings are different
  measurements (data version, model size, mouse set), not errors. Named versions with a
  default keep them all legitimate and visible.
- **One exempt list for old reports, like `provenance-baseline.json`.** Rejected: AGENTS.md
  forbids growing the baseline to get green. The frontmatter `schema_version` already
  separates old and new rules.
- **Call the gate "audit" or "domain review".** Rejected: "domain review" is the separate,
  advisory reviewer (scientific blind spots, scores, ideas), and mixing the two jobs lowers
  recall. The gate is the **claims check**.

## Consequences

- `schemas/report-frontmatter` gains version 2 (`claims`, `draft_reason`, `prose_reviewed`);
  the checker enforces the skeleton and claim pointers for version-2 reports, warns on
  unreviewed prose and unsourced prose numbers, and `make` generates INDEX and README
  verdicts (#62).
- A new `studies/util/references.py` holds the registry of named versions, with defaults
  chosen in review (#63). It is a new module because editing `cross_study.py` would mark
  every artifact that imports it stale.
- `CONTEXT.md` gains metric and methods glossaries (#64).
- The claims check is measured on the known-flaw versions of 13 r1, 13 r2, 13 r3, 01 r7 and
  04 r1 before it is relied on; its catch rate decides whether the prose-number lint stays.
  13 r1 is the first retrofit and study 14's first report the first new one (#65).
- The claims check's output describes a report and never feeds the frontmatter, INDEX or
  README, so it cannot become a fourth verdict source.
