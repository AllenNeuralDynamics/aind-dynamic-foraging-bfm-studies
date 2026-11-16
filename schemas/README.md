# schemas/

The single source of truth for how a study is laid out and how its provenance is recorded.
Vocabulary is in [`../CONTEXT.md`](../CONTEXT.md); the launch / W&B-group rule is
[ADR 0001](../docs/adr/0001-one-launch-one-wandb-group.md). Skills and AGENTS.md link here
rather than restating fields.

| File | Governs |
|---|---|
| `layout.yaml` | study and variant directories, root allowlist, Makefile targets |
| `study-frontmatter.v1.schema.json` | YAML frontmatter of `studies/<study>/README.md` |
| `report-frontmatter.v1.schema.json` | YAML frontmatter of `analysis/reports/rN-<slug>.md` written before version 2 |
| `report-frontmatter.v2.schema.json` | the same for new reports: version 1 plus `claims`, `draft_reason`, `prose_reviewed` ([below](#version-2-reports)) |
| `tags.yaml` | the study and report tag vocabulary: values, labels, tooltip text |
| `meta.v2.schema.json` | `_meta` of every analysis JSON (`build_meta(..., code=[__file__])`) |
| `launch-record.v2.schema.json` | `variants/<v>/launch_record/*.json` |
| `provenance-baseline.json` | findings that predate a rule; may only shrink |

`scripts/check_provenance.py` (`make validate`) is the only implementation of the checks; CI
runs it as the `provenance` job.

**Versioning.** Each file type has its own `schema_version`. A file without one is the
previous version: frozen history that is reported, never rewritten. Changing a rule means
editing the schema and the checker together, and only that change may add baseline entries,
for the rule it introduces.

**Digests.** Paths in `_meta.code` / `inputs` / `outputs` are repository-relative. A JSON
artifact's own entry covers its body without `_meta`, serialised with sorted keys and
`(",", ":")` separators; a result block (`<report>.md#<name>`) is hashed with trailing
whitespace and surrounding blank lines stripped. Both are implemented once, in
`studies/util/_meta.py`.

## Version-2 reports

A report with `schema_version: 2` (what `make new-report` writes) follows the rules in the
dispatcher `posthoc-reporting` skill, "Report content"; the reasons are
[ADR 0003](../docs/adr/0003-report-claims-and-claims-check.md). A version-1 report keeps the
version-1 rules, checked exactly as before, until its results, claims or result blocks change;
that change is the bump to 2. A typo, link or wording fix leaves it on version 1. The machine-checkable parts:

| Finding | Level | Rule |
|---|---|---|
| `report-frontmatter` | error | the v2 schema: `claims` (at least one) and `prose_reviewed` are required, `draft_reason` while `status: draft` |
| `report-skeleton` | note | headings `# rN — <the finding>`, then `## Question`, `## Answer`, `## Methods`, `## Results`, `## What it means`, `## Caveats`, `## Next`, `## Provenance`, in that order; other `##` sections may sit between them; headings inside BEGIN/END blocks do not count |
| `report-claim-unresolved` | error when live; note when draft | every claim pointer names a value |
| `report-prose-stale` | note | every BEGIN/END block still has the digest recorded in `prose_reviewed` |
| `report-prose-number` | note | a decimal or percentage in hand-written prose matches a number in the JSON the report cites |
| `verdict-stale` | warn | the generated INDEX and README blocks copy the current claims |
| `report-claims-check-missing` | error when live; note when draft | the report has a claims-check record, `analysis/provenance/claims_check_<rN>_<date>[_<k>].md` (studies#160) |
| `report-claims-check-stale` | warn when live; note when draft | the newest record's `content_digest` (the report's claims + body) and the evidence digest of every file in its `inputs` still match (`context` and `code` are not checked) |
| `claims-check-record` | error | the record header follows `claims-check-record.v1.schema.json` and names this report |

`warn` and `error` fail CI (baseline ratchet). A **note** is level `info`: it never fails, and
`make validate` prints it under `note`. A superseded or retracted report is history: only its
schema and skeleton are checked.

**Claims.** `claims: [{id: F1, text: "...", pointer: "analysis/r1_fit.json#/delta/mean"}]`. The
pointer is a study-relative JSON path, `#`, and an RFC 6901 JSON Pointer (`~1` for `/`, `~0` for
`~` in a key; array items by index). A draft may point ahead of its producer, which is why a
fresh scaffold passes; a live report may not.

**`prose_reviewed`.** `{date: 2026-10-28, blocks: {methods: <sha256>, result-1: <sha256>}}`:
the day the hand-written prose was last re-read, and the digest of every BEGIN/END block it
was read against, hashed like a result block in `_meta.outputs` (`block_digest`, below). When
a producer rewrites a block, `report-prose-stale` names the block and prints its new digest.
Re-read the prose (and every sibling report that says "same as rN"), then paste the digest and
set today's date. The digests make the check exact: regenerating unchanged numbers changes
nothing, and a changed table is caught on the same day.

**Prose numbers** (advisory, ADR 0003). Hand-written prose is the body outside BEGIN/END
blocks, HTML comments, fenced and inline code, link targets and URLs. Its decimals
(`0.7213`, `5.3e-5`) and percentages (`12%`) are matched against every number in the JSON
files the report cites in `inputs` or in a claim pointer: values, numeric keys and numbers
inside strings, `_meta` excluded. A printed number matches if a value rounds to it at its
printed precision, sign included (`0.721` matches 0.72128; an unsigned `0.016` does not match
−0.016, so write the sign); a percentage may also be a fraction (`12%` matches 0.12). Integers are not checked: most are counts, sizes, years or ids. Expect a note
on a derived number, such as a difference the JSON does not store: store it, or cite the
finding by id. A constant that is not a result is written as code (`0.5`); `95%` is allowed
as it stands (`PROSE_CONSTANTS` in the checker).

**Claims-check records** (studies#160). Each claims check is committed as
`studies/<study>/analysis/provenance/claims_check_<rN>_<YYYY-MM-DD>[_<k>].md`, written by
`studies/util/claims_check.py` `write_record()` (or its CLI, which reads the reviewer's own output file, so the copy is verbatim; studies#164): a YAML header
([`claims-check-record.v1`](claims-check-record.v1.schema.json)), then the prompt given to the reviewer,
its response verbatim, and the answer to each finding. The header's `content_digest` hashes the
report's `claims` and body only, so `status`, `draft_reason` and `prose_reviewed` can change without
making the check stale; any change to a claim or the text does. `inputs` maps every other file the
reviewer read (repository-relative) to its evidence digest: an analysis JSON's body without `_meta`
(regenerating it with the same numbers changes nothing), any other file's sha256. The report itself
is never an input. Reading context (`CONTEXT.md`, the study `README.md`, `PLAN.md`, `CHANGELOG.md`)
is recorded under `context`, and code (`.py` files, e.g. the producers) under `code`; neither makes
a record stale, because a code change that alters a number or a sentence already shows in the JSON
body or the report text (studies#164). The prompt and
the response are stored verbatim, with their sha256 in the header (`prompt_sha256`, `response_sha256`).
The newest record (by date, then `_<k>`) is the one checked.
Like the claims check itself, the record never feeds the frontmatter, INDEX or README.

**Generated verdicts.** `scripts/report_verdicts.py studies/<study>` (`make -C studies/<study>
verdicts`) writes one block between `<!-- BEGIN verdicts -->` and `<!-- END verdicts -->` in each
of two files. In `analysis/reports/INDEX.md` the block is a table of every version-2 report
(title, status with `draft_reason`, the first paragraph of `## Question`, the claims). In the
README under `## Verdict` it holds the claims of every live version-2 report. Hand-written text
around a block is kept, and a version-1 report's hand row is never touched. A study with no
version-2 report and no block is left alone. One block holds the whole table because an HTML
comment line ends a Markdown table, so rows cannot carry their own markers.
