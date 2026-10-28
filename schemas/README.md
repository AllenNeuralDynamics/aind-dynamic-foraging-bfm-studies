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
version-1 rules, checked exactly as before, until it is next changed; the retrofit is the bump
to 2. The machine-checkable parts:

| Finding | Level | Rule |
|---|---|---|
| `report-frontmatter` | error | the v2 schema: `claims` (at least one) and `prose_reviewed` are required, `draft_reason` while `status: draft` |
| `report-skeleton` | error | headings `# rN — <the finding>`, then `## Question`, `## Answer`, `## Methods`, `## Results`, `## What it means`, `## Caveats`, `## Next`, `## Provenance`, in that order; other `##` sections may sit between them; headings inside BEGIN/END blocks do not count |
| `report-claim-unresolved` | error when live; note when draft | every claim pointer names a value |
| `report-prose-stale` | warn | every BEGIN/END block still has the digest recorded in `prose_reviewed` |
| `report-prose-number` | note | a decimal or percentage in hand-written prose matches a number in the JSON the report cites |
| `verdict-stale` | warn | the generated INDEX and README blocks copy the current claims |

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
printed precision (`0.721` matches 0.72128); a percentage may also be a fraction (`12%`
matches 0.12). Integers are not checked: most are counts, sizes, years or ids. Expect a note
on a derived number, such as a difference the JSON does not store: store it, or cite the
finding by id. A constant that is not a result is written as code (`0.5`); `95%` is allowed
as it stands (`PROSE_CONSTANTS` in the checker).

**Generated verdicts.** `scripts/report_verdicts.py studies/<study>` (`make -C studies/<study>
verdicts`) writes one block between `<!-- BEGIN verdicts -->` and `<!-- END verdicts -->` in each
of two files. In `analysis/reports/INDEX.md` the block is a table of every version-2 report
(title, status with `draft_reason`, the first paragraph of `## Question`, the claims). In the
README under `## Verdict` it holds the claims of every live version-2 report. Hand-written text
around a block is kept, and a version-1 report's hand row is never touched. A study with no
version-2 report and no block is left alone. One block holds the whole table because an HTML
comment line ends a Markdown table, so rows cannot carry their own markers.
