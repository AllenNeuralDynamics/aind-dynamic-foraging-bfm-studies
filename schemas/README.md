# schemas/

The single source of truth for how a study is laid out and how its provenance is recorded.
Vocabulary is in [`../CONTEXT.md`](../CONTEXT.md); the launch / W&B-group rule is
[ADR 0001](../docs/adr/0001-one-launch-one-wandb-group.md). Skills and AGENTS.md link here
rather than restating fields.

| File | Governs |
|---|---|
| `layout.yaml` | study and variant directories, root allowlist, Makefile targets |
| `study-frontmatter.v1.schema.json` | YAML frontmatter of `studies/<study>/README.md` |
| `report-frontmatter.v1.schema.json` | YAML frontmatter of `analysis/reports/rN-<slug>.md` |
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
