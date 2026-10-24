# MIGRATION.md

This repository was extracted from
[`aind-dynamic-foraging-bfm-dispatcher`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher)'s
`studies/` tree with its full history, following that repository's `docs/repo-split-plan.md`
(tracking issue dispatcher #124, operation #174).

| | |
|---|---|
| source repository | https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher |
| `SOURCE_DISPATCHER_COMMIT` | `818cc6ef556eedf2600eb28e6dca89035c3fa2d8` |
| `AGENTS_SOURCE_COMMIT` | `818cc6ef556eedf2600eb28e6dca89035c3fa2d8` (`AGENTS.md` here is the plan's template as reviewed in dispatcher #178, at that commit) |
| tool | `git-filter-repo==2.47.0`, `--path studies/` (the `studies/` prefix is kept) |
| extracted | 2026-10-24 00:09 PT |
| operator | Han Hou (hanhou), executed by Claude Code |
| abandoned branches | none: no remote branch carried a `studies/` commit absent from `main` at extraction |
| initial direct push | the one documented exception to "every change through a PR": an empty repository has no protected base branch |
| tag | `studies-split-v1`, annotated, on the first published tip |
| assembly commit | `76ed77a1ab54a000c3be49807127e08f251e70e3` |

## What was verified before publishing

- `studies/util/validate_provenance.py --strict` over all 86 variants gives byte-identical
  findings on the source checkout and on the assembled tree
  (`migration/provenance-findings.txt`, sha256 `ae853256e2217a1f...`). 35 variants are clean; 51
  carry findings that predate the split (36 ERROR, 17 WARN and 1 INFO lines): variants with no
  launch record at all (19), launcher records without a dispatcher commit (13), and records
  older than the intervention schema (17 warnings). The split introduced none. This is the
  explicit exception dispatcher #174 Step 4 allows; #158 stays open, untouched by it.
- `migration/source-tree.tsv` (`git ls-tree -r` of `studies/` at the source commit) is identical
  to the extracted tree: modes, blob IDs and paths.
- `migration/source-launch-records.sha256` verifies every `launch_record*/**` file byte for byte,
  before and after assembly.
- `migration/assembly-changes.tsv` lists every path the assembly added or changed, each with an
  allowed reason (`root-seed`, `ci`, `active-path-rewrite`); no launch record and no committed
  analysis artifact changed.
- `git log --follow` reaches pre-split history, e.g. study 01's `analysis/rl_baseline.py` back to
  dispatcher `1e30716` (rewritten SHA here; same subject and date).
- `scripts/check_regeneration.py` passes in a fresh clone with no credentials and no sibling
  checkouts.

## Path references

51 active files were rewritten so dispatcher paths resolve through the sibling checkout
(`../aind-dynamic-foraging-bfm-dispatcher/code/...`). Every remaining match of the old path
patterns is listed with its class in `migration/reference-classification.tsv`:

```
   362  immutable: launch record (what actually ran)
    60  intentional: already resolves through the sibling dispatcher
    30  intentional: GitHub URL to a dispatcher issue/PR/file (still valid)
    30  intentional: study 09 runtime path, pending dispatcher #179
    10  intentional: analysis/ text or producer string (regenerated output; reports are code)
     4  intentional: root seed prose naming a directory of the dispatcher (reviewed in dispatcher #178)
     4  immutable: pre-rename repository name in historical prose (ADR-0007)
     2  intentional: wrapper path, not a dispatcher path
     2  intentional: script comment describing what the job's dispatcher checkout holds
```

Launch records keep the paths and commits that actually ran. Study 09's runtime paths into the
dispatcher checkout are left as they are until dispatcher #179 decides how the runtime finds
study scripts.

## Deviations from the plan

- `.gitattributes` (git-LFS) from the reviewed seed is not applied at publish: LFS attributes on
  files that history stores as plain blobs make every fresh clone report them as modified.
  Enabling LFS needs a renormalize commit with git-lfs installed; tracked as a follow-up.

## Migrating a study branch that is still in the dispatcher

Follow [`docs/migrate-a-live-study-branch.md`](docs/migrate-a-live-study-branch.md): archive the
old branch, move its `studies/` commits with `git format-patch` / `git am --3way`, fix what the
split changed, and open PRs here.
