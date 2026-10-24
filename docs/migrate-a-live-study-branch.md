# Migrating a live study branch from the dispatcher

For anyone (person or agent) with a study branch that was started in
`aind-dynamic-foraging-bfm-dispatcher` before the split, where studies lived under its
`studies/`. Studies now live here. The dispatcher's `studies/` is removed by dispatcher #181,
and a branch left there can never merge.

**When to migrate:** soon, at the next clean break in the work: everything committed, no
launch in flight, no analysis half-written. Not at the very end of the branch's work, for
three reasons:
- every check this repo enforces (offline `make all`, the 10 MiB file limit, launch
  provenance) would surface at once on a large branch;
- every launch made from the dispatcher records commits that the migrated copy will not
  share;
- merging dispatcher `main` into the old branch after #181 would delete the study files.

Budget about an hour. The commands below were checked on a scratch repository: commits that
touch only non-study files are dropped, mixed commits keep only their `studies/` part, and
authors and dates are preserved.

## 0. Preconditions

- [ ] **The skills pack is version 0.11.0 or later.** Older versions still say studies live
      in the dispatcher's `studies/`, so an agent following them would migrate the study wrong.
      Update, then restart Claude Code; `/plugin` should list
      `aind-behavior-foundation-model-skills` at 0.11.0 or later:

```bash
claude plugin marketplace update aind-behavior-foundation-model
claude plugin update aind-behavior-foundation-model-skills@aind-behavior-foundation-model
```

- [ ] No launch is running that will still write to the branch: no Beaker experiment, SLURM
      array or watcher appending to `launch_record*/`.
- [ ] `git status` is clean on the study branch, or the changes are committed.
- [ ] A clone of this repository sits beside the dispatcher clone:

```bash
cd <parent of the dispatcher clone>
git clone git@github.com:AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies.git
```

On `/allen`, ACLs make git report mode-only changes; run `git config core.fileMode false` in
each clone. **Never name a checkout directory `studies`**: older launchers took the first
`studies` path component as the study name (dispatcher #182, fixed in #183).

## 1. Archive the old branch (provenance)

Launch records written so far name dispatcher commits. Keep those commits reachable, then
stop committing to the branch:

```bash
cd aind-dynamic-foraging-bfm-dispatcher
git switch <study-branch>
git fetch origin
git push origin HEAD:refs/heads/archive/<study-branch>     # read-only from now on
```

## 2. Preview what moves

```bash
BASE=$(git merge-base HEAD origin/main)
git log --oneline "$BASE"..HEAD -- studies/              # these commits move (their studies/ part)
git diff --stat "$BASE"..HEAD -- . ':!studies/'          # these changes do NOT move (step 5)
git diff --stat "$BASE"..HEAD -- studies/ ':!studies/<NN-study>/'   # touches shared/other studies?
```

If the third command lists anything, typically `studies/util/` or another study, it moves too,
but give it its own PR in step 6: shared helpers have changed here since the split.

## 3. Export and apply

```bash
# in the dispatcher clone
git format-patch -o /tmp/<study>-patches "$BASE"..HEAD -- studies/

# in the studies clone
cd ../aind-dynamic-foraging-bfm-studies
git switch main && git pull --ff-only
git switch -c study/<NN-study> origin/main
git am --3way /tmp/<study>-patches/*.patch
```

- The `studies/` path prefix is the same in both repositories, so paths need no rewriting.
- `--3way` resolves most context drift, because this repository kept the dispatcher's blobs.
- On a conflict: fix the files, `git add` them, then `git am --continue`. `git am --abort`
  returns to a clean branch.
- Merge commits are not exported: the moved history is linear, one commit per original commit
  that touched `studies/`.

## 4. Fix what the split changed

Run these from `studies/<NN-study>/` and fix every hit in a **live** file: sweeps,
experiment templates, scripts. Never edit anything under `launch_record*/`. Those are frozen
records of what actually ran, and the old paths in them are correct history; the greps
below already exclude them.

**Beaker jobs have no studies checkout.** A sweep or experiment that runs a script out of the
study at job time will fail:

```bash
grep -rn --exclude-dir='launch_record*' '/workspace/aind-dynamic-foraging-bfm-dispatcher/studies' . --include='*.yaml'
```

Move such a runner into the wrapper and call it as `python -m <module>`, as study 09 did
(`external_transfer.gru_transfer`, dispatcher #179). Pass anything it reads from the study on
the command line. HPC jobs can still run study scripts from a studies checkout (next item).

**HPC scripts take the study from `STUDIES_ROOT`:**

```bash
grep -rn --exclude-dir='launch_record*' 'DISPATCHER_ROOT}*/studies' . --include='*.slurm' --include='*.sh' --include='*.sbatch'
```

Replace with `${STUDIES_ROOT}/studies/...`. Add
`STUDIES_ROOT="${STUDIES_ROOT:?STUDIES_ROOT is required}"` next to the other roots, and export
`STUDIES_COMMIT` beside `DISPATCHER_COMMIT`. Model this on study 09's `variants/*/*.slurm`.

**Paths to dispatcher code.** `external_bandit_datasets` and the launchers stay in the
dispatcher. Reach them through a sibling checkout, `../aind-dynamic-foraging-bfm-dispatcher/code`
from the repository root, overridable with `DISPATCHER_ROOT` (see study 09's `Makefile`), not
through `../../code`.

**`make all` must run offline.** CI regenerates every study with no W&B, Beaker or `/allen`
credentials and compares with `HEAD`; only `_meta`, float noise and image bytes may differ.
Anything that pulls live data goes in a `freeze-*` target that writes a committed frozen file
keyed by run id and digest; see studies 01/02/05/07 (dispatcher #177). A study with no
`Makefile` is reported as `EXEMPT` rather than failing.

**No file over 10 MiB** unless allowlisted in `scripts/check_file_sizes.py`. Raw data, W&B or
Beaker caches, checkpoints and notebooks with large outputs belong in S3, a Beaker dataset or
a W&B artifact, committed here only as id plus digest. Caches match `studies/*/analysis/_cache_*`
in `.gitignore`. A large file merged into `main` stays in history for good.
If the CI gate fails on a pull request, deleting the file in a new commit does not fix it,
because the merge would still carry the commit that added it. Rewrite your own PR branch
instead; the failure message prints the commands (`git reset --soft` and recommit, or
`git rebase -i`, then `git push --force-with-lease`). Only `main` refuses rewrites.

## 5. Non-study changes

Changes outside `studies/` (launchers, `code/`, configs, skills) stay in the dispatcher. Open
a dispatcher PR from a fresh branch off dispatcher `main`, carrying only those changes:

```bash
cd ../aind-dynamic-foraging-bfm-dispatcher      # back in the dispatcher clone
git switch <study-branch>                       # the original (now archived) branch
BASE=$(git merge-base HEAD origin/main)
git format-patch -o /tmp/<study>-code "$BASE"..HEAD -- . ':!studies/'
git switch -c <study>-code origin/main          # a fresh branch for the dispatcher PR
git am --3way /tmp/<study>-code/*.patch
```

Merge it before the studies PR if the study depends on it.

## 6. Verify locally, then open PRs

On Allen HPC never run these on the login node (AGENTS.md section 5); wrap each in
`srun -n 1 -p aind ...` (`-n 1`: srun defaults to two tasks there, which run the command twice).

```bash
python3 scripts/check_file_sizes.py --range origin/main..HEAD   # every incoming commit, as CI does
python -m unittest discover -s tests -t .
python scripts/check_regeneration.py --study <NN-study>
```

Open the PRs against `main` here, in small pieces: shared helpers, then the study, then large
analysis drops. Each needs CI green and one approval, and merges with a merge commit (squash
is disabled). In the study's `CHANGELOG.md`, note that launches before the migration record
dispatcher commits, preserved on `archive/<study-branch>`.

## 7. Launching afterwards

Launch from the dispatcher, pointing at the sweep in this repository:

```bash
cd aind-dynamic-foraging-bfm-dispatcher
python code/launch_beaker_resumable.py \
  --sweep ../aind-dynamic-foraging-bfm-studies/studies/<NN-study>/variants/<variant>/sweep.yaml \
  --experiment ../aind-dynamic-foraging-bfm-studies/studies/<NN-study>/variants/<variant>/experiment.yaml \
  --label <label> --note "<why>"
```

`launch_hpc.py --sweep-yaml ...` works the same way. Both launchers record
`studies_git_commit` (this repository) separately from `dispatcher_git_commit`. They refuse
to launch when `studies/<NN-study>/` has uncommitted changes. Commit first, or pass
`--diagnostic-dirty-ok "<reason>"` for a diagnostic run.

## Checklist for an agent

- [ ] Skills pack >= 0.11.0 loaded (updated, then Claude Code restarted)
- [ ] Preconditions met; old branch pushed as `archive/<study-branch>`
- [ ] `format-patch ... -- studies/` applied with `git am --3way` on `study/<NN-study>`
- [ ] No `/workspace/.../dispatcher/studies` command left in any sweep or experiment template
- [ ] HPC scripts use `STUDIES_ROOT`; no `../../code` paths
- [ ] `make all` runs offline; live pulls are `freeze-*` targets
- [ ] `check_file_sizes.py --range origin/main..HEAD`, unit tests and `check_regeneration.py --study` pass (compute node on HPC)
- [ ] Non-study changes in a separate dispatcher PR
- [ ] Studies PRs opened in pieces; CHANGELOG notes the archive branch
