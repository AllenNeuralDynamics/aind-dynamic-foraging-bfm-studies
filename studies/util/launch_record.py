"""Schema + writers for launch records (all studies): interventions, and initial records for runs
no dispatcher launcher submits (MEvolve, see :func:`write_initial`).

WHY THIS EXISTS. A study's *first* launch is well instrumented: the launchers write
`launch_record/beaker_resumable.json` with a fixed schema, and stamp the portable `meta.*` fields into each W&B run config.
wedged tasks, probing a failure, changing priority tier, backfilling a lost metric — has
historically been done with a bespoke script that invented its own JSON shape.

Study 06 is the worked example of the cost. Its five launch records looked like this:

    beaker_resumable.json  (launcher)  {chunk_size, mode, n_chunks, n_tasks, parts}
    beaker_resubmit20.json (by hand)   {bad_node, parts, reason}
    beaker_nanretry1.json  (by hand)   {experiment_id, purpose, tasks}
    beaker_rescue1.json    (by hand)   {bad_node, bad_node_hostname, experiment_id,
                                        progress_discarded, purpose, tasks}
    beaker_tier1.json      (by hand)   {experiment_id, purpose, superseded_experiment,
                                        tasks, thrash_evidence}

No two alike, **none carrying a timestamp**, none using `_meta`. Interventions are exactly
the surprising events provenance exists for, and they were the least well recorded. Prose in
notes.md filled the gap, but prose is not queryable and drifts from reality.

This module fixes the shape, not the prose. `write_intervention()` wraps the existing
`_meta.build_meta()` so every record gets a Seattle-time timestamp and both git SHAs for free.

    from launch_record import write_intervention

    write_intervention(
        path=variant / "launch_record" / "beaker_rescue1.json",
        kind="rescue",
        platform="beaker",
        wandb_group="mult-d-grid@20260718-151409",
        job_refs=[{"type": "beaker_experiment", "id": "01KYXNJYA7KEY47JXWJGEG4M3Y"}],
        study_root=STUDY,
        supersedes=["01KYB43A78GH0X95K1W70EFNSZ"],
        trigger={"symptom": "3 tasks pending 5 days",
                 "cause": "node aidc-h200-prd2 image-pull failure",
                 "evidence": {"node": "01KPVKJYXNWNJCH7ZFK0TBXPW5"}},
        tasks=[{"orig_task": "...-060", "new_task": "rescue1-060",
                "orig_wandb_run": "...", "new_wandb_run": "..."}],
        deviations={"fresh_run_ids": True, "renamed_tasks": True},
        cost={"progress_discarded_steps": {"060": 76120}},
    )

BEAKER *AND* HPC. The two backends are handled by one schema because a study can and does
use both (AGENTS.md section 13: GPU jobs to Beaker, CPU jobs to HPC SLURM, one repo driving
both). They differ only in what identifies a submitted unit, which is why `job_refs` is a
list of typed refs rather than a Beaker-shaped `experiment_ids`:

    beaker  ->  {"type": "beaker_experiment",  "id": "01K..."}
    hpc     ->  {"type": "wandb_sweep",        "id": "abc123"}     # the durable handle
                {"type": "slurm_array_job",    "id": "12345"}      # ages out of sacct
                {"type": "slurm_job",          "id": "12345_7"}
    cedar   ->  {"type": "cedar_session",      "id": "sess_01M...", "name": "cedar-smoke-..."}

For HPC prefer the **W&B sweep id**: SLURM job ids are recycled and drop out of `sacct`
history, so a sweep id is the only ref that stays resolvable months later. `launch_hpc.py`
writes `hpc_sweep.json` for the first launch (since #172); interventions after it still use
this helper.

Times are Seattle per AGENTS.md section 7.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from _meta import build_meta

# Kinds of post-launch intervention. Keep this list short and meaningful: it is what makes
# records queryable ("show me every rescue across all studies"). Add a kind only when an
# existing one genuinely does not describe the action.
KINDS = (
    "resubmit",      # re-run tasks that never produced usable data (bad node, lost image, ...)
    "rescue",        # re-run tasks wedged/stuck, discarding partial progress
    "probe",         # diagnostic re-run to test a hypothesis, not to fill a cell
    "tier-change",   # same work resubmitted at a different priority/preemptible tier
    "backfill",      # recover a metric post-hoc WITHOUT recompute (no new Beaker job)
    "extend",        # continue a finished run to a longer horizon
)

_REQUIRED_TRIGGER_KEYS = ("symptom", "cause")

# Compute backends. "none" is for kinds that create no job at all (backfill), so the field
# stays required and meaningful rather than being left blank.
PLATFORMS = ("beaker", "hpc", "cedar", "none")

# Ref types per platform. Beaker identifies a submitted unit by experiment; HPC by W&B sweep
# (durable) and SLURM job/array ids (recycled, and they age out of sacct — prefer the sweep);
# Cedar by session (one per grid point; names are permanent per workspace).
REF_TYPES = ("beaker_experiment", "wandb_sweep", "slurm_array_job", "slurm_job", "cedar_session")


def write_intervention(
    path: Path | str,
    *,
    kind: str,
    platform: str,
    wandb_group: str,
    study_root: Path | str,
    job_refs: list[dict[str, str]] | None = None,
    supersedes: list[str] | None = None,
    trigger: dict[str, Any] | None = None,
    tasks: list[dict[str, Any]] | None = None,
    deviations: dict[str, Any] | None = None,
    cost: dict[str, Any] | None = None,
    label: str | None = None,
    note: str | None = None,
    manifests: list[str] | None = None,
    revisions: dict[str, str] | None = None,
    data: dict[str, Any] | None = None,
    image: dict[str, Any] | None = None,
    produced_by: str = "launch_record.write_intervention",
) -> dict:
    """Write a schema-conformant intervention record and return it.

    Args:
        path: destination JSON, conventionally
            ``studies/<study>/variants/<variant>/launch_record/beaker_<label>.json``.
        kind: one of :data:`KINDS`.
        platform: one of :data:`PLATFORMS` — ``"beaker"``, ``"hpc"``, or ``"none"`` for a
            kind that submits nothing (backfill).
        wandb_group: the group these runs belong to — the join key that lets the
            validator reconcile records against what actually landed in W&B.
        study_root: ``studies/<study>/``, for the wrapper SHA in ``_meta``.
        job_refs: typed handles for the submitted unit(s), e.g.
            ``[{"type": "beaker_experiment", "id": "01K..."}]`` on Beaker or
            ``[{"type": "wandb_sweep", "id": "abc123"}]`` on HPC. Omit for
            ``kind="backfill"``, which by definition creates no job. On HPC prefer the
            W&B sweep id — SLURM ids are recycled and age out of ``sacct``.
        supersedes: experiment ids (or run ids) this action replaces, so a reader can
            follow the chain backwards without reading prose.
        trigger: why this was needed — requires ``symptom`` and ``cause``, plus any
            ``evidence``. The point is to record the *observation*, not just the verdict.
        tasks: per-task mapping, typically
            ``{orig_task, new_task, orig_wandb_run, new_wandb_run}``.
        deviations: deliberate departures from a plain resubmit — e.g.
            ``{"fresh_run_ids": True, "renamed_tasks": True,
               "context_changed": {"priority": "normal", "preemptible": False}}``.
            Every study-06 intervention made such a choice for a real reason; capturing it
            here makes the reason auditable instead of buried in a commit message.
        cost: what the action gave up, e.g. ``{"progress_discarded_steps": {...}}``.
        label, note: as for a launch; ``label`` defaults to the record's file stem and
            ``note`` to the trigger's symptom and cause.
        manifests: variant-relative paths of the manifests this action submitted (e.g. its
            own snapshot under ``launch_record/``). Each must equal the launch's manifest of
            the same file name, as submitted then, except for keys named in ``deviations``
            (dotted paths, e.g. ``"parameters.n_steps"``); only ``extend`` may change the step
            budget (docs/adr/0001). Defaults to the launch's own manifests.
        revisions: repo -> full SHA of the code that ran; defaults to the initial launch's.
        data, image: what the runs trained on and ran in (studies#25); default to the initial
            launch's, and are left out of a backfill (it submits nothing) unless given.

    The record is v2 (schemas/launch-record.v2.schema.json) and must join a W&B group that
    has an initial launch record in the same variant.

    Raises:
        ValueError: on an unknown ``kind``/``platform``/ref type, a ``trigger`` missing
            required keys, or a job-creating kind with no ``job_refs`` (a silent no-op
            record is worse than no record).
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    if platform not in PLATFORMS:
        raise ValueError(f"unknown platform {platform!r}; expected one of {PLATFORMS}")

    trigger = dict(trigger or {})
    missing = [k for k in _REQUIRED_TRIGGER_KEYS if not trigger.get(k)]
    if missing:
        raise ValueError(
            f"trigger is missing required key(s) {missing}. An intervention record without "
            "an observed symptom and a cause is not provenance, it is a rumour."
        )

    job_refs = [dict(r) for r in (job_refs or [])]
    for r in job_refs:
        if r.get("type") not in REF_TYPES:
            raise ValueError(f"job_ref {r!r} has unknown type; expected one of {REF_TYPES}")
        if not r.get("id"):
            raise ValueError(f"job_ref {r!r} has no id")

    if kind == "backfill":
        if platform != "none":
            raise ValueError('kind="backfill" submits nothing, so platform must be "none"')
        if job_refs:
            raise ValueError('kind="backfill" submits nothing, so job_refs must be empty')
    else:
        if platform == "none":
            raise ValueError('platform="none" is only valid for kind="backfill"')
        if not job_refs:
            raise ValueError(
                f"kind={kind!r} submits work, so job_refs must be non-empty "
                '(beaker: [{"type":"beaker_experiment","id":...}]; '
                'hpc: [{"type":"wandb_sweep","id":...}])'
            )

    allowed_ref_types = {
        "beaker": {"beaker_experiment"},
        "hpc": {"wandb_sweep", "slurm_array_job", "slurm_job"},
        "cedar": {"cedar_session"},
    }
    allowed = allowed_ref_types.get(platform)
    if allowed is not None:
        for r in job_refs:
            if r.get("type") not in allowed:
                raise ValueError(
                    f"job_ref {r!r} not valid for platform {platform!r}; expected type in {sorted(allowed)}"
                )

    path = Path(path).resolve()      # a repo-relative destination must not mix with resolved paths
    launch_dir = _launch_record_root(path)
    initial = _initial_launch(launch_dir, wandb_group)
    variant_dir = launch_dir.parent
    deviations = dict(deviations or {})
    if note is not None and len(note.strip()) < 10:
        raise ValueError("note must say why this action exists (at least 10 characters; "
                         "schemas/launch-record.v2.schema.json)")
    inherited = manifests is None
    if inherited:
        manifests = list(initial.get("manifests") or [])
    if not manifests:
        raise ValueError("manifests is empty; a record must name what it submitted "
                         "(schemas/launch-record.v2.schema.json)")
    resolved = _resolve_manifests(path.parent, variant_dir, initial, manifests, inherited)
    if kind != "backfill":            # a backfill submits nothing, so there is no submission to diff
        _check_manifest_diff(variant_dir, initial, kind, resolved, deviations)
    # stored variant-relative, so scripts/check_provenance.py finds each one from the variant
    manifests = [p.relative_to(variant_dir).as_posix() for p in resolved.values()]
    revisions = dict(revisions or initial.get("revisions") or {})
    if platform != "none":
        data = data if data is not None else initial.get("data")
        # an image only carries over to the same backend (an HPC rescue of a Beaker launch runs
        # in a conda env, not the Beaker image); otherwise it stays unrecorded unless given
        inherited = initial.get("image")
        if image is None and isinstance(inherited, dict) and inherited.get("platform") == platform:
            image = inherited

    meta = build_meta(produced_by, [wandb_group], study_root=study_root)
    record = {
        "_meta": {"schema_version": 2, **meta},
        "kind": kind,
        "platform": platform,
        "wandb_group": wandb_group,
        "label": label or path.stem,
        "note": note or f"{kind}: {trigger['symptom']} (cause: {trigger['cause']})",
        "revisions": revisions,
        "manifests": list(manifests),
        "job_refs": job_refs,
        "supersedes": list(supersedes or []),
        "trigger": trigger,
        "tasks": list(tasks or []),
        "deviations": deviations,
        "cost": dict(cost or {}),
        **({"data": data} if data is not None else {}),
        **({"image": image} if image is not None else {}),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def conda_image(prefix: Path | str) -> dict:
    """The launch record's HPC ``image`` for a conda env, digested as the dispatcher's launcher
    does (``image_record_hpc`` in the dispatcher's ``code/launch_inputs.py``): sha256 over the
    sorted conda package filenames in ``conda-meta`` and then the sorted pip ``*.dist-info`` names
    in each site-packages, so it changes when any installed version does."""
    prefix = Path(prefix).expanduser().resolve()
    site_packages = sorted(prefix.glob("lib/python*/site-packages"))
    if not site_packages:
        raise ValueError(f"{prefix} has no lib/python*/site-packages; is it a conda env prefix?")
    names = sorted(p.name for p in (prefix / "conda-meta").glob("*.json"))
    names += sorted(p.name for sp in site_packages for p in sp.glob("*.dist-info"))
    return {"platform": "hpc", "conda_env": str(prefix),
            "lock_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest()}


_SHA = re.compile(r"^[0-9a-f]{40}$")


def write_initial(
    path: Path | str,
    *,
    platform: str,
    wandb_group: str,
    study_root: Path | str,
    label: str,
    note: str,
    job_refs: list[dict[str, str]],
    revisions: dict[str, str],
    manifests: list[str],
    data: dict[str, Any],
    image: dict[str, Any],
    recorded_after_launch: bool = False,
    command: str | None = None,
    n_tasks: int | None = None,
    produced_by: str = "launch_record.write_initial",
) -> dict:
    """Write the v2 initial record of a launch no dispatcher launcher wrote, and return it.

    For runs of tools outside the dispatcher, such as MEvolve (study 12's 2-level EM scoring,
    study 40's searches): ``revisions`` names ``studies`` and ``mevolve`` (full SHAs) and need not
    name the dispatcher or the wrapper; ``data`` may be ``{"source": "freeze", ...}`` with the
    freeze's snapshot id, path (``uri``), manifest sha256 and the data config it was built from.

    Args:
        path: destination JSON under the variant's ``launch_record/``.
        platform: ``"beaker"``, ``"hpc"`` or ``"cedar"``.
        wandb_group: ``<variant>@<launch_id>``; the prefix must be the variant's directory name.
        study_root: ``studies/<study>/``, for ``_meta``.
        label, note: the launch's label and why it exists (at least 10 characters).
        job_refs: typed handles, e.g. ``[{"type": "slurm_job", "id": "27567407"}]``.
        revisions: repo -> full SHA; ``studies`` plus either ``dispatcher`` and ``wrapper`` or
            ``mevolve``.
        manifests: variant-relative paths of what was submitted (e.g. the sbatch script with its
            command-line arguments); each must exist, and is copied beside the record as the
            launch's snapshot. Several launches in one variant (one per model, say) each need
            their own subdirectory of ``launch_record/``.
        data, image: what the run read and ran in (studies#25); both required.
        recorded_after_launch: True when written after submission from the launch facts kept in
            the variant's notes; stored as ``_meta.recorded_after_launch``. The manifests are then
            snapshotted from git at ``revisions["studies"]``, not from the working copy.
        command: the exact command line submitted, when the manifest alone does not carry its
            arguments (e.g. ``sbatch --array=0-17 run_2level.sbatch``).

    Raises:
        ValueError: on a bad platform, ref, group prefix, SHA, note, missing manifest, missing
            ``data``/``image``, or a second initial record for the same group.
    """
    if platform not in ("beaker", "hpc", "cedar"):
        raise ValueError(f'an initial launch runs on "beaker", "hpc" or "cedar", not {platform!r}')
    allowed = {"beaker": {"beaker_experiment"}, "hpc": {"wandb_sweep", "slurm_array_job", "slurm_job"},
               "cedar": {"cedar_session"}}[platform]
    job_refs = [dict(r) for r in job_refs or []]
    if not job_refs:
        raise ValueError("job_refs is empty; an initial launch submitted something")
    for r in job_refs:
        if r.get("type") not in allowed or not r.get("id"):
            raise ValueError(f"job_ref {r!r} not valid for platform {platform!r}; expected type in {sorted(allowed)} and an id")
    revisions = dict(revisions or {})
    bad = {k: v for k, v in revisions.items() if not _SHA.match(str(v))}
    if bad:
        raise ValueError(f"revisions must be full 40-character SHAs: {bad}")
    if "studies" not in revisions or not ({"dispatcher", "wrapper"} <= revisions.keys() or "mevolve" in revisions):
        raise ValueError("revisions needs studies, plus dispatcher and wrapper or mevolve "
                         "(schemas/launch-record.v2.schema.json)")
    if not note or len(note.strip()) < 10:
        raise ValueError("note must say why this launch exists (at least 10 characters)")
    if not data or not image:
        raise ValueError("data and image are required: what the run read and the environment it ran in (studies#25)")

    path = Path(path).resolve()
    if path.exists():
        raise ValueError(f"{path} already exists; a launch record is never overwritten")
    launch_dir = _launch_record_root(path)
    variant_dir = launch_dir.parent
    if wandb_group.split("@")[0] != variant_dir.name:
        raise ValueError(f"group {wandb_group} must start with {variant_dir.name}@ (docs/adr/0001)")
    try:
        _initial_launch(launch_dir, wandb_group)
    except ValueError:
        pass
    else:
        raise ValueError(f"{wandb_group} already has an initial launch record; a later action is an intervention")
    if (image or {}).get("platform") != platform:
        raise ValueError(f"image describes {(image or {}).get('platform')!r}, but the launch ran on {platform!r}")
    if not manifests:
        raise ValueError("manifests is empty; a record must name what it submitted")
    names = [Path(m).name for m in manifests]
    if len(set(names)) != len(names):     # snapshots are stored by file name beside the record
        raise ValueError(f"manifests {list(manifests)} share a file name; rename one so each "
                         "snapshot beside the record is distinct")
    snapshots = {}
    for m in manifests:
        if Path(m).suffix == ".json":   # a JSON snapshot in launch_record/ would read as a record
            raise ValueError(f"manifest {m} is JSON; submit YAML or a script, which cannot be "
                             "mistaken for a launch record")
        if not recorded_after_launch and not (variant_dir / m).exists():
            raise ValueError(f"manifest {m} not found in {variant_dir}; commit it beside the record")
        if recorded_after_launch:      # the working copy may have changed since the submission
            text = _text_at_commit(variant_dir / m, revisions["studies"])
            if text is None:
                raise ValueError(f"manifest {m} is not at studies {revisions['studies']}; a record "
                                 "written after launch snapshots what was committed when it ran")
        else:
            text = (variant_dir / m).read_text()
        snap = path.parent / Path(m).name
        if snap.exists() and snap.read_text() != text:
            raise ValueError(f"{snap} already holds a different manifest; write each launch's record "
                             "in its own subdirectory of launch_record/")
        if snap != (variant_dir / m).resolve():
            snapshots[snap] = text

    meta = build_meta(produced_by, [wandb_group], study_root=study_root)
    record = {
        "_meta": {"schema_version": 2, **meta,
                  **({"recorded_after_launch": True} if recorded_after_launch else {})},
        "kind": "initial",
        "platform": platform,
        "wandb_group": wandb_group,
        "label": label,
        "note": note,
        "revisions": revisions,
        "manifests": list(manifests),
        "job_refs": job_refs,
        **({"command": command} if command else {}),
        **({"n_tasks": n_tasks} if n_tasks is not None else {}),
        "data": dict(data),
        "image": dict(image),
    }
    from jsonschema import Draft202012Validator   # dev requirement, as for scripts/check_provenance.py
    schema = json.loads((Path(__file__).resolve().parents[2] / "schemas" / "launch-record.v2.schema.json").read_text())
    errors = sorted(f"{'/'.join(map(str, e.absolute_path)) or '<record>'}: {e.message}"
                    for e in Draft202012Validator(schema).iter_errors(record))
    if errors:
        raise ValueError("the record does not match schemas/launch-record.v2.schema.json: " + "; ".join(errors))
    # snapshot each manifest beside the record, as the dispatcher launchers do, so a later
    # intervention compares against what was submitted even if the variant's copy changes;
    # written only once the record is known to be valid
    path.parent.mkdir(parents=True, exist_ok=True)
    for snap, text in snapshots.items():
        snap.write_text(text)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def _launch_record_root(path: Path) -> Path:
    """The variant's launch_record/ directory that ``path`` lives in (subdirectories allowed)."""
    for parent in [path.parent, *path.parent.parents]:
        if parent.name == "launch_record":
            return parent
    raise ValueError(f"{path} is not under a variant's launch_record/ directory")


def _initial_launch(launch_dir: Path, wandb_group: str) -> dict:
    """The v2 initial launch record of ``wandb_group`` in this variant (ADR 0001)."""
    for p in sorted(launch_dir.rglob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (isinstance(rec, dict) and rec.get("kind") == "initial"
                and rec.get("wandb_group") == wandb_group):
            rec["_dir"] = p.parent
            return rec
    raise ValueError(
        f"no initial launch record for {wandb_group} in {launch_dir}; an intervention joins an "
        "existing launch's group (docs/adr/0001). A different configuration is a new launch.")


_OVERRIDE = re.compile(r"^[+~]*([A-Za-z_][\w.]*)=(.*)$")


def _manifest_keys(name: str, text: str) -> dict[str, Any]:
    """A manifest's keys for the diff: YAML or JSON flattened; any other file (an sbatch script
    for a MEvolve run) is one key, ``script``, so any edit to it must be declared, plus one key
    per Hydra-style override token in it (``script.=model.training.n_steps``), so declaring the
    script edit cannot slip a step-budget change past the extend-only rule."""
    if Path(name).suffix in (".yaml", ".yml", ".json"):
        return _flatten(yaml.safe_load(text) or {})
    keys: dict[str, Any] = {"script": text}
    for token in text.split():
        token = token.strip("'\"")             # a quoted override is still an override
        m = _OVERRIDE.match(token)
        if m:                                   # a repeated key keeps every value, in order
            keys[f"script.={m[1]}"] = keys.get(f"script.={m[1]}", ()) + (token,)
    return keys


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Dotted key -> leaf value. Lists are flattened too, so a Hydra override inside a sweep
    `command` or a Beaker `tasks[*].command` (``model.training.n_steps=150000``) becomes its
    own key (``...command.=model.training.n_steps``) rather than hiding in one list leaf."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out or {prefix.rstrip("."): {}}
    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            m = _OVERRIDE.match(v) if isinstance(v, str) else None
            if m:                     # keep the whole token: `~k=v` (delete) differs from `k=v`
                out[f"{prefix}={m[1]}"] = v
            else:
                out.update(_flatten(v, f"{prefix}{i}."))
        return out or {prefix.rstrip("."): []}
    return {prefix.rstrip("."): obj}


# Only `extend` may change the step budget (docs/adr/0001); any other kind that changes it is
# a new launch, whatever `deviations` declares.
_EXTEND_ONLY = ("n_steps",)


def _text_at_commit(path: Path, sha: str) -> str | None:
    """``path``'s text in the studies repo at commit ``sha``, or None if git cannot show it."""
    # run git from the nearest directory that still exists: the manifest's own may be gone
    here = next((d for d in path.resolve().parents if d.is_dir()), Path("/"))
    try:
        top = subprocess.run(["git", "-C", str(here), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=True).stdout.strip()
        rel = path.resolve().relative_to(Path(top).resolve()).as_posix()
        return subprocess.run(["git", "-C", top, "show", f"{sha}:{rel}"],
                              capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, ValueError, OSError):
        return None


def _launch_manifest_text(name: str, initial: dict, variant_dir: Path) -> str:
    """The launch's manifest as submitted: its immutable snapshot beside the launch record,
    else the file at the launch's studies commit. Never the variant's current file, which
    may have been edited since."""
    snap = initial["_dir"] / Path(name).name
    if snap.exists():
        return snap.read_text()
    sha = (initial.get("revisions") or {}).get("studies")
    if sha:
        text = _text_at_commit(variant_dir / name, sha)
        if text is not None:
            return text
    raise ValueError(f"cannot recover the launch's {name} (no snapshot beside its record, and not "
                     f"at studies {sha or '<no revision>'}); refusing to compare against a file "
                     "that may have changed since the launch")


def _resolve_manifests(record_dir: Path, variant_dir: Path, initial: dict,
                       manifests: list[str], inherited: bool) -> dict[str, Path]:
    """name -> file. Explicit manifests are variant-relative paths that must exist as given;
    only the launch's own (inherited) manifests fall back to beside the new or launch record,
    so a missing submission artifact is never silently replaced by the launch's file."""
    out = {}
    for name in manifests:
        candidates = ((variant_dir / name, record_dir / Path(name).name, initial["_dir"] / Path(name).name)
                      if inherited else (variant_dir / name,))
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            raise ValueError(f"manifest {name} not found in {variant_dir}, {record_dir} or beside "
                             "the launch record")
        out[name] = found.resolve()
    return out


def _check_manifest_diff(variant_dir: Path, initial: dict, kind: str,
                         manifests: dict[str, Path], deviations: dict) -> None:
    """Refuse a submitted manifest that differs from the launch's in an undeclared key.

    Each submitted file is matched to the launch manifest with the same file name and compared
    key by key (a key present in one and absent in the other differs, even if null).
    """
    launch = {Path(m).name: m for m in initial.get("manifests") or []}
    declared = set(deviations)
    missing = object()
    for name, new in manifests.items():
        if Path(name).name not in launch:
            raise ValueError(f"{name} matches none of the launch's manifests {sorted(launch.values())}; "
                             "a different spec is a new launch (docs/adr/0001)")
        a = _manifest_keys(name, _launch_manifest_text(launch[Path(name).name], initial, variant_dir))
        b = _manifest_keys(name, new.read_text())
        changed = sorted(k for k in a.keys() | b.keys() if a.get(k, missing) != b.get(k, missing))
        covered = lambda k: any(k == d or k.startswith(d + ".") for d in declared)
        undeclared = [k for k in changed if not covered(k)]
        if undeclared:
            raise ValueError(
                f"{name} differs from the launch's in undeclared key(s) {undeclared}; declare each "
                "in deviations, or launch anew if the configuration changed (docs/adr/0001)")
        restricted = [k for k in changed
                      if any(part.lstrip("=") in _EXTEND_ONLY for part in k.split("."))]
        if restricted and kind != "extend":
            raise ValueError(f"{kind} changes {restricted}; only an extend may change the step "
                             "budget (docs/adr/0001). Use kind='extend', or launch anew")


def load_records(launch_record_dir: Path | str) -> list[dict]:
    """Load every JSON in a ``launch_record/`` dir, newest schema or legacy alike.

    Legacy records (pre-schema, e.g. study 06's hand-written ones) are returned as-is with
    ``_schema: "legacy"`` so the validator can report them without crashing. Being able to
    read the old shape is what lets a study adopt this incrementally.
    """
    out = []
    root = Path(launch_record_dir)
    directories = [root, *sorted(root.parent.glob(root.name + "_*"))]
    for p in sorted(path for directory in directories for path in directory.rglob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            out.append({"_schema": "unreadable", "_path": str(p), "_error": f"{type(e).__name__}: {e}"})
            continue
        if isinstance(d.get("_meta"), dict) and d["_meta"].get("schema_version") == 2:
            schema = "v2"            # schemas/launch-record.v2.schema.json (launchers and interventions)
        elif "_meta" in d and "kind" in d:
            schema = "current"
        elif d.get("mode") in ("resumable-grid", "hpc-sweep") or "sweep_id" in d:
            schema = "launcher"
        else:
            schema = "legacy"
        d.setdefault("_schema", schema)
        d["_path"] = str(p)
        out.append(d)
    return out


def refs_in(records: list[dict], ref_type: str) -> set[str]:
    """Every id of ``ref_type`` mentioned by any record, current schema or legacy.

    Legacy shapes are handled explicitly rather than guessed at, so a study can adopt the
    schema without a migration: study 06 used both a flat ``experiment_id`` and a
    ``parts: [{experiment_id: ...}]`` list, and `launch_hpc.py` wrote nothing at all.
    """
    ids: set[str] = set()
    for r in records:
        for ref in r.get("job_refs", []) or []:
            if isinstance(ref, dict) and ref.get("type") == ref_type and ref.get("id"):
                ids.add(str(ref["id"]))
        if ref_type == "wandb_sweep":
            # launcher records: `hpc_sweep.json` (short id) and `beaker_sweep.json`
            # (entity/project/id); the validator compares short ids.
            for key in ("wandb_sweep_id", "sweep_id"):
                v = r.get(key)
                if isinstance(v, str) and v:
                    ids.add(v.rpartition("/")[2])
        if ref_type != "beaker_experiment":
            continue
        # legacy Beaker shapes
        for key in ("experiment_ids", "experiment_id"):
            v = r.get(key)
            if isinstance(v, str):
                ids.add(v)
            elif isinstance(v, list):
                ids.update(x for x in v if isinstance(x, str))
        for part in r.get("parts", []) or []:
            if isinstance(part, dict) and isinstance(part.get("experiment_id"), str):
                ids.add(part["experiment_id"])
    return ids


def experiment_ids_in(records: list[dict]) -> set[str]:
    """Beaker experiment ids across current and legacy records."""
    return refs_in(records, "beaker_experiment")
