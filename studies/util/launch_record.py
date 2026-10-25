"""Schema + writer for INTERVENTION launch records (all studies).

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

For HPC prefer the **W&B sweep id**: SLURM job ids are recycled and drop out of `sacct`
history, so a sweep id is the only ref that stays resolvable months later. `launch_hpc.py`
writes `hpc_sweep.json` for the first launch (since #172); interventions after it still use
this helper.

Times are Seattle per AGENTS.md section 7.
"""
from __future__ import annotations

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
PLATFORMS = ("beaker", "hpc", "none")

# Ref types per platform. Beaker identifies a submitted unit by experiment; HPC by W&B sweep
# (durable) and SLURM job/array ids (recycled, and they age out of sacct — prefer the sweep).
REF_TYPES = ("beaker_experiment", "wandb_sweep", "slurm_array_job", "slurm_job")


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
    }
    allowed = allowed_ref_types.get(platform)
    if allowed is not None:
        for r in job_refs:
            if r.get("type") not in allowed:
                raise ValueError(
                    f"job_ref {r!r} not valid for platform {platform!r}; expected type in {sorted(allowed)}"
                )

    path = Path(path)
    launch_dir = _launch_record_root(path)
    initial = _initial_launch(launch_dir, wandb_group)
    variant_dir = launch_dir.parent
    deviations = dict(deviations or {})
    if note is not None and len(note.strip()) < 10:
        raise ValueError("note must say why this action exists (at least 10 characters; "
                         "schemas/launch-record.v2.schema.json)")
    if manifests is None:
        manifests = list(initial.get("manifests") or [])
    if kind != "backfill":            # a backfill submits nothing, so there is no submission to diff
        _check_manifest_diff(path.parent, variant_dir, initial, kind, manifests, deviations)
    revisions = dict(revisions or initial.get("revisions") or {})

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
    }

    path.parent.mkdir(parents=True, exist_ok=True)
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
            if m:
                out[f"{prefix}={m[1]}"] = m[2]
            else:
                out.update(_flatten(v, f"{prefix}{i}."))
        return out or {prefix.rstrip("."): []}
    return {prefix.rstrip("."): obj}


# Only `extend` may change the step budget (docs/adr/0001); any other kind that changes it is
# a new launch, whatever `deviations` declares.
_EXTEND_ONLY = ("n_steps",)


def _launch_manifest_text(name: str, initial: dict, variant_dir: Path) -> str:
    """The launch's manifest as submitted: its immutable snapshot beside the launch record,
    else the file at the launch's studies commit. Never the variant's current file, which
    may have been edited since."""
    snap = initial["_dir"] / Path(name).name
    if snap.exists():
        return snap.read_text()
    sha = (initial.get("revisions") or {}).get("studies")
    if sha:
        target = (variant_dir / name).resolve()
        try:
            top = subprocess.run(["git", "-C", str(variant_dir), "rev-parse", "--show-toplevel"],
                                 capture_output=True, text=True, check=True).stdout.strip()
            rel = target.relative_to(Path(top).resolve()).as_posix()
            return subprocess.run(["git", "-C", top, "show", f"{sha}:{rel}"],
                                  capture_output=True, text=True, check=True).stdout
        except (subprocess.CalledProcessError, ValueError, OSError):
            pass
    raise ValueError(f"cannot recover the launch's {name} (no snapshot beside its record, and not "
                     f"at studies {sha or '<no revision>'}); refusing to compare against a file "
                     "that may have changed since the launch")


def _check_manifest_diff(record_dir: Path, variant_dir: Path, initial: dict, kind: str,
                         manifests: list[str], deviations: dict) -> None:
    """Refuse a submitted manifest that differs from the launch's in an undeclared key.

    ``manifests`` are variant-relative paths of what this action submitted; each is matched
    to the launch manifest with the same file name and compared key by key (a key present
    in one and absent in the other differs, even if its value is null).
    """
    if not manifests:
        raise ValueError("manifests is empty; a record must name what it submitted "
                         "(schemas/launch-record.v2.schema.json)")
    launch = {Path(m).name: m for m in initial.get("manifests") or []}
    declared = set(deviations)
    missing = object()
    for name in manifests:
        candidates = (variant_dir / name, record_dir / Path(name).name, initial["_dir"] / Path(name).name)
        new = next((p for p in candidates if p.exists()), None)
        if new is None:
            raise ValueError(f"manifest {name} not found in {variant_dir}, {record_dir} or beside "
                             "the launch record")
        if Path(name).name not in launch:
            raise ValueError(f"{name} matches none of the launch's manifests {sorted(launch.values())}; "
                             "a different spec is a new launch (docs/adr/0001)")
        a = _flatten(yaml.safe_load(_launch_manifest_text(launch[Path(name).name], initial, variant_dir)) or {})
        b = _flatten(yaml.safe_load(new.read_text()) or {})
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
