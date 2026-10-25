"""Shared provenance helper for analysis JSON outputs (all studies).

Per docs/posthoc-analysis.md "JSON output contract": every JSON written by an
analysis script must carry an opening `_meta` block. This helper centralises
the construction so producer scripts stay terse and the schema stays uniform
across studies.

Usage from a study's analysis script:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "util"))  # studies/util
    from _meta import build_meta

    STUDY = Path(__file__).resolve().parents[1]          # studies/<study>
    out = {"_meta": build_meta("analysis/scaling.py", WANDB_GROUPS, study_root=STUDY), ...}
    out_json.write_text(json.dumps(out, indent=2))

Times stamped in America/Los_Angeles per AGENTS.md section 7.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")

# _meta key -> the launch-record keys that hold that repo's SHA, newest schema first, then
# the legacy launcher key, then the `launch.revisions` block hand-written records carry.
# The bare repo names are the keys of a v2 record's `revisions` block
# (schemas/launch-record.v2.schema.json).
_SOURCE_SHA_KEYS = {
    "source_studies_git_shas": ("studies_git_commit", "studies_commit", "studies"),
    "source_dispatcher_git_shas": ("dispatcher_git_commit", "dispatcher_commit", "dispatcher"),
    "source_wrapper_git_shas": ("wrapper_git_commit", "wrapper_commit", "wrapper"),
    "source_foraging_models_git_shas": ("foraging_models_git_commit", "foraging_models_commit",
                                        "foraging_models"),
}


# Variables that redirect git's repository discovery away from cwd. The sandbox exports
# GIT_DIR/GIT_WORK_TREE for the dispatcher checkout (git-session-isolation); after the split
# a producer here runs in a different checkout, and must not be described as the dispatcher.
_GIT_DISCOVERY_VARS = (
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR",
)


def _git_env(cwd: Path) -> dict[str, str]:
    """Ambient git env only when ``cwd`` is inside the exported ``GIT_WORK_TREE``."""
    env = dict(os.environ)
    work_tree = env.get("GIT_WORK_TREE")
    if work_tree:
        try:
            Path(cwd).resolve().relative_to(Path(work_tree).resolve())
            return env
        except ValueError:
            pass
    for var in _GIT_DISCOVERY_VARS:
        env.pop(var, None)
    return env


def _git(*args: str) -> str | None:
    """Best-effort git query from this file's directory; None outside a checkout.

    `git` walks up to the repo root, so running it from studies/util (or any subdir)
    describes the checkout that holds the studies -- the dispatcher before the split
    (#124), the studies repository after it. It is therefore recorded as `studies_git_sha`,
    never as the dispatcher's. Same per-worktree env rule as ``code/provenance.py``
    (duplicated: studies/util must not import from code/).
    """
    cwd = Path(__file__).parent
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=cwd, timeout=10, env=_git_env(cwd),
        )
    except Exception:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _studies_state() -> tuple[str | None, bool | None]:
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return (sha or None), (None if status is None else bool(status))


def _record_groups(record: dict) -> set[str]:
    groups = set()
    for key in ("wandb_group", "group"):
        if isinstance(record.get(key), str):
            groups.add(record[key])
    groups.update((record.get("_meta") or {}).get("wandb_groups") or [])
    launch = record.get("launch") or {}
    if isinstance(launch, dict):
        g = (launch.get("wandb") or {}).get("group") if isinstance(launch.get("wandb"), dict) else None
        if isinstance(g, str):
            groups.add(g)
    return groups


def _source_shas(study_root: Path | None, wandb_groups: list[str]) -> tuple[dict, list[str]]:
    """Per-repo SHAs of the launches that produced ``wandb_groups``, from their launch records.

    The arrays are what makes an analysis traceable to the exact code that produced the runs
    it reads, once that code lives in two or three repositories. A group with no record gets
    no guessed value -- it is listed in the findings so the gap is visible in the artifact.
    """
    found = {key: set() for key in _SOURCE_SHA_KEYS}
    covered: set[str] = set()                                   # groups with any record
    covered_by_key = {key: set() for key in _SOURCE_SHA_KEYS}   # groups with that repo's SHA
    wanted = set(wandb_groups)
    findings: list[str] = []
    if study_root is None:
        findings.append("no study_root given, so source launch records were not read")
    else:
        for path in sorted(Path(study_root).glob("variants/*/launch_record*/**/*.json")):
            try:
                rec = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(rec, dict):
                continue
            groups = _record_groups(rec) & wanted
            if not groups:
                continue
            covered |= groups
            revisions = (rec.get("launch") or {}).get("revisions") if isinstance(rec.get("launch"), dict) else None
            sources = [rec] + [r for r in (revisions, rec.get("revisions")) if isinstance(r, dict)]
            for key, aliases in _SOURCE_SHA_KEYS.items():
                for src in sources:
                    for alias in aliases:
                        values = src.get(alias)
                        for v in (values if isinstance(values, list) else [values]):
                            if isinstance(v, str) and _FULL_SHA.fullmatch(v):
                                found[key].add(v)
                                covered_by_key[key] |= groups
    if study_root is not None:
        missing = sorted(wanted - covered)
        if missing:
            findings.append("no launch record found for W&B group(s): " + ", ".join(missing))
        for key in _SOURCE_SHA_KEYS:
            # a record that names the group but carries no SHA for this repo is a gap too
            gap = sorted((wanted & covered) - covered_by_key[key])
            if gap:
                repo = key[len("source_"):-len("_git_shas")]
                findings.append(f"no {repo} SHA in the launch record(s) of W&B group(s): "
                                + ", ".join(gap))
    return {key: sorted(v) for key, v in found.items()}, findings


def _wrapper_sha(study_root: Path) -> str | None:
    """Wrapper commit pinned in the study's environment.lock, or None.

    This is the version of aind-dynamic-foraging-bfm-wrapper (or its legacy
    aind-disrnn-wrapper name) whose post_training_analysis produced the W&B
    keys these analyses read.
    """
    lock = Path(study_root) / "environment.lock"
    try:
        m = re.search(
            r"aind-(?:disrnn|dynamic-foraging-bfm)-wrapper\.git@([0-9a-f]{7,40})",
            lock.read_text(),
        )
        return m.group(1) if m else None
    except Exception:
        return None


def build_meta(produced_by: str, wandb_groups: list[str], study_root: Path | str | None = None,
               *, kind: str | None = None, code: list | None = None, inputs: list | None = None,
               sources: list[dict] | None = None) -> dict:
    """Return a _meta dict per the JSON output contract.

    Args:
        produced_by: path to the producer script, relative to the study root
            (e.g. ``"analysis/scaling.py"``).
        wandb_groups: W&B group names this run pulled data from.
        study_root: path to ``studies/<study>/`` (where environment.lock and the variants'
            launch records live). If None, the wrapper sha and the ``source_*_git_shas``
            arrays are omitted, and ``provenance_findings`` says so.
        kind, code, inputs, sources: passing ``code`` opts into ``_meta`` v2
            (schemas/meta.v2.schema.json). ``kind`` is ``"producer"`` (default) or ``"freeze"``;
            ``code`` is the producer script (pass ``__file__``), to which every imported
            ``studies/util`` module is added; ``inputs`` are the committed files it read;
            ``sources`` the external objects a freeze read. Write the artifact with
            ``write_artifact()``, which adds the ``outputs`` digests. Without ``code`` the
            v1 shape is returned unchanged, so existing producers regenerate identically.
    """
    now_pt = datetime.datetime.now(ZoneInfo("America/Los_Angeles"))
    studies_sha, studies_dirty = _studies_state()
    root = Path(study_root) if study_root else None
    source_shas, findings = _source_shas(root, list(wandb_groups))
    meta = {
        "produced_by": produced_by,
        "produced_at_pt": now_pt.isoformat(timespec="seconds"),
        # The checkout this producer ran from (studies repo after #124). Pre-split outputs
        # carried this value as `dispatcher_git_sha`; new outputs name what it is.
        "studies_git_sha": studies_sha,
        "studies_git_dirty": studies_dirty,
        "wrapper_git_sha": _wrapper_sha(study_root) if study_root else None,
        "wandb_groups": list(wandb_groups),
        **source_shas,
        "provenance_findings": findings,
    }
    if code is None:
        return meta
    kind = kind or "producer"
    if kind not in ("producer", "freeze"):
        raise ValueError(f"kind must be 'producer' or 'freeze', not {kind!r}")
    if kind == "freeze" and not sources:
        raise ValueError("a freeze must list the external `sources` it read")
    meta = {"schema_version": 2, "kind": kind, **meta,
            "code": [digest_entry(p) for p in _with_util_modules(code)],
            "inputs": [digest_entry(p) for p in inputs or []]}
    if sources:
        meta["sources"] = list(sources)
    return meta


# --- _meta v2 digests ---------------------------------------------------------------------
# One implementation shared by producers (write) and scripts/check_provenance.py (verify), so
# the two can never disagree on what a digest covers.

REPO_ROOT = Path(__file__).resolve().parents[2]
_UTIL_DIR = Path(__file__).resolve().parent


def repo_path(path: Path | str) -> str:
    """Repository-relative POSIX path; relative inputs are resolved against the cwd."""
    return Path(path).resolve().relative_to(REPO_ROOT).as_posix()


def file_digest(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def body_digest(obj: dict, default=None) -> str:
    """Digest of a JSON artifact without its `_meta`, canonically serialised.

    A file cannot contain its own hash, so the artifact's entry in its own `outputs` covers
    the body only. Sorted keys and fixed separators make the bytes independent of insertion
    order and indentation. The body is round-tripped through JSON first, so an in-memory
    object hashes the same as the file read back: int keys ({10: .., 614: ..}) become the
    strings they are on disk, and `default` (e.g. for numpy values) is applied.
    """
    body = json.loads(json.dumps({k: v for k, v in obj.items() if k != "_meta"}, default=default))
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


RESULT_BLOCK = re.compile(r"<!-- BEGIN ([\w.-]+) -->\n?(.*?)<!-- END \1 -->", re.S)


def block_digest(text: str) -> str:
    """Digest of a result block, ignoring trailing whitespace and surrounding blank lines."""
    norm = "\n".join(line.rstrip() for line in text.strip("\n").splitlines()).strip()
    return hashlib.sha256(norm.encode()).hexdigest()


def result_blocks(report: Path | str) -> dict[str, str]:
    return dict(RESULT_BLOCK.findall(Path(report).read_text()))


def digest_entry(spec: Path | str) -> dict:
    """``{path, sha256}`` for a file, or for ``<report.md>#<block>`` a result block."""
    spec = str(spec)
    if "#" in spec:
        report, block = spec.split("#", 1)
        blocks = result_blocks(report)
        if block not in blocks:
            raise KeyError(f"{report} has no result block {block!r}")
        return {"path": f"{repo_path(report)}#{block}", "sha256": block_digest(blocks[block])}
    return {"path": repo_path(spec), "sha256": file_digest(spec)}


def _with_util_modules(code: list) -> list:
    """The given code files plus every loaded module from studies/util (except this one)."""
    paths = [Path(p).resolve() for p in code]
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and Path(f).resolve().parent == _UTIL_DIR and Path(f).resolve() != Path(__file__).resolve():
            paths.append(Path(f).resolve())
    paths.append(Path(__file__).resolve())       # _meta.py itself shapes every artifact
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def write_artifact(path: Path | str, obj: dict, outputs: list | None = None, indent: int = 2,
                   default=None) -> None:
    """Write a JSON artifact whose ``_meta`` is v2, filling ``_meta.outputs``.

    ``outputs`` are the other files this run wrote (figures, CSVs, ``report.md#block``), written
    before this call. The artifact's own entry is the digest of its body without ``_meta``.
    """
    meta = obj.get("_meta") or {}
    if meta.get("schema_version") != 2:
        raise ValueError("write_artifact needs a v2 _meta: call build_meta(..., code=[__file__])")
    path = Path(path)
    meta["outputs"] = [{"path": repo_path(path), "sha256": body_digest(obj, default)}] + \
        [digest_entry(p) for p in outputs or []]
    path.write_text(json.dumps(obj, indent=indent, ensure_ascii=False, default=default) + "\n")
