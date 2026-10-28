#!/usr/bin/env python3
"""Check every study against the layout and provenance schemas in ``schemas/``.

Rules live in ``schemas/`` (JSON Schemas + ``layout.yaml``); vocabulary in ``CONTEXT.md``;
the launch/group rule in ``docs/adr/0001``. This script is the only implementation of the
checks: CI runs it, and the atlas embeds its ``--format json`` output.

Levels:
  error/warn  a broken or missing link in the provenance chain. Fails CI unless the finding
              is in ``schemas/provenance-baseline.json``.
  info        worth knowing, never fails (frozen v1 launch records, whole-checkout dirty flag).
              The advisory ones (ADVISORY: version-2 report notes) are printed as `note`.

The baseline holds the findings that existed when a rule was introduced. It may only shrink:
a finding that disappears must be removed from it (``--prune-baseline``), and with
``--base-ref`` new entries are refused unless the same change edits ``schemas/`` or this
script, i.e. introduces the rule that produced them. A new file is therefore always strict,
and a modified file may keep only the findings it already had.

    python scripts/check_provenance.py                         # report, exit 1 on new findings
    python scripts/check_provenance.py --base-ref origin/main  # CI: also guard the baseline
    python scripts/check_provenance.py --prune-baseline        # drop entries that are fixed
    python scripts/check_provenance.py --write-baseline        # only with a rule change
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import posixpath
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import yaml
    from jsonschema import Draft202012Validator
except ImportError as e:              # e.g. the HPC's system python
    sys.exit(f"check_provenance: {e}. Install the dev requirements first:\n"
             "    python -m pip install -r requirements-dev.txt")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "studies" / "util"))
sys.path.insert(0, str(REPO / "scripts"))
from _meta import block_digest, body_digest, file_digest, result_blocks  # noqa: E402
import report_verdicts  # noqa: E402

SCHEMAS = REPO / "schemas"
BASELINE = SCHEMAS / "provenance-baseline.json"
# Files whose change introduces or alters a rule, and so may add baseline entries.
RULE_FILE = re.compile(r"^(schemas/[^/]+\.schema\.json|schemas/layout\.yaml|scripts/check_provenance\.py)$")
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
GROUP = re.compile(r"\b([A-Za-z0-9][\w.-]*@\d{8}-\d{6})\b")
MAKE_TARGET = re.compile(r"^([A-Za-z0-9][\w.-]*(?:\s+[A-Za-z0-9][\w.-]*)*)\s*:(?!=)", re.M)

# Version-2 reports (schemas/README.md). The skeleton: these headings, in this order.
SKELETON = ["## Question", "## Answer", "## Methods", "## Results", "## What it means",
            "## Caveats", "## Next", "## Provenance"]
# A number in hand-written prose: a decimal, scientific notation or a percentage. Integers are
# left alone (years, D=614, n=149, r1, F2, seed counts): too many are not results.
PROSE_NUMBER = re.compile(r"(?<![\w.@/\\])(\d*\.\d+(?:[eE][-+−]?\d+)?|\d+[eE][-+−]?\d+|\d+(?:\.\d+)?%)(?![\w%]|\.\d)")
JSON_NUMBER = re.compile(r"\d*\.?\d+(?:[eE][-+]?\d+)?")
# Prose numbers that are conventions, not results. Write any other constant as code (`0.5`).
PROSE_CONSTANTS = {"95%"}
# Hand-written text the prose-number lint ignores besides blocks, comments and fenced code.
NOT_PROSE = re.compile(r"`[^`\n]*`|\]\([^)\n]*\)|https?://\S+")
# Advisory findings: `info`, so they never fail CI, but `make validate` prints them.
ADVISORY = {"report-claim-unresolved", "report-prose-number"}

# How to fix each finding: agents fix what CI tells them, so every check says how.
FIX = {
    "study-name": "rename to NN-{model}-{purpose} (lowercase, hyphens); never reuse a number",
    "study-missing": "add it; `make new-study` scaffolds every required file",
    "study-root-extra": "move code/data/figures under analysis/ and launch records under "
                        "variants/<v>/launch_record/; only *.md docs and reference/ may sit at the root",
    "study-environment-lock": "a study with a freeze needs environment.lock (pip freeze of the env that ran it)",
    "study-motivation-unresolved": "point `motivated_by` at an existing study (and report), or an issue",
    "study-motivation-order": "a study can only be motivated by an earlier study (lower accession number)",
    "tag-unknown": "use a value listed in schemas/tags.yaml, or add it there (with a label and description) in the same PR",
    "study-tag-unreported": "tag the report that does this work (report frontmatter `tags`), or drop the study tag",
    "study-motivation-many": "more than 3 motivations usually mixes in what the study uses; move those to its reports' `upstream`",
    "study-frontmatter-missing": "add README frontmatter (schemas/study-frontmatter.v1.schema.json)",
    "study-frontmatter": "fix the README frontmatter field named in the message "
                         "(schemas/study-frontmatter.v1.schema.json)",
    "makefile-target": "add the Makefile target named in the message",
    "variant-name": "rename the variant to lowercase words joined by - _ or .",
    "variant-notes": "add notes.md saying what the variant is and why it exists",
    "variant-launch-record-dir": "keep one launch_record/ per variant; use subdirectories inside it",
    "record-unreadable": "fix the JSON so it parses (a frozen record: leave it, it stays in the baseline)",
    "record-schema": "fix the launch record field named in the message (schemas/launch-record.v2.schema.json)",
    "record-group-prefix": "a launch's W&B group must be <variant>@<launch_id> (ADR 0001); launch again",
    "record-orphan-intervention": "an intervention joins an existing launch's group (ADR 0001); "
                                  "record the initial launch, or launch anew",
    "record-manifest-missing": "commit the manifest the record names, beside it in the variant",
    "record-data-unrecorded": "launch with a pinned data snapshot from a dispatcher that records `data` (studies#25); "
                              "a null digest means the manifest could not be read at launch",
    "record-image-unrecorded": "launch with a dispatcher that records `image` (studies#25) and with BEAKER_TOKEN set, "
                               "so the image name resolves to its immutable id",
    "report-name": "name reports rN-<slug>.md",
    "report-frontmatter-missing": "add frontmatter (schemas/report-frontmatter.v2.schema.json; "
                                  "`make new-report` writes it)",
    "report-frontmatter": "fix the frontmatter field named in the message "
                          "(schemas/report-frontmatter.v<schema_version>.schema.json)",
    "report-skeleton": "give the report the version-2 headings, in order: `# rN — <the finding>`, "
                       + ", ".join(f"`{h}`" for h in SKELETON) + " (`make new-report` writes them)",
    "report-claim-unresolved": "point the claim at a value that exists, `<study-relative JSON path>#<JSON Pointer>` "
                               "(e.g. analysis/r1_fit.json#/delta/mean), or regenerate the JSON; "
                               "a draft may point ahead (advisory), a live report may not",
    "report-prose-stale": "re-read every hand-written sentence against the changed block, and every sibling "
                          "report that says \"same as rN\"; then set `prose_reviewed.blocks` to the digests "
                          "the message prints and `prose_reviewed.date` to today",
    "report-prose-number": "cite the finding by its id instead of retyping the number, or print a number the "
                           "report's JSON holds (cite that JSON in `inputs`); write a constant that is not a "
                           "result as code (`0.5`). Advisory: never fails CI",
    "verdict-stale": "regenerate it: `make -C studies/<study> verdicts` (scripts/report_verdicts.py). "
                     "The generated block copies the version-2 reports' claims: edit the claims, never the block",
    "report-id-mismatch": "make frontmatter id/slug equal the filename rN-<slug>.md",
    "report-reproduce": "point `reproduce:` at `make -C studies/<study> rN` and add that Makefile target",
    "report-input-missing": "`inputs` holds existing study-relative paths; commands go in `upstream`",
    "report-group-unrecorded": "cite exact groups that have a launch record; a missing record is "
                               "recovered with studies/util/launch_record.py write_intervention(kind='backfill')",
    "report-ref-unresolved": "reference an existing {study, report}",
    "report-upstream-stale": "the cited report no longer stands as cited. Corrected: also cite the correcting "
                             "report in `upstream`. Superseded or retracted: replace the ref with its successor "
                             "(or drop it). Then check this report's conclusion still holds and regenerate",
    "report-upstream-undeclared": "declare what the producer reads from that study in `upstream`: "
                                  "{study, report} for a result, {study, wandb_group} for raw data",
    "report-not-indexed": "link the report from analysis/reports/INDEX.md",
    "report-no-result-block": "wrap the report's numbers in <!-- BEGIN result-N --> / <!-- END result-N -->",
    "artifact-unreadable": "fix the JSON so it parses",
    "artifact-no-meta": "write it with build_meta(..., code=[__file__]) + write_artifact()",
    "artifact-meta-v1": "migrate the producer: build_meta(..., code=[__file__], inputs=[...]) + write_artifact()",
    "artifact-meta-schema": "fix the _meta field named in the message (schemas/meta.v2.schema.json)",
    "artifact-stale-code": "the producer changed after this artifact was written: commit, then regenerate",
    "artifact-stale-input": "an input changed after this artifact was written: regenerate it",
    "artifact-input-uncommitted": "inputs must be committed files; freeze the data first (a gitignored cache is never an input)",
    "artifact-stale-output": "an output was edited by hand or by another run: regenerate with the producer",
    "artifact-group-unrecorded": "every W&B group a freeze read needs a launch record in its study",
    "artifact-dirty": "",
}


@dataclass
class Finding:
    level: str
    check: str
    path: str
    item: str
    message: str

    @property
    def key(self) -> str:
        return f"{self.check}|{self.path}|{self.item}"

    def as_dict(self) -> dict:
        return {**asdict(self), "key": self.key, "fix": FIX.get(self.check, "")}


class Checker:
    def __init__(self, root: Path):
        self.root = root
        self.layout = yaml.safe_load((SCHEMAS / "layout.yaml").read_text())
        self.schema = {name: Draft202012Validator(json.loads((SCHEMAS / f"{name}.schema.json").read_text()))
                       for name in ("report-frontmatter.v1", "report-frontmatter.v2", "study-frontmatter.v1",
                                    "meta.v2", "launch-record.v2")}
        self._json_numbers: dict[Path, list[float] | None] = {}
        self.tags = yaml.safe_load((SCHEMAS / "tags.yaml").read_text())["facets"]
        self.findings: list[Finding] = []
        self._keys: set[str] = set()
        # committed (in the index): what may be an input. visible: also new, not-yet-added
        # files, so `make validate` before `git add` sees what CI will see.
        self.tracked = set(self._git("ls-files").splitlines())
        self.visible = self.tracked | set(self._git("ls-files", "--others", "--exclude-standard").splitlines())
        self.studies = sorted(p for p in (root / "studies").iterdir()
                              if p.is_dir() and re.match(r"^\d{2}-", p.name))
        self.groups = self._group_index()
        self.reports, self.corrected_by = self._report_index()

    # -- helpers ------------------------------------------------------------------------------
    def _git(self, *args: str) -> str:
        return subprocess.run(["git", "-C", str(self.root), *args], capture_output=True,
                              text=True, check=True).stdout

    def rel(self, p: Path) -> str:
        return p.relative_to(self.root).as_posix()

    def add(self, level: str, check: str, path: Path | str, msg: str, item: str = "") -> None:
        path = path if isinstance(path, str) else self.rel(path)
        f = Finding(level, check, path, item, msg)
        if f.key not in self._keys:
            self._keys.add(f.key)
            self.findings.append(f)

    def schema_errors(self, name: str, obj, check: str, path: Path) -> None:
        for e in sorted(self.schema[name].iter_errors(obj), key=lambda e: list(e.path)):
            where = "/".join(str(x) for x in e.absolute_path) or "(top)"
            # several errors can share a location (two missing required keys), so the key
            # includes the failing keyword and, for `required`, the missing property
            what = e.message.split("'")[1] if e.validator == "required" else e.validator
            self.add("error", check, path, f"{where}: {e.message}", item=f"{where}:{what}")

    @staticmethod
    def frontmatter(path: Path) -> tuple[dict | None, str, str | None]:
        text = path.read_text(errors="replace")
        m = FRONTMATTER.match(text)
        if not m:
            return None, text, None
        try:
            fm = yaml.safe_load(m.group(1))
        except yaml.YAMLError as e:
            return None, text[m.end():], f"YAML error: {e}"
        return (fm if isinstance(fm, dict) else None), text[m.end():], None

    def _group_index(self) -> dict[str, list[dict]]:
        """W&B group -> launch records naming it, across every study.

        Legacy records at a study root (study 01) are indexed too: they are misplaced, which
        is a layout finding, but they still record the launch.
        """
        index: dict[str, list[dict]] = {}
        for study in self.studies:
            files = [p for d in study.glob("variants/*/launch_record*") if d.is_dir()
                     for p in d.rglob("*") if p.suffix in (".json", ".yaml", ".yml")]
            files += [p for p in study.glob("*") if p.suffix in (".json", ".yaml", ".yml")]
            for p in files:
                text = p.read_text(errors="replace")
                rec = None
                if p.suffix == ".json":
                    try:
                        rec = json.loads(text)
                    except json.JSONDecodeError:
                        pass
                for g in set(GROUP.findall(text)):
                    index.setdefault(g, []).append({"path": p, "record": rec if isinstance(rec, dict) else None})
        return index

    def _report_index(self) -> tuple[dict[str, dict], dict[str, set[str]]]:
        """`<study>/rN` -> report frontmatter, and -> the reports that correct it.

        A corrected report stays `live` (CONTEXT.md, Correction), so whether it still stands
        is read from every other report's `corrects:`, never from its own status.
        """
        A = self.layout["analysis"]
        reports: dict[str, dict] = {}
        self.report_paths: dict[str, Path] = {}
        corrected_by: dict[str, set[str]] = {}
        for study in self.studies:
            for p in sorted((study / A["reports_dir"]).glob("r*.md")):
                m = re.match(A["report_name_pattern"], p.name)
                if not m:
                    continue
                fm = self.frontmatter(p)[0] or {}
                key = f"{study.name}/r{m['n']}"
                reports[key] = fm
                self.report_paths[key] = p
                for c in fm.get("corrects") or []:
                    if isinstance(c, dict):
                        corrected_by.setdefault(f"{c.get('study')}/{c.get('report')}", set()).add(key)

        def standing(key: str) -> str | None:
            """The report that now carries `key`'s correction: itself if it stands, its
            successor if superseded, nothing if retracted (a withdrawn correction binds no one)."""
            seen = set()
            while key in reports and key not in seen:
                seen.add(key)
                fm = reports[key]
                if fm.get("status") == "retracted":
                    return None
                succ = fm.get("superseded_by")
                if fm.get("status") != "superseded" or not isinstance(succ, dict):
                    return key
                key = f"{succ.get('study')}/{succ.get('report')}"
            return None
        corrected_by = {t: {s for c in cs if (s := standing(c))} for t, cs in corrected_by.items()}
        return reports, {t: cs for t, cs in corrected_by.items() if cs}

    def makefile_targets(self, study: Path) -> set[str]:
        mk = study / "Makefile"
        if not mk.exists():
            return set()
        return {t for line in MAKE_TARGET.findall(mk.read_text()) for t in line.split()}

    # -- checks -------------------------------------------------------------------------------
    def run(self) -> list[Finding]:
        for study in self.studies:
            self.check_study(study)
        return self.findings

    def check_study(self, study: Path) -> None:
        L = self.layout["study"]
        if not re.match(L["name_pattern"], study.name):
            self.add("error", "study-name", study, f"{study.name} does not match NN-{{model}}-{{purpose}}")
        for req in L["required"]:
            if not (study / req.rstrip("/")).exists():
                self.add("error", "study-missing", study, f"missing {req}", item=req)
        allowed = {a.rstrip("/") for a in L["allowed"]}
        for p in sorted(study.iterdir()):
            if p.name in allowed or p.name == "__pycache__" or not self._is_tracked_tree(p):
                continue
            if p.is_file() and any(re.match(pat, p.name) for pat in L["allowed_patterns"]):
                continue
            self.add("warn", "study-root-extra", p, f"{p.name} is not allowed at the study root",
                     item=p.name)

        fm, _, err = self.frontmatter(study / "README.md") if (study / "README.md").exists() else (None, "", None)
        if fm is None:
            self.add("warn", "study-frontmatter-missing", study / "README.md", err or "README has no frontmatter")
        else:
            self.schema_errors("study-frontmatter.v1", fm, "study-frontmatter", study / "README.md")
            if fm.get("id") not in (None, study.name):
                self.add("error", "study-frontmatter", study / "README.md",
                         f"id {fm.get('id')!r} != directory {study.name}", item="id")
            self.check_motivation(study, fm)
            self.check_study_tags(study, fm)

        targets = self.makefile_targets(study)
        for t in self.layout["makefile"]["required_targets"]:
            if (study / "Makefile").exists() and t not in targets:
                self.add("error", "makefile-target", study / "Makefile", f"no `{t}` target", item=t)

        variants = sorted(p for p in (study / "variants").glob("*") if p.is_dir() and p.name != "__pycache__")
        for v in variants:
            self.check_variant(study, v)
        self.check_reports(study, targets)
        has_freeze = self.check_artifacts(study)
        if has_freeze:
            if not (study / "environment.lock").exists():
                self.add("warn", "study-environment-lock", study, "study has a freeze but no environment.lock")
            if not any(re.match(self.layout["makefile"]["per_freeze_target"], t) for t in targets):
                self.add("error", "makefile-target", study / "Makefile", "no `freeze-*` target", item="freeze")

    def check_motivation(self, study: Path, fm: dict) -> None:
        """`motivated_by` points back in time at results that exist (CONTEXT.md, Motivation)."""
        entries = fm.get("motivated_by") if isinstance(fm.get("motivated_by"), list) else []  # else: a schema error
        refs = [m for m in entries if isinstance(m, dict) and "study" in m]
        readme = study / "README.md"
        for m in refs:
            target = str(m["study"])
            ref = f"{target}/{m['report']}" if "report" in m else target
            if not (self.root / "studies" / target).is_dir() or (
                    "report" in m and ref not in self.reports):
                self.add("error", "study-motivation-unresolved", readme, f"motivated_by {ref} not found", item=ref)
            elif target[:2].isdigit() and int(target[:2]) >= int(study.name[:2]):
                self.add("error", "study-motivation-order", readme,
                         f"motivated_by {ref} is not an earlier study than {study.name}", item=ref)
        if len(entries) > 3:
            self.add("info", "study-motivation-many", readme, f"{len(entries)} motivations")

    def check_tags(self, tags, path: Path) -> None:
        """Every tag value is in schemas/tags.yaml (CONTEXT.md, Tag); shape errors are the schema's."""
        for facet, values in (tags.items() if isinstance(tags, dict) else []):
            known = (self.tags.get(facet) or {}).get("values") or {}
            for v in values if isinstance(values, list) and known else []:
                if isinstance(v, str) and v not in known:       # other types: the schema reports them
                    self.add("error", "tag-unknown", path, f"tags.{facet}: {v!r} is not in schemas/tags.yaml", item=f"{facet}:{v}")

    def check_study_tags(self, study: Path, fm: dict) -> None:
        """A complete study's tags are each carried by one of its standing (live or draft) reports;
        `cognitive/bh` on a report covers `cognitive` on the study. Planned and active studies have
        nothing to show yet, and a study none of whose reports has frontmatter is already flagged
        by report-frontmatter-missing."""
        tags = fm.get("tags")
        self.check_tags(tags, study / "README.md")
        mine = [r for k, r in self.reports.items() if k.startswith(study.name + "/")]
        if fm.get("status") != "complete" or not isinstance(tags, dict) or not any(mine):
            return
        live = [r for r in mine if r.get("status") in ("live", "draft")]
        for facet, values in tags.items():
            carried = set()
            for r in live:
                rt = (r.get("tags") or {}).get(facet) if isinstance(r.get("tags"), dict) else None
                for t in rt if isinstance(rt, list) else []:
                    if not isinstance(t, str):
                        continue
                    parts = t.split("/")
                    carried |= {"/".join(parts[:i + 1]) for i in range(len(parts))}
            for v in values if isinstance(values, list) else []:
                if isinstance(v, str) and v not in carried:
                    self.add("warn", "study-tag-unreported", study / "README.md",
                             f"tags.{facet}: {v!r} is on no live or draft report of this complete study", item=f"{facet}:{v}")

    def _is_tracked_tree(self, p: Path) -> bool:
        r = self.rel(p)
        return r in self.visible or any(t.startswith(r + "/") for t in self.visible)

    def check_variant(self, study: Path, v: Path) -> None:
        V = self.layout["variant"]
        if not re.match(V["name_pattern"], v.name):
            self.add("warn", "variant-name", v, f"variant name {v.name!r} is not lowercase words")
        for req in V["required"]:
            if not (v / req).exists():
                self.add("warn", "variant-notes", v, f"missing {req}", item=req)
        for d in sorted(v.glob(V["launch_record_dir"] + "_*")):
            if d.is_dir():
                self.add("warn", "variant-launch-record-dir", d,
                         f"{d.name}/ beside launch_record/; use a subdirectory of launch_record/", item=d.name)
        initial_groups: set[str] = set()
        records = []
        for d in [v / V["launch_record_dir"], *sorted(v.glob(V["launch_record_dir"] + "_*"))]:
            for p in sorted(d.rglob("*.json")) if d.is_dir() else []:
                try:
                    rec = json.loads(p.read_text())
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    self.add("warn", "record-unreadable", p, f"{type(e).__name__}: {e}")
                    continue
                if not isinstance(rec, dict):
                    self.add("warn", "record-unreadable", p, "top level is not an object")
                    continue
                records.append((p, rec))
        for p, rec in records:
            if (rec.get("_meta") or {}).get("schema_version") == 2 and rec.get("kind") == "initial":
                initial_groups.add(rec.get("wandb_group"))
        for p, rec in records:
            if (rec.get("_meta") or {}).get("schema_version") != 2:
                continue                                   # v1: frozen history, never re-judged
            self.schema_errors("launch-record.v2", rec, "record-schema", p)
            group = rec.get("wandb_group") or ""
            if rec.get("kind") == "initial" and group.split("@")[0] != v.name:
                self.add("error", "record-group-prefix", p, f"group {group} does not start with {v.name}@")
            if rec.get("kind") not in (None, "initial") and group not in initial_groups:
                self.add("error", "record-orphan-intervention", p,
                         f"{rec.get('kind')} joins {group}, which has no initial launch record in {v.name}")
            for m in rec.get("manifests") or []:
                if not (v / m).exists() and not (p.parent / m).exists():
                    self.add("error", "record-manifest-missing", p, f"manifest {m} not found", item=m)
            # what the runs trained on and ran in (studies#25); a backfill submits nothing.
            # The schema accepts a null digest or id (a launcher without network or token
            # writes one rather than failing), so the checker warns on those too.
            if rec.get("platform") in ("beaker", "hpc"):
                data, image = rec.get("data"), rec.get("image")
                if not isinstance(data, dict):
                    self.add("warn", "record-data-unrecorded", p, "the launch record has no `data`")
                elif data.get("source") == "live":
                    self.add("warn", "record-data-unrecorded", p, "the runs read the live database (no snapshot)")
                elif data.get("source") == "snapshot" and not (data.get("uri") and data.get("manifest_sha256")):
                    self.add("warn", "record-data-unrecorded", p, "the snapshot has no uri or manifest digest")
                # the image must describe this record's backend: a Beaker id for Beaker, a conda
                # lock digest for HPC; the other backend's identity proves nothing about this run
                identity = {"beaker": "id", "hpc": "lock_sha256"}[rec["platform"]]
                if not isinstance(image, dict):
                    self.add("warn", "record-image-unrecorded", p, "the launch record has no `image`")
                elif image.get("platform") != rec["platform"]:
                    self.add("warn", "record-image-unrecorded", p,
                             f"`image` describes {image.get('platform')}, but the launch ran on {rec['platform']}")
                elif not image.get(identity):
                    self.add("warn", "record-image-unrecorded", p, f"`image` has no immutable {identity} ({image.get('name') or image.get('conda_env')})")

    def check_reports(self, study: Path, targets: set[str]) -> None:
        A = self.layout["analysis"]
        rdir = study / A["reports_dir"]
        index_path = study / A["index"]
        index = index_path.read_text() if index_path.exists() else ""
        for p in sorted(rdir.glob("*.md")) if rdir.is_dir() else []:
            if p.name == "INDEX.md":
                continue
            m = re.match(A["report_name_pattern"], p.name)
            if not m:
                self.add("warn", "report-name", p, f"{p.name} is not rN-<slug>.md")
                continue
            if p.name not in index:
                self.add("warn", "report-not-indexed", p, "not linked from INDEX.md")
            fm, body, err = self.frontmatter(p)
            if not result_blocks(p):
                self.add("warn", "report-no-result-block", p, "no result block")
            if fm is None:
                self.add("warn", "report-frontmatter-missing", p, err or "no frontmatter")
                continue
            if fm.get("schema_version") == 2:
                # YAML reads an unquoted 2026-10-28 as a date; the schema sees the string
                fm = json.loads(json.dumps(fm, default=str))
                self.check_report_v2(study, p, m, fm, body)
            self.check_report_frontmatter(study, p, m, fm, targets)
        self.check_verdicts(study)

    def check_report_frontmatter(self, study: Path, p: Path, m, fm: dict, targets: set[str]) -> None:
        version = "v2" if fm.get("schema_version") == 2 else "v1"
        self.schema_errors(f"report-frontmatter.{version}", fm, "report-frontmatter", p)
        self.check_tags(fm.get("tags"), p)
        if fm.get("id") != f"r{m['n']}" or (fm.get("slug") and fm.get("slug") != m["slug"]):
            self.add("error", "report-id-mismatch", p,
                     f"frontmatter id/slug {fm.get('id')}/{fm.get('slug')} != filename r{m['n']}-{m['slug']}")
        rid = f"r{m['n']}"
        if rid not in targets:          # the command's shape is the schema's job
            self.add("warn", "report-reproduce", p, f"Makefile has no `{rid}` target", item=rid)
        inputs = fm.get("inputs")
        for role, val in (inputs.items() if isinstance(inputs, dict) else []):
            for item in val if isinstance(val, list) else [val]:
                if isinstance(item, str) and not (study / item).exists():
                    self.add("error", "report-input-missing", p, f"inputs.{role}: {item} does not exist",
                             item=f"{role}:{item}")
        for g in fm.get("wandb_groups") or []:
            if isinstance(g, str) and GROUP.fullmatch(g) and g not in self.groups:
                self.add("warn", "report-group-unrecorded", p, f"W&B group {g} has no launch record", item=g)
        refs = [r for r in (fm.get("upstream") or []) if isinstance(r, dict) and "report" in r]
        refs += [r for r in (fm.get("corrects") or []) if isinstance(r, dict)]
        if isinstance(fm.get("superseded_by"), dict):
            refs.append(fm["superseded_by"])
        for r in refs:
            target = self.root / "studies" / str(r.get("study")) / "analysis" / "reports"
            if not any(target.glob(f"{r.get('report')}-*.md")):
                self.add("error", "report-ref-unresolved", p, f"{r.get('study')}/{r.get('report')} not found",
                         item=f"{r.get('study')}/{r.get('report')}")
        self.check_upstream_stands(study, p, m, fm)

    def check_upstream_stands(self, study: Path, p: Path, m, fm: dict) -> None:
        """A cited report must still stand: not superseded or retracted, and not corrected by
        a report this one does not also cite. A superseded or retracted report is history and
        is not re-judged."""
        if fm.get("status") in ("superseded", "retracted"):
            return
        me = f"{study.name}/r{m['n']}"
        cited = {f"{r['study']}/{r['report']}" for r in fm.get("upstream") or []
                 if isinstance(r, dict) and "report" in r and "study" in r}   # else: a schema error
        corrects = {f"{r.get('study')}/{r.get('report')}" for r in fm.get("corrects") or [] if isinstance(r, dict)}
        for ref in sorted(cited):
            target = self.reports.get(ref)
            if target is None:
                continue                                 # report-ref-unresolved covers it
            if target.get("status") in ("superseded", "retracted"):
                succ = target.get("superseded_by")
                then = (f"; cite {succ.get('study')}/{succ.get('report')} instead"
                        if isinstance(succ, dict) else "")
                self.add("warn", "report-upstream-stale", p, f"upstream {ref} is {target['status']}{then}", item=ref)
                continue
            missed = sorted(self.corrected_by.get(ref, set()) - cited - {me})
            if missed and ref not in corrects:
                self.add("warn", "report-upstream-stale", p,
                         f"upstream {ref} is corrected by {', '.join(missed)}, which this report does not cite",
                         item=ref)

    # -- version-2 reports (schemas/README.md, Version-2 reports) --------------------------------
    def check_report_v2(self, study: Path, p: Path, m, fm: dict, body: str) -> None:
        """Skeleton, claim pointers, stale prose and unsourced prose numbers. A superseded or
        retracted report is history: only its skeleton is checked."""
        self.check_skeleton(p, m, body)
        if fm.get("status") in ("superseded", "retracted"):
            return
        self.check_claims(study, p, fm)
        self.check_prose_reviewed(p, fm)
        self.check_prose_numbers(study, p, fm, body)

    def check_skeleton(self, p: Path, m, body: str) -> None:
        heads = report_verdicts.headings(body)
        if not heads or not re.fullmatch(rf"# r{m['n']} — \S.*", heads[0]):
            self.add("error", "report-skeleton", p, f"the first heading must be `# r{m['n']} — <the finding>`, "
                     f"not {heads[0] if heads else 'nothing'!r}", item="title")
        h2, i = [h for h in heads if h.startswith("## ")], 0
        for want in SKELETON:
            if want in h2[i:]:
                i = h2.index(want, i) + 1
            else:
                self.add("error", "report-skeleton", p, f"`{want}` is {'out of order' if want in h2 else 'missing'}",
                         item=want)

    def check_claims(self, study: Path, p: Path, fm: dict) -> None:
        """Every claim points at a value. A draft may point ahead of its producer (advisory)."""
        level = "error" if fm.get("status") == "live" else "info"
        for c in fm.get("claims") if isinstance(fm.get("claims"), list) else []:
            if not (isinstance(c, dict) and isinstance(c.get("pointer"), str) and "#" in c["pointer"]):
                continue                                 # the schema reports it
            path, pointer = c["pointer"].split("#", 1)
            try:
                doc = json.loads((study / path).read_text())
                problem = None if self._resolves(doc, pointer) else f"{path} has no {pointer}"
            except FileNotFoundError:
                problem = f"{path} does not exist"
            except (OSError, ValueError):
                problem = f"{path} is not readable JSON"
            if problem:
                self.add(level, "report-claim-unresolved", p, f"{c.get('id')}: {problem}", item=str(c.get("id")))

    @staticmethod
    def _resolves(doc, pointer: str) -> bool:
        """Whether an RFC 6901 JSON Pointer names a value in `doc`."""
        if pointer and not pointer.startswith("/"):
            return False
        for token in pointer.split("/")[1:]:
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(doc, dict) and token in doc:
                doc = doc[token]
            elif isinstance(doc, list) and re.fullmatch(r"0|[1-9][0-9]*", token) and int(token) < len(doc):
                doc = doc[int(token)]
            else:
                return False
        return True

    def check_prose_reviewed(self, p: Path, fm: dict) -> None:
        """The prose was re-read after the last change to any BEGIN/END block."""
        reviewed = fm.get("prose_reviewed")
        recorded = reviewed.get("blocks") if isinstance(reviewed, dict) else None
        if not isinstance(recorded, dict):
            return                                       # the schema reports it
        now = {name: block_digest(text) for name, text in result_blocks(p).items()}
        for name in sorted(set(now) | set(recorded)):
            if now.get(name) == recorded.get(name):
                continue
            if name not in now:
                msg = f"block {name} is gone; after re-reading the prose, remove prose_reviewed.blocks.{name}"
            else:
                what = "is new" if name not in recorded else f"changed after the prose review of {reviewed.get('date')}"
                msg = f"block {name} {what}; after re-reading the prose, set prose_reviewed.blocks.{name}: {now[name]}"
            self.add("warn", "report-prose-stale", p, msg, item=name)

    def check_prose_numbers(self, study: Path, p: Path, fm: dict, body: str) -> None:
        """Advisory: a decimal or percentage in hand-written prose matches a number in the JSON
        the report cites (`inputs`, claim pointers), to the precision it is printed with."""
        inputs = fm.get("inputs") if isinstance(fm.get("inputs"), dict) else {}
        claims = fm.get("claims") if isinstance(fm.get("claims"), list) else []
        cited = [i for v in inputs.values() for i in (v if isinstance(v, list) else [v])]
        cited += [c["pointer"].split("#")[0] for c in claims if isinstance(c, dict) and isinstance(c.get("pointer"), str)]
        files = sorted({posixpath.normpath(f) for f in cited if isinstance(f, str) and f.endswith(".json")})
        values = sorted(v for f in files for v in self.json_numbers(study / f))
        prose = report_verdicts.blank(report_verdicts.hand_written(body), NOT_PROSE)
        offset = p.read_text(errors="replace").count("\n") - body.count("\n")
        for n in PROSE_NUMBER.finditer(prose):
            token = n.group(1)
            if token in PROSE_CONSTANTS or self._sourced(token, values):
                continue
            line = offset + prose.count("\n", 0, n.start()) + 1
            self.add("info", "report-prose-number", p,
                     f"{token} (line {line}) matches no number in {', '.join(files) or 'the cited JSON (none cited)'}",
                     item=token)

    def json_numbers(self, path: Path) -> list[float]:
        """Every number in a JSON file outside `_meta` (values, numeric keys, numbers inside
        strings), as sorted absolute values; [] when the file is missing or unreadable."""
        if path not in self._json_numbers:
            try:
                doc = json.loads(path.read_text())
            except (OSError, ValueError):
                doc = None
            out: list[float] = []
            stack = [{k: v for k, v in doc.items() if k != "_meta"} if isinstance(doc, dict) else doc]
            while stack:
                x = stack.pop()
                if isinstance(x, bool) or x is None:
                    continue
                if isinstance(x, (int, float)):
                    out.append(abs(float(x)))
                elif isinstance(x, str):
                    out += [abs(float(s)) for s in JSON_NUMBER.findall(x)]
                elif isinstance(x, dict):
                    stack += [*x.keys(), *x.values()]
                elif isinstance(x, list):
                    stack += x
            self._json_numbers[path] = sorted(v for v in out if math.isfinite(v))
        return self._json_numbers[path]

    @staticmethod
    def _sourced(token: str, values: list[float]) -> bool:
        """Whether a printed number is one of `values` rounded to its printed precision; a
        percentage may also be a fraction (12% ~ 0.12)."""
        number = token.rstrip("%").replace("−", "-")
        mantissa, _, exponent = number.lower().partition("e")
        decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
        x, tol = float(number), 0.5 * 10.0 ** (int(exponent or 0) - decimals)
        for target, t in [(x, tol)] + ([(x / 100, tol / 100)] if token.endswith("%") else []):
            slack = t * (1 + 1e-9) + 1e-15
            i = bisect.bisect_left(values, target - slack)
            if i < len(values) and values[i] <= target + slack:
                return True
        return False

    def check_verdicts(self, study: Path) -> None:
        """The generated INDEX table and README verdict still copy the version-2 reports' claims."""
        for path, inner in report_verdicts.expected(study).items():
            have = report_verdicts.current(path.read_text())
            if have != inner:
                self.add("warn", "verdict-stale", path,
                         f"the generated verdicts block is {'missing' if have is None else 'out of date'}; "
                         f"run `make -C {self.rel(study)} verdicts`", item="verdicts")

    def check_artifacts(self, study: Path) -> bool:
        """Check analysis JSON; return whether the study has a freeze."""
        has_freeze = False
        cache = self.layout["analysis"]["cache_pattern"]
        for p in sorted((study / "analysis").rglob("*.json")):
            if any(cache in part for part in p.relative_to(study).parts) or self.rel(p) not in self.visible:
                continue
            try:
                obj = json.loads(p.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self.add("error", "artifact-unreadable", p, f"{type(e).__name__}: {e}")
                continue
            meta = obj.get("_meta") if isinstance(obj, dict) else None
            if not isinstance(meta, dict):
                self.add("warn", "artifact-no-meta", p, "no _meta block")
                continue
            if meta.get("studies_git_dirty"):
                self.add("info", "artifact-dirty", p, "written from a checkout with uncommitted changes")
            if meta.get("schema_version") != 2:
                self.add("warn", "artifact-meta-v1", p, "_meta is v1 (no code/input/output digests)")
                has_freeze |= str(meta.get("produced_by", "")).split("/")[-1].startswith("freeze")
                continue
            has_freeze |= meta.get("kind") == "freeze"
            self.schema_errors("meta.v2", meta, "artifact-meta-schema", p)
            self.check_digests(p, obj, meta)
        return has_freeze

    def check_digests(self, p: Path, obj: dict, meta: dict) -> None:
        for e in meta.get("code") or []:
            self._check_file_digest(p, e, "artifact-stale-code")
        for e in meta.get("inputs") or []:
            path = e.get("path", "")
            if path not in self.tracked:
                self.add("error", "artifact-input-uncommitted", p, f"input {path} is not a committed file", item=path)
                continue
            self._check_file_digest(p, e, "artifact-stale-input")
        for e in meta.get("outputs") or []:
            path, want = e.get("path", ""), e.get("sha256")
            if path == self.rel(p):
                got = body_digest(obj)
            elif "#" in path:
                report, block = path.split("#", 1)
                blocks = result_blocks(self.root / report) if (self.root / report).exists() else {}
                got = block_digest(blocks[block]) if block in blocks else None
            else:
                got = file_digest(self.root / path) if (self.root / path).exists() else None
            if got != want:
                self.add("error", "artifact-stale-output", p,
                         f"output {path} {'is missing' if got is None else 'does not match its digest'}", item=path)
        self.check_upstream_declared(p, meta)
        if meta.get("kind") == "freeze":
            for g in meta.get("wandb_groups") or []:
                if g not in self.groups:
                    self.add("error", "artifact-group-unrecorded", p, f"W&B group {g} has no launch record", item=g)

    def check_upstream_declared(self, p: Path, meta: dict) -> None:
        """A report written from another study's files names that study in `upstream`, so a
        correction there can reach it (report-upstream-stale). "Written from" is a result block
        this artifact wrote, or the artifact listed in the report's `inputs`."""
        own = self.rel(p).split("/")[1]
        read = {path.split("/")[1] for e in meta.get("inputs") or []
                if re.match(r"^studies/\d{2}-[^/]+/", path := str(e.get("path", ""))) and path.split("/")[1] != own}
        written = {str(e.get("path", "")).split("#", 1)[0] for e in meta.get("outputs") or []
                   if "#" in str(e.get("path", ""))}
        # ... and the reports that cite this artifact in `inputs`, whose prose may use its numbers
        study, art = self.rel(p).split("/", 2)[1], self.rel(p).split("/", 2)[2]
        for key, path in self.report_paths.items():
            fm = self.reports[key]
            if key.startswith(f"{study}/") and isinstance(fm.get("inputs"), dict) and any(
                    isinstance(i, str) and posixpath.normpath(i) == art
                    for v in fm["inputs"].values() for i in (v if isinstance(v, list) else [v])):
                written.add(self.rel(path))
        for report in sorted(written):
            if not (self.root / report).exists():
                continue                                 # artifact-stale-output covers it
            fm = self.frontmatter(self.root / report)[0] or {}
            declared = {u.get("study") for u in fm.get("upstream") or [] if isinstance(u, dict)}
            for other in sorted(read - declared):
                self.add("warn", "report-upstream-undeclared", report,
                         f"it is written from studies/{other} files via {self.rel(p)}, "
                         f"but `upstream` names nothing in {other}", item=other)

    def _check_file_digest(self, p: Path, e: dict, check: str) -> None:
        path = e.get("path", "")
        f = self.root / path
        if not f.exists() or file_digest(f) != e.get("sha256"):
            self.add("error", check, p, f"{path} {'is missing' if not f.exists() else 'changed since this artifact was written'}",
                     item=path)


# -- baseline --------------------------------------------------------------------------------
def load_baseline(text: str | None) -> set[str]:
    return set(json.loads(text)["findings"]) if text else set()


def write_baseline(keys: set[str]) -> None:
    BASELINE.write_text(json.dumps({
        "_comment": "Known provenance findings in files that predate a rule. May only shrink, "
                    "except in a change that edits schemas/ or scripts/check_provenance.py. "
                    "See scripts/check_provenance.py.",
        "findings": sorted(keys)}, indent=1) + "\n")


def git(*args: str) -> str | None:
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", choices=("md", "json"), default="md")
    ap.add_argument("--base-ref", help="refuse baseline growth unless this change edits the rules")
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--prune-baseline", action="store_true")
    args = ap.parse_args(argv)

    findings = Checker(REPO).run()
    gating = {f.key: f for f in findings if f.level != "info"}
    baseline = load_baseline(BASELINE.read_text() if BASELINE.exists() else None)

    if args.write_baseline:
        write_baseline(set(gating))
        print(f"wrote {BASELINE.relative_to(REPO)} ({len(gating)} findings)")
        return 0
    if args.prune_baseline:
        kept = baseline & set(gating)
        write_baseline(kept)
        print(f"pruned {len(baseline) - len(kept)} fixed entries; {len(kept)} remain")
        return 0

    new = [gating[k] for k in sorted(set(gating) - baseline)]
    fixed = sorted(baseline - set(gating))
    grown: list[str] = []
    if args.base_ref:
        base = load_baseline(git("show", f"{args.base_ref}:schemas/provenance-baseline.json"))
        changed = (git("diff", "--name-only", f"{args.base_ref}...HEAD") or "").splitlines()
        added = set((git("diff", "--name-only", "--diff-filter=A", f"{args.base_ref}...HEAD") or "").splitlines())
        rule_change = any(RULE_FILE.match(c) for c in changed)
        if base:
            # A rule change may baseline what its new rule finds in existing files; nothing
            # may baseline a finding on a file this change adds, which is always strict.
            grown = sorted(k for k in baseline - base
                           if not rule_change or k.split("|")[1].split("#")[0] in added)

    if args.format == "json":
        json.dump({"findings": [f.as_dict() for f in findings], "new": [f.key for f in new],
                   "fixed": fixed, "baseline_grown": grown, "baseline_size": len(baseline)},
                  sys.stdout, indent=1)
        print()
    else:
        levels = {lv: sum(f.level == lv for f in findings) for lv in ("error", "warn", "info")}
        print(f"provenance: {levels} — {len(baseline)} baselined, {len(new)} new, {len(fixed)} fixed")
        for f in new:
            print(f"\nNEW {f.level} [{f.check}] {f.path}\n    {f.message}\n    fix: {FIX.get(f.check, '')}")
        for f in findings:
            if f.level == "info" and f.check in ADVISORY:
                print(f"\nnote [{f.check}] {f.path}\n    {f.message}\n    fix: {FIX.get(f.check, '')}")
        if fixed:
            print(f"\n{len(fixed)} baseline entries are fixed; remove them: "
                  "python scripts/check_provenance.py --prune-baseline")
            for k in fixed[:20]:
                print(f"    {k}")
        for k in grown:
            print(f"\nBASELINE GROWN: {k}\n    only a change to a schema, layout.yaml or the checker may add "
                  "entries, and never for a file it adds\n    fix: fix the finding instead of baselining it")
    return 1 if new or fixed or grown else 0


if __name__ == "__main__":
    raise SystemExit(main())
