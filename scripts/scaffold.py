#!/usr/bin/env python3
"""Scaffold a study, a variant, or a report that passes `make validate` from the first commit.

    make new-study   NAME=12-gru-foo QUESTION="Does ...?" PROJECT=gru_foo VARIANT=base
    make new-variant STUDY=12-gru-foo NAME=wider
    make new-report  STUDY=12-gru-foo SLUG=heldout-curve TITLE="Held-out vs D"

The layout and field rules are schemas/ (layout.yaml, *-frontmatter.v1.schema.json); the
vocabulary is CONTEXT.md. The scaffolder only writes what those require plus TODO prompts.
It never overwrites an existing file.
"""
from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

REPO = Path(__file__).resolve().parents[1]
STUDIES = REPO / "studies"
LAYOUT = yaml.safe_load((REPO / "schemas" / "layout.yaml").read_text())
TODAY = datetime.datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()


def fail(msg: str) -> None:
    sys.exit(f"scaffold: {msg}")


def commit(files: dict[Path, str]) -> None:
    """Write every planned file, or none: all checks run before the first write, so a
    refusal never leaves a half-made study, variant or report behind."""
    for path in files:
        if path.exists():
            fail(f"{path.relative_to(REPO)} already exists; not overwriting")
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"  wrote {path.relative_to(REPO)}")


def study_dir(name: str) -> Path:
    if not name or not re.match(LAYOUT["study"]["name_pattern"], name):
        fail(f"STUDY {name!r} is not an NN-{{model}}-{{purpose}} study name")
    d = STUDIES / name
    if not d.is_dir():
        fail(f"no study {name!r} under studies/")
    return d


def check_variant_name(name: str) -> None:
    if not name or not re.match(LAYOUT["variant"]["name_pattern"], name):
        fail(f"variant {name!r} must be lowercase words joined by - _ or .")


def parse_issue(ref: str) -> int | str:
    """`14` -> 14 (this repo); `AllenNeuralDynamics/<repo>#N` kept as is; anything else refused."""
    if re.fullmatch(r"[1-9][0-9]*", ref):
        return int(ref)
    if re.fullmatch(r"AllenNeuralDynamics/[\w.-]+#[1-9][0-9]*", ref):
        return ref
    fail(f"ISSUE {ref!r} must be a number (this repo) or AllenNeuralDynamics/<repo>#N")


def make_var(study: Path) -> str:
    """The interpreter variable a study's Makefile already defines (PY or PYTHON)."""
    text = (study / "Makefile").read_text() if (study / "Makefile").exists() else ""
    for var in ("PY", "PYTHON"):
        if re.search(rf"^{var}\s*[?:]?=", text, re.M):
            return f"$({var})"
    return "python"


def new_study(a: argparse.Namespace) -> None:
    m = re.match(LAYOUT["study"]["name_pattern"], a.name)
    if not m:
        fail(f"{a.name!r} is not NN-{{model}}-{{purpose}} (e.g. 12-gru-timing-inputs)")
    taken = {p.name[:2] for p in STUDIES.iterdir() if p.is_dir()}
    if m["number"] in taken:
        fail(f"accession number {m['number']} is taken; numbers are never reused "
             f"(next free: {max(int(n) for n in taken if n.isdigit()) + 1:02d})")
    if len(a.question) < 10:
        fail("QUESTION is the study's one scientific question; write it out")
    if not a.project:
        fail("PROJECT is the study's W&B project (one per study)")
    check_variant_name(a.variant)
    issue = parse_issue(a.issue) if a.issue else None
    d = STUDIES / a.name
    front = {"schema_version": 1, "id": a.name, "question": a.question, "status": "planned",
             "wandb": {"entity": a.entity, "project": a.project}}
    if issue is not None:
        front["issues"] = [issue]
    print(f"new study {a.name}")
    files = {}
    files[d / "README.md"] = ( "---\n" + yaml.safe_dump(front, sort_keys=False, allow_unicode=True) + "---\n"
          f"# Study: {a.name}\n\n"
          f"W&B project [`{a.project}`](https://wandb.ai/{a.entity}/{a.project}).\n\n"
          f"**Question.** {a.question}\n\n"
          "## Design\n\nTODO: the variants, what each varies, and what result would answer the question.\n\n"
          "## Verdict\n\nTODO once reports are live; also set `status` and `verdict` in the frontmatter.\n")
    files[d / "CHANGELOG.md"] = f"# Changelog — {a.name}\n\n## {TODAY}\n\n- Study created.\n"
    files[d / "Makefile"] = (
          f"# studies/{a.name}/Makefile\n#\n"
          "# `all` regenerates every report OFFLINE from committed frozen inputs (CI runs it).\n"
          "# One target per report (rN); one `freeze-*` target per freeze, never part of `all`.\n\n"
          "PY ?= python\n.PHONY: all\n\nall:\n")
    files[d / ".gitignore"] = ("# W&B pull caches are never committed and never an input\n"
                               "analysis/*cache*\n__pycache__/\n")
    files[d / "analysis" / "reports" / "INDEX.md"] = (
        f"# {a.name} — reports\n\n| report | question | verdict |\n|---|---|---|\n")
    files.update(_variant_files(d, a.variant))
    commit(files)
    print(f"  variant {a.variant}: add sweep.yaml / experiment.yaml next to notes.md before launching")
    print("next: fill in README.md (Design), then `make new-report` when a result exists; "
          "run `make validate` before opening the PR")


def _variant_files(study: Path, name: str) -> dict[Path, str]:
    check_variant_name(name)
    v = study / "variants" / name
    if v.exists():
        fail(f"{v.relative_to(REPO)} already exists")
    return {v / "notes.md": (
          f"# Variant `{name}`\n\n"
          "**What.** TODO: the configuration, and what it changes relative to the other variants.\n\n"
          "**Why.** TODO: what this variant should teach us.\n\n"
          "**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group "
          f"`{name}@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join "
          "that group.\n\n**Status.** Not launched.\n")}


def new_variant(a: argparse.Namespace) -> None:
    d = study_dir(a.study)
    files = _variant_files(d, a.name)
    print(f"new variant {a.study}/{a.name}")
    commit(files)
    print(f"  variant {a.name}: add sweep.yaml / experiment.yaml next to notes.md before launching")


def new_report(a: argparse.Namespace) -> None:
    d = study_dir(a.study)
    reports = d / "analysis" / "reports"
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", a.slug):
        fail(f"SLUG {a.slug!r} must be lowercase words joined by -")
    nums = [int(m["n"]) for p in reports.glob("r*.md")
            if (m := re.match(LAYOUT["analysis"]["report_name_pattern"], p.name))]
    # a deleted draft can leave its rN target behind; never hand its number to a new report
    targets = re.findall(r"^r([0-9]+)\s*:", (d / "Makefile").read_text(), re.M) if (d / "Makefile").exists() else []
    n = max(nums + [int(x) for x in targets], default=0) + 1
    rid, name = f"r{n}", f"r{n}-{a.slug}.md"
    front = {"schema_version": 1, "id": rid, "slug": a.slug, "status": "draft",
             "authors": [a.author], "wandb_groups": [],
             "inputs": {"script": f"analysis/r{n}_{a.slug.replace('-', '_')}.py"},
             "reproduce": f"make -C studies/{a.study} {rid}"}
    producer = d / front["inputs"]["script"]
    output = producer.with_suffix(".json")     # report-specific, so it cannot clobber a frozen input
    if output.exists():
        fail(f"{output.relative_to(REPO)} already exists; the new producer would overwrite it")
    files = {}
    files[reports / name] = ("---\n" + yaml.safe_dump(front, sort_keys=False) + "---\n"
          f"# {rid} — {a.title}\n\n"
          "**Question.** TODO: the sub-question this report answers.\n\n"
          f"<!-- BEGIN result-1 -->\nTODO: written by the producer, never by hand.\n<!-- END result-1 -->\n\n"
          "## What it says\n\nTODO\n")
    files[producer] = (
          f'"""Producer for {rid}: TODO describe what it computes from which frozen inputs."""\n'
          "import sys\nfrom pathlib import Path\n\n"
          "HERE = Path(__file__).resolve().parent\n"
          "sys.path.insert(0, str(HERE.parents[2] / 'studies' / 'util'))\n"
          "from _meta import build_meta, write_artifact  # noqa: E402\n\n"
          "WANDB_GROUPS: list[str] = []   # exact groups read, if any (via a committed freeze)\n\n\n"
          "def main() -> None:\n"
          "    inputs: list[Path] = []      # committed frozen inputs read; _meta v2 needs at least one\n"
          "    result = {}                  # TODO compute\n"
          f"    meta = build_meta('{front['inputs']['script']}', WANDB_GROUPS, study_root=HERE.parent,\n"
          "                      code=[__file__], inputs=inputs)\n"
          f"    # TODO: rewrite the result block in reports/{name}, then list it in outputs as\n"
          f"    #       HERE / 'reports' / '{name}#result-1' alongside any figures.\n"
          f"    write_artifact(HERE / '{output.name}', {{'_meta': meta, **result}}, outputs=[])\n\n\n"
          "if __name__ == '__main__':\n    main()\n")
    print(f"new report {a.study}/{rid}")
    commit(files)
    mk = d / "Makefile"
    text = mk.read_text()
    if not re.search(rf"^{rid}\s*:", text, re.M):
        mk.write_text(text.rstrip("\n") + f"\n\n# {rid} = {a.title}\n.PHONY: {rid}\n{rid}:\n"
                      f"\t{make_var(d)} {front['inputs']['script']}\n")
        print(f"  added `{rid}` to Makefile (add it to `all` once the producer runs offline)")
    index = reports / "INDEX.md"
    if index.exists() and name not in index.read_text():
        index.write_text(index.read_text().rstrip("\n") + f"\n| [{rid} — {a.title}]({name}) | TODO | — |\n")
        print("  linked from INDEX.md")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("study")
    s.add_argument("--name", required=True)
    s.add_argument("--question", required=True)
    s.add_argument("--project", required=True, help="W&B project (one per study)")
    s.add_argument("--entity", default="AIND-disRNN")
    s.add_argument("--variant", required=True, help="first variant")
    s.add_argument("--issue", help="e.g. AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies#20")
    v = sub.add_parser("variant")
    v.add_argument("--study", required=True)
    v.add_argument("--name", required=True)
    r = sub.add_parser("report")
    r.add_argument("--study", required=True)
    r.add_argument("--slug", required=True)
    r.add_argument("--title", required=True)
    r.add_argument("--author", default="han")
    a = ap.parse_args(argv)
    {"study": new_study, "variant": new_variant, "report": new_report}[a.cmd](a)


if __name__ == "__main__":
    main()
