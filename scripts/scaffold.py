#!/usr/bin/env python3
"""Scaffold a study, a variant, or a report that passes `make validate` from the first commit.

    make new-study   NAME=12-gru-foo QUESTION="Does ...?" PROJECT=gru_foo VARIANT=base MOTIVATED_BY=06/r2 \
                     MODEL=gru QUESTION_TYPE=scaling,comparison
    make new-variant STUDY=12-gru-foo NAME=wider
    make new-report  STUDY=12-gru-foo SLUG=heldout-curve TITLE="Held-out vs D"

The layout and field rules are schemas/ (layout.yaml, study-frontmatter.v1.schema.json,
report-frontmatter.v2.schema.json and the version-2 report rules in schemas/README.md); the
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
sys.path.insert(0, str(REPO / "studies" / "util"))
sys.path.insert(0, str(REPO / "scripts"))
import _meta  # noqa: E402
import report_verdicts  # noqa: E402

STUDIES = REPO / "studies"
LAYOUT = yaml.safe_load((REPO / "schemas" / "layout.yaml").read_text())
TAGS = yaml.safe_load((REPO / "schemas" / "tags.yaml").read_text())["facets"]
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


MOTIVATION_TODO = "TODO what this result showed that raised this question"


def parse_motivation(spec: str | None, new_number: str) -> list[dict]:
    """`none` -> [] (a root study); else comma-separated `05/r1`, `05` (the study's verdict) or `#23` /
    `AllenNeuralDynamics/<repo>#N` (an issue). Each target must exist and predate the new study."""
    if not spec:
        fail("MOTIVATED_BY is required: the earlier results this study follows up (e.g. 05/r1,03/r2), "
             "or `none` for a root study")
    if spec.strip() == "none":
        return []
    out = []
    for part in (p.strip() for p in spec.split(",")):
        if m := re.fullmatch(r"#([1-9][0-9]*)", part):
            out.append({"issue": int(m[1]), "note": MOTIVATION_TODO})
            continue
        if re.fullmatch(r"AllenNeuralDynamics/[\w.-]+#[1-9][0-9]*", part):
            out.append({"issue": part, "note": MOTIVATION_TODO})
            continue
        m = re.fullmatch(r"([0-9]{2})(-[a-z0-9-]+)?(?:/(r[0-9]+))?", part)
        if not m:
            fail(f"MOTIVATED_BY entry {part!r} must look like 05/r1, 05, #23 or AllenNeuralDynamics/<repo>#N")
        # a full name must match its directory exactly; the shorthand `05` resolves by accession number
        found = ([STUDIES / (m[1] + m[2])] if m[2] else sorted(STUDIES.glob(f"{m[1]}-*")))
        found = [p for p in found if p.is_dir()]
        if not found:
            fail(f"MOTIVATED_BY {part!r}: no study {m[1] + (m[2] or '')} under studies/")
        if int(m[1]) >= int(new_number):
            fail(f"MOTIVATED_BY {part!r}: a study can only be motivated by an earlier one")
        entry = {"study": found[0].name}
        if m[3]:
            if not any((found[0] / "analysis" / "reports").glob(f"{m[3]}-*.md")):
                fail(f"MOTIVATED_BY {part!r}: {found[0].name} has no report {m[3]}")
            entry["report"] = m[3]
        out.append({**entry, "note": MOTIVATION_TODO})
    return out


# the study target that regenerates the INDEX table and README verdict from the reports' claims
VERDICTS_TARGET = ("\n# verdicts = the INDEX.md table and README `## Verdict`, generated from the version-2\n"
                   "# reports' frontmatter claims (scripts/report_verdicts.py); edit the claims, never the output\n"
                   ".PHONY: verdicts\nverdicts:\n\t{py} ../../scripts/report_verdicts.py .\n")


# a rule line names one or more targets before `:` (`r1 r2:`), never `:=` assignments
MAKE_RULE = re.compile(r"^([A-Za-z0-9][\w.-]*(?:[ \t]+[A-Za-z0-9][\w.-]*)*)[ \t]*:(?!=)", re.M)


def parse_tags(facet: str, spec: str | None, var: str) -> list[str]:
    """A comma-separated list of schemas/tags.yaml values for one facet; at least one."""
    known = TAGS[facet]["values"]
    values = [v.strip() for v in (spec or "").split(",") if v.strip()]
    if not values:
        fail(f"{var} is required: comma-separated, from {', '.join(known)} (schemas/tags.yaml)")
    if bad := [v for v in values if v not in known]:
        fail(f"{var}: {', '.join(bad)} not in schemas/tags.yaml ({', '.join(known)})")
    return list(dict.fromkeys(values))


def make_targets(study: Path) -> set[str]:
    mk = study / "Makefile"
    text = mk.read_text() if mk.exists() else ""
    return {t for rule in MAKE_RULE.findall(text) for t in rule.split()}


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
    motivated_by = parse_motivation(a.motivated_by, m["number"])
    tags = {"model": parse_tags("model", a.model, "MODEL"),
            "question": parse_tags("question", a.question_type, "QUESTION_TYPE")}
    d = STUDIES / a.name
    front = {"schema_version": 1, "id": a.name, "question": a.question, "status": "planned",
             "wandb": {"entity": a.entity, "project": a.project}, "motivated_by": motivated_by, "tags": tags}
    if issue is not None:
        front["issues"] = [issue]
    print(f"new study {a.name}")
    files = {}
    files[d / "README.md"] = ( "---\n" + yaml.safe_dump(front, sort_keys=False, allow_unicode=True) + "---\n"
          f"# Study: {a.name}\n\n"
          f"W&B project [`{a.project}`](https://wandb.ai/{a.entity}/{a.project}).\n\n"
          f"**Question.** {a.question}\n\n"
          "## Design\n\nTODO: the variants, what each varies, and what result would answer the question.\n\n"
          "## Verdict\n\n" + report_verdicts.wrap(report_verdicts.readme_block(d)) + "\n\n"
          "TODO once reports are live: also set `status` and `verdict` in the frontmatter.\n")
    files[d / "CHANGELOG.md"] = f"# Changelog — {a.name}\n\n## {TODAY}\n\n- Study created.\n"
    files[d / "Makefile"] = (
          f"# studies/{a.name}/Makefile\n#\n"
          "# `all` regenerates every report OFFLINE from committed frozen inputs (CI runs it).\n"
          "# One target per report (rN); one `freeze-*` target per freeze, never part of `all`.\n\n"
          "PY ?= python\n.PHONY: all\n\nall: verdicts\n" + VERDICTS_TARGET.format(py="$(PY)"))
    files[d / ".gitignore"] = ("# W&B pull caches are never committed and never an input\n"
                               "analysis/*cache*\n__pycache__/\n")
    files[d / "analysis" / "reports" / "INDEX.md"] = (
        f"# {a.name} — reports\n\n" + report_verdicts.wrap(report_verdicts.index_block(d)) + "\n")
    files.update(_variant_files(d, a.variant))
    commit(files)
    print(f"  variant {a.variant}: add sweep.yaml / experiment.yaml next to notes.md before launching")
    if motivated_by:
        print("  motivated_by: replace each TODO note with what that result showed")
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


# The version-2 report skeleton (schemas/README.md, Version-2 reports). Keep decimals and
# percentages out of it: the prose-number lint reads the hand-written parts.
REPORT_BODY = """\
# {rid} — {title}

## Question

TODO: the question in one sentence, and why it matters.

## Answer

- **F1.** TODO: claim F1, the same sentence. Order the findings as an argument: the headline
  first, then what supports it, then what limits it.

## Methods

<!-- BEGIN methods -->
TODO: written by the producer from the frozen inputs: mice, sessions, split, seeds, model
versions, interval type, and what n counts.
<!-- END methods -->

As in the study methods (README), except: TODO.

## Results

### F1 — TODO

<!-- BEGIN result-1 -->
TODO: written by the producer, never by hand. Figure first, then table; every number with its
n and its uncertainty.
<!-- END result-1 -->

## What it means

TODO: the interpretation. Cite findings by id (F1) instead of retyping their numbers, and
explain each specialised term where it first appears.

## Caveats

TODO: what this result does not show, and the other explanations not ruled out.

## Next

TODO: the next step, or "none".

## Provenance

W&B groups and inputs are in the frontmatter; reproduce with `make -C studies/{study} {rid}`.
"""


def new_report(a: argparse.Namespace) -> None:
    d = study_dir(a.study)
    reports = d / "analysis" / "reports"
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", a.slug):
        fail(f"SLUG {a.slug!r} must be lowercase words joined by -")
    if LAYOUT["analysis"]["cache_pattern"] in a.slug:
        fail(f"SLUG {a.slug!r} contains {LAYOUT['analysis']['cache_pattern']!r}: files named so are "
             "gitignored caches and skipped by the checker; pick another word")
    nums = [int(m["n"]) for p in reports.glob("r*.md")
            if (m := re.match(LAYOUT["analysis"]["report_name_pattern"], p.name))]
    # a deleted draft can leave its rN target behind; never hand its number to a new report
    targets = make_targets(d)
    n = max(nums + [int(t[1:]) for t in targets if re.fullmatch(r"r[0-9]+", t)], default=0) + 1
    rid, name = f"r{n}", f"r{n}-{a.slug}.md"
    script = f"analysis/r{n}_{a.slug.replace('-', '_')}.py"
    producer = d / script
    output = producer.with_suffix(".json")     # report-specific, so it cannot clobber a frozen input
    if output.exists():
        fail(f"{output.relative_to(REPO)} already exists; the new producer would overwrite it")
    body = REPORT_BODY.format(rid=rid, title=a.title, study=a.study)
    blocks = {k: _meta.block_digest(v) for k, v in _meta.RESULT_BLOCK.findall(body)}
    front = {"schema_version": 2, "id": rid, "slug": a.slug, "status": "draft",
             "draft_reason": "scaffolded: producer, claims and prose are still TODO",
             "authors": [a.author], "wandb_groups": [], "inputs": {"script": script},
             "claims": [{"id": "F1", "text": "TODO: the headline finding, in one sentence",
                         "pointer": f"{output.relative_to(d).as_posix()}#/TODO"}],
             "prose_reviewed": {"date": TODAY, "blocks": blocks},
             "reproduce": f"make -C studies/{a.study} {rid}"}
    files = {}
    files[reports / name] = "---\n" + yaml.safe_dump(front, sort_keys=False, allow_unicode=True) + "---\n" + body
    files[producer] = (
          f'"""Producer for {rid}: TODO describe what it computes from which frozen inputs."""\n'
          "import sys\nfrom pathlib import Path\n\n"
          "HERE = Path(__file__).resolve().parent\n"
          "sys.path.insert(0, str(HERE.parents[2] / 'studies' / 'util'))\n"
          "from _meta import build_meta, write_artifact  # noqa: E402\n"
          "# Figures: `import figure_style as fs`, `fs.apply_paper_style()`, draw at final size\n"
          "# (`figsize=fs.mm(183, 80)`), then `fs.save_figure(fig, HERE / 'fig_<slug>.svg')`.\n\n"
          "WANDB_GROUPS: list[str] = []   # exact groups read, if any (via a committed freeze)\n\n\n"
          "def main() -> None:\n"
          "    inputs: list[Path] = []      # committed frozen inputs read; _meta v2 needs at least one\n"
          "    result = {}                  # TODO compute; the report's claims point at values in here.\n"
          "    #   A verdict word is a field chosen from the numbers, never a fixed conclusion sentence.\n"
          f"    meta = build_meta('{script}', WANDB_GROUPS, study_root=HERE.parent,\n"
          "                      code=[__file__], inputs=inputs)\n"
          f"    # TODO: rewrite the methods and result-1 blocks in reports/{name} (match the marker pairs),\n"
          f"    #       then list them in outputs as HERE / 'reports' / '{name}#methods' and '...#result-1',\n"
          "    #       alongside any figures.\n"
          f"    write_artifact(HERE / '{output.name}', {{'_meta': meta, **result}}, outputs=[])\n\n\n"
          "if __name__ == '__main__':\n    main()\n")
    print(f"new report {a.study}/{rid}")
    commit(files)
    mk = d / "Makefile"
    targets = make_targets(d)
    if rid not in targets:
        mk.write_text(mk.read_text().rstrip("\n") + f"\n\n# {rid} = {a.title}\n.PHONY: {rid}\n{rid}:\n"
                      f"\t{make_var(d)} {script}\n")
        print(f"  added `{rid}` to Makefile (add it to `all` once the producer runs offline)")
    if "verdicts" not in targets:
        mk.write_text(mk.read_text().rstrip("\n") + "\n" + VERDICTS_TARGET.format(py=make_var(d)))
        print("  added `verdicts` to Makefile (add it to `all`)")
    for path in report_verdicts.write(d):
        print(f"  wrote the generated verdicts in {path.relative_to(REPO)}")
    print("  tags: add `tags: {model: [...], question: [...]}` for what this report does (schemas/tags.yaml)")
    print("  claims: point each at a value in the producer's JSON; when a block changes, re-read the prose and "
          "update `prose_reviewed` (make validate prints the digests)")


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
    s.add_argument("--motivated-by", help="earlier results this study follows up: 05/r1,03/r2 | 05 | #23 | none")
    s.add_argument("--model", help="models the study trains or fits, from schemas/tags.yaml: gru,cognitive")
    s.add_argument("--question-type", help="kinds of question it asks, from schemas/tags.yaml: scaling,comparison")
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
