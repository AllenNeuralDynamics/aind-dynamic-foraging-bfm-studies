"""Claims-check records: where a report's claims check is kept, and whether it is still current.

A claims check (ADR 0003; the dispatcher `posthoc-reporting` skill and its claims-check rubric) is
a reader who did not write the report comparing its text with its evidence. Each run is committed
as one record (studies#160):

    studies/<study>/analysis/provenance/claims_check_<rN>_<YYYY-MM-DD>[_<k>].md

A record is Markdown with a YAML header (schemas/claims-check-record.v1.schema.json) and three
sections: the prompt given to the reviewer, its response verbatim, and the response to each
finding. The header keys the run:

- ``content_digest``: sha256 of the report's ``claims`` and body (everything after the
  frontmatter). ``status``, ``draft_reason`` and ``prose_reviewed`` are left out, so flipping
  draft -> live does not make the check stale, while any change to a claim or the text does.
- ``inputs``: repository-relative path -> evidence digest of every other file the reviewer was
  given: for an analysis JSON, the digest of its body without ``_meta`` (``_meta.body_digest``,
  as in ``_meta.outputs``), so regenerating it with the same numbers changes nothing; for any
  other file, its sha256. The report itself is never an input: it is keyed by ``content_digest``
  alone, and a raw-byte hash of it would go stale on a ``status`` flip.
- ``context``: the same for reading context the reviewer was given (the repository glossary
  ``CONTEXT.md``, the study ``README.md``, ``PLAN.md``, ``CHANGELOG.md``). It is recorded but does
  not make a record stale: a glossary edit elsewhere is not a change to this report's evidence.
- ``prompt_sha256`` / ``response_sha256``: sha256 of the prompt and the response as given, so the
  record's copies can be checked to be verbatim.

`scripts/check_provenance.py` compares the newest record with the report and the files as they
are now (`report-claims-check-missing`, `report-claims-check-stale`). The record describes the
report; it never feeds the frontmatter, INDEX or README (ADR 0003).

    from claims_check import write_record
    write_record(study, report, inputs=[...], rubric_version="claims-check/2 @ <sha>",
                 model="claude-opus-5-5, effort high", prompt=Path("brief.txt"),
                 response=Path("response.md"), responses=...)

or from the shell (studies repo root), with the reviewer's own output file as the response:

    python studies/util/claims_check.py --report studies/<study>/analysis/reports/rN-<slug>.md \
        --inputs inputs.txt --rubric-version "claims-check/2 @ <sha>" --model "..." \
        --prompt brief.txt --response response.md --responses answers.md [--pr-comment URL]

``inputs.txt`` lists one repository-relative path per line (the files given to the reviewer).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _meta import body_digest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORD_DIR = "analysis/provenance"
RECORD_NAME = re.compile(r"^claims_check_(r\d+)_(\d{4}-\d{2}-\d{2})(?:_(\d+))?\.md$")
_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
# Reading context, recorded under `context` and never making a record stale: exactly the
# repository glossary and the report's own study README, PLAN and CHANGELOG.
STUDY_CONTEXT = ("README.md", "PLAN.md", "CHANGELOG.md")


def split_frontmatter(text: str) -> tuple[dict, str]:
    """(frontmatter dict, body) of a Markdown file with a YAML header."""
    m = _FRONTMATTER.match(text)
    if not m:
        raise ValueError("no YAML frontmatter")
    return yaml.safe_load(m.group(1)) or {}, text[m.end():]


def content_digest(report: Path | str) -> str:
    """sha256 of a report's claims and body: what a claims check reads and judges."""
    fm, body = split_frontmatter(Path(report).read_text())
    payload = json.dumps({"claims": fm.get("claims"), "body": body.rstrip()}, sort_keys=True,
                         ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_digest(path: Path) -> str:
    """An analysis JSON (one with `_meta`) by its body without `_meta`; any other file by sha256."""
    if path.suffix == ".json":
        try:
            obj = json.loads(path.read_text())
        except ValueError:
            obj = None
        if isinstance(obj, dict) and "_meta" in obj:
            return body_digest(obj)
    return file_sha256(path)


def records(study: Path, rid: str) -> list[Path]:
    """The claims-check records of report `rid` in a study, oldest first."""
    found = []
    for p in (study / RECORD_DIR).glob(f"claims_check_{rid}_*.md"):
        m = RECORD_NAME.match(p.name)
        if m and m.group(1) == rid:
            found.append((m.group(2), int(m.group(3) or 0), p))
    return [p for *_, p in sorted(found)]


def read_header(record: Path) -> dict:
    fm, _ = split_frontmatter(record.read_text())
    return json.loads(json.dumps(fm, default=str))    # YAML dates -> strings, as the schema sees them


def stale_reasons(root: Path, study: Path, header: dict) -> list[str]:
    """Why a record no longer describes the report and its evidence (empty: it is current)."""
    reasons = []
    report = study / str(header.get("report", ""))
    report_rel = str(report.resolve().relative_to(root.resolve())) if report.exists() else None
    if not report.is_file():
        return [f"report {header.get('report')} does not exist"]
    if content_digest(report) != header.get("content_digest"):
        reasons.append("the report's claims or text changed")
    changed = []
    for rel, digest in sorted((header.get("inputs") or {}).items()):
        if rel == report_rel:
            continue                                     # the report is keyed by content_digest only
        p = root / rel
        if not p.is_file():
            changed.append(f"{rel} (gone)")
        elif evidence_digest(p) != digest:
            changed.append(rel)
    if changed:
        reasons.append("input(s) changed: " + ", ".join(changed[:5]) + (f" and {len(changed) - 5} more" if len(changed) > 5 else ""))
    return reasons


def _fenced(text: str) -> str:
    """`text` unchanged inside a fence longer than any run of tildes in it."""
    runs = re.findall(r"~+", text)
    fence = "~" * max(4, max((len(r) for r in runs), default=0) + 1)
    return f"{fence}text\n{text}{'' if text.endswith(chr(10)) else chr(10)}{fence}"


def _text(x: str | Path) -> str:
    """A string as given, or a file's text read unchanged (a Path)."""
    return Path(x).read_text() if isinstance(x, Path) else x


def write_record(study: Path, report: Path, *, inputs: list[Path], rubric_version: str, model: str,
                 prompt: str | Path, response: str | Path, responses: str | Path, pr_comment: str | None = None,
                 date: str | None = None, root: Path = REPO_ROOT) -> Path:
    """Write one claims-check record and return its path.

    Call it with the report and input files exactly as the reviewer read them: the digests are
    taken now. The report is keyed by its content digest and is dropped from `inputs` if listed
    there; `CONTEXT.md` and a study's `README.md`, `PLAN.md` and `CHANGELOG.md` go under `context`. `prompt` and `response` are kept verbatim (their sha256 goes in the header);
    `responses` is Markdown (what was fixed, or why not, per finding). Each of the three may be a
    Path, read unchanged: pass the file the reviewer wrote its output to, so the copy is verbatim
    by construction."""
    prompt, response, responses = _text(prompt), _text(response), _text(responses)
    study, report, root = Path(study).resolve(), Path(report).resolve(), Path(root).resolve()
    rid = re.match(r"(r\d+)-", report.name).group(1)
    date = date or dt.date.today().isoformat()
    out_dir = study / RECORD_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out, k = out_dir / f"claims_check_{rid}_{date}.md", 1
    while out.exists():
        k += 1
        out = out_dir / f"claims_check_{rid}_{date}_{k}.md"
    files = [Path(p).resolve() for p in inputs if Path(p).resolve() != report]
    context_paths = {root / "CONTEXT.md", *(study / n for n in STUDY_CONTEXT)}
    evidence = [p for p in files if p not in context_paths]
    context = [p for p in files if p in context_paths]
    header = {"schema_version": 1, "report": str(report.relative_to(study)), "checked_on": date,
              "content_digest": content_digest(report),
              "inputs": {str(p.relative_to(root)): evidence_digest(p) for p in evidence},
              "context": {str(p.relative_to(root)): file_sha256(p) for p in context},
              "rubric_version": rubric_version, "model": model,
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "response_sha256": hashlib.sha256(response.encode()).hexdigest()}
    if pr_comment:
        header["pr_comment"] = pr_comment
    out.write_text("---\n" + yaml.safe_dump(header, sort_keys=False, allow_unicode=True) + "---\n"
                   f"# Claims check — {rid}, {date}\n\n"
                   "The record describes the report; it never feeds the frontmatter, INDEX or README (ADR 0003).\n\n"
                   f"## Prompt given to the reviewer\n\n{_fenced(prompt)}\n\n"
                   f"## Reviewer's response\n\n{_fenced(response)}\n\n"
                   f"## Responses to the findings\n\n{responses.strip()}\n")
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Write one claims-check record (studies#160, #164).")
    ap.add_argument("--report", required=True, type=Path, help="the report checked (studies/<study>/analysis/reports/rN-*.md)")
    ap.add_argument("--inputs", required=True, type=Path, help="file listing the reviewer's inputs, one repo-relative path per line")
    ap.add_argument("--rubric-version", required=True)
    ap.add_argument("--model", required=True)
    for k in ("--prompt", "--response", "--responses"):
        ap.add_argument(k, required=True, type=Path)
    ap.add_argument("--pr-comment")
    ap.add_argument("--date")
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    a = ap.parse_args(argv)
    root = a.root.resolve()
    report = (a.report if a.report.is_absolute() else root / a.report).resolve()
    study = report.parents[2]
    inputs = [root / line.strip() for line in a.inputs.read_text().splitlines() if line.strip() and not line.startswith("#")]
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing:
        raise SystemExit(f"inputs not found: {missing}")
    out = write_record(study, report, inputs=inputs, rubric_version=a.rubric_version, model=a.model,
                       prompt=a.prompt, response=a.response, responses=a.responses,
                       pr_comment=a.pr_comment, date=a.date, root=root)
    print(out)


if __name__ == "__main__":
    main()
