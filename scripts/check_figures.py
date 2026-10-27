#!/usr/bin/env python3
"""Every study figure is an editable, paper-style SVG (studies/util/figure_style.py).

For each study not listed in LEGACY_STUDIES, every figure it commits (tracked or new, not
gitignored; launch_record/ and reference/ excluded) must be an SVG that passes
studies/util/figure_audit.py: text kept as Arial <text>, no label split into per-letter
<tspan>s, and the paper limits (5-8 pt text, at most 183 mm wide), or the slide limits for
figures under a slides/ directory. Raster or PDF figures (.png, .jpg, .gif, .pdf,
.eps, .tif) are findings: save with figure_style.save_figure(), which writes SVG only.

LEGACY_STUDIES predate the style (#44) and are printed as EXEMPT, never silently skipped;
remove a study from the list when its figures are redrawn. A new study is checked from its
first figure. Standard library only, so the CI provenance job runs it without matplotlib.

Usage: python3 scripts/check_figures.py [--study 13-hb-subject-vs-gru-heldout ...]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "studies" / "util"))
import figure_audit  # noqa: E402

# Drawn before the paper style existed (#44). Shrinks as studies are redrawn; never grows.
LEGACY_STUDIES = {
    "01-gru-scaling-law", "02-gru-scaling-law-ignore", "03-disrnn-beta-scan",
    "04-gru-vs-disrnn-embedding-recovery", "05-disrnn-scaling-law",
    "06-disrnn-operating-point-at-scale", "07-gru-timing-inputs", "08-hb-vs-gru-heldout",
    "09-gru-cross-species-transfer", "10-gru-reservoir-ablation", "11-disrnn-near-gru-ceiling",
}
RASTER = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".eps", ".tif", ".tiff", ".webp"}
EXCLUDED_DIRS = {"launch_record", "reference"}


def study_files(study: Path) -> list[Path]:
    """Tracked and new files under the study, minus gitignored ones."""
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--",
                          str(study.relative_to(ROOT))], cwd=ROOT, capture_output=True, check=True)
    files = [ROOT / p for p in out.stdout.decode().split("\0") if p]
    return [f for f in files if f.is_file() and not EXCLUDED_DIRS & set(f.relative_to(study).parts)]


def check_study(study: Path) -> list[str]:
    findings = []
    for f in sorted(study_files(study)):
        rel = f.relative_to(ROOT)
        if f.suffix.lower() in RASTER:
            findings.append(f"{rel}: {f.suffix} figure; save it with figure_style.save_figure() (SVG only)")
        elif f.suffix.lower() == ".svg":
            findings += [f"{rel}: {p}" for p in figure_audit.audit_svg(f, figure_audit.style_for(f))]
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", action="append", help="check only these studies (repeatable)")
    a = ap.parse_args(argv)
    studies = sorted(p for p in (ROOT / "studies").iterdir() if p.is_dir() and p.name[:2].isdigit())
    if a.study:
        unknown = set(a.study) - {s.name for s in studies}
        if unknown:
            ap.error(f"no such study: {', '.join(sorted(unknown))}")
        studies = [s for s in studies if s.name in a.study]
    findings = []
    for s in studies:
        if s.name in LEGACY_STUDIES:
            print(f"EXEMPT {s.name}: drawn before the paper figure style (#44)")
            continue
        found = check_study(s)
        print(f"{'FAIL' if found else 'ok'}   {s.name}" + (f": {len(found)} finding(s)" if found else ""))
        findings += found
    for f in findings:
        print(f"  {f}")
    if findings:
        print(f"\n{len(findings)} figure finding(s). Redraw with studies/util/figure_style.py "
              "(apply_paper_style + save_figure).")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
