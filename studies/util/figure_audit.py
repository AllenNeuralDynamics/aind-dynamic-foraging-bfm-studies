"""Check that a saved SVG figure is editable in Illustrator and fits the paper style.

Standard library only, so the CI provenance job (no matplotlib) can run it through
``scripts/check_figures.py``; ``figure_style.save_figure()`` runs it on every figure it writes.

What it catches, each a real failure seen in this repo:

- letters saved as outlines (matplotlib's default ``svg.fonttype: path``);
- labels split into one ``<tspan>`` per letter (math text such as default log ticks, ``$10^2$``),
  which Illustrator imports as separate characters;
- a font other than Arial or its stand-ins, e.g. DejaVu Sans, the fallback on the compute nodes;
- text under 5 pt or over 8 pt, or a figure wider than 183 mm or taller than 170 mm (Nature's
  limits: 5-7 pt text, 8 pt panel letters, 89 or 183 mm wide).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

MM_PER_PT = 25.4 / 72
MAX_WIDTH_MM, MAX_HEIGHT_MM = 183.0, 170.0
MIN_PT, MAX_PT = 5.0, 8.0
FONTS = {"Arial", "Liberation Sans", "Helvetica", "sans-serif"}

_SVG = "{http://www.w3.org/2000/svg}"
_GLYPH_ID = re.compile(r"^[A-Za-z][\w]*-[0-9a-f]+$")   # matplotlib glyph outlines: DejaVuSans-31
_TOL_MM = 0.5


def _pt(value: str | None) -> float | None:
    if not value:
        return None
    m = re.fullmatch(r"\s*([\d.]+)\s*(pt)?\s*", value)
    return float(m.group(1)) if m else None


def audit_svg(path: Path | str) -> list[str]:
    """Problems with one SVG, each a short sentence; an empty list means it passes."""
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        return [f"not a parseable SVG ({e})"]
    problems = []
    width, height = _pt(root.get("width")), _pt(root.get("height"))
    if width is None or height is None:
        problems.append("no width/height in pt on the <svg> element (save with figure_style.save_figure)")
    else:
        if width * MM_PER_PT > MAX_WIDTH_MM + _TOL_MM:
            problems.append(f"{width * MM_PER_PT:.0f} mm wide; the limit is {MAX_WIDTH_MM:.0f} mm")
        if height * MM_PER_PT > MAX_HEIGHT_MM + _TOL_MM:
            problems.append(f"{height * MM_PER_PT:.0f} mm tall; the limit is {MAX_HEIGHT_MM:.0f} mm")

    glyphs = [p.get("id") for p in root.iter(f"{_SVG}path")
              if p.get("id") and _GLYPH_ID.match(p.get("id"))]
    if glyphs:
        problems.append(f"{len(glyphs)} letters saved as outlines, not text (svg.fonttype must be 'none')")
    split = sum(1 for t in root.iter(f"{_SVG}text") if t.find(f"{_SVG}tspan") is not None)
    if split:
        problems.append(f"{split} labels split into one <tspan> per letter (math text; "
                        "use plain text or figure_style.plain_log_axis)")

    families, sizes = set(), []
    for el in root.iter():
        style = el.get("style") or ""
        m = re.search(r"font-family:\s*([^;]+)", style)
        if m:
            families |= {f.strip(" '\"") for f in m.group(1).split(",")}
        m = re.search(r"font-size:\s*([\d.]+)px", style)
        if m:
            sizes.append(float(m.group(1)))      # matplotlib's SVG user unit is 1 pt
    if families - FONTS:
        problems.append(f"fonts {sorted(families - FONTS)}; only {sorted(FONTS)} are allowed")
    small = sorted({s for s in sizes if s < MIN_PT - 1e-6})
    large = sorted({s for s in sizes if s > MAX_PT + 1e-6})
    if small:
        problems.append(f"text at {', '.join(f'{s:g}' for s in small)} pt; the minimum is {MIN_PT:g} pt")
    if large:
        problems.append(f"text at {', '.join(f'{s:g}' for s in large)} pt; the maximum is {MAX_PT:g} pt")
    return problems
