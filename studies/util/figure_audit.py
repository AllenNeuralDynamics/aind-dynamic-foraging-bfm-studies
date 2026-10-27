"""Check that a saved SVG figure is editable in Illustrator and fits its style (paper or slide).

Standard library only, so the CI provenance job (no matplotlib) can run it through
``scripts/check_figures.py``; ``figure_style.save_figure()`` runs it on every figure it writes.

What it catches, each a real failure seen in this repo:

- letters saved as outlines (matplotlib's default ``svg.fonttype: path``);
- labels split into one ``<tspan>`` per letter (math text such as default log ticks, ``$10^2$``),
  which Illustrator imports as separate characters;
- a font other than Arial or its stand-ins, e.g. DejaVu Sans, the fallback on the compute nodes;
- text or a figure outside its style's limits (LIMITS): a paper figure has 5-8 pt text and is
  at most 183 mm wide (any height), drawn at the size it is printed (Nature: 5-7 pt text, 8 pt
  panel letters); a slide figure, kept under an `analysis/slides/` directory, has 12-28 pt text and
  fits a 16:9 slide.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

MM_PER_PT = 25.4 / 72
# style -> max width, max height (mm; None = any height), text range (pt). A paper figure is
# drawn at the size it is printed and never rescaled, so its text sizes are final; its height is
# free (a prototype grid can be tall). A slide figure must fit the slide.
LIMITS = {"paper": {"max_w": 183.0, "max_h": None, "pt": (5.0, 8.0)},
          "slide": {"max_w": 338.7, "max_h": 190.5, "pt": (12.0, 28.0)}}
SLIDES_DIR = "slides"
NAMED_FONTS = ("Arial", "Liberation Sans", "Helvetica")   # the first font in a stack must be one of these
FONTS = {*NAMED_FONTS, "sans-serif"}

_SVG = "{http://www.w3.org/2000/svg}"
# matplotlib glyph outlines are ids "<PostScript name>-<hex code>": DejaVuSans-31, LiberationSans-Bold-41
_GLYPH_ID = re.compile(r"^[A-Za-z][\w-]*-[0-9a-f]+$")
_TOL_MM = 0.5


def _pt(value: str | None) -> float | None:
    if not value:
        return None
    m = re.fullmatch(r"\s*([\d.]+)\s*(pt)?\s*", value)
    return float(m.group(1)) if m else None


def _fonts(el, family, size, scaled, out: list, problems: set) -> None:
    """Walk the tree carrying the inherited font; record (family stack, size) for every <text>.

    Only the encoding matplotlib writes is accepted (inline ``font-family`` / ``font-size: Npx``
    longhand, no scaling transform above the text); anything else is reported rather than assumed
    compliant."""
    css = el.get("style") or ""
    scaled = scaled or bool(re.search(r"\b(scale|matrix|skew[XY]?)\s*\(", el.get("transform") or ""))
    if el.get("font-family") is not None or el.get("font-size") is not None or el.get("font") is not None:
        problems.add("font set by an SVG attribute, not inline style (save with figure_style.save_figure)")
    if re.search(r"(^|;)\s*font\s*:", css):
        problems.add("font set by the CSS 'font:' shorthand (save with figure_style.save_figure)")
    m = re.search(r"(^|;)\s*font-family:\s*([^;]+)", css)
    if m:
        family = tuple(f.strip(" '\"") for f in m.group(2).split(","))
    m = re.search(r"(^|;)\s*font-size:\s*([^;]+)", css)
    if m:
        n = re.fullmatch(r"\s*([\d.]+)px\s*", m.group(2))
        if n:
            size = float(n.group(1))              # matplotlib's SVG user unit is 1 pt
        else:
            problems.add(f"font-size {m.group(2).strip()!r}; only px (= pt) is written by matplotlib")
    if el.tag == f"{_SVG}text":
        if family is None or size is None:
            problems.add("text with no font-family or font-size of its own or inherited")
        if scaled:
            problems.add("text under a scale/matrix transform, so its printed size is not its font-size")
        out.append((family, size))
    for child in el:
        _fonts(child, family, size, scaled, out, problems)


def style_for(path: Path | str) -> str:
    """"slide" for a figure whose own directory is `slides/`, else "paper" (ancestors do not count,
    so a checkout under some /work/slides/ is still paper)."""
    return "slide" if Path(path).parent.name == SLIDES_DIR else "paper"


def audit_svg(path: Path | str, style: str = "paper") -> list[str]:
    """Problems with one SVG under ``style`` (see LIMITS), each a short sentence; [] means it passes."""
    lim = LIMITS[style]
    min_pt, max_pt = lim["pt"]
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
        w, h = width * MM_PER_PT, height * MM_PER_PT
        if w > lim["max_w"] + _TOL_MM:
            problems.append(f"{w:.0f} mm wide; the {style} limit is {lim['max_w']:g} mm")
        if lim["max_h"] and h > lim["max_h"] + _TOL_MM:
            problems.append(f"{h:.0f} mm tall; the {style} limit is {lim['max_h']:g} mm")
        box = (root.get("viewBox") or "").replace(",", " ").split()
        size = [_pt(b) for b in box[2:]] if len(box) == 4 else [None]   # no viewBox: user units are px
        if None in size or not all(abs(b - s) <= 1e-3 * s for b, s in zip(size, (width, height))):
            # font sizes are in user units; they are points only when the viewBox is the page in pt
            problems.append(f"viewBox {' '.join(box) or '(missing)'} is not the page size in pt, so font sizes are not points")

    glyphs = [p.get("id") for p in root.iter(f"{_SVG}path")
              if p.get("id") and _GLYPH_ID.match(p.get("id"))]
    if glyphs:
        problems.append(f"{len(glyphs)} letters saved as outlines, not text (svg.fonttype must be 'none')")
    split = sum(1 for t in root.iter(f"{_SVG}text") if t.find(f"{_SVG}tspan") is not None)
    if split:
        problems.append(f"{split} labels split into one <tspan> per letter (math text; "
                        "use plain text or figure_style.plain_log_axis)")

    texts, font_problems = [], set()
    _fonts(root, None, None, False, texts, font_problems)
    problems += sorted(font_problems)
    families = set().union(*(f for f, _ in texts if f))
    sizes = {s for _, s in texts if s is not None}
    if families - FONTS:
        problems.append(f"fonts {sorted(families - FONTS)}; only {sorted(FONTS)} are allowed")
    generic_first = sorted({f[0] for f, _ in texts if f and f[0] not in NAMED_FONTS})
    if generic_first:      # a renderer may pick its own default (DejaVu Sans) for a generic family
        problems.append(f"a font stack starts with {generic_first}; it must start with one of {list(NAMED_FONTS)}")
    small = sorted(s for s in sizes if s < min_pt - 1e-6)
    large = sorted(s for s in sizes if s > max_pt + 1e-6)
    if small:
        problems.append(f"text at {', '.join(f'{s:g}' for s in small)} pt; "
                        f"the {style} minimum is {min_pt:g} pt")
    if large:
        problems.append(f"text at {', '.join(f'{s:g}' for s in large)} pt; "
                        f"the {style} maximum is {max_pt:g} pt")
    return problems
