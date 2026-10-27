"""The figure styles study figures use, and the one way to save a figure.

Paper style, the default for every report figure: Nature-family final size
(research-figure-guide.nature.com), 89 mm (single column) or 183 mm (double) wide, at most 170 mm
tall, 5-7 pt sans-serif text, 8 pt bold lowercase panel letters.

Slide style, for a talk: a 16:9 slide (13.33 x 7.5 in) at 1:1, 14-22 pt text and heavy marks.
Slide figures live under an ``analysis/slides/`` directory and are saved with
``save_figure(..., style="slide")``; reports embed the paper figures.

Figures are saved as SVG only, with text kept as editable ``<text>`` in Arial, and are checked as
they are written (``figure_audit``), so a figure that would break in Illustrator never lands.

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "util"))
    import figure_style as fs

    fs.apply_paper_style()
    fig, ax = plt.subplots(figsize=fs.mm(89, 60), layout="constrained")
    ...
    fs.save_figure(fig, HERE / "fig_r1_curve.svg")

Fonts: the SVG names Arial, so Illustrator, journals and browsers on Mac/Windows/iOS draw Arial.
Layout is computed with Liberation Sans (vendored in ``fonts/``, SIL Open Font License), which has
Arial's exact letter widths, so the text sits where matplotlib put it on any machine, CI included.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import font_manager, ticker

import figure_audit

MM = 1 / 25.4                                   # inches per mm
SINGLE_MM, DOUBLE_MM, MAX_HEIGHT_MM = 89.0, 183.0, 170.0
SLIDE_IN = (13.333, 7.5)                        # 16:9 slide; leave room for the slide's own title
SANS = ["Arial", "Liberation Sans", "Helvetica"]
FONT_DIR = Path(__file__).resolve().parent / "fonts"

# Okabe-Ito: colour-blind safe, distinct in greyscale print.
OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
             "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}

_EXPORT = {
    "svg.fonttype": "none",                     # text stays <text>, never glyph outlines
    "svg.hashsalt": "bfm-studies",              # fixed ids: regenerating writes identical bytes
    "pdf.fonttype": 42, "ps.fonttype": 42,      # TrueType if a PDF is ever made for submission
    "axes.formatter.use_mathtext": False,
}


def mm(width: float, height: float) -> tuple[float, float]:
    """A matplotlib figsize (inches) from millimetres: ``figsize=fs.mm(183, 90)``."""
    return width * MM, height * MM


def _register_fonts() -> None:
    known = {f.fname for f in font_manager.fontManager.ttflist}
    for f in sorted(FONT_DIR.glob("*.ttf")):
        if str(f) not in known:
            font_manager.fontManager.addfont(str(f))
    found = Path(font_manager.findfont(font_manager.FontProperties(family=SANS),
                                       fallback_to_default=False))
    if "dejavu" in found.name.lower():
        raise RuntimeError(f"text would be laid out in {found.name}, not Arial or Liberation Sans; "
                           f"is {FONT_DIR} present?")


def _apply(sizes: dict) -> None:
    mpl.rcdefaults()
    _register_fonts()
    mpl.rcParams.update({
        **_EXPORT,
        "font.family": "sans-serif", "font.sans-serif": SANS,
        "axes.spines.top": False, "axes.spines.right": False,
        "errorbar.capsize": 0, "legend.frameon": False,
        "figure.constrained_layout.use": True,
        "axes.prop_cycle": mpl.cycler(color=[OKABE_ITO[k] for k in
                                             ("blue", "vermillion", "green", "orange", "purple", "sky",
                                              "yellow", "black")]),
        **sizes,
    })


def apply_paper_style() -> None:
    """Reset matplotlib and apply the paper style: 5-7 pt Arial, 0.5 pt axes, Okabe-Ito colours."""
    _apply({
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "figure.titlesize": 7,
        "figure.labelsize": 7, "legend.fontsize": 6, "legend.title_fontsize": 6,
        "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.linewidth": 0.5, "axes.titlepad": 3, "axes.labelpad": 2,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "lines.linewidth": 1.0, "lines.markersize": 2.5, "patch.linewidth": 0.5,
        "hatch.linewidth": 0.5, "grid.linewidth": 0.4,
        "legend.handlelength": 1.2, "legend.borderaxespad": 0.2,
        "figure.dpi": 150, "savefig.dpi": 300,  # 300 dpi applies only to rasterized artists
    })


def apply_slide_style() -> None:
    """Reset matplotlib and apply the slide style: 14-20 pt Arial, 1.5 pt axes, heavy lines."""
    _apply({
        "font.size": 18, "axes.labelsize": 20, "axes.titlesize": 20, "figure.titlesize": 20,
        "figure.labelsize": 20, "legend.fontsize": 16, "legend.title_fontsize": 16,
        "xtick.labelsize": 16, "ytick.labelsize": 16,
        "axes.linewidth": 1.5, "axes.titlepad": 8, "axes.labelpad": 6,
        "xtick.major.width": 1.5, "ytick.major.width": 1.5, "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0, "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.minor.size": 3.5, "ytick.minor.size": 3.5,
        "lines.linewidth": 3.0, "lines.markersize": 7, "patch.linewidth": 1.5,
        "hatch.linewidth": 1.5, "grid.linewidth": 1.0,
        "figure.dpi": 100, "savefig.dpi": 200,
    })


def num(value: float, spec: str = "g") -> str:
    """A number for figure text, with a true minus sign (−) as matplotlib's tick labels use."""
    return format(value, spec).replace("-", "−")


def plain_log_axis(ax, which: str = "both") -> None:
    """Log-axis ticks as plain text (0.5, 1, 10) instead of math text, which splits per letter."""
    fmt = ticker.FuncFormatter(lambda v, _: f"{v:g}")
    for axis in ((ax.xaxis, ax.yaxis) if which == "both" else (getattr(ax, f"{which}axis"),)):
        axis.set_major_formatter(fmt)
        axis.set_minor_formatter(ticker.NullFormatter())


def panel_label(ax, letter: str, dx_pt: float = -24, dy_pt: float = 6) -> None:
    """Panel letter, bold lowercase, 1 pt above the body text (8 pt in the paper style), above the
    axes' top-left corner, offset in points."""
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(dx_pt, dy_pt),
                textcoords="offset points", fontsize=mpl.rcParams["font.size"] + 1, fontweight="bold",
                ha="left", va="bottom")


def violins(ax, groups, colors, labels=None, width: float = 0.7, alpha: float = 0.35):
    """One violin per group, with its interquartile range as a bar and its median as a white dot.

    Returns the x positions (1, 2, ...). ``labels`` become the x tick labels. Marks scale with the
    style's line width, so the same call reads on a page and on a slide."""
    lw = mpl.rcParams["lines.linewidth"]
    pos = np.arange(1, len(groups) + 1)
    parts = ax.violinplot(groups, positions=pos, widths=width, showextrema=False)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(alpha)
        body.set_edgecolor("none")
    for x, g, c in zip(pos, groups, colors):
        q1, med, q3 = np.percentile(g, [25, 50, 75])
        ax.vlines(x, q1, q3, color=c, lw=1.2 * lw, zorder=2)
        ax.scatter([x], [med], s=6 * lw ** 2, color="white", edgecolor=c, linewidth=0.6 * lw, zorder=3)
    ax.set_xticks(pos, labels if labels is not None else [""] * len(pos))
    ax.set_xlim(0.4, len(groups) + 0.6)
    return pos


def save_figure(fig, path: Path | str, style: str = "paper") -> Path:
    """Save ``fig`` as SVG at its own size and check it; raises ValueError if it would not pass.

    ``style`` sets the size and text limits checked: "paper" (default, report figures) or "slide"
    (figures under an ``analysis/slides/`` directory). Dense layers (thousands of points,
    image-like heatmaps) keep the file small when drawn with ``rasterized=True``; axes and text
    stay vector.
    """
    path = Path(path)
    if path.suffix != ".svg":
        raise ValueError(f"{path.name}: figures are saved as .svg only")
    if (style == "slide") != (figure_audit.SLIDES_DIR in path.parts):
        raise ValueError(f"{path.name}: slide figures, and only they, go under an "
                         f"'{figure_audit.SLIDES_DIR}/' directory (style={style!r})")
    with mpl.rc_context(_EXPORT):
        fig.savefig(path, format="svg", metadata={"Date": None})
    problems = figure_audit.audit_svg(path, style)
    if problems:
        raise ValueError(f"{path.name}: " + "; ".join(problems))
    return path
