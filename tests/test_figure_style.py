"""studies/util/figure_audit.py, figure_style.py and scripts/check_figures.py."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "studies" / "util"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _meta  # noqa: E402
import check_figures  # noqa: E402
import figure_audit  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figure_style as fs
except ImportError:          # the provenance job has no matplotlib; the regenerate job does
    fs = None

ARIAL = "font-family: 'Arial', 'Liberation Sans', 'Helvetica', sans-serif"


def svg(body: str, width: str = "250pt", height: str = "150pt") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{width}" height="{height}" viewBox="0 0 250 150">{body}</svg>')


GOOD = svg(f'<text style="font-size: 7px; {ARIAL}" x="10" y="20">held-out likelihood</text>')


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def audit(self, text: str) -> list[str]:
        p = Path(self.tmp.name) / "f.svg"
        p.write_text(text)
        return figure_audit.audit_svg(p)

    def test_paper_style_text_passes(self):
        self.assertEqual(self.audit(GOOD), [])

    def test_outlined_letters(self):
        problems = self.audit(svg('<defs><path id="DejaVuSans-31" d="M0 0"/></defs>'))
        self.assertTrue(any("outlines" in p for p in problems), problems)

    def test_per_letter_tspans(self):
        body = (f'<text style="font-size: 6px; {ARIAL}"><tspan x="1" y="2">1</tspan>'
                '<tspan x="5" y="2">0</tspan></text>')
        self.assertTrue(any("<tspan>" in p for p in self.audit(svg(body))))

    def test_other_font(self):
        body = '<text style="font-size: 7px; font-family: \'DejaVu Sans\', sans-serif">x</text>'
        self.assertTrue(any("DejaVu Sans" in p for p in self.audit(svg(body))))

    def test_text_size_limits(self):
        small = self.audit(svg(f'<text style="font-size: 4px; {ARIAL}">x</text>'))
        large = self.audit(svg(f'<text style="font-size: 15px; {ARIAL}">x</text>'))
        self.assertTrue(any("minimum" in p for p in small), small)
        self.assertTrue(any("maximum" in p for p in large), large)

    def test_figure_size_limits(self):
        problems = self.audit(svg("", width="936pt", height="360pt"))   # 13 x 5 in
        self.assertTrue(any("wide" in p for p in problems), problems)
        problems = self.audit(svg("", width="250pt", height="500pt"))
        self.assertTrue(any("tall" in p for p in problems), problems)

    def test_slide_limits(self):
        p = Path(self.tmp.name) / "s.svg"
        p.write_text(svg(f'<text style="font-size: 18px; {ARIAL}">x</text>', width="936pt", height="446pt"))
        self.assertEqual(figure_audit.audit_svg(p, "slide"), [])
        self.assertTrue(figure_audit.audit_svg(p, "paper"))          # too wide, text too large
        p.write_text(svg(f'<text style="font-size: 7px; {ARIAL}">x</text>'))
        self.assertTrue(any("slide minimum" in x for x in figure_audit.audit_svg(p, "slide")))


class CheckFiguresTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        env = {k: v for k, v in os.environ.items() if k not in _meta._GIT_DISCOVERY_VARS}
        subprocess.run(["git", "init", "-q", str(self.root)], check=True, env=env)
        p = mock.patch.object(check_figures, "ROOT", self.root)
        p.start()
        self.addCleanup(p.stop)

    def study(self, name: str, files: dict[str, str]) -> None:
        for rel, text in files.items():
            f = self.root / "studies" / name / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)

    def test_new_study_png_and_bad_svg_fail(self):
        self.study("14-gru-new", {"analysis/fig_a.png": "x", "analysis/fig_b.svg": svg("", "936pt"),
                                  "analysis/fig_c.svg": GOOD,
                                  "variants/v/launch_record/old.png": "x", "reference/paper.png": "x"})
        findings = check_figures.check_study(self.root / "studies" / "14-gru-new")
        self.assertEqual(len(findings), 2, findings)
        self.assertTrue(any("fig_a.png" in f for f in findings))
        self.assertTrue(any("fig_b.svg" in f and "wide" in f for f in findings))

    def test_slides_dir_uses_slide_limits(self):
        slide = svg(f'<text style="font-size: 18px; {ARIAL}">x</text>', width="936pt", height="446pt")
        self.study("14-gru-new", {"analysis/slides/fig_talk.svg": slide, "analysis/fig_talk.svg": slide})
        findings = check_figures.check_study(self.root / "studies" / "14-gru-new")
        self.assertTrue(findings and all("slides/" not in f for f in findings), findings)

    def test_gitignored_and_legacy_are_skipped(self):
        self.study("14-gru-new", {".gitignore": "_cache*\n", "analysis/_cache_plot.png": "x"})
        self.study("01-gru-scaling-law", {"analysis/fig.png": "x"})
        with mock.patch("sys.stdout"):
            self.assertEqual(check_figures.main([]), 0)


@unittest.skipIf(fs is None, "matplotlib not installed")
class SaveFigureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        fs.apply_paper_style()

    def tearDown(self):
        plt.close("all")
        matplotlib.rcdefaults()

    def plot(self, plain: bool):
        fig, ax = plt.subplots(figsize=fs.mm(89, 60))
        ax.plot([1, 10, 100], [1, 2, 3])
        ax.set_xscale("log")
        if plain:
            fs.plain_log_axis(ax, "x")
        ax.set_xlabel("trials (log)")
        fs.panel_label(ax, "a")
        return fig

    def test_paper_figure_passes_and_regenerates_identically(self):
        a, b = Path(self.tmp.name) / "a.svg", Path(self.tmp.name) / "b.svg"
        fs.save_figure(self.plot(plain=True), a)
        fs.save_figure(self.plot(plain=True), b)
        self.assertEqual(figure_audit.audit_svg(a), [])
        self.assertEqual(a.read_bytes(), b.read_bytes())
        self.assertIn("Arial", a.read_text())

    def test_default_log_ticks_are_refused(self):
        with self.assertRaisesRegex(ValueError, "tspan"):
            fs.save_figure(self.plot(plain=False), Path(self.tmp.name) / "c.svg")

    def test_only_svg(self):
        with self.assertRaisesRegex(ValueError, "svg only"):
            fs.save_figure(self.plot(plain=True), Path(self.tmp.name) / "c.png")

    def test_slide_style_saves_under_slides(self):
        fs.apply_slide_style()
        fig, ax = plt.subplots(figsize=(fs.SLIDE_IN[0], 6))
        fs.violins(ax, [[1, 2, 3, 4], [2, 3, 4, 5]], ["C0", "C1"], ["HB", "GRU"])
        ax.set_ylabel("held-out likelihood")
        slides = Path(self.tmp.name) / "slides"
        slides.mkdir()
        self.assertEqual(figure_audit.audit_svg(fs.save_figure(fig, slides / "s.svg", style="slide"), "slide"), [])
        with self.assertRaisesRegex(ValueError, "slides/"):
            fs.save_figure(fig, Path(self.tmp.name) / "s.svg", style="slide")
        with self.assertRaisesRegex(ValueError, "slides/"):
            fs.save_figure(fig, slides / "t.svg")          # paper style may not go in slides/


if __name__ == "__main__":
    unittest.main()
