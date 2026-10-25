"""scripts/scaffold.py output passes scripts/check_provenance.py with no findings."""

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
import check_provenance as cp  # noqa: E402
import scaffold  # noqa: E402


class ScaffoldTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "studies" / "11-gru-existing").mkdir(parents=True)
        env = {k: v for k, v in os.environ.items() if k not in _meta._GIT_DISCOVERY_VARS}
        subprocess.run(["git", "init", "-q", str(self.root)], check=True, env=env)
        patches = [mock.patch.object(scaffold, "REPO", self.root),
                   mock.patch.object(scaffold, "STUDIES", self.root / "studies")]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def findings(self):
        return [f.key for f in cp.Checker(self.root).run()
                if f.level != "info" and "11-gru-existing" not in f.path]

    def new_study(self, name="12-gru-example", motivated_by="none"):
        scaffold.main(["study", "--name", name, "--question", "Does the example hold up?",
                       "--project", "gru_example", "--variant", "base", "--motivated-by", motivated_by])

    def test_study_variant_report_validate_clean(self):
        self.new_study()
        scaffold.main(["variant", "--study", "12-gru-example", "--name", "wider"])
        scaffold.main(["report", "--study", "12-gru-example", "--slug", "heldout-curve",
                       "--title", "Held-out vs D"])
        scaffold.main(["report", "--study", "12-gru-example", "--slug", "second", "--title", "Second"])
        self.assertEqual(self.findings(), [])
        reports = sorted(p.name for p in (self.root / "studies/12-gru-example/analysis/reports").glob("r*.md"))
        self.assertEqual(reports, ["r1-heldout-curve.md", "r2-second.md"])
        self.assertIn("r2:", (self.root / "studies/12-gru-example/Makefile").read_text())

    def test_reused_accession_number_is_refused(self):
        with self.assertRaises(SystemExit):
            self.new_study("11-gru-other")

    def test_bad_name_and_existing_files_are_refused(self):
        with self.assertRaises(SystemExit):
            self.new_study("12_gru")
        self.new_study()
        with self.assertRaises(SystemExit):
            scaffold.main(["variant", "--study", "12-gru-example", "--name", "base"])

    def tree(self):
        return sorted(str(p.relative_to(self.root)) for p in (self.root / "studies").rglob("*") if p.is_file())

    def test_refusals_leave_the_tree_unchanged(self):
        before = self.tree()
        for argv in (["study", "--name", "12-gru-example", "--question", "Does the example hold up?",
                      "--project", "p", "--variant", "Bad Name", "--motivated-by", "none"],   # bad first variant
                     ["study", "--name", "12-gru-example", "--question", "Does the example hold up?",
                      "--project", "p", "--variant", "base", "--issue", "#14",
                      "--motivated-by", "none"],                                       # malformed issue
                     ["study", "--name", "12-gru-example", "--question", "Does the example hold up?",
                      "--project", "p", "--variant", "base"],                          # no MOTIVATED_BY
                     ["variant", "--study", "", "--name", "wider"]):                   # missing STUDY
            with self.assertRaises(SystemExit):
                scaffold.main(argv)
            self.assertEqual(self.tree(), before)
        self.new_study()
        scaffold.main(["report", "--study", "12-gru-example", "--slug", "curve", "--title", "Curve"])
        (self.root / "studies/12-gru-example/analysis/r2_again.py").write_text("")   # producer collision
        before = self.tree()
        with self.assertRaises(SystemExit):
            scaffold.main(["report", "--study", "12-gru-example", "--slug", "again", "--title", "Again"])
        self.assertEqual(self.tree(), before)

    def test_cache_slug_is_refused(self):
        self.new_study()
        before = self.tree()
        with self.assertRaises(SystemExit):
            scaffold.main(["report", "--study", "12-gru-example", "--slug", "cache-sensitivity", "--title", "C"])
        self.assertEqual(self.tree(), before)

    def test_local_issue_number_becomes_an_int(self):
        scaffold.main(["study", "--name", "12-gru-example", "--question", "Does the example hold up?",
                       "--project", "p", "--variant", "base", "--issue", "14", "--motivated-by", "none"])
        self.assertIn("- 14\n", (self.root / "studies/12-gru-example/README.md").read_text())
        self.assertEqual(self.findings(), [])

    def test_motivated_by_resolves_earlier_results_and_issues(self):
        reports = self.root / "studies/11-gru-existing/analysis/reports"
        reports.mkdir(parents=True)
        (self.root / "studies/13-gru-later").mkdir()
        for bad in ("11/r2", "13", "14/r1", "05-x/r1/extra"):    # no such report, later, missing, malformed
            with self.assertRaises(SystemExit):
                self.new_study(motivated_by=bad)
        (self.root / "studies/13-gru-later").rmdir()             # only there to test the order refusal
        (reports / "r1-curve.md").write_text("---\nid: r1\n---\n")
        with self.assertRaises(SystemExit):                      # r1 exists, but in 11-gru-existing:
            self.new_study(motivated_by="11-gru-other/r1")       # a full name must match exactly
        self.new_study(motivated_by="11-gru-existing/r1, #23")
        readme = (self.root / "studies/12-gru-example/README.md").read_text()
        self.assertIn("study: 11-gru-existing\n  report: r1\n", readme)
        self.assertIn("- issue: 23\n", readme)
        self.assertIn(scaffold.MOTIVATION_TODO, readme)
        self.assertEqual(self.findings(), [])

    def test_root_study_has_an_empty_list(self):
        self.new_study()
        self.assertIn("motivated_by: []", (self.root / "studies/12-gru-example/README.md").read_text())

    def test_report_numbers_skip_leftover_targets_and_outputs_are_unique(self):
        self.new_study()
        mk = self.root / "studies/12-gru-example/Makefile"
        mk.write_text(mk.read_text() + "\nr0 r1:\n\t$(PY) analysis/old.py\n")   # multi-target rule left behind
        scaffold.main(["report", "--study", "12-gru-example", "--slug", "curve", "--title", "Curve"])
        self.assertTrue((self.root / "studies/12-gru-example/analysis/reports/r2-curve.md").exists())
        self.assertIn("analysis/r2_curve.py", mk.read_text())
        (self.root / "studies/12-gru-example/analysis/r3_other.json").write_text("{}")   # frozen input
        before = self.tree()
        with self.assertRaises(SystemExit):
            scaffold.main(["report", "--study", "12-gru-example", "--slug", "other", "--title", "Other"])
        self.assertEqual(self.tree(), before)

    def test_report_target_uses_the_studys_interpreter_variable(self):
        self.new_study()
        mk = self.root / "studies/12-gru-example/Makefile"
        mk.write_text(mk.read_text().replace("PY ?= python", "PYTHON ?= python"))
        scaffold.main(["report", "--study", "12-gru-example", "--slug", "curve", "--title", "Curve"])
        self.assertIn("\t$(PYTHON) analysis/r1_curve.py", mk.read_text())


class MakeTargetTest(unittest.TestCase):
    """Through `make`, in a temp copy of the repo skeleton: shell characters arrive intact."""

    def setUp(self):
        import shutil
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for rel in ("Makefile", "scripts/scaffold.py", "schemas/layout.yaml"):
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO_ROOT / rel, self.root / rel)
        (self.root / "studies").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_quotes_in_question_and_title(self):
        q = 'Does "high D" help, and isn\'t $HOME expanded?'
        run = lambda *a: subprocess.run(["make", "-s", "-C", str(self.root), *a, f"PY={sys.executable}"],
                                        check=True, capture_output=True, text=True)
        run("new-study", "NAME=12-gru-quoting", f"QUESTION={q}", "PROJECT=p", "VARIANT=base", "MOTIVATED_BY=none")
        run("new-report", "STUDY=12-gru-quoting", "SLUG=curve", 'TITLE=The "curve"')
        study = self.root / "studies" / "12-gru-quoting"
        self.assertIn('Does "high D" help', (study / "README.md").read_text())
        self.assertIn("$HOME", (study / "README.md").read_text())
        self.assertIn('# r1 — The "curve"', (study / "analysis/reports/r1-curve.md").read_text())

if __name__ == "__main__":
    unittest.main()
