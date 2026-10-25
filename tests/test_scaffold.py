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

    def new_study(self, name="12-gru-example"):
        scaffold.main(["study", "--name", name, "--question", "Does the example hold up?",
                       "--project", "gru_example", "--variant", "base"])

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
                      "--project", "p", "--variant", "Bad Name"],                     # bad first variant
                     ["study", "--name", "12-gru-example", "--question", "Does the example hold up?",
                      "--project", "p", "--variant", "base", "--issue", "#14"],        # malformed issue
                     ["variant", "--study", "", "--name", "wider"]):                   # missing STUDY
            with self.assertRaises(SystemExit):
                scaffold.main(argv)
            self.assertEqual(self.tree(), before)
        self.new_study()
        scaffold.main(["report", "--study", "12-gru-example", "--slug", "curve", "--title", "Curve"])
        (self.root / "studies/12-gru-example/analysis/reports/r1-curve.md").unlink()
        before = self.tree()                                   # producer analysis/curve.py remains
        with self.assertRaises(SystemExit):
            scaffold.main(["report", "--study", "12-gru-example", "--slug", "curve", "--title", "Again"])
        self.assertEqual(self.tree(), before)

    def test_local_issue_number_becomes_an_int(self):
        scaffold.main(["study", "--name", "12-gru-example", "--question", "Does the example hold up?",
                       "--project", "p", "--variant", "base", "--issue", "14"])
        self.assertIn("- 14\n", (self.root / "studies/12-gru-example/README.md").read_text())
        self.assertEqual(self.findings(), [])

    def test_report_target_uses_the_studys_interpreter_variable(self):
        self.new_study()
        mk = self.root / "studies/12-gru-example/Makefile"
        mk.write_text(mk.read_text().replace("PY ?= python", "PYTHON ?= python"))
        scaffold.main(["report", "--study", "12-gru-example", "--slug", "curve", "--title", "Curve"])
        self.assertIn("\t$(PYTHON) analysis/curve.py", mk.read_text())


class MakeTargetTest(unittest.TestCase):
    """Through `make`, in this checkout: shell characters in values must arrive intact."""

    NAME = "98-gru-scaffold-quoting"

    def tearDown(self):
        import shutil
        shutil.rmtree(REPO_ROOT / "studies" / self.NAME, ignore_errors=True)

    def test_quotes_in_question_and_title(self):
        q = 'Does "high D" help, and isn\'t $HOME expanded?'
        run = lambda *a: subprocess.run(["make", "-s", "-C", str(REPO_ROOT), *a, f"PY={sys.executable}"],
                                        check=True, capture_output=True, text=True)
        run("new-study", f"NAME={self.NAME}", f"QUESTION={q}", "PROJECT=p", "VARIANT=base")
        run("new-report", f"STUDY={self.NAME}", "SLUG=curve", 'TITLE=The "curve"')
        study = REPO_ROOT / "studies" / self.NAME
        self.assertIn('Does "high D" help', (study / "README.md").read_text())
        self.assertIn("$HOME", (study / "README.md").read_text())
        self.assertIn('# r1 — The "curve"', (study / "analysis/reports/r1-curve.md").read_text())


if __name__ == "__main__":
    unittest.main()
