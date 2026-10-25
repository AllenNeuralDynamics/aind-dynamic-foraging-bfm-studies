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


if __name__ == "__main__":
    unittest.main()
