"""scripts/check_provenance.py against small studies-shaped git repos.

Each test builds a compliant study in a temp repo, breaks one thing, and asserts on the
finding keys. Schemas are read from this checkout, so these tests also pin the schemas.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "studies" / "util"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _meta  # noqa: E402
import check_provenance as cp  # noqa: E402

STUDY = "studies/12-gru-example"
GROUP = "base@20261024-101500"
SHA = "a" * 40

README = textwrap.dedent("""\
    ---
    schema_version: 1
    id: 12-gru-example
    question: Does the example question have an answer?
    status: active
    wandb: {entity: AIND-disRNN, project: example}
    ---
    # Study: example
    """)

REPORT = textwrap.dedent(f"""\
    ---
    schema_version: 1
    id: r1
    slug: example
    status: live
    authors: [han]
    wandb_groups: [{GROUP}]
    inputs:
      script: analysis/produce.py
      data: analysis/grid.csv
    reproduce: make -C {STUDY} r1
    ---
    # r1 — example

    <!-- BEGIN result-1 -->
    | D | 10 |
    |---|---|
    | x | 0.7 |
    <!-- END result-1 -->
    """)

RECORD = {
    "_meta": {"schema_version": 2, "produced_by": "launch_beaker_resumable.py",
              "produced_at_pt": "2026-10-24T10:15:00-07:00"},
    "kind": "initial", "platform": "beaker", "wandb_group": GROUP, "label": "base",
    "note": "baseline launch for the example study", "n_tasks": 1,
    "job_refs": [{"type": "beaker_experiment", "id": "01KEXAMPLE"}],
    "revisions": {"studies": SHA, "dispatcher": SHA, "wrapper": SHA},
    "manifests": ["sweep.yaml"],
}


def git(cwd: Path, *args: str) -> None:
    env = {k: v for k, v in os.environ.items() if k not in _meta._GIT_DISCOVERY_VARS}
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, env=env)


class CheckProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.study = self.root / STUDY
        files = {
            "README.md": README,
            "CHANGELOG.md": "# Changelog\n",
            "Makefile": "all: r1\nr1:\n\tpython analysis/produce.py\n",
            "variants/base/notes.md": "# base\n",
            "variants/base/sweep.yaml": "name: base\n",
            "variants/base/launch_record/beaker.json": json.dumps(RECORD),
            "analysis/produce.py": "print('produce')\n",
            "analysis/grid.csv": "D,x\n10,0.7\n",
            "analysis/reports/INDEX.md": "[r1](r1-example.md)\n",
            "analysis/reports/r1-example.md": REPORT,
        }
        for rel, text in files.items():
            (self.study / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.study / rel).write_text(text)
        git(self.root, "init", "-q")
        self.commit()

    def tearDown(self):
        self.tmp.cleanup()

    def commit(self):
        git(self.root, "add", "-A")
        git(self.root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x", "--allow-empty")

    def keys(self) -> set[str]:
        return {f.key for f in cp.Checker(self.root).run() if f.level != "info"}

    def write_v2_artifact(self) -> Path:
        """Produce summary.json + fig.png the way a migrated producer would."""
        a = self.study / "analysis"
        (a / "fig.png").write_bytes(b"\x89PNG fake")
        with mock.patch.object(_meta, "REPO_ROOT", self.root.resolve()), \
                mock.patch.object(_meta, "_studies_state", return_value=(SHA, False)):
            meta = _meta.build_meta("analysis/produce.py", [GROUP], study_root=self.study,
                                    code=[a / "produce.py"], inputs=[a / "grid.csv"])
            # _meta.py itself is recorded as code; it lives outside the temp repo
            meta["code"] = [c for c in meta["code"] if not c["path"].startswith("..")]
            out = a / "summary.json"
            _meta.write_artifact(out, {"_meta": meta, "x": 0.7},
                                 outputs=[a / "fig.png", f"{a}/reports/r1-example.md#result-1"])
        return out

    # -- compliant baseline ---------------------------------------------------------------------
    def test_compliant_study_has_no_findings(self):
        self.assertEqual(self.keys(), set())

    def test_v2_artifact_round_trip_is_clean(self):
        with mock.patch.object(_meta, "_with_util_modules", side_effect=lambda code: code):
            self.write_v2_artifact()
        self.commit()
        self.assertEqual(self.keys(), set())

    # -- artifacts --------------------------------------------------------------------------------
    def test_new_artifact_without_meta_is_a_finding(self):
        (self.study / "analysis/out.json").write_text('{"x": 1}')
        self.commit()
        self.assertIn(f"artifact-no-meta|{STUDY}/analysis/out.json|", self.keys())

    def test_edited_figure_and_input_are_stale(self):
        with mock.patch.object(_meta, "_with_util_modules", side_effect=lambda code: code):
            out = self.write_v2_artifact()
        (self.study / "analysis/fig.png").write_bytes(b"\x89PNG edited by hand")
        (self.study / "analysis/grid.csv").write_text("D,x\n10,0.8\n")
        self.commit()
        rel = f"{STUDY}/analysis"
        keys = self.keys()
        self.assertIn(f"artifact-stale-output|{rel}/summary.json|{rel}/fig.png", keys)
        self.assertIn(f"artifact-stale-input|{rel}/summary.json|{rel}/grid.csv", keys)
        self.assertTrue(out.exists())

    def test_edited_result_block_is_stale_but_whitespace_is_not(self):
        with mock.patch.object(_meta, "_with_util_modules", side_effect=lambda code: code):
            self.write_v2_artifact()
        report = self.study / "analysis/reports/r1-example.md"
        report.write_text(report.read_text().replace("| x | 0.7 |", "| x | 0.7 |   "))
        self.commit()
        self.assertEqual(self.keys(), set())
        report.write_text(report.read_text().replace("0.7", "0.9"))
        self.commit()
        self.assertTrue(any(k.startswith("artifact-stale-output|") for k in self.keys()))

    def test_uncommitted_input_is_refused(self):
        (self.study / ".gitignore").write_text("analysis/cache.csv\n")
        (self.study / "analysis/cache.csv").write_text("D\n1\n")
        a = self.study / "analysis"
        with mock.patch.object(_meta, "REPO_ROOT", self.root.resolve()), \
                mock.patch.object(_meta, "_studies_state", return_value=(SHA, False)), \
                mock.patch.object(_meta, "_with_util_modules", side_effect=lambda code: code):
            meta = _meta.build_meta("analysis/produce.py", [], code=[a / "produce.py"],
                                    inputs=[a / "cache.csv"])
            _meta.write_artifact(a / "s.json", {"_meta": meta, "x": 1})
        self.commit()
        self.assertIn(f"artifact-input-uncommitted|{STUDY}/analysis/s.json|{STUDY}/analysis/cache.csv",
                      self.keys())

    def test_freeze_requires_sources(self):
        with self.assertRaises(ValueError):
            _meta.build_meta("analysis/freeze.py", [GROUP], kind="freeze", code=[__file__])

    def test_v1_build_meta_is_unchanged(self):
        meta = _meta.build_meta("analysis/produce.py", [GROUP])
        self.assertNotIn("schema_version", meta)
        self.assertNotIn("code", meta)

    # -- layout and reports -----------------------------------------------------------------------
    def test_code_at_study_root_is_a_finding(self):
        (self.study / "analyze.py").write_text("")
        (self.study / "PLAN.md").write_text("")          # docs are allowed
        self.commit()
        self.assertEqual(self.keys(), {f"study-root-extra|{STUDY}/analyze.py|analyze.py"})

    def test_glob_group_and_command_input(self):
        report = self.study / "analysis/reports/r1-example.md"
        report.write_text(report.read_text()
                          .replace(GROUP, "base@*")
                          .replace("data: analysis/grid.csv", "data: python run.py --all"))
        self.commit()
        keys = self.keys()
        path = f"{STUDY}/analysis/reports/r1-example.md"
        self.assertIn(f"report-frontmatter|{path}|wandb_groups/0:pattern", keys)
        self.assertIn(f"report-input-missing|{path}|data:python run.py --all", keys)

    def test_superseded_report_needs_successor(self):
        report = self.study / "analysis/reports/r1-example.md"
        report.write_text(report.read_text().replace("status: live", "status: superseded"))
        self.commit()
        self.assertIn(f"report-frontmatter|{STUDY}/analysis/reports/r1-example.md|(top):superseded_by",
                      self.keys())

    def test_unresolved_correction(self):
        report = self.study / "analysis/reports/r1-example.md"
        report.write_text(report.read_text().replace(
            "reproduce:", "corrects: [{study: 05-disrnn-scaling-law, report: r9, note: wrong}]\nreproduce:"))
        self.commit()
        self.assertIn(f"report-ref-unresolved|{STUDY}/analysis/reports/r1-example.md|05-disrnn-scaling-law/r9",
                      self.keys())

    # -- launch records (ADR 0001) ----------------------------------------------------------------
    def test_group_prefix_must_be_variant(self):
        rec = dict(RECORD, wandb_group="other@20261024-101500")
        (self.study / "variants/base/launch_record/beaker.json").write_text(json.dumps(rec))
        self.commit()
        self.assertIn(f"record-group-prefix|{STUDY}/variants/base/launch_record/beaker.json|", self.keys())

    def test_intervention_must_join_an_initial_launch(self):
        rescue = dict(RECORD, kind="rescue", wandb_group="base@20261101-000000",
                      trigger={"symptom": "3 tasks wedged"})
        (self.study / "variants/base/launch_record/rescue1.json").write_text(json.dumps(rescue))
        self.commit()
        self.assertIn(f"record-orphan-intervention|{STUDY}/variants/base/launch_record/rescue1.json|",
                      self.keys())

    def test_v1_record_is_never_rejudged(self):
        (self.study / "variants/base/launch_record/old.json").write_text(json.dumps({"mode": "resumable-grid"}))
        self.commit()
        self.assertEqual(self.keys(), set())


class BodyDigestTest(unittest.TestCase):
    def test_key_order_and_meta_do_not_matter(self):
        a = {"_meta": {"x": 1}, "b": [1, 2], "a": {"y": 0.5}}
        b = {"a": {"y": 0.5}, "b": [1, 2], "_meta": {"x": 2}}
        self.assertEqual(_meta.body_digest(a), _meta.body_digest(b))


if __name__ == "__main__":
    unittest.main()
