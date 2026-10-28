"""scripts/check_provenance.py against small studies-shaped git repos.

Each test builds a compliant study in a temp repo, breaks one thing, and asserts on the
finding keys. Schemas are read from this checkout, so these tests also pin the schemas.
"""

import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "studies" / "util"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _meta  # noqa: E402
import check_provenance as cp  # noqa: E402
import report_verdicts  # noqa: E402

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
    motivated_by: []
    tags: {model: [gru], question: [scaling]}
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
    "data": {"source": "snapshot", "config": "mice_snapshot", "type": "mice_snapshot", "snapshot": "20260603",
             "uri": "s3://aind-scratch-data/aind-dynamic-foraging-cache/snapshots/20260603/", "manifest_sha256": "f" * 64},
    "image": {"platform": "beaker", "name": "han-hou/disrnn-wrapper-main-20260712", "id": "01KIMAGE"},
}


def git(cwd: Path, *args: str) -> None:
    env = {k: v for k, v in os.environ.items() if k not in _meta._GIT_DISCOVERY_VARS}
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, env=env)


class StudyRepo(unittest.TestCase):
    """A compliant study in a temp git repo; subclasses break one thing per test."""

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


class CheckProvenanceTest(StudyRepo):
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

    def test_untracked_new_files_are_seen_before_git_add(self):
        """`make validate` runs before `git add`; it must see what CI will see."""
        (self.study / "analysis/out.json").write_text('{"x": 1}')
        (self.study / "analyze.py").write_text("")
        keys = self.keys()
        self.assertIn(f"artifact-no-meta|{STUDY}/analysis/out.json|", keys)
        self.assertIn(f"study-root-extra|{STUDY}/analyze.py|analyze.py", keys)

    def test_int_keyed_artifact_is_not_stale_after_reading_back(self):
        a = self.study / "analysis"
        with mock.patch.object(_meta, "REPO_ROOT", self.root.resolve()), \
                mock.patch.object(_meta, "_studies_state", return_value=(SHA, False)), \
                mock.patch.object(_meta, "_with_util_modules", side_effect=lambda code: code):
            meta = _meta.build_meta("analysis/produce.py", [], code=[a / "produce.py"],
                                    inputs=[a / "grid.csv"])
            _meta.write_artifact(a / "by_d.json", {"_meta": meta, "by_d": {10: 0.71, 614: 0.72}})
        self.commit()
        self.assertEqual(self.keys(), set())

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

    def test_code_digests_skip_meta_module(self):
        paths = _meta._with_util_modules([__file__])
        self.assertEqual(paths[0], Path(__file__).resolve())
        self.assertNotIn(Path(_meta.__file__).resolve(), paths)

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

    def test_study_context_and_adrs_are_allowed(self):
        (self.study / "CONTEXT.md").write_text("")
        (self.study / "adr").mkdir()
        (self.study / "adr/0001-example.md").write_text("")
        self.commit()
        self.assertEqual(self.keys(), set())

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

    # -- cross-study upstream ---------------------------------------------------------------------
    OTHER = "studies/05-disrnn-other"

    def other_report(self, n: int, extra: str = "", status: str = "live") -> None:
        """A report in a second study, enough for the report index (its own findings are ignored)."""
        p = self.root / self.OTHER / f"analysis/reports/r{n}-other.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"---\nschema_version: 1\nid: r{n}\nslug: other\nstatus: {status}\n{extra}---\n# r{n}\n")

    def cite(self, *refs: str) -> None:
        report = self.study / "analysis/reports/r1-example.md"
        lines = "".join(f"  - {{study: {r.split('/')[0]}, report: {r.split('/')[1]}}}\n" for r in refs)
        report.write_text(report.read_text().replace("reproduce:", f"upstream:\n{lines}reproduce:"))

    def upstream_keys(self) -> set[str]:
        return {k for k in self.keys() if k.startswith("report-upstream-")}

    def test_corrected_upstream_is_stale_until_the_correction_is_cited(self):
        self.other_report(1)
        self.other_report(2, f"corrects: [{{study: {self.OTHER.split('/')[1]}, report: r1, note: wrong}}]\n")
        other = self.OTHER.split("/")[1]
        self.cite(f"{other}/r1")
        self.commit()
        self.assertEqual(self.upstream_keys(),
                         {f"report-upstream-stale|{STUDY}/analysis/reports/r1-example.md|{other}/r1"})
        report = self.study / "analysis/reports/r1-example.md"
        report.write_text(report.read_text().replace(
            f"  - {{study: {other}, report: r1}}\n",
            f"  - {{study: {other}, report: r1}}\n  - {{study: {other}, report: r2}}\n"))
        self.commit()
        self.assertEqual(self.upstream_keys(), set())

    def test_withdrawn_or_replaced_correction_resolves_to_what_stands(self):
        other = self.OTHER.split("/")[1]
        stale = {f"report-upstream-stale|{STUDY}/analysis/reports/r1-example.md|{other}/r1"}
        corrects = f"corrects: [{{study: {other}, report: r1, note: wrong}}]\n"
        self.other_report(1)
        # r2 corrects r1 but is superseded by r3: r3 carries the correction
        self.other_report(2, corrects + f"superseded_by: {{study: {other}, report: r3}}\n", status="superseded")
        self.other_report(3)
        self.cite(f"{other}/r1", f"{other}/r3")
        self.commit()
        self.assertEqual(self.upstream_keys(), set())
        report = self.study / "analysis/reports/r1-example.md"
        report.write_text(report.read_text().replace(f"  - {{study: {other}, report: r3}}\n", ""))
        self.commit()
        self.assertEqual(self.upstream_keys(), stale)
        # a retracted correction binds no one
        self.other_report(2, corrects, status="retracted")
        self.commit()
        self.assertEqual(self.upstream_keys(), set())

    def test_superseded_upstream_is_stale_and_the_corrector_itself_is_not(self):
        other = self.OTHER.split("/")[1]
        self.other_report(1, status="superseded")
        self.cite(f"{other}/r1")
        self.commit()
        self.assertEqual(self.upstream_keys(),
                         {f"report-upstream-stale|{STUDY}/analysis/reports/r1-example.md|{other}/r1"})
        # a corrected (still live) report cited by the very report that corrects it is fine
        self.other_report(1)
        report = self.study / "analysis/reports/r1-example.md"
        report.write_text(report.read_text().replace(
            "reproduce:", f"corrects: [{{study: {other}, report: r1, note: wrong}}]\nreproduce:"))
        self.commit()
        self.assertEqual(self.upstream_keys(), set())

    def test_incomplete_upstream_ref_is_a_schema_error_not_a_crash(self):
        report = self.study / "analysis/reports/r1-example.md"
        report.write_text(report.read_text().replace("reproduce:", "upstream: [{report: r1}]\nreproduce:"))
        self.commit()
        self.assertTrue(any(k.startswith(f"report-frontmatter|{STUDY}/analysis/reports/r1-example.md|upstream/0")
                            for k in self.keys()))

    def test_reading_another_study_needs_upstream(self):
        other = self.OTHER.split("/")[1]
        src = self.root / self.OTHER / "analysis/curve.json"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text('{"614": 0.7268}')
        self.commit()
        a = self.study / "analysis"
        with mock.patch.object(_meta, "REPO_ROOT", self.root.resolve()), \
                mock.patch.object(_meta, "_studies_state", return_value=(SHA, False)), \
                mock.patch.object(_meta, "_with_util_modules", side_effect=lambda code: code):
            meta = _meta.build_meta("analysis/produce.py", [GROUP], study_root=self.study,
                                    code=[a / "produce.py"], inputs=[a / "grid.csv", src])
            _meta.write_artifact(a / "summary.json", {"_meta": meta, "x": 0.7},
                                 outputs=[f"{a}/reports/r1-example.md#result-1"])
        self.commit()
        self.assertEqual(self.upstream_keys(),
                         {f"report-upstream-undeclared|{STUDY}/analysis/reports/r1-example.md|{other}"})
        # declaring it (here as raw data) clears it; the result block is untouched, so not stale
        report = self.study / "analysis/reports/r1-example.md"
        report.write_text(report.read_text().replace(
            "reproduce:", f"upstream: [{{study: {other}, wandb_group: base@20261001-000000}}]\nreproduce:"))
        self.commit()
        self.assertEqual(self.upstream_keys(), set())
        self.assertFalse(any(k.startswith("artifact-stale-") for k in self.keys()))

    def test_report_citing_the_artifact_in_inputs_needs_upstream_too(self):
        """No result-block output: the report is linked through its `inputs`, however spelled."""
        other = self.OTHER.split("/")[1]
        src = self.root / self.OTHER / "analysis/curve.json"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text('{"614": 0.7268}')
        report = self.study / "analysis/reports/r1-example.md"
        report.write_text(report.read_text().replace("  data: analysis/grid.csv\n",
                                                     "  data: analysis/grid.csv\n  summary: ./analysis/summary.json\n"))
        a = self.study / "analysis"
        with mock.patch.object(_meta, "REPO_ROOT", self.root.resolve()), \
                mock.patch.object(_meta, "_studies_state", return_value=(SHA, False)), \
                mock.patch.object(_meta, "_with_util_modules", side_effect=lambda code: code):
            meta = _meta.build_meta("analysis/produce.py", [GROUP], study_root=self.study,
                                    code=[a / "produce.py"], inputs=[a / "grid.csv", src])
            _meta.write_artifact(a / "summary.json", {"_meta": meta, "x": 0.7})
        self.commit()
        self.assertEqual(self.upstream_keys(),
                         {f"report-upstream-undeclared|{STUDY}/analysis/reports/r1-example.md|{other}"})

    # -- motivation (ADR 0002) --------------------------------------------------------------------
    def motivate(self, entries: str) -> None:
        readme = self.study / "README.md"
        readme.write_text(readme.read_text().replace("motivated_by: []", f"motivated_by: {entries}"))
        self.commit()

    def test_motivation_must_resolve_and_point_back_in_time(self):
        self.other_report(1)                                 # studies/05-disrnn-other/.../r1
        later = self.root / "studies/13-gru-later/analysis/reports/r1-x.md"
        later.parent.mkdir(parents=True)
        later.write_text("---\nid: r1\n---\n")
        readme = f"{STUDY}/README.md"
        self.motivate("[{study: 05-disrnn-other, report: r1, note: the earlier curve peaked},"
                      " {study: 05-disrnn-other, report: r9, note: no such report here},"
                      " {study: 13-gru-later, report: r1, note: a later study cannot motivate},"
                      " {issue: 23, note: a roadmap item asked for it}]")
        mine = {k for k in self.keys() if k.startswith("study-motivation")}
        self.assertEqual(mine, {f"study-motivation-unresolved|{readme}|05-disrnn-other/r9",
                                f"study-motivation-order|{readme}|13-gru-later/r1"})
        info = {f.key for f in cp.Checker(self.root).run() if f.level == "info"}
        self.assertIn(f"study-motivation-many|{readme}|", info)

    def test_missing_or_malformed_motivated_by_is_a_schema_error(self):
        readme = self.study / "README.md"
        self.motivate("23")                                  # a scalar must not crash the run
        self.assertIn(f"study-frontmatter|{STUDY}/README.md|motivated_by:type", self.keys())
        readme.write_text(readme.read_text().replace("motivated_by: 23\n", ""))
        self.commit()
        self.assertIn(f"study-frontmatter|{STUDY}/README.md|(top):motivated_by", self.keys())

    # -- tags (CONTEXT.md, Tag) ------------------------------------------------------------------
    def set_tags(self, path: Path, old: str, new: str) -> None:
        path.write_text(path.read_text().replace(old, new))
        self.commit()

    def test_tag_values_come_from_the_vocabulary(self):
        report = self.study / "analysis/reports/r1-example.md"
        self.set_tags(self.study / "README.md", "tags: {model: [gru], question: [scaling]}",
                      "tags: {model: [gru, lstm], question: [scaling]}")
        self.set_tags(report, "reproduce:", "tags: {question: [vibes]}\nreproduce:")
        mine = {k for k in self.keys() if k.startswith("tag-")}
        self.assertEqual(mine, {f"tag-unknown|{STUDY}/README.md|model:lstm",
                                f"tag-unknown|{STUDY}/analysis/reports/r1-example.md|question:vibes"})

    def test_a_complete_study_shows_each_tag_on_a_live_report(self):
        readme, report = self.study / "README.md", self.study / "analysis/reports/r1-example.md"
        self.set_tags(readme, "tags: {model: [gru], question: [scaling]}",
                      "tags: {model: [gru, cognitive], question: [scaling, comparison]}")
        unreported = lambda: {k for k in self.keys() if k.startswith("study-tag-unreported")}
        self.assertEqual(unreported(), set())                 # active: nothing to show yet
        self.set_tags(readme, "status: active", "status: complete")
        self.assertEqual(unreported(), {f"study-tag-unreported|{STUDY}/README.md|{t}" for t in
                                        ("model:gru", "model:cognitive", "question:scaling", "question:comparison")})
        # a nested report tag covers its parent; a parent never covers a child
        self.set_tags(report, "reproduce:", "tags: {model: [gru, cognitive/bh], question: [scaling]}\nreproduce:")
        self.assertEqual(unreported(), {f"study-tag-unreported|{STUDY}/README.md|question:comparison"})
        # a retracted report is no evidence; malformed entries are the schema's, never a crash
        self.set_tags(report, "status: live", "status: retracted")
        self.assertEqual(len(unreported()), 4)
        self.set_tags(readme, "model: [gru, cognitive]", "model: [{name: gru}]")
        self.assertIn(f"study-frontmatter|{STUDY}/README.md|tags/model/0:type", self.keys())

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

    def test_launch_without_data_or_image_is_a_warning(self):
        rec = {k: v for k, v in RECORD.items() if k not in ("data", "image")}
        (self.study / "variants/base/launch_record/beaker.json").write_text(json.dumps(rec))
        self.commit()
        path = f"{STUDY}/variants/base/launch_record/beaker.json"
        self.assertEqual({k for k in self.keys() if k.startswith("record-")},
                         {f"record-data-unrecorded|{path}|", f"record-image-unrecorded|{path}|"})

    def test_unpinned_data_or_image_is_a_warning_even_when_present(self):
        path = f"{STUDY}/variants/base/launch_record/beaker.json"
        for data, image in (({**RECORD["data"], "manifest_sha256": None}, RECORD["image"]),
                            ({"source": "live", "config": "mice_snapshot"}, {**RECORD["image"], "id": None}),
                            (RECORD["data"], {"platform": "hpc", "conda_env": "/envs/x", "lock_sha256": "e" * 64})):
            (self.study / "variants/base/launch_record/beaker.json").write_text(json.dumps({**RECORD, "data": data, "image": image}))
            self.commit()
            got = {k for k in self.keys() if k.startswith("record-")}
            if data is not RECORD["data"]:
                self.assertIn(f"record-data-unrecorded|{path}|", got)
            if image is not RECORD["image"]:     # a null id, or an HPC lock on a Beaker launch: no pin
                self.assertIn(f"record-image-unrecorded|{path}|", got)

    def test_v1_record_is_never_rejudged(self):
        (self.study / "variants/base/launch_record/old.json").write_text(json.dumps({"mode": "resumable-grid"}))
        self.commit()
        self.assertEqual(self.keys(), set())


FIT = {"delta": {"mean": 0.0123, "ci": [0.0101, 0.0146]}, "verdict": "non-inferior",
       "frac_fail": 0.12, "a/b": {"c~d": [0.5, 0.6]}}

V2_BODY = textwrap.dedent("""\
    # r2 — The fit improves

    ## Question

    Does the fit improve on the baseline?

    ## Answer

    - **F1.** The fit improves by 0.0123, and 12% of sessions fail.

    ## Methods

    <!-- BEGIN methods -->
    149 mice, seeds 0-2.
    <!-- END methods -->

    As in the study methods.

    ## Results

    ### F1 — The fit improves

    <!-- BEGIN result-1 -->
    ## not a section: a producer may write headings
    | delta | 0.0123 [0.0101, 0.0146] | 0.333 |
    <!-- END result-1 -->

    ## What it means

    F1 holds within its 95% CI (0.010 to 0.015); `v0.4.1` and [a link](fig_0.25.svg) are not numbers.

    ## Caveats

    Single seed, no error bar; v2.1 is a version.

    ## Next

    none

    ## Provenance

    `make -C studies/12-gru-example r2`
    """)


class ReportV2Test(StudyRepo):
    """Version-2 report rules (schemas/README.md), on r2 beside the version-1 r1."""

    REL = f"{STUDY}/analysis/reports/r2-fit.md"

    def setUp(self):
        super().setUp()
        a = self.study / "analysis"
        with mock.patch.object(_meta, "REPO_ROOT", self.root.resolve()), \
                mock.patch.object(_meta, "_studies_state", return_value=(SHA, False)), \
                mock.patch.object(_meta, "_with_util_modules", side_effect=lambda code: code):
            meta = _meta.build_meta("analysis/produce.py", [GROUP], study_root=self.study,
                                    code=[a / "produce.py"], inputs=[a / "grid.csv"])
            _meta.write_artifact(a / "fit.json", {"_meta": meta, **FIT})
        (self.study / "Makefile").write_text("all: r1 r2\nr1:\n\tpython analysis/produce.py\nr2:\n\tpython analysis/produce.py\n")
        index = a / "reports/INDEX.md"
        index.write_text(index.read_text() + "| [r2](r2-fit.md) | a hand row, kept |\n")
        self.write_report()
        report_verdicts.write(self.study)
        self.commit()

    def write_report(self, body: str = V2_BODY, status: str = "live", extra: str = "",
                     claims: str = "  - {id: F1, text: \"The fit improves.\", pointer: \"analysis/fit.json#/delta/mean\"}\n",
                     reviewed: dict | None = None) -> None:
        blocks = reviewed or {k: _meta.block_digest(v) for k, v in _meta.RESULT_BLOCK.findall(body)}
        front = (f"schema_version: 2\nid: r2\nslug: fit\nstatus: {status}\n{extra}authors: [han]\n"
                 f"wandb_groups: [{GROUP}]\ninputs:\n  script: analysis/produce.py\n  data: analysis/fit.json\n"
                 f"claims:\n{claims}prose_reviewed:\n  date: 2026-10-28\n  blocks:\n"
                 + "".join(f"    {k}: {v}\n" for k, v in blocks.items())
                 + f"reproduce: make -C {STUDY} r2\n")
        (self.root / self.REL).write_text(f"---\n{front}---\n{body}")

    def findings(self, prefix: str = "", level: str | None = None) -> set[str]:
        return {f.key for f in cp.Checker(self.root).run()
                if f.check.startswith(prefix) and (f.level == level if level else f.level != "info")}

    def test_v2_report_round_trip_is_clean(self):
        """Unquoted date, sourced prose numbers (to printed precision, 12% as 0.12), 95%, code,
        links, versions and a `##` inside a block: no finding, not even a note."""
        self.assertEqual(self.findings(), set())
        self.assertEqual(self.findings("report-", "info"), set())

    def test_v1_report_is_unaffected(self):
        """r1 stays version 1: no skeleton, no claims, an unsourced decimal; nothing new, and the
        verdict generator leaves its hand-written INDEX row alone."""
        r1 = self.study / "analysis/reports/r1-example.md"
        r1.write_text(r1.read_text() + "\n## Discussion\n\nThe gain is 0.5555.\n")
        self.commit()
        self.assertEqual(self.findings(), set())
        self.assertFalse(any("r1-example" in k for k in self.findings("report-", "info")))
        self.assertIn("[r1](r1-example.md)\n", (self.study / "analysis/reports/INDEX.md").read_text())

    def test_skeleton_headings_in_order(self):
        """A title without `rN — `, a missing section and two swapped ones (the `##` the producer
        wrote inside result-1 does not count)."""
        body = (V2_BODY.replace("# r2 — The fit improves", "# The fit improves")
                .replace("## Caveats\n\nSingle seed, no error bar; v2.1 is a version.\n\n", "")
                .replace("## Answer", "## TMP").replace("## Methods", "## Answer").replace("## TMP", "## Methods"))
        self.write_report(body)
        self.commit()
        self.assertEqual(self.findings("report-skeleton"),
                         {f"report-skeleton|{self.REL}|title", f"report-skeleton|{self.REL}|## Caveats",
                          f"report-skeleton|{self.REL}|## Methods"})

    def test_claim_pointer_must_resolve_when_live(self):
        bad = ("  - {id: F1, text: \"Escapes resolve.\", pointer: \"analysis/fit.json#/a~1b/c~0d/1\"}\n"
               "  - {id: F2, text: \"No such key.\", pointer: \"analysis/fit.json#/delta/median\"}\n"
               "  - {id: F3, text: \"No such file.\", pointer: \"analysis/nope.json#/x\"}\n"
               "  - {id: F4, text: \"Index out of range.\", pointer: \"analysis/fit.json#/delta/ci/2\"}\n")
        self.write_report(claims=bad)
        self.commit()
        want = {f"report-claim-unresolved|{self.REL}|F{k}" for k in (2, 3, 4)}
        self.assertEqual(self.findings("report-claim"), want)
        # a draft may point ahead of its producer: the same findings, advisory
        self.write_report(claims=bad, status="draft", extra="draft_reason: producer pending\n")
        self.commit()
        self.assertEqual(self.findings("report-claim"), set())
        self.assertEqual(self.findings("report-claim", "info"), want)

    def test_draft_needs_a_reason_and_claims_are_required(self):
        self.write_report(status="draft", claims="")
        report = self.root / self.REL
        report.write_text(report.read_text().replace("claims:\n", "claims: []\n"))
        self.commit()
        keys = self.findings("report-frontmatter")
        self.assertIn(f"report-frontmatter|{self.REL}|(top):draft_reason", keys)
        self.assertIn(f"report-frontmatter|{self.REL}|claims:minItems", keys)

    def test_changed_block_needs_a_prose_review(self):
        body = V2_BODY.replace("| 0.333 |", "| 0.334 |")
        self.write_report(body, reviewed={k: _meta.block_digest(v) for k, v in _meta.RESULT_BLOCK.findall(V2_BODY)})
        self.commit()
        found = [f for f in cp.Checker(self.root).run() if f.check == "report-prose-stale"]
        self.assertEqual([f.key for f in found], [f"report-prose-stale|{self.REL}|result-1"])
        new = _meta.block_digest(dict(_meta.RESULT_BLOCK.findall(body))["result-1"])
        self.assertIn(new, found[0].message)          # the message prints the digest to record
        self.write_report(body)                       # ... and recording it clears the warning
        self.commit()
        self.assertEqual(self.findings("report-prose-stale"), set())
        # a block the review never saw is stale too
        self.write_report(body.replace("## What it means", "<!-- BEGIN result-2 -->\nx\n<!-- END result-2 -->\n\n## What it means"),
                          reviewed={k: _meta.block_digest(v) for k, v in _meta.RESULT_BLOCK.findall(body)})
        self.commit()
        self.assertEqual(self.findings("report-prose-stale"), {f"report-prose-stale|{self.REL}|result-2"})

    def test_unsourced_prose_number_is_an_advisory_note(self):
        body = V2_BODY.replace("none\n", "Seed spread ~0.0004, a 7.5% gain; the block's 0.333 is not cited.\n")
        self.write_report(body)
        self.commit()
        self.assertEqual(self.findings(), set())      # advisory: never fails CI
        self.assertEqual(self.findings("report-prose-number", "info"),
                         {f"report-prose-number|{self.REL}|{t}" for t in ("0.0004", "7.5%", "0.333")})
        out = io.StringIO()
        with mock.patch.object(cp, "REPO", self.root), mock.patch.object(cp, "BASELINE", self.root / "none.json"), \
                redirect_stdout(out):
            self.assertEqual(cp.main([]), 0)
        self.assertIn(f"note [report-prose-number] {self.REL}", out.getvalue())
        self.assertIn("0.0004 (line ", out.getvalue())

    def test_generated_verdicts_follow_the_claims(self):
        index, readme = self.study / "analysis/reports/INDEX.md", self.study / "README.md"
        self.assertIn("**F1.** The fit improves.", index.read_text())
        self.assertIn("- **F1.** The fit improves.", readme.read_text())
        self.assertIn("[r1](r1-example.md)\n", index.read_text())          # hand content kept
        self.write_report(claims="  - {id: F1, text: \"The fit improves a lot.\", pointer: \"analysis/fit.json#/delta/mean\"}\n")
        self.commit()
        self.assertEqual(self.findings("verdict-stale"),
                         {f"verdict-stale|{STUDY}/analysis/reports/INDEX.md|verdicts",
                          f"verdict-stale|{STUDY}/README.md|verdicts"})
        self.assertEqual({p.name for p in report_verdicts.write(self.study)}, {"INDEX.md", "README.md"})
        self.commit()
        self.assertEqual(self.findings(), set())
        self.assertEqual(report_verdicts.write(self.study), [])           # idempotent
        # a draft is listed in INDEX with its reason, and leaves the README verdict
        self.write_report(status="draft", extra="draft_reason: k-hat rescue pending\n")
        report_verdicts.write(self.study)
        self.assertIn("| draft: k-hat rescue pending |", index.read_text())
        self.assertIn(report_verdicts.NO_LIVE, readme.read_text())

    def test_a_study_without_v2_reports_is_left_alone(self):
        (self.root / self.REL).unlink()
        index = self.study / "analysis/reports/INDEX.md"
        index.write_text("[r1](r1-example.md)\n")
        readme = self.study / "README.md"
        readme.write_text(README)
        self.assertEqual(report_verdicts.write(self.study), [])
        self.assertEqual(index.read_text(), "[r1](r1-example.md)\n")
        self.assertEqual(readme.read_text(), README)


class BodyDigestTest(unittest.TestCase):
    def test_key_order_and_meta_do_not_matter(self):
        a = {"_meta": {"x": 1}, "b": [1, 2], "a": {"y": 0.5}}
        b = {"a": {"y": 0.5}, "b": [1, 2], "_meta": {"x": 2}}
        self.assertEqual(_meta.body_digest(a), _meta.body_digest(b))


if __name__ == "__main__":
    unittest.main()
