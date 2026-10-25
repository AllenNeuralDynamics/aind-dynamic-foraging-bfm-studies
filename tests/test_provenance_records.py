"""Native launcher records and labeled launch directories are valid provenance."""
import json
import types
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies" / "util"))
from launch_record import load_records
from validate_provenance import check_beaker, check_records_parse, recorded_for_group
from launch_record import experiment_ids_in


class ProvenanceRecordsTests(unittest.TestCase):
    def test_native_launcher_in_labeled_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "launch_record"
            labeled = Path(temp) / "launch_record_paired-rerun"
            labeled.mkdir()
            record = {
                "mode": "resumable-grid", "n_tasks": 3,
                "experiment_id": "experiment-1", "created_utc": "2026-09-08T11:57:48Z",
                "sweep_file": "studies/10/variants/reservoir/sweep.yaml",
                "dispatcher_commit": "a" * 40,
            }
            (labeled / "beaker_resumable.json").write_text(json.dumps(record))
            records = load_records(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["_schema"], "launcher")
            findings = []
            check_records_parse(records, findings)
            self.assertEqual(findings, [])

    def test_records_in_per_launch_subdirectories_are_loaded(self):
        """`launch_beaker_resumable.py` output lives in `launch_record/<label>/`.

        A non-recursive glob never saw these, so a strict check passed without auditing
        them (study 08 had 13 such records and the validator reported one).
        """
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "launch_record"
            for label in ("beaker_a", "beaker_b"):
                (root / label).mkdir(parents=True)
                record = {
                    "mode": "resumable-grid", "n_tasks": 1,
                    "experiment_id": f"experiment-{label}", "created_utc": "2026-09-08T11:57:48Z",
                    "sweep_file": "sweep.yaml", "dispatcher_commit": "a" * 40,
                }
                (root / label / "beaker_resumable.json").write_text(json.dumps(record))
            records = load_records(root)
            self.assertEqual(sorted(r["experiment_id"] for r in records),
                             ["experiment-beaker_a", "experiment-beaker_b"])
            findings = []
            check_records_parse(records, findings)
            self.assertEqual(findings, [])

    def test_beaker_reconciliation_is_scoped_to_the_group_being_checked(self):
        """Recursive discovery loads every launch of a variant, which can span several W&B groups.

        `check_beaker` flags recorded experiments it did not see in the scan; that comparison must
        only use the records for the group being checked, or every other group's launches are
        reported as missing.
        """
        records = [
            {"experiment_id": "exp-a1", "wandb_group": "grp-a"},
            {"experiment_id": "exp-a2", "_meta": {"wandb_groups": ["grp-a", "grp-b"]}},
            {"experiment_id": "exp-b1", "wandb_group": "grp-b"},
            {"experiment_id": "exp-none"},
        ]
        found_in_scan_of_a = {"exp-a1", "exp-a2"}
        for_a = experiment_ids_in(recorded_for_group(records, "grp-a"))
        self.assertEqual(for_a, {"exp-a1", "exp-a2"})
        self.assertEqual(for_a - found_in_scan_of_a, set())          # nothing falsely missing
        # The unscoped comparison, which is what the old code did, reports other groups' launches.
        self.assertEqual(experiment_ids_in(records) - found_in_scan_of_a, {"exp-b1", "exp-none"})
        self.assertEqual(experiment_ids_in(recorded_for_group(records, "grp-b")), {"exp-a2", "exp-b1"})
        self.assertEqual(recorded_for_group(records, "grp-unknown"), [])

    def test_check_beaker_does_not_report_other_groups_launches_as_missing(self):
        """End to end through `check_beaker`, against a fake `beaker` module."""
        def env(group):
            return [types.SimpleNamespace(name="WANDB_RUN_GROUP", value=group)]
        def group_a_spec(envs):
            return types.SimpleNamespace(tasks=[types.SimpleNamespace(env_vars=envs)])

        specs = {"exp-a": group_a_spec(env("grp-a")), "exp-b": group_a_spec(env("grp-b"))}

        class FakeBeaker:
            def __init__(self, *_a, **_k):
                self.workspace = types.SimpleNamespace(
                    experiments=lambda limit=None: [types.SimpleNamespace(id=i) for i in specs])
                self.experiment = types.SimpleNamespace(spec=lambda exp_id: specs[exp_id])

        fake = types.ModuleType("beaker")
        fake.Beaker, fake.Config = FakeBeaker, lambda **_k: None
        records = [{"experiment_id": "exp-a", "wandb_group": "grp-a", "job_refs": [
                        {"type": "beaker_experiment", "id": "exp-a"}]},
                   {"experiment_id": "exp-b", "wandb_group": "grp-b", "job_refs": [
                        {"type": "beaker_experiment", "id": "exp-b"}]},
                   {"experiment_id": "exp-gone", "wandb_group": "grp-b", "job_refs": [
                        {"type": "beaker_experiment", "id": "exp-gone"}]}]
        findings = []
        with mock.patch.dict(sys.modules, {"beaker": fake}), mock.patch.dict(os.environ, {"BEAKER_TOKEN": "x"}):
            check_beaker("grp-a", records, "ws", 50, findings)
        text = " ".join(str(f) for f in findings)
        self.assertNotIn("exp-b", text)       # another group's launch, seen or not, is not this group's business
        self.assertNotIn("exp-gone", text)    # recorded under grp-b, so not expected in grp-a's scan
        self.assertIn("all 1 Beaker experiment(s) for this group are recorded", text)

    def test_native_launcher_still_requires_a_real_commit(self):
        findings = []
        check_records_parse([{
            "_schema": "launcher", "_path": "beaker_resumable.json",
            "mode": "resumable-grid", "n_tasks": 3,
            "experiment_id": "experiment-1", "created_utc": "2026-09-08T11:57:48Z",
            "sweep_file": "sweep.yaml", "dispatcher_commit": None,
        }], findings)
        self.assertEqual(findings[0]["level"], "ERROR")

    def test_interventions_still_require_a_cause(self):
        findings = []
        check_records_parse([{
            "_schema": "current", "_path": "rescue.json", "kind": "rescue",
            "trigger": {"symptom": "stalled"},
        }], findings)
        self.assertEqual(findings[0]["level"], "ERROR")



class WriteInterventionV2Tests(unittest.TestCase):
    """write_intervention emits v2, joins an initial launch, and refuses undeclared changes."""

    GROUP = "base@20261024-101500"

    def setUp(self):
        import yaml  # noqa: F401  (dependency of launch_record)
        self.tmp = tempfile.TemporaryDirectory()
        self.variant = Path(self.tmp.name) / "studies" / "12-gru-x" / "variants" / "base"
        self.lr = self.variant / "launch_record"
        self.lr.mkdir(parents=True)
        self.SWEEP = "parameters:\n  n_steps: {value: 60000}\n  lr: {value: 0.001}\n"
        (self.variant / "sweep.yaml").write_text(self.SWEEP)
        (self.lr / "sweep.yaml").write_text(self.SWEEP)      # the launcher's immutable snapshot
        initial = {"_meta": {"schema_version": 2, "produced_by": "launch_beaker_resumable.py",
                             "produced_at_pt": "2026-10-24T10:15:00-07:00"},
                   "kind": "initial", "platform": "beaker", "wandb_group": self.GROUP, "label": "base",
                   "note": "baseline launch of the example", "manifests": ["sweep.yaml"],
                   "job_refs": [{"type": "beaker_experiment", "id": "01K"}],
                   "revisions": {"studies": "a" * 40, "dispatcher": "b" * 40, "wrapper": "c" * 40}}
        (self.lr / "beaker_resumable.json").write_text(json.dumps(initial))
        self.study = self.variant.parents[1]

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name="rescue1.json", **kw):
        from launch_record import write_intervention
        args = dict(kind="rescue", platform="beaker", wandb_group=self.GROUP, study_root=self.study,
                    job_refs=[{"type": "beaker_experiment", "id": "01KNEW"}],
                    trigger={"symptom": "3 tasks wedged", "cause": "bad node"})
        args.update(kw)
        with mock.patch("launch_record.build_meta", return_value={
                "produced_by": "x", "produced_at_pt": "2026-10-25T00:00:00-07:00", "wandb_groups": [self.GROUP]}):
            return write_intervention(self.lr / name, **args)

    def schema_errors(self, rec):
        from jsonschema import Draft202012Validator
        schema = json.loads((Path(__file__).resolve().parents[1] / "schemas" / "launch-record.v2.schema.json").read_text())
        return [e.message for e in Draft202012Validator(schema).iter_errors(rec)]

    def test_rescue_is_v2_and_inherits_the_launch(self):
        rec = self.write()
        self.assertEqual(rec["_meta"]["schema_version"], 2)
        self.assertEqual(rec["revisions"]["dispatcher"], "b" * 40)
        self.assertEqual(rec["manifests"], ["sweep.yaml"])
        self.assertEqual(self.schema_errors(rec), [])
        findings = []
        check_records_parse(load_records(self.lr), findings)
        self.assertEqual(findings, [])

    def test_backfill_validates(self):
        rec = self.write("backfill1.json", kind="backfill", platform="none", job_refs=[])
        self.assertEqual(self.schema_errors(rec), [])

    def test_orphan_group_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no initial launch record"):
            self.write(wandb_group="base@20261101-000000")

    def resubmit(self, text: str) -> str:
        (self.lr / "rescue1").mkdir(exist_ok=True)
        (self.lr / "rescue1" / "sweep.yaml").write_text(text)
        return "launch_record/rescue1/sweep.yaml"

    def test_undeclared_manifest_change_is_refused(self):
        m = self.resubmit(self.SWEEP.replace("60000", "120000"))
        with self.assertRaisesRegex(ValueError, "undeclared key.*n_steps"):
            self.write("rescue1/extend1.json", kind="extend", manifests=[m])
        rec = self.write("rescue1/extend1.json", kind="extend", manifests=[m],
                         deviations={"parameters.n_steps": 120000})
        self.assertEqual(rec["manifests"], [m])               # the submitted path, not the launch's

    def test_only_extend_may_change_the_step_budget(self):
        m = self.resubmit(self.SWEEP.replace("60000", "120000"))
        with self.assertRaisesRegex(ValueError, "only an extend"):
            self.write("rescue1/r.json", manifests=[m], deviations={"parameters.n_steps": 120000})

    def test_variant_edited_after_launch_is_compared_with_the_snapshot(self):
        (self.variant / "sweep.yaml").write_text(self.SWEEP.replace("0.001", "0.005"))
        with self.assertRaisesRegex(ValueError, "undeclared key.*lr"):
            self.write()                                       # default: the launch's manifests

    def test_added_null_key_is_a_change(self):
        m = self.resubmit(self.SWEEP + "  extra: {value: null}\n")
        with self.assertRaisesRegex(ValueError, "undeclared key.*extra"):
            self.write("rescue1/r.json", manifests=[m])

    def test_empty_manifests_and_unknown_manifest_are_refused(self):
        with self.assertRaisesRegex(ValueError, "manifests is empty"):
            self.write(manifests=[])
        (self.variant / "other.yaml").write_text("a: 1\n")
        with self.assertRaisesRegex(ValueError, "matches none"):
            self.write(manifests=["other.yaml"])

    def test_launch_manifest_is_read_at_the_launch_commit(self):
        import subprocess
        root = Path(self.tmp.name)
        git = lambda *a: subprocess.run(["git", "-C", str(root), *a], check=True,
                                        capture_output=True, text=True).stdout.strip()
        (self.lr / "sweep.yaml").unlink()                      # no snapshot: fall back to git
        git("init", "-q"); git("add", "-A")
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "launch")
        rec_path = self.lr / "beaker_resumable.json"
        rec = json.loads(rec_path.read_text())
        rec["revisions"]["studies"] = git("rev-parse", "HEAD")
        rec_path.write_text(json.dumps(rec))
        (self.variant / "sweep.yaml").write_text(self.SWEEP.replace("0.001", "0.005"))   # edited since
        with self.assertRaisesRegex(ValueError, "undeclared key.*lr"):
            self.write()

    def test_step_budget_override_in_a_command_list_is_extend_only(self):
        cmd = ("command: [python, -m, run, model.training.n_steps=%d, '${args_no_hyphens}']\n")
        (self.lr / "sweep.yaml").write_text(self.SWEEP + cmd % 60000)
        (self.variant / "sweep.yaml").write_text(self.SWEEP + cmd % 60000)
        m = self.resubmit(self.SWEEP + cmd % 150000)
        with self.assertRaisesRegex(ValueError, "only an extend"):
            self.write("rescue1/r.json", manifests=[m], deviations={"command": True})
        rec = self.write("rescue1/e.json", kind="extend", manifests=[m], deviations={"command": True})
        self.assertEqual(rec["kind"], "extend")

    def test_inherited_record_local_manifest_is_found_beside_the_launch(self):
        rec_path = self.lr / "beaker_resumable.json"
        rec = json.loads(rec_path.read_text())
        rec["manifests"] = ["sweep.yaml", "experiment_resumable_submitted.yaml"]
        rec_path.write_text(json.dumps(rec))
        (self.lr / "experiment_resumable_submitted.yaml").write_text("tasks: []\n")
        (self.lr / "rescue2").mkdir()
        out = self.write("rescue2/rescue2.json")
        # variant-relative paths the CI checker resolves from the variant directory
        self.assertEqual(out["manifests"], ["sweep.yaml", "launch_record/experiment_resumable_submitted.yaml"])
        self.assertTrue(all((self.variant / m).exists() for m in out["manifests"]))

    def test_backfill_skips_the_submission_diff(self):
        (self.variant / "sweep.yaml").write_text(self.SWEEP.replace("0.001", "0.005"))   # edited since
        rec = self.write("backfill1.json", kind="backfill", platform="none", job_refs=[])
        self.assertEqual(rec["manifests"], ["sweep.yaml"])

    def test_backfill_with_no_manifests_is_refused(self):
        with self.assertRaisesRegex(ValueError, "manifests is empty"):
            self.write("backfill1.json", kind="backfill", platform="none", job_refs=[], manifests=[])

    def test_hydra_delete_operator_is_a_change(self):
        cmd = "command: [python, -m, run, '%smodel.training.lr=0.001']\n"
        (self.lr / "sweep.yaml").write_text(self.SWEEP + cmd % "")
        (self.variant / "sweep.yaml").write_text(self.SWEEP + cmd % "")
        m = self.resubmit(self.SWEEP + cmd % "~")
        with self.assertRaisesRegex(ValueError, "undeclared key"):
            self.write("rescue1/r.json", manifests=[m])

    def test_missing_explicit_manifest_is_refused_not_substituted(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            self.write("rescue1.json", manifests=["launch_record/rescue1/sweep.yaml"])

    def test_relative_destination(self):
        cwd = os.getcwd()
        os.chdir(self.tmp.name)
        try:
            from launch_record import write_intervention
            with mock.patch("launch_record.build_meta", return_value={"produced_by": "x"}):
                rec = write_intervention("studies/12-gru-x/variants/base/launch_record/backfill.json",
                                         kind="backfill", platform="none", wandb_group=self.GROUP,
                                         study_root=self.study, trigger={"symptom": "lost metric",
                                                                         "cause": "logger crash"})
        finally:
            os.chdir(cwd)
        self.assertEqual(rec["manifests"], ["sweep.yaml"])

    def test_short_note_is_refused(self):
        with self.assertRaisesRegex(ValueError, "at least 10"):
            self.write(note="oops")

    def test_unrecoverable_launch_manifest_is_refused(self):
        (self.lr / "sweep.yaml").unlink()                      # no snapshot, not a git repo
        with self.assertRaisesRegex(ValueError, "cannot recover"):
            self.write()

if __name__ == "__main__":
    unittest.main()
