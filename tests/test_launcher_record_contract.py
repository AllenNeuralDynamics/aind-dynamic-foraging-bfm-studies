"""The dispatcher launchers' records are records this repo's validator reads.

The launchers live in the dispatcher and the validator here, so neither repo's tests can run
both since the split (dispatcher #175). ``fixtures/launcher_records/`` holds one record per
launcher, written by the launchers themselves at dispatcher ``0a3a820`` (dispatcher #195: the
records carry ``data`` and ``image``, studies#25; only the temp-dir path normalized, ids fixed).
When a launcher's record schema changes, regenerate them with ``fixtures/launcher_records/
regenerate.py`` and keep this test green. Each record must validate against
schemas/launch-record.v2.schema.json, still carry the legacy keys older readers use, and record
what its runs trained on and ran in, as the checker's record-data/image-unrecorded rules ask.
"""

import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "launcher_records"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies" / "util"))

from launch_record import load_records, refs_in  # noqa: E402
from validate_provenance import check_records_parse  # noqa: E402

SCHEMA = Draft202012Validator(json.loads(
    (Path(__file__).resolve().parents[1] / "schemas" / "launch-record.v2.schema.json").read_text()))
FOUR_SHAS = ("studies_git_commit", "dispatcher_git_commit", "wrapper_git_commit",
             "foraging_models_git_commit")


class TestLauncherRecords(unittest.TestCase):
    def _records(self, launcher: str) -> list[dict]:
        records = load_records(FIXTURES / launcher)
        self.assertEqual([r["_schema"] for r in records], ["v2"])
        rec = {k: v for k, v in records[0].items() if k not in ("_schema", "_path")}
        self.assertEqual([e.message for e in SCHEMA.iter_errors(rec)], [])
        self.assertEqual(rec["kind"], "initial")
        findings = []
        check_records_parse(records, findings)
        self.assertEqual(findings, [])
        return records

    def test_resumable_beaker_record(self):
        (rec,) = self._records("beaker_resumable")
        self.assertEqual(rec["mode"], "resumable-grid")
        for key in FOUR_SHAS:
            self.assertRegex(rec[key], r"^[0-9a-f]{40}$", key)

    def test_native_beaker_record(self):
        records = self._records("beaker_native")
        self.assertEqual(refs_in(records, "wandb_sweep"), {"abc123"})

    def test_hpc_record(self):
        records = self._records("hpc")
        self.assertEqual(records[0]["mode"], "hpc-sweep")
        self.assertEqual(refs_in(records, "wandb_sweep"), {"xyz789"})

    def test_beaker_records_pin_a_snapshot_and_an_image_id(self):
        for launcher in ("beaker_resumable", "beaker_native"):
            (rec, *_) = self._records(launcher)
            data, image = rec["data"], rec["image"]
            self.assertEqual(data["source"], "snapshot", launcher)
            self.assertTrue(data["uri"].endswith(f"/snapshots/{data['snapshot']}/"), launcher)
            self.assertRegex(data["manifest_sha256"], r"^[0-9a-f]{64}$", launcher)
            self.assertEqual(image["platform"], "beaker", launcher)
            self.assertTrue(image["name"] and image["id"], launcher)

    def test_hpc_record_names_its_data_and_conda_env(self):
        (rec,) = self._records("hpc")
        self.assertEqual(rec["data"]["source"], "generated")
        self.assertEqual(rec["image"]["platform"], "hpc")
        self.assertRegex(rec["image"]["lock_sha256"], r"^[0-9a-f]{64}$")

    def test_every_record_names_its_wandb_group(self):
        for launcher in ("beaker_resumable", "beaker_native", "hpc"):
            rec = json.loads(next((FIXTURES / launcher).glob("*.json")).read_text())
            self.assertRegex(rec["wandb_group"], r"^stan-arm@\d{8}-\d{6}$", launcher)


if __name__ == "__main__":
    unittest.main()
