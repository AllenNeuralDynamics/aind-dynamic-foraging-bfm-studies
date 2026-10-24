"""The dispatcher launchers' records are records this repo's validator reads.

The launchers live in the dispatcher and the validator here, so neither repo's tests can run
both since the split (dispatcher #175). ``fixtures/launcher_records/`` holds one record per
launcher, written by the launchers themselves at dispatcher ``818cc6e`` (only the temp-dir
sweep path and branch name normalized). When a launcher's record schema changes, regenerate
these from the dispatcher and keep this test green.
"""

import json
import sys
import unittest
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "launcher_records"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies" / "util"))

from launch_record import load_records, refs_in  # noqa: E402
from validate_provenance import check_records_parse  # noqa: E402

FOUR_SHAS = ("studies_git_commit", "dispatcher_git_commit", "wrapper_git_commit",
             "foraging_models_git_commit")


class TestLauncherRecords(unittest.TestCase):
    def _records(self, launcher: str) -> list[dict]:
        records = load_records(FIXTURES / launcher)
        self.assertEqual([r["_schema"] for r in records], ["launcher"])
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

    def test_every_record_names_its_wandb_group(self):
        for launcher in ("beaker_resumable", "beaker_native", "hpc"):
            rec = json.loads(next((FIXTURES / launcher).glob("*.json")).read_text())
            self.assertRegex(rec["wandb_group"], r"^stan-arm@\d{8}-\d{6}$", launcher)


if __name__ == "__main__":
    unittest.main()
