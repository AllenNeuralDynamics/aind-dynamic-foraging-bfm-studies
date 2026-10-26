"""studies/util/freeze_run_provenance.py: what a group's W&B runs say they trained on and ran in.

W&B, Beaker and the snapshot manifest are stubbed; only the per-group summary is under test.
"""

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies" / "util"))

import freeze_run_provenance as frp  # noqa: E402

SHA = {"w": "1" * 40, "d": "2" * 40, "m": "3" * 40}


def run(rid, data, job=None, commits=True, state="finished"):
    cfg = {"data": data}
    if commits:
        cfg.update(wrapper_commit=SHA["w"], dispatcher_commit=SHA["d"], foraging_models_commit=SHA["m"])
    if job:
        cfg["BEAKER_JOB_ID"] = job
    return types.SimpleNamespace(id=rid, state=state, config=cfg)


class GroupProvenanceTest(unittest.TestCase):
    def summarize(self, runs):
        images = {"J1": "01KIMAGE", "J2": "01KIMAGE", "J3": None}          # J3: a purged job
        with mock.patch.object(frp, "manifest", return_value=("e" * 64, {"latest_build_id": "B", "n_sessions": 7},
                                                              "https://example/snapshot_manifest.json")):
            return frp.group_provenance(runs, "AIND-disRNN/p", images.get)

    def test_snapshot_runs_record_digest_commits_and_image(self):
        g, sources = self.summarize([run("a", {"type": "mice_snapshot", "snapshot": "20260603"}, "J1"),
                                     run("b", {"type": "mice_snapshot", "snapshot": "20260603"}, "J2")])
        self.assertEqual(g["data"], [{"source": "snapshot", "type": "mice_snapshot", "snapshot": "20260603",
                                      "uri": f"{frp.SNAPSHOT_PREFIX}/snapshots/20260603/", "manifest_sha256": "e" * 64,
                                      "build_id": "B", "n_sessions": 7, "n_runs": 2}])
        self.assertEqual(g["revisions"]["wrapper"], {SHA["w"]: 2})
        self.assertEqual(g["image"], [{"platform": "beaker", "id": "01KIMAGE", "via": "beaker job", "n_runs": 2}])
        self.assertEqual(g["unrecovered"], {})
        self.assertIn({"type": "wandb_run", "id": "AIND-disRNN/p/a", "state": "finished"}, sources)

    def test_live_unrecorded_and_unrecoverable_are_told_apart(self):
        g, _ = self.summarize([run("a", {"type": "mice_snapshot"}, "J3"),       # database, no snapshot: live
                               run("b", {}, commits=False),                     # re-scoring job: nothing logged
                               run("c", {"type": "synthetic_hierarchical"}, "J1")])
        self.assertEqual({d["source"] for d in g["data"]}, {"live", "unrecorded", "generated"})
        self.assertIn({"platform": "beaker", "id": None, "via": "beaker job", "n_runs": 1}, g["image"])   # JobNotFound
        self.assertEqual(g["unrecovered"]["wrapper commit"], 1)
        self.assertEqual(g["unrecovered"]["image (no Beaker job id logged)"], 1)        # not "HPC": unknown

    def test_an_unreadable_manifest_stops_the_freeze(self):
        with mock.patch.object(frp.urllib.request, "urlopen", side_effect=OSError("S3 down")):
            with self.assertRaises(SystemExit):
                frp.manifest("20260603")

    def test_jobless_runs_take_the_launch_record_experiment_image(self):
        with mock.patch.object(frp, "manifest", return_value=(None, {}, "u")):
            g, sources = frp.group_provenance([run("a", {}, commits=False)], "e/p", lambda j: None,
                                              {"01KEXP": {"01KIMG"}})
        self.assertEqual(g["image"], [{"platform": "beaker", "id": "01KIMG", "via": "launch-record experiment", "n_runs": 1}])
        self.assertIn({"type": "url", "id": "https://beaker.org/ex/01KEXP"}, sources)


if __name__ == "__main__":
    unittest.main()
