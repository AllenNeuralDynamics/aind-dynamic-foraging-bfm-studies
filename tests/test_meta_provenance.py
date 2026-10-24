"""Analysis ``_meta`` names the studies checkout and the source SHAs of the groups it reads.

Moved from the dispatcher's ``tests/test_provenance_helpers.py`` (dispatcher #175) with the
helpers it tests. Every test builds a studies-shaped git repo in a temp dir, so nothing here
depends on this checkout's state beyond it being a git repo.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "studies" / "util"))

import _meta  # noqa: E402
from _meta import build_meta  # noqa: E402

SWEEP = """\
entity: AIND-disRNN
project: ai_hub_test
name: unit-test-sweep
method: grid
command: [python, -m, run_hpc, data=synthetic, "${args_no_hyphens}"]
parameters:
  seed:
    values: [1]
"""


def _git(cwd: Path, *args: str) -> str:
    """git in ``cwd`` with git's discovery variables cleared, so a sandbox ``GIT_DIR`` /
    ``GIT_WORK_TREE`` exported for another checkout never makes a fixture command touch it."""
    env = {k: v for k, v in os.environ.items() if k not in _meta._GIT_DISCOVERY_VARS}
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True,
                          text=True, env=env).stdout.strip()


def make_studies_repo(tmp: Path) -> tuple[Path, Path, str]:
    """A git repo laid out like this one; returns (root, sweep, head sha)."""
    root = tmp / "studies-repo"
    variant = root / "studies" / "08-hb-vs-gru-heldout" / "variants" / "stan-arm"
    variant.mkdir(parents=True)
    sweep = variant / "sweep.yaml"
    sweep.write_text(SWEEP)
    (root / "README.md").write_text("studies\n")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root, sweep, _git(root, "rev-parse", "HEAD")


class TestAnalysisMetaNamesTheStudiesCheckout(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="prov_meta_"))
        self.root, self.sweep, self.head = make_studies_repo(self.tmp)
        self.study_root = self.root / "studies" / "08-hb-vs-gru-heldout"
        rec_dir = self.study_root / "variants" / "stan-arm" / "launch_record" / "beaker_x"
        rec_dir.mkdir(parents=True)
        (rec_dir / "beaker_resumable.json").write_text(json.dumps({
            "mode": "resumable-grid", "wandb_group": "stan-arm@1",
            "dispatcher_git_commit": "d" * 40, "dispatcher_commit": "d" * 40,
            "studies_git_commit": "1" * 40,
            "wrapper_git_commit": "2" * 40, "foraging_models_git_commit": "f" * 40,
        }))
        (self.study_root / "variants" / "stan-arm" / "launch_record" / "hpc_stan.json").write_text(
            json.dumps({"_meta": {"wandb_groups": ["stan-arm@2"]}, "kind": "resubmit",
                        "launch": {"revisions": {"dispatcher_commit": "e" * 40,
                                                 "wrapper_commit": "2" * 40}}}))

    def test_source_shas_come_from_the_launch_records_of_the_groups_read(self):
        meta = build_meta("analysis/x.py", ["stan-arm@1", "stan-arm@2"], study_root=self.study_root)
        self.assertEqual(meta["source_dispatcher_git_shas"], ["d" * 40, "e" * 40])
        self.assertEqual(meta["source_wrapper_git_shas"], ["2" * 40])
        self.assertEqual(meta["source_foraging_models_git_shas"], ["f" * 40])
        self.assertEqual(meta["source_studies_git_shas"], ["1" * 40])
        # stan-arm@2's record names the group but carries no studies / models SHA: a gap, not
        # silence, even though every array above is non-empty
        self.assertEqual(meta["provenance_findings"], [
            "no studies SHA in the launch record(s) of W&B group(s): stan-arm@2",
            "no foraging_models SHA in the launch record(s) of W&B group(s): stan-arm@2",
        ])
        self.assertNotIn("dispatcher_git_sha", meta)   # never its own HEAD under that name
        self.assertRegex(meta["studies_git_sha"], r"^[0-9a-f]{40}$")
        self.assertIn(meta["studies_git_dirty"], (True, False))

    def test_a_group_without_a_record_is_an_explicit_finding_not_a_guess(self):
        meta = build_meta("analysis/x.py", ["stan-arm@1", "stan-arm@none"], study_root=self.study_root)
        self.assertEqual(meta["source_dispatcher_git_shas"], ["d" * 40])
        self.assertEqual(meta["provenance_findings"],
                         ["no launch record found for W&B group(s): stan-arm@none"])
        full = build_meta("analysis/x.py", ["stan-arm@1"], study_root=self.study_root)
        self.assertEqual(full["provenance_findings"], [])   # every repo's SHA present

    def test_meta_git_env_is_ambient_only_inside_the_exported_work_tree(self):
        env = {"GIT_DIR": "/elsewhere/store", "GIT_WORK_TREE": str(self.root)}
        with mock.patch.dict(os.environ, env):
            self.assertIn("GIT_DIR", _meta._git_env(self.root / "studies"))      # inside: kept
            self.assertNotIn("GIT_DIR", _meta._git_env(REPO_ROOT / "studies" / "util"))
            self.assertNotIn("GIT_WORK_TREE", _meta._git_env(REPO_ROOT))  # outside: stripped

    def test_without_a_study_root_the_arrays_are_empty_with_a_finding(self):
        meta = build_meta("analysis/x.py", ["g@1"])
        self.assertEqual(meta["source_dispatcher_git_shas"], [])
        self.assertTrue(meta["provenance_findings"])


if __name__ == "__main__":
    unittest.main()
