"""Which W&B key a producer reads (studies#157). Fake values; no network."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies" / "util"))

import wandb_key  # noqa: E402


class KeyForTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for patcher in (mock.patch.dict(os.environ, {wandb_key.KEY_DIR_ENV: str(self.dir)}),
                        mock.patch.object(wandb_key, "_netrc_key", return_value=None)):
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop("WANDB_API_KEY", None)

    def tearDown(self):
        self.tmp.cleanup()

    def test_file_beats_a_stale_env_var(self):
        (self.dir / "api_key.NeuralDynamics").write_text("from-file\n")
        with mock.patch.dict(os.environ, {"WANDB_API_KEY": "stale-personal"}):
            self.assertEqual(wandb_key.key_for("NeuralDynamics"), "from-file")

    def test_without_a_file_the_env_var_then_netrc_are_used(self):
        with mock.patch.dict(os.environ, {"WANDB_API_KEY": "from-env"}):
            self.assertEqual(wandb_key.key_for("AIND-disRNN"), "from-env")
        with mock.patch.object(wandb_key, "_netrc_key", return_value="from-netrc"):
            self.assertEqual(wandb_key.key_for("AIND-disRNN"), "from-netrc")

    def test_no_key_names_the_file(self):
        with self.assertRaisesRegex(RuntimeError, "api_key.NeuralDynamics"):
            wandb_key.key_for("NeuralDynamics")


if __name__ == "__main__":
    unittest.main()
