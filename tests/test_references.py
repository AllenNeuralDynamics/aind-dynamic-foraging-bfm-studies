"""The registry of named cross-study reference versions (``studies/util/references.py``, #63).

Every entry must resolve against the committed JSON it names, every comparison must have a
registered default, and no entry may store the number it stands for.
"""

import csv
import json
import os
import re
import statistics
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "studies" / "util"))

import _meta  # noqa: E402
import references  # noqa: E402
from references import ReferenceUnresolved, ref, resolve_pointer, versions  # noqa: E402

S01 = REPO_ROOT / "studies" / "01-gru-scaling-law"
NAME = re.compile(r"^[a-z][a-z0-9_]*\.[a-z0-9_]+$")
ENTRY_KEYS = {"study", "report", "json", "pointer", "checks", "conditions"}
# git's discovery variables cleared, so a GIT_DIR exported for another checkout is not asked
GIT_ENV = {k: v for k, v in os.environ.items() if k not in _meta._GIT_DISCOVERY_VARS}


def _leaves(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _leaves(v)
    else:
        yield obj


class TestEveryEntry(unittest.TestCase):
    def test_resolves_against_the_committed_json(self):
        self.assertTrue(versions())
        for name in versions():
            with self.subTest(name=name):
                r = ref(name)
                self.assertEqual(r.name, name)
                self.assertIsInstance(r.value, float)
                self.assertTrue(r.source.is_file())

    def test_source_json_is_committed(self):
        for name in versions():
            with self.subTest(name=name):
                rel = ref(name).source.relative_to(REPO_ROOT).as_posix()
                out = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", rel],
                                     capture_output=True, text=True, env=GIT_ENV)
                self.assertEqual(out.returncode, 0, f"{rel} is not tracked")

    def test_names_the_report_its_readers_cite_in_upstream(self):
        for name in versions():
            with self.subTest(name=name):
                r = ref(name)
                reports = REPO_ROOT / "studies" / r.study / "analysis" / "reports"
                self.assertEqual(len(list(reports.glob(f"{r.report}-*.md"))), 1)
                self.assertEqual(r.upstream, {"study": r.study, "report": r.report})

    def test_shape_and_conditions(self):
        for name, entry in references._ENTRIES.items():
            with self.subTest(name=name):
                self.assertRegex(name, NAME)
                self.assertLessEqual(set(entry), ENTRY_KEYS)
                self.assertLessEqual(ENTRY_KEYS - {"checks"}, set(entry))
                self.assertTrue(entry["pointer"].startswith("/"))
                self.assertEqual(set(entry["conditions"]), set(references.CONDITION_KEYS))
                for key, text in entry["conditions"].items():
                    self.assertIsInstance(text, str, key)
                    self.assertTrue(text.strip(), key)


class TestNoStoredValues(unittest.TestCase):
    def test_numbers_appear_only_as_check_coordinates(self):
        for name, entry in references._ENTRIES.items():
            with self.subTest(name=name):
                value = ref(name).value
                for key, part in entry.items():
                    if key == "checks":
                        self.assertNotIn(value, list(part.values()))
                        continue
                    for leaf in _leaves(part):
                        self.assertIsInstance(leaf, str, f"{key} holds {leaf!r}")

    def test_module_retypes_no_decimal(self):
        # a likelihood or correlation pasted anywhere in the module, docstrings included
        text = Path(references.__file__).read_text()
        self.assertEqual(re.findall(r"\d\.\d{3,}", text), [])


class TestDefaults(unittest.TestCase):
    def test_one_registered_default_per_comparison(self):
        self.assertEqual(set(references.DEFAULTS), set(references.comparisons()))
        for comparison, version in references.DEFAULTS.items():
            with self.subTest(comparison=comparison):
                r = ref(comparison)
                self.assertEqual(r.name, f"{comparison}.{version}")
                self.assertTrue(r.is_default)

    def test_other_versions_are_not_default(self):
        for name in versions():
            comparison, version = name.split(".", 1)
            self.assertEqual(ref(name).is_default, references.DEFAULTS[comparison] == version)

    def test_versions_lists_one_comparison(self):
        names = versions("gru_ceiling")
        self.assertIn("gru_ceiling.v2_h128", names)
        self.assertTrue(all(n.startswith("gru_ceiling.") for n in names))


class TestWhatTheRegistryRecords(unittest.TestCase):
    def test_v2_h128_is_one_measurement_in_three_files(self):
        """01 r1's v2 D=614 cell, r7's N=128 x D=614 grid cell and the seed mean of the
        `v2-sc-active` D=614 rows (what studies 08 and 10 quote) are the same runs."""
        value = ref("gru_ceiling.v2_h128").value
        nxd = json.loads((S01 / "analysis" / "nxd_scaling.json").read_text())
        self.assertEqual((nxd["Ns"][2], nxd["Ds"][3]), (128, 614))
        self.assertEqual(value, nxd["mean_grid"][2][3])
        with open(S01 / "scaling_results.csv", newline="") as f:
            rows = [float(r["heldout_likelihood"]) for r in csv.DictReader(f)
                    if r["variant"] == "v2-sc-active" and r["D"] == "614"]
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(value, statistics.fmean(rows), places=12)

    def test_cite_prints_the_version_name(self):
        r = ref("gru_ceiling.v2_h128")
        self.assertEqual(r.cite(), f"{r.value:.4f} (gru_ceiling.v2_h128)")
        self.assertEqual(r.cite(".6f"), f"{r.value:.6f} (gru_ceiling.v2_h128)")

    def test_source_goes_into_build_meta_inputs(self):
        """A producer passes `ref.source` to build_meta, so check_provenance sees it read study 01
        (report-upstream-undeclared) and that its input changed (artifact-stale-input)."""
        r = ref("rl_baseline.hattori")
        meta = _meta.build_meta("analysis/x.py", [], code=[__file__], inputs=[r.source])
        self.assertEqual([e["path"] for e in meta["inputs"]],
                         ["studies/01-gru-scaling-law/analysis/rl_baseline.json"])

    def test_importing_does_not_load_cross_study(self):
        """build_meta digests every loaded studies/util module; references must not pull in
        cross_study.py, or every reader's artifact would stale on a cross_study edit."""
        code = ("import sys; sys.path.insert(0, 'studies/util'); import references; "
                "print('cross_study' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True,
                             text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False")


class TestFailsLoudly(unittest.TestCase):
    def _entry(self, **overrides):
        entry = dict(references._ENTRIES["gru_ceiling.v2_h256"])
        entry.update(overrides)
        return {"test_cmp.bad": entry}

    def _assert_unresolved(self, entries, *fragments):
        with mock.patch.dict(references._ENTRIES, entries):
            with self.assertRaises(ReferenceUnresolved) as cm:
                ref("test_cmp.bad")
        for fragment in ("test_cmp.bad", *fragments):
            self.assertIn(fragment, str(cm.exception))

    def test_missing_key(self):
        self._assert_unresolved(self._entry(pointer="/mean_grid_typo/3/3"), "mean_grid_typo")

    def test_index_out_of_range(self):
        self._assert_unresolved(self._entry(pointer="/mean_grid/9/3"), "out of range")

    def test_null_value(self):
        # nxd_scaling.json has no H512 x D614 cell (host-RAM OOM), stored as null
        self._assert_unresolved(self._entry(pointer="/mean_grid/4/3", checks={"/Ns/4": 512}), "not a number")

    def test_non_scalar_value(self):
        self._assert_unresolved(self._entry(pointer="/mean_grid/3"), "not a number")

    def test_failed_coordinate_check(self):
        self._assert_unresolved(self._entry(checks={"/Ns/3": 128}), "/Ns/3")

    def test_missing_file(self):
        self._assert_unresolved(self._entry(json="analysis/no_such.json"), "no_such.json")

    def test_unknown_names(self):
        with self.assertRaises(KeyError):
            ref("gru_ceiling.v9_h1")
        with self.assertRaises(KeyError):
            ref("no_such_comparison")


class TestResolvePointer(unittest.TestCase):
    DOC = {"a": {"b/c": [10, 20], "m~n": 1, "": 2}}

    def test_rfc6901(self):
        self.assertIs(resolve_pointer(self.DOC, ""), self.DOC)
        self.assertEqual(resolve_pointer(self.DOC, "/a/b~1c/1"), 20)
        self.assertEqual(resolve_pointer(self.DOC, "/a/m~0n"), 1)
        self.assertEqual(resolve_pointer(self.DOC, "/a/"), 2)

    def test_bad_pointers_raise(self):
        for pointer in ("a", "/x", "/a/b~1c/2", "/a/b~1c/01", "/a/b~1c/-", "/a/m~0n/q"):
            with self.subTest(pointer=pointer), self.assertRaises(ReferenceUnresolved):
                resolve_pointer(self.DOC, pointer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
