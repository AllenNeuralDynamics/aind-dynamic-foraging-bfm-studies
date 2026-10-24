#!/usr/bin/env python3
"""Keep study 09's GRU transfer sweeps pointing at the wrapper runner with an inlined manifest.

The transfer runner lives in the wrapper (``python -m external_transfer.gru_transfer``,
dispatcher #179), and a job reads no study files, so each sweep carries its source manifest
inline: ``--source-manifest-name=<file>`` plus ``--source-manifest-json <exact file text>``.
The runner hashes those bytes as ``manifest_sha256``, which this study's freeze scripts compare
with the file, so the inlined copy must be the file verbatim. Re-run this after editing any
``source_runs*.json``; ``--check`` exits 1 when a sweep is stale (CI runs it).

Converts the pre-split form (``python /workspace/aind-dynamic-foraging-bfm-dispatcher/studies/
09-.../run_gru_transfer.py [--source-manifest=<file>]``) on first use. Launch records are never
touched.

Usage: python sync_transfer_sweeps.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

STUDY = Path(__file__).resolve().parent
MODULE = ["python", "-m", "external_transfer.gru_transfer"]
OLD_SCRIPT_SUFFIX = "/run_gru_transfer.py"
DEFAULT_MANIFEST = "source_runs.json"  # the old runner's default when no --source-manifest was given


def _sweeps() -> list[Path]:
    out = []
    for path in sorted((STUDY / "variants").glob("*/*.yaml")):
        if any(p.startswith("launch_record") for p in path.relative_to(STUDY).parts):
            continue
        cfg = yaml.safe_load(path.read_text())
        cmd = cfg.get("command") if isinstance(cfg, dict) else None
        if isinstance(cmd, list) and (
            any(isinstance(t, str) and t.endswith(OLD_SCRIPT_SUFFIX) for t in cmd)
            or cmd[: len(MODULE)] == MODULE
        ):
            out.append(path)
    return out


def _rebuild(cmd: list) -> list:
    """The command with the wrapper module and the current manifest text inlined."""
    manifest = DEFAULT_MANIFEST
    rest: list = []
    tokens = iter(cmd)
    for t in tokens:
        if t == "python":
            continue
        if isinstance(t, str) and t.endswith(OLD_SCRIPT_SUFFIX):
            continue
        if t == "-m":
            next(tokens)              # the module name; re-added below
            continue
        if isinstance(t, str) and t.startswith("--source-manifest="):
            manifest = t.split("=", 1)[1]
            continue
        if isinstance(t, str) and t.startswith("--source-manifest-name="):
            manifest = t.split("=", 1)[1]
            continue
        if t == "--source-manifest-json":
            next(tokens)              # the stale inlined text
            continue
        rest.append(t)
    text = (STUDY / manifest).read_text()
    head = [t for t in rest if t != "${args_no_hyphens}"]
    tail = [t for t in rest if t == "${args_no_hyphens}"]
    return MODULE + head + [f"--source-manifest-name={manifest}", "--source-manifest-json", text] + tail


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="exit 1 if any sweep is stale; write nothing")
    args = p.parse_args()
    stale = []
    for path in _sweeps():
        cfg = yaml.safe_load(path.read_text())
        new_cmd = _rebuild(cfg["command"])
        if new_cmd == cfg["command"]:
            continue
        stale.append(path.relative_to(STUDY).as_posix())
        if args.check:
            continue
        cfg["command"] = new_cmd
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, width=100_000, allow_unicode=True))
        # the inlined text must survive the YAML round trip byte for byte
        back = yaml.safe_load(path.read_text())["command"]
        name = next(t for t in back if t.startswith("--source-manifest-name=")).split("=", 1)[1]
        inline = back[back.index("--source-manifest-json") + 1]
        assert hashlib.sha256(inline.encode()).digest() == hashlib.sha256((STUDY / name).read_bytes()).digest(), path
    verb = "stale" if args.check else "updated"
    print(f"{len(stale)} sweep(s) {verb}" + "".join(f"\n  {s}" for s in stale))
    return 1 if (args.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
