"""Freeze round 2 of study 14 (variant ``refit-prior``: refits pooled toward the global fit).

Reads each pinned run's output files on /allen (written by the wrapper's GLMHMMTrainer) and
writes ``frozen_round2.json``: per cell (emission, K, weight-prior SD, transition strength), the
scalar scores and the per-subject held-out scores. ``r2_refit_prior.py`` reads only that file
and ``frozen_round1.json``.

Every cell loaded round 1's global fit for its (emission, K). The freeze checks that it loaded
the file the variant's manifest names (sha256) and that its zero-shot score equals round 1's,
the score of that same global fit on the same trials. It refuses to write otherwise.

Needs the /allen mount, so it runs on an HPC compute node:

    make -C studies/14-glmhmm-vs-gru-heldout freeze-round2
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

WANDB_GROUPS = ["refit-prior@20261101-130226"]
RUN_ROOT = Path("/allen/aind/scratch/han.hou/outputs/disrnn/wandb")
RUN_GLOB = "run-202611*"
PROJECT = "AIND-disRNN/mice_glmhmm"
OUT = HERE / "frozen_round2.json"
ROUND1 = HERE / "frozen_round1.json"
MANIFEST = HERE.parent / "variants" / "refit-prior" / "global_fits_manifest.json"
EMISSIONS = ("hurdle", "reference")
GRID = {"num_states": range(1, 11), "prior_sigma_subject": (0.25, 0.5, 1.0, 2.0),
        "transition_strength_subject": (None, 10.0, 100.0, 1000.0)}
ZERO_SHOT_TOL = 1e-5
SUMMARY_PREFIXES = ("glmhmm/", "heldout/")
# Per-subject held-out scores kept per cell, in the order of the top-level subject list.
PER_SUBJECT_SCORES = ("eval_likelihood", "eval_likelihood_LR_engaged")


def sha256(path):
    """sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_group(files):
    """The run's W&B group, from its config.yaml, or None."""
    try:
        meta = yaml.safe_load((files / "config.yaml").read_text())["meta"]["value"]
    except (OSError, yaml.YAMLError, KeyError, TypeError):
        return None
    return f"{meta.get('variant')}@{meta.get('launch_id')}"


def cell_key(config):
    """(emission, K, sigma, c) of a fit.json config; c is None when the transitions are not pooled."""
    c = config.get("transition_strength_subject")
    return (config["emission"], int(config["num_states"]), float(config["prior_sigma_subject"]),
            None if c is None else float(c))


def find_runs():
    """Every finished run of the pinned group, one per cell."""
    runs = {}
    for run_dir in sorted(RUN_ROOT.glob(RUN_GLOB)):
        files = run_dir / "files"
        fit_path = files / "outputs" / "glmhmm" / "fit.json"
        if run_group(files) not in WANDB_GROUPS or not fit_path.exists():
            continue
        try:
            summary = json.loads((files / "wandb-summary.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "glmhmm/select/eval_likelihood" not in summary:
            continue
        config = json.loads(fit_path.read_text())["config"]
        key = cell_key(config)
        if key in runs:
            raise RuntimeError(f"two finished runs for {key}: {runs[key]['dir']} and {run_dir}")
        runs[key] = {"dir": run_dir, "files": files, "summary": summary, "config": config,
                     "wandb_run_id": run_dir.name.split("-")[-1]}
    return runs


def round1_zero_shot():
    """Round 1's zero-shot held-out score per (emission, K), from its committed freeze."""
    frozen = json.loads(ROUND1.read_text())
    return {(r["emission"], r["num_states"]): r["summary"]["heldout/zero_shot/eval_likelihood"]
            for r in frozen["runs"] if r["group"].split("@")[0] in EMISSIONS}


def main():
    runs = find_runs()
    expected = {(e, K, s, c) for e in EMISSIONS for K in GRID["num_states"]
                for s in GRID["prior_sigma_subject"] for c in GRID["transition_strength_subject"]}
    missing = sorted(expected - set(runs), key=str)
    extra = sorted(set(runs) - expected, key=str)
    if extra:
        raise RuntimeError(f"cells outside the grid: {extra}")
    manifest = {f["file"]: f["sha256"] for f in json.loads(MANIFEST.read_text())["files"]}
    zero_shot_r1 = round1_zero_shot()

    cells, sources, worst_zero_shot, subjects, n_trials = [], [], 0.0, None, None
    for key in sorted(runs, key=lambda k: (k[0], k[1], k[2], -1.0 if k[3] is None else k[3])):
        run = runs[key]
        emission, K, sigma, c = key
        s = run["summary"]
        expected_file = f"{emission}_K{K}_L10.json"
        if s.get("glmhmm/global/loaded_sha256") != manifest[expected_file]:
            raise RuntimeError(f"{run['wandb_run_id']}: did not load {expected_file} from the manifest")
        if run["config"].get("subject_prior_center") != "global":
            raise RuntimeError(f"{run['wandb_run_id']}: weight prior not centred on the global fit")
        gap = abs(s["heldout/zero_shot/eval_likelihood"] - zero_shot_r1[(emission, K)])
        if gap > ZERO_SHOT_TOL:
            raise RuntimeError(f"{run['wandb_run_id']}: zero-shot differs from round 1 by {gap:.2e}")
        worst_zero_shot = max(worst_zero_shot, gap)

        out = run["files"] / "outputs" / "glmhmm"
        paths = [run["files"] / "wandb-summary.json", out / "fit.json", out / "heldout_per_subject.csv"]
        sources.append({"type": "wandb_run", "id": f"{PROJECT}/{run['wandb_run_id']}"})
        sources += [{"type": "file", "id": str(p), "sha256": sha256(p)} for p in paths]
        per_subject = pd.read_csv(out / "heldout_per_subject.csv", dtype={"subject_id": str})
        per_subject = per_subject.sort_values("subject_id").reset_index(drop=True)
        if subjects is None:
            subjects = per_subject["subject_id"].tolist()
            n_trials = per_subject["n_eval_trials"].astype(int).tolist()
        elif per_subject["subject_id"].tolist() != subjects:
            raise RuntimeError(f"{run['wandb_run_id']}: scored other held-out subjects")
        cells.append({
            "emission": emission, "num_states": K, "prior_sigma_subject": sigma,
            "transition_strength_subject": c, "wandb_run_id": run["wandb_run_id"],
            "summary": {k: v for k, v in s.items()
                        if k.startswith(SUMMARY_PREFIXES) and isinstance(v, (int, float, str, bool))
                        and not (isinstance(v, float) and not math.isfinite(v))},
            "per_subject": {k: per_subject[k].round(6).tolist() for k in PER_SUBJECT_SCORES},
        })

    meta = build_meta("analysis/freeze_round2.py", WANDB_GROUPS, study_root=HERE.parent,
                      kind="freeze", code=[__file__], inputs=[ROUND1, MANIFEST], sources=sources)
    write_artifact(OUT, {
        "_meta": meta,
        "group": WANDB_GROUPS[0],
        "grid": {k: list(v) for k, v in GRID.items()},
        "checks": {"n_cells": len(cells), "n_expected": len(expected),
                   "missing": [list(m) for m in missing],
                   "loaded_sha256_matches_manifest": True,
                   "max_zero_shot_gap_to_round1": worst_zero_shot},
        "subjects": subjects, "subject_n_eval_trials": n_trials,
        "cells": cells,
    })
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB): {len(cells)} of {len(expected)} cells, "
          f"{len(missing)} missing; max zero-shot gap to round 1 {worst_zero_shot:.1e}")


if __name__ == "__main__":
    main()
