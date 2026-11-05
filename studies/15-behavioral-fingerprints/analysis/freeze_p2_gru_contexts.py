#!/usr/bin/env python
"""P2: the GRU's context for every score session of its held-out subjects, so it can play the gym.

The fine-tuned GRU (variant ``gru-heldout-replay``, launch ``LAUNCH_ID``) predicts trial t as
``readout(step(h, obs[t-1], c))``: a GRU update on the previous trial's (choice, reward), with
[-1, -1] before the first trial and a zero state at each session's start, read out as 2 logits.
Its context c is the subject's embedding plus a session offset, computed by the network's own
context module from the subject's index and the session's 1-based position among the subject's
sessions (``session_context_map.json``). Everything but c is a plain GRU step
(``post_training_analysis/gru_step.py``), so this freeze evaluates c once per session with the
wrapper's network, and the rollout freeze steps the GRU in numpy.

Check, per seed and session: the first trial's p(right) from c alone (``GruStep`` on [-1, -1]
from a zero state) equals the held-out CSV's ``probability_choice_1`` (``TOL_FIRST``). The full
replay gate (every trial) runs in ``freeze_p2_gru_rollouts.py``, where the rewards are.

Needs /allen and the wrapper; on a compute node:

    srun --ntasks=1 --cpus-per-task=4 --mem=16G -p aind --time=00:30:00 \\
      env PYTHONPATH=<wrapper>/code <dynamic-foraging-bfm-cpu env>/bin/python \\
      analysis/freeze_p2_gru_contexts.py --wrapper <wrapper at the launch's SHA>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

SPECS, GRU = HERE / "p2_task_specs.json", HERE / "p1_gru.json"
LAUNCH_ID = "20261104-150009"
OUTPUTS = Path("/allen/aind/scratch/han.hou/outputs/study15/gru-heldout-replay/hpc") / LAUNCH_ID
OUT = HERE / "p2_gru_contexts.json"
SEEDS, TOL_FIRST = (0, 1, 2), 1e-5


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_dir(seed: int) -> Path:
    found = sorted((OUTPUTS / f"s{seed}" / "heldout").rglob("outputs/params.json"))
    assert len(found) == 1, found
    return found[0].parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrapper", type=Path, required=True)
    a = ap.parse_args()
    import jax.numpy as jnp
    import pandas as pd
    from models.gru_network import make_gru_network
    from models.session_conditioning import resolve_session_conditioning_from_architecture
    from post_training_analysis.gru_step import GruStep, subject_context

    gru = json.loads(GRU.read_text())
    assert gru["launch"]["launch_id"] == LAUNCH_ID
    trained = set(gru["seeds"]["seed0"]["subjects_in_training"])
    specs = [s for s in json.loads(SPECS.read_text())["sessions"] if s["subject_id"] not in trained]
    out, sources, worst = {}, [], 0.0
    for seed in SEEDS:
        d = seed_dir(seed)
        params = {k: {kk: jnp.asarray(vv) for kk, vv in v.items()}
                  for k, v in json.loads((d / "params.json").read_text()).items()}
        arch = json.loads((d / "gru_config.json").read_text())["architecture"]
        meta = json.loads((d / "multisubject_metadata.json").read_text())
        index = json.loads((d / "subject_index_map.json").read_text())["subject_id_to_index"]
        cmap = json.loads((d / "session_context_map.json").read_text())
        assert cmap["indexing"] == "1_based"
        positions = {}
        for e in cmap["per_subject"]:
            if str(e["subject_id"]) in index and e["subject_index"] == index[str(e["subject_id"])]:
                positions[str(e["subject_id"])] = {s: i + 1 for i, s in enumerate(e["ordered_source_session_ids"])}
        n_subjects = int(meta["num_subjects"])
        cfg = resolve_session_conditioning_from_architecture(
            architecture=arch, metadata=meta, multisubject=True, max_n_subjects=n_subjects,
            subject_embedding_size=int(arch["subject_embedding_size"]), context="study 15 P2 GRU contexts")
        make = make_gru_network(
            hidden_size=int(arch["hidden_size"]), output_size=2, multisubject=True, max_n_subjects=n_subjects,
            subject_embedding_size=int(arch["subject_embedding_size"]),
            subject_embedding_init=str(arch.get("subject_embedding_init", "zeros")),
            session_encoding_type=str(cfg["session_encoding_type"]),
            session_integration_type=str(cfg["session_integration_type"]),
            session_fourier_k=int(cfg["session_fourier_k"]),
            session_delta_n_layers=int(cfg["session_delta_n_layers"]),
            session_delta_hidden_size=int(cfg["session_delta_hidden_size"]),
            session_max_index_by_subject_index=list(meta.get("session_max_index_by_subject_index") or []))
        subj = np.array([index[s["subject_id"]] for s in specs])
        sess = np.array([positions[s["subject_id"]][s["session_id"]] for s in specs])
        c = subject_context(make, params, subj, sess)
        step = GruStep.from_params(params)
        first = np.asarray(step.logit_diff(step.step(jnp.zeros((len(specs), step.hidden_size)),
                                                     -jnp.ones((len(specs), 2)), jnp.asarray(c))))
        p_first = 1.0 / (1.0 + np.exp(-first))
        csv = sorted((d).glob("test_trial_predictions.csv"))[0]
        df = pd.read_csv(csv, usecols=["subject_id", "source_ses_idx", "trial", "probability_choice_1"],
                         dtype={"subject_id": str})
        want = df.sort_values("trial").groupby("source_ses_idx").probability_choice_1.first()
        diff = np.abs(p_first - want.loc[[s["session_id"] for s in specs]].to_numpy())
        worst = max(worst, float(diff.max()) if np.all(np.isfinite(diff)) else np.inf)   # NaN fails
        out[f"seed{seed}"] = {"params": str(d / "params.json"), "params_sha256": _sha256(d / "params.json"),
                              "max_first_trial_diff": float(diff.max()),
                              "contexts": {s["session_id"]: [float(x) for x in ci] for s, ci in zip(specs, c)}}
        sources += [{"type": "file", "id": str(d / f), "sha256": _sha256(d / f)}
                    for f in ("params.json", "gru_config.json", "multisubject_metadata.json",
                              "subject_index_map.json", "session_context_map.json")]
        print(seed, "sessions", len(specs), "max first-trial diff", float(diff.max()), flush=True)

    import subprocess

    wrapper_sha = subprocess.run(["git", "-C", str(a.wrapper), "rev-parse", "HEAD"], capture_output=True,
                                 text=True, check=True).stdout.strip()
    meta = build_meta("analysis/freeze_p2_gru_contexts.py", [], study_root=HERE.parent, kind="freeze",
                      code=[__file__], inputs=[SPECS, GRU], sources=sources)
    write_artifact(OUT, {"_meta": meta, "wrapper_git_sha": wrapper_sha, "launch_id": LAUNCH_ID,
                         "tolerance_first_trial": TOL_FIRST, "max_first_trial_diff": worst,
                         "passed": worst <= TOL_FIRST, "n_sessions": len(specs), "seeds": out})
    print(f"{'PASS' if worst <= TOL_FIRST else 'FAIL'} (max {worst:.1e}) -> {OUT}")


if __name__ == "__main__":
    main()
