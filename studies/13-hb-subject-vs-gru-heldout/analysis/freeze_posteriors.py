#!/usr/bin/env python
"""Freeze per-subject posterior summaries of the `hattori` runs into ``analysis/hattori_posteriors.json``
(``make freeze-posteriors``).

Reads each finished run's saved draws (``heldout_two_level_draws.npz``: ``mu_p`` and ``log_sigma``,
6,000 posterior draws x 5 parameters, on the unconstrained scale) from the HPC artifact directory and
keeps, per subject and parameter:

- ``subject``: the subject's typical session, ``mu_p`` mapped to the parameter's own scale
  (quantiles over posterior draws);
- ``sigma``: the between-session spread on the unconstrained scale (quantiles over draws);
- ``session_width``: how far the parameter moves across this subject's sessions, on its own scale:
  for each posterior draw, the width between the 10th and 90th percentile session,
  ``f(mu_p + 1.2816 sigma) - f(mu_p - 1.2816 sigma)`` (exact, since every transform is monotone),
  summarized over draws. Conditional on each draw, so uncertainty about ``mu_p`` does not widen it.

The mapping is foraging-models' ``hattori2019_session_params`` at the commit the runs used, so it cannot
drift from the model. Each row is keyed by the W&B run id and the sha256 of the draws file it read.
Needs the HPC filesystem, so it runs on a compute node. ``r3_posteriors_vs_mle.py`` is offline.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

from aind_dynamic_foraging_models.hierarchical_bayes.model import (  # noqa: E402
    HATTORI2019_PARAMS,
    hattori2019_session_params,
)

WANDB_GROUPS = ["hattori@20261026-002131"]
RUNS = HERE / "hattori_runs.json"
ARTIFACT_ROOT = Path("/home/han.hou/outputs/disrnn/hb_artifacts")
OUT = HERE / "hattori_posteriors.json"
QUANTILES = (5, 50, 95)


def _bounded(theta: np.ndarray, beta_max: float) -> np.ndarray:
    """Unconstrained (..., 5) -> each parameter on its own scale, same layout."""
    params = hattori2019_session_params(theta, beta_max=beta_max)
    return np.stack([np.asarray(params[name]) for name in HATTORI2019_PARAMS], axis=-1)


Z90 = 1.2815515655446004   # standard-normal 90th percentile


def _summary(path: Path, beta_max: float) -> dict:
    """Posterior summaries for one subject from its draws file."""
    draws = np.load(path)
    mu = np.asarray(draws["mu_p"])[:, 0, :]                  # (draws, params)
    sigma = np.exp(np.asarray(draws["log_sigma"])[:, 0, :])
    width = _bounded(mu + Z90 * sigma, beta_max) - _bounded(mu - Z90 * sigma, beta_max)
    subject_q = np.percentile(_bounded(mu, beta_max), QUANTILES, axis=0)
    sigma_q = np.percentile(sigma, QUANTILES, axis=0)
    width_q = np.percentile(width, QUANTILES, axis=0)
    return {name: {"subject": subject_q[:, k].tolist(), "sigma": sigma_q[:, k].tolist(),
                   "session_width": width_q[:, k].tolist()}
            for k, name in enumerate(HATTORI2019_PARAMS)}


def main() -> None:
    """One row per finished run: its subject, draws digest, and posterior summaries."""
    rows, sources = [], []
    finished = [r for r in json.loads(RUNS.read_text())["rows"] if r["state"] == "finished"]
    for run in sorted(finished, key=lambda r: r["subject_id"]):
        run_dir = ARTIFACT_ROOT / run["wandb_run_id"]
        path = run_dir / "heldout_two_level_draws.npz"
        fit_json = run_dir / "heldout_two_level_fit.json"   # beta_max sets the temperature scale
        beta_max = json.loads(fit_json.read_text())["beta_max"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        fit_digest = hashlib.sha256(fit_json.read_bytes()).hexdigest()
        rows.append({"wandb_run_id": run["wandb_run_id"], "subject_id": run["subject_id"],
                     "draws_sha256": digest, "fit_json_sha256": fit_digest, "beta_max": beta_max,
                     "params": _summary(path, beta_max)})
        sources += [{"type": "file", "id": str(path), "sha256": digest},
                    {"type": "file", "id": str(fit_json), "sha256": fit_digest}]
    meta = build_meta("analysis/freeze_posteriors.py", WANDB_GROUPS, study_root=HERE.parent,
                      kind="freeze", code=[__file__], inputs=[RUNS], sources=sources)
    write_artifact(OUT, {"_meta": meta, "quantiles": {"subject": QUANTILES, "sigma": QUANTILES,
                                                      "session_width": QUANTILES},
                         "param_order": list(HATTORI2019_PARAMS), "rows": rows}, outputs=[])
    print(f"froze posteriors for {len(rows)} subjects to {OUT}")


if __name__ == "__main__":
    main()
