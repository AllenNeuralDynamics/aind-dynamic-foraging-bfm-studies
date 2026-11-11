#!/usr/bin/env python
"""Freeze the HB plug-in control into ``analysis/hattori_plugin.json`` (``make freeze-plugin``).

The plug-in scores each held-out subject's eval sessions with ONE fixed parameter set, the posterior
median of its typical session (``mu_p``, per parameter), run through each session by teacher forcing.
It is the HB's fit with no between-session spread and no within-session inference: the same
predictor shape as the per-subject MLE, but with HB's prior and partial pooling across sessions. So:

- plug-in minus MLE = how the parameters were fitted (prior, pooling);
- zero-spread minus plug-in = averaging over posterior uncertainty in ``mu_p`` (the zero-spread
  control keeps every posterior draw of ``mu_p``, sets the between-session spread to zero, and scores
  each session by the session-marginal over those draws);
- session-marginal minus zero-spread = modelling between-session spread, and inferring within each
  session where it sits.

The eval sessions are reloaded through the wrapper's own data loader and split helpers (the same path
``HBTrainer`` used), from the pinned code worktree the production runs used, and every subject's
scored-trial count is checked against the production run's. Log probabilities are exact (log-softmax of
the logits), as in the session-marginal scorer. Needs the HPC filesystem and the data snapshot, so it
runs on a compute node.
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
sys.path.insert(0, str(HERE))
import final_runs  # noqa: E402

# The code the production runs used (launch_record/production/hpc_sweep.json).
CODE_ROOT = Path("/home/han.hou/code/wt/hpc13b")
sys.path.insert(0, str(CODE_ROOT / "aind-dynamic-foraging-bfm-wrapper" / "code"))

WANDB_GROUPS = final_runs.PRODUCTION_GROUPS + final_runs.FINAL_GROUPS
RUNS = final_runs.FINAL_RUNS          # the final set (ADR 0004); draws via final_runs.draws_run
LAUNCH = HERE.parent / "variants" / "hattori" / "launch_record" / "production" / "hpc_sweep.json"
ARTIFACT_ROOT = Path("/home/han.hou/outputs/disrnn/hb_artifacts")
OUT = HERE / "hattori_plugin.json"
EVAL_EVERY_N = 2


def _heldout_frame():
    """The held-out cohort's trials, through the wrapper's own loader and data config."""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    config_dir = CODE_ROOT / "aind-dynamic-foraging-bfm-dispatcher" / "code" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config", overrides=["data=mice_snapshot_scaling",
                                                       "data.subject_ratio=0.016"])
    assert str(cfg.data.snapshot) == "20260603", cfg.data.snapshot
    return instantiate(cfg.data, split="heldout", multisubject=True).load().raw


def _per_draw_log_lik(choices, rewards, thetas, beta_max):
    """Exact summed log probability of one session under each of many fixed parameter sets."""
    import jax
    import jax.numpy as jnp
    from aind_dynamic_foraging_models.hierarchical_bayes.likelihood import hattori2019_value_trajectory
    from aind_dynamic_foraging_models.hierarchical_bayes.model import hattori2019_session_params

    choices_j, rewards_j = jnp.asarray(choices), jnp.asarray(rewards, dtype=jnp.float32)

    def one(theta):
        """One parameter set's summed log probability."""
        p = hattori2019_session_params(theta, beta_max=beta_max)
        q, _ = hattori2019_value_trajectory(choices_j, rewards_j, **{k: p[k] for k in p})
        logits = p["softmax_inverse_temperature"] * q.T + jnp.array([p["bias_l"], 0.0])
        log_p = jax.nn.log_softmax(logits, axis=-1)[jnp.arange(len(choices)), choices_j]
        return jnp.sum(log_p)

    return np.asarray(jax.jit(jax.vmap(one))(jnp.asarray(thetas)), dtype=np.float64)


def _log_lik(choices, rewards, theta, beta_max) -> float:
    """Exact summed log probability of one session's choices under one fixed parameter set."""
    import jax
    import jax.numpy as jnp
    from aind_dynamic_foraging_models.hierarchical_bayes.likelihood import hattori2019_value_trajectory
    from aind_dynamic_foraging_models.hierarchical_bayes.model import hattori2019_session_params

    p = hattori2019_session_params(jnp.asarray(theta), beta_max=beta_max)
    q, _ = hattori2019_value_trajectory(choices, rewards, **{k: p[k] for k in p})
    logits = p["softmax_inverse_temperature"] * q.T + jnp.array([p["bias_l"], 0.0])
    log_p = jax.nn.log_softmax(logits, axis=-1)[jnp.arange(len(choices)), jnp.asarray(choices)]
    return float(jnp.sum(log_p))


def main() -> None:
    """One row per subject: plug-in summed log likelihood and trials on its eval sessions."""
    from model_trainers.hb_trainer import _extract_subject_sessions
    from utils.multisubject import compute_train_eval_session_ids

    choices, rewards, session_ids = _extract_subject_sessions(_heldout_frame())
    runs = {r["subject_id"]: r for r in final_runs.final_rows()}
    rows, sources = [], []
    for subject in sorted(runs):
        key = next(k for k in session_ids if str(k) == subject)
        _, eval_ids = compute_train_eval_session_ids(session_ids[key], EVAL_EVERY_N)
        index_of = {sid: i for i, sid in enumerate(session_ids[key])}
        draws_run = final_runs.draws_run(runs[subject])
        run_dir = ARTIFACT_ROOT / draws_run
        draws_path = run_dir / "heldout_two_level_draws.npz"
        fit_json = run_dir / "heldout_two_level_fit.json"   # beta_max sets the temperature scale
        beta_max = json.loads(fit_json.read_text())["beta_max"]
        mu_draws = np.asarray(np.load(draws_path)["mu_p"])[:, 0, :]
        theta = np.median(mu_draws, axis=0)
        total, zero_spread, trials = 0.0, 0.0, 0
        for sid in eval_ids:
            c, r = choices[key][index_of[sid]], rewards[key][index_of[sid]]
            total += _log_lik(c, r, theta, beta_max)
            per_draw = _per_draw_log_lik(c, r, mu_draws, beta_max)
            top = per_draw.max()
            zero_spread += float(top + np.log(np.mean(np.exp(per_draw - top))))
            trials += len(c)
        assert trials == runs[subject]["heldout/num_eval_trials"], (subject, trials)
        digest = hashlib.sha256(draws_path.read_bytes()).hexdigest()
        fit_digest = hashlib.sha256(fit_json.read_bytes()).hexdigest()
        rows.append({"subject_id": subject, "wandb_run_id": runs[subject]["wandb_run_id"],
                     "draws_run_id": draws_run,
                     "draws_sha256": digest, "fit_json_sha256": fit_digest, "beta_max": beta_max,
                     "theta_unconstrained": theta.tolist(),
                     "plugin_log_lik_sum": total, "zero_spread_log_lik_sum": zero_spread,
                     "n_trials": trials})
        sources += [{"type": "file", "id": str(draws_path), "sha256": digest},
                    {"type": "file", "id": str(fit_json), "sha256": fit_digest}]
    data = json.loads(LAUNCH.read_text())["data"]
    sources.append({"type": "url", "id": data["uri"], "sha256": data["manifest_sha256"]})
    meta = build_meta("analysis/freeze_plugin.py", WANDB_GROUPS, study_root=HERE.parent,
                      kind="freeze", code=[__file__, final_runs.__file__],
                      inputs=[RUNS, final_runs.PRODUCTION_RUNS, LAUNCH], sources=sources)
    write_artifact(OUT, {"_meta": meta, "theta": "posterior median of mu_p per parameter",
                         "rows": rows}, outputs=[])
    print(f"froze plug-in scores for {len(rows)} subjects to {OUT}")


if __name__ == "__main__":
    main()
