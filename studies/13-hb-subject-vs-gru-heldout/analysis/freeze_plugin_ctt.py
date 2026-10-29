#!/usr/bin/env python
"""Freeze the HB-CTT plug-in and zero-spread controls into ``analysis/ctt_plugin.json``
(``make freeze-plugin-ctt``).

The same two controls as ``freeze_plugin.py`` (read its docstring for what each step isolates), for
the ``ctt`` variant's final fits:

- plug-in: ONE fixed parameter set per subject, the posterior median of its typical session
  (``mu_p``), run through each eval session by teacher forcing;
- zero-spread: every posterior draw of ``mu_p``, no between-session spread, each session scored by
  the session-marginal over those draws.

The eval sessions come through the wrapper's own loader, reused from ``freeze_plugin.py``, and each
subject's scored-trial count is checked against its final run. Every ctt run saved its own draws.
The likelihood is foraging-models' CTT (``get_model("ctt")``): exact log-softmax of its logits, as
in the session-marginal scorer. Needs the HPC filesystem and the data snapshot, so it runs on a
compute node.
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
import freeze_plugin as fp  # noqa: E402  (data loader and its pinned code root)

WANDB_GROUPS = ["ctt@20261028-003745", "ctt@20261028-020002"]
RUNS = HERE / "ctt_final_runs.json"
LAUNCH = HERE.parent / "variants" / "ctt" / "launch_record" / "production" / "hpc_sweep.json"
OUT = HERE / "ctt_plugin.json"
MODEL = "ctt"


def _per_draw_log_lik(choices, rewards, thetas, beta_max):
    """Exact summed log probability of one session under each of many fixed parameter sets."""
    import jax
    import jax.numpy as jnp
    from aind_dynamic_foraging_models.hierarchical_bayes.model import get_model

    spec = get_model(MODEL)
    choices_j, rewards_j = jnp.asarray(choices), jnp.asarray(rewards, dtype=jnp.float32)

    def one(theta):
        """One parameter set's summed log probability."""
        logits = spec.logits(spec.session_params(theta, beta_max=beta_max), choices_j, rewards_j)
        return jnp.sum(jax.nn.log_softmax(logits, axis=-1)[jnp.arange(len(choices)), choices_j])

    return np.asarray(jax.jit(jax.vmap(one))(jnp.asarray(thetas)), dtype=np.float64)


def main() -> None:
    """One row per subject: plug-in and zero-spread summed log likelihoods on its eval sessions."""
    from model_trainers.hb_trainer import _extract_subject_sessions
    from utils.multisubject import compute_train_eval_session_ids

    choices, rewards, session_ids = _extract_subject_sessions(fp._heldout_frame())
    runs = {r["subject_id"]: r for r in json.loads(RUNS.read_text())["rows"]
            if r["state"] == "finished"}
    assert len(runs) == 149, len(runs)
    rows, sources = [], []
    for subject in sorted(runs):
        key = next(k for k in session_ids if str(k) == subject)
        _, eval_ids = compute_train_eval_session_ids(session_ids[key], fp.EVAL_EVERY_N)
        index_of = {sid: i for i, sid in enumerate(session_ids[key])}
        run_dir = fp.ARTIFACT_ROOT / runs[subject]["wandb_run_id"]
        draws_path = run_dir / "heldout_two_level_draws.npz"
        fit_json = run_dir / "heldout_two_level_fit.json"
        fit = json.loads(fit_json.read_text())
        assert fit.get("hb_model") == MODEL, (subject, fit.get("hb_model"))
        beta_max = fit["beta_max"]
        mu_draws = np.asarray(np.load(draws_path)["mu_p"])[:, 0, :]
        theta = np.median(mu_draws, axis=0)
        total, zero_spread, trials = 0.0, 0.0, 0
        for sid in eval_ids:
            c, r = choices[key][index_of[sid]], rewards[key][index_of[sid]]
            total += float(_per_draw_log_lik(c, r, theta[None], beta_max)[0])
            per_draw = _per_draw_log_lik(c, r, mu_draws, beta_max)
            top = per_draw.max()
            zero_spread += float(top + np.log(np.mean(np.exp(per_draw - top))))
            trials += len(c)
        assert trials == runs[subject]["heldout/num_eval_trials"], (subject, trials)
        digest = hashlib.sha256(draws_path.read_bytes()).hexdigest()
        fit_digest = hashlib.sha256(fit_json.read_bytes()).hexdigest()
        rows.append({"subject_id": subject, "wandb_run_id": runs[subject]["wandb_run_id"],
                     "draws_sha256": digest, "fit_json_sha256": fit_digest, "beta_max": beta_max,
                     "theta_unconstrained": theta.tolist(),
                     "plugin_log_lik_sum": total, "zero_spread_log_lik_sum": zero_spread,
                     "n_trials": trials})
        sources += [{"type": "file", "id": str(draws_path), "sha256": digest},
                    {"type": "file", "id": str(fit_json), "sha256": fit_digest}]
    data = json.loads(LAUNCH.read_text())["data"]
    sources.append({"type": "url", "id": data["uri"], "sha256": data["manifest_sha256"]})
    meta = build_meta("analysis/freeze_plugin_ctt.py", WANDB_GROUPS, study_root=HERE.parent,
                      kind="freeze", code=[__file__, fp.__file__], inputs=[RUNS, LAUNCH],
                      sources=sources)
    write_artifact(OUT, {"_meta": meta, "theta": "posterior median of mu_p per parameter",
                         "rows": rows}, outputs=[])
    print(f"froze ctt plug-in scores for {len(rows)} subjects to {OUT}")


if __name__ == "__main__":
    main()
