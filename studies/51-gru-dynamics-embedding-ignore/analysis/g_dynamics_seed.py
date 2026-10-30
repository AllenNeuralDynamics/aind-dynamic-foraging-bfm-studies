"""G1 (stage A): the D-phase dynamics of one C3 seed — ignore attractor and relaxation.

    python analysis/g_dynamics_seed.py C3-s1

Loads the seed's split (data bundle, ~17 GB) for its session contexts, then in float64:
  * the attractor of each constant input at the mean context, by N_ITER forward iterations
    from the mean engaged state (side logit, P(ignore), last-step change);
  * relaxation from N_RELAX real engaged states, each with its own session context: |side|
    as a fraction of its start after k ignores, and mean P(ignore) after k ignores.
Merges the seed's entry into ``g_dynamics.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk02 as s13  # noqa: E402

INPUTS = {"R+": (1.0, 1.0), "R−": (1.0, 0.0), "L+": (0.0, 1.0), "L−": (0.0, 0.0), "ignore": (2.0, 0.0)}
N_ITER = 3000
N_RELAX = 5000
K_SHOWN = (1, 2, 3, 10)
SEED = 0


def main(key: str) -> None:
    import jax  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp  # noqa: PLC0415

    from post_training_analysis.gru_counterfactual import load_gru_split  # noqa: PLC0415
    from post_training_analysis.gru_step import GruStep, split_inputs, subject_context  # noqa: PLC0415

    split = load_gru_split(s13.CACHE / "models" / key / "run")
    step = GruStep.from_params(split.params)
    w, b = np.asarray(step.readout_w, np.float64), np.asarray(step.readout_b, np.float64)

    def readouts(h):
        z = np.asarray(h, np.float64) @ w + b
        return z[..., 1] - z[..., 0], np.logaddexp(z[..., 0], z[..., 1]) - z[..., 2]

    subject_idx, session_idx, _ = split_inputs(split.xs, session_conditioning=split.run.session_conditioning_enabled)
    contexts = subject_context(split.make_network, split.params, subject_idx[0], session_idx[0]).astype(np.float64)
    trials = s13.load_trials(["ses_idx", "action", *s13.STATE_COLUMNS], key=key)
    col = trials["ses_idx"].map({s: i for i, s in enumerate(split.session_ids)}).to_numpy()
    engaged = np.flatnonzero(trials["action"].to_numpy() != s13.IGNORE)
    states = trials[s13.STATE_COLUMNS].to_numpy(np.float64)
    h_mean = states[engaged].mean(axis=0)
    c_mean = jnp.asarray(contexts.mean(axis=0))

    attractors = {}
    for name, u in INPUTS.items():
        u = jnp.asarray(u)
        h = jax.lax.fori_loop(0, N_ITER, lambda _, x: step.step(x, u, c_mean), jnp.asarray(h_mean))
        change = float(np.abs(np.asarray(step.step(h, u, c_mean)) - np.asarray(h)).max())
        s, e = readouts(h)
        attractors[name] = {"side_logit": float(s), "p_ignore": float(1 / (1 + np.exp(e))), "last_step_change": change}

    rows = np.random.default_rng(SEED).choice(engaged, size=N_RELAX, replace=False)
    h0 = jnp.asarray(states[rows])
    c_rows = jnp.asarray(contexts[col[rows]])
    u = jnp.broadcast_to(jnp.asarray(INPUTS["ignore"]), (N_RELAX, 2))
    s0, _ = readouts(h0)
    h, frac, p_ign = h0, {}, {}
    for k in range(1, max(K_SHOWN) + 1):
        h = step.step(h, u, c_rows)
        if k in K_SHOWN:
            s, e = readouts(h)
            frac[str(k)] = float(np.median(np.abs(s) / np.abs(s0)))
            p_ign[str(k)] = float(np.mean(1 / (1 + np.exp(e))))

    path = s13.ANALYSIS / "g_dynamics.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    seeds = existing.get("seeds", {})
    seeds[key] = {"attractors_mean_context": attractors, "relaxation_side_fraction_median": frac,
                  "relaxation_p_ignore_mean": p_ign}
    s13.write_json("g_dynamics.json", {
        "seeds": dict(sorted(seeds.items())), "n_iter": N_ITER, "n_relax": N_RELAX, "precision": "float64, highest",
    }, "analysis/g_dynamics_seed.py")
    print(key, {n: (round(a["side_logit"], 2), round(a["p_ignore"], 3), f"{a['last_step_change']:.0e}")
                for n, a in attractors.items()}, frac, p_ign)


if __name__ == "__main__":
    main(sys.argv[1])
