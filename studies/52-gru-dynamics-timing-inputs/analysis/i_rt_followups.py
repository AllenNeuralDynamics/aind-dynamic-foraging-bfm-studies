"""Tier 3 follow-ups (stage A): is RT a gain or a push, and does RT change how outcomes are used?

    PYTHONPATH=<wrapper>/code python analysis/i_rt_followups.py R-s0   # then R-s1; one process each

Same replay machinery as ``i_rt_mechanism.py`` (50k real states from RT-present sessions,
sampling seed 0), for the given seed of R. Eligibility here is τ ≥ 2 (the previous step must
carry a real RT), so the sampled states differ from i_rt_mechanism's. All logits are in switch coordinates:
s = logit toward the side NOT chosen on the trial whose outcome is being fed.

  F1  gain vs push: for each swept RT input v, fit s_v = a(v) + b(v)·s_0 across states (s_0 =
      the same state's switch-logit with RT input 0), per outcome. A pure push moves a(v) with
      b = 1; a pure gain (scaling the current preference) moves b(v) with a = 0. Also the raw
      Δs = s_v − s_0 binned by s_0 for two RT values.
  F2  same-input interaction: the effect of flipping that step's reward, as a function of that
      step's RT input v (both in the same input vector).
  F3  stored RT: the reward-flip effect binned by the RT input of the step before (drawn in
      ``fig_i_rt_followups.py`` with the current step's RT input held at 0).

Writes ``_cache/i_rt_followups_<key>.npz`` (per-state arrays) and ``i_rt_followups_<key>.json``;
``fig_i_rt_followups.py`` draws the figures from the npz files of both seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import i_rt_mechanism as mech  # noqa: E402
from _pk03 import CACHE, write_json  # noqa: E402

PRODUCER = "analysis/i_rt_followups.py"
SWEEP = mech.SWEEP


def main(key: str) -> None:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    split, step, obs, contexts = mech.build(key)
    t = mech.trial_index(split, key)
    cols, taus = t["col"].to_numpy(), t["tau"].to_numpy()
    present = t["rt_present"].to_numpy()
    rng = np.random.default_rng(mech.SEED)
    # τ ≥ 2 so the previous step's RT input is a real RT (row 0 is padding).
    eligible = np.flatnonzero(present & (taus >= 2) & (taus + mech.N_AHEAD < split.n_trials[cols]))
    rows = np.sort(rng.choice(eligible, mech.N_SAMPLE, replace=False))
    c_rows, t_rows = cols[rows], taus[rows]
    h_prev = mech.states_at(step, obs, contexts, c_rows, t_rows - 1)
    base = obs[t_rows, c_rows]
    prev_rt = obs[t_rows - 1, c_rows, 2]
    away = np.where(base[:, 0] == 1, -1.0, 1.0)
    h, c = jnp.asarray(h_prev), jnp.asarray(contexts[c_rows])

    @jax.jit
    def switch_logit(rt, reward):
        o = jnp.asarray(base).at[:, 2].set(rt).at[:, 1].set(reward)
        return step.logit_diff(step.step(h, o, c)) * jnp.asarray(away)

    reward = base[:, 1]
    flipped = 1.0 - reward
    s = np.stack([np.asarray(switch_logit(jnp.full(len(rows), v, jnp.float32), reward)) for v in SWEEP])
    s_flip = np.stack([np.asarray(switch_logit(jnp.full(len(rows), v, jnp.float32), flipped)) for v in SWEEP])
    s0 = np.asarray(switch_logit(jnp.zeros(len(rows), jnp.float32), reward))
    s_obs = np.asarray(switch_logit(jnp.asarray(base[:, 2]), reward))
    s_obs_flip = np.asarray(switch_logit(jnp.asarray(base[:, 2]), flipped))

    fits = {}
    for r in (0.0, 1.0):
        m = reward == r
        x = np.column_stack([np.ones(m.sum()), s0[m]])
        coef = np.linalg.lstsq(x, s[:, m].T, rcond=None)[0]  # [2, V]
        resid = s[:, m].T - x @ coef
        r2 = 1 - resid.var(0) / s[:, m].T.var(0)
        fits[str(int(r))] = {"a": coef[0], "b": coef[1], "r2": r2}

    np.savez_compressed(CACHE / f"i_rt_followups_{key}.npz", sweep=SWEEP, s=s.astype(np.float32),
                        s_flip=s_flip.astype(np.float32), s0=s0, s_obs=s_obs, s_obs_flip=s_obs_flip,
                        reward=reward, rt_input=base[:, 2], prev_rt_input=prev_rt)
    write_json(f"i_rt_followups_{key}.json", {
        "key": key, "n_states": int(len(rows)), "F1_fits_by_reward": fits,
        "median_s0_by_reward": {str(int(r)): float(np.median(s0[reward == r])) for r in (0.0, 1.0)},
        "fraction_states_favouring_switch": {str(int(r)): float(np.mean(s0[reward == r] > 0)) for r in (0.0, 1.0)},
    }, PRODUCER)
    print({r: {k: np.round(v[[0, 5, 10, 15, 20]], 3).tolist() for k, v in f.items()} for r, f in fits.items()})


if __name__ == "__main__":
    main(sys.argv[1])
