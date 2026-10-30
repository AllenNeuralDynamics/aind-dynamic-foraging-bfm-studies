"""Licks-model follow-ups (stage A): where L reads the outcome, and how other-side licks act.

    PYTHONPATH=<wrapper>/code python analysis/i_licks_followups.py L-s0   # then L-s1

Same machinery and state sample as ``i_licks_mechanism.py`` (50k real states, sessions with
licks, sampling seed 0). Switch coordinates: s = logit toward the side NOT chosen on the fed trial.
"Typical licks" for an outcome = (median chosen-side count after that outcome, 0 other-side).

  FL1  decomposing the outcome: change one step's outcome by (a) flipping the reward input only
       (licks as observed — an input combination the data rarely contain), (b) setting the licks
       to the other outcome's typical pattern with the reward input unchanged, (c) both together
       (a consistent flip). Δ s on the next choice, relative to the same state with the step's
       licks set to its OWN outcome's typical pattern (so (a)–(c) are compared from one baseline).
  FL2  does other-side licking change how the outcome counts? Effect of a consistent outcome flip
       as a function of the step's other-side count (chosen side at typical).
  FL3  push vs gain for other-side licks: s_k = a(k) + b(k)·s_0 across states for k other-side
       licks (chosen side at typical), per outcome.

Writes ``_cache/i_licks_followups_<key>.npz`` and ``i_licks_followups_<key>.json``;
``fig_i_licks_followups.py`` draws both seeds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import i_rt_mechanism as mech  # noqa: E402
from i_licks_mechanism import N_AHEAD, N_SAMPLE, SEED, enc, side_columns, trial_index  # noqa: E402
from _pk03 import CACHE, write_json  # noqa: E402
import _pk03  # noqa: E402

PRODUCER = "analysis/i_licks_followups.py"
OTHER_K = np.array([0, 1, 2, 3, 4, 6, 8])


def main(key: str) -> None:
    _pk03.use_keys(key)
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    split, step, obs, contexts = mech.build(key, n_obs=4)
    t = trial_index(split, key)
    cols, taus = t["col"].to_numpy(), t["tau"].to_numpy()
    present = t["licks_present"].to_numpy()
    action = t["action"].to_numpy()
    chosen_count = np.where(action == 1, t["lick_right_count"], t["lick_left_count"])
    typical = {r: float(np.median(chosen_count[(t["reward"].to_numpy() == r) & present])) for r in (0, 1)}
    rng = np.random.default_rng(SEED)
    eligible = np.flatnonzero(present & (taus >= 1) & (taus + N_AHEAD < split.n_trials[cols]))
    rows = np.sort(rng.choice(eligible, N_SAMPLE, replace=False))
    c_rows, t_rows = cols[rows], taus[rows]
    h = jnp.asarray(mech.states_at(step, obs, contexts, c_rows, t_rows - 1))
    c = jnp.asarray(contexts[c_rows])
    base = obs[t_rows, c_rows]
    choice, reward = base[:, 0], base[:, 1]
    ch_col, ot_col = side_columns(choice)
    away = np.where(choice == 1, -1.0, 1.0)
    idx = np.arange(len(rows))

    @jax.jit
    def s_of(o):
        return step.logit_diff(step.step(h, o, c)) * jnp.asarray(away)

    def inputs(rew, chosen, other):
        o = base.copy()
        o[:, 1] = rew
        o[idx, ch_col] = enc(chosen)
        o[idx, ot_col] = enc(other)
        return jnp.asarray(o)

    own = np.where(reward == 1, typical[1], typical[0])
    flipped_typ = np.where(reward == 1, typical[0], typical[1])
    zero = np.zeros(len(rows))
    s_own = np.asarray(s_of(inputs(reward, own, zero)))
    fl1 = {
        "reward only": np.asarray(s_of(inputs(1 - reward, own, zero))) - s_own,
        "licks only": np.asarray(s_of(inputs(reward, flipped_typ, zero))) - s_own,
        "both (consistent flip)": np.asarray(s_of(inputs(1 - reward, flipped_typ, zero))) - s_own,
    }
    # FL2 / FL3: other-side count k with chosen side at its typical value.
    s_k = np.stack([np.asarray(s_of(inputs(reward, own, np.full(len(rows), k)))) for k in OTHER_K])
    s_k_flip = np.stack([np.asarray(s_of(inputs(1 - reward, flipped_typ, np.full(len(rows), k)))) for k in OTHER_K])
    fits = {}
    for r in (0.0, 1.0):
        m = reward == r
        x = np.column_stack([np.ones(m.sum()), s_k[0, m]])
        coef = np.linalg.lstsq(x, s_k[:, m].T, rcond=None)[0]
        resid = s_k[:, m].T - x @ coef
        fits[str(int(r))] = {"a": coef[0], "b": coef[1], "r2": 1 - resid.var(0) / s_k[:, m].T.var(0)}
    np.savez_compressed(CACHE / f"i_licks_followups_{key}.npz", other_k=OTHER_K, reward=reward,
                        s_own=s_own, s_k=s_k.astype(np.float32), s_k_flip=s_k_flip.astype(np.float32),
                        **{f"fl1_{n.split()[0]}": v for n, v in fl1.items()})
    summary = {r: {n: {"median": float(np.median(v[reward == float(r)])), "mean": float(np.mean(v[reward == float(r)]))}
                   for n, v in fl1.items()} for r in (0, 1)}
    write_json(f"i_licks_followups_{key}.json", {
        "key": key, "typical_chosen_count": typical, "n_states": int(len(rows)),
        "FL1_by_original_reward": summary, "FL3_fits_by_reward": fits,
        "fraction_states_favouring_switch_s0": {str(int(r)): float(np.mean(s_k[0, reward == r] > 0)) for r in (0.0, 1.0)},
    }, PRODUCER)
    print(json.dumps(summary, indent=1))
    print({r: {k: np.round(v[[0, 2, 4, 6]], 3).tolist() for k, v in f.items()} for r, f in fits.items()})


if __name__ == "__main__":
    main(sys.argv[1])
