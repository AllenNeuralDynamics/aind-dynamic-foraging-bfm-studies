"""Licks model mechanism (stage A): what one trial's licking does in L's dynamics.

    PYTHONPATH=<wrapper>/code python analysis/i_licks_mechanism.py [L-s1]   # default L-s0

Replays L with the wrapper's ``GruStep`` (via ``i_rt_mechanism.build``); the replay must
reproduce the logged likelihood and the lick columns of ``xs`` must equal the A1 table's
``prev_lick_*_input``. Inputs are [choice, reward, left licks, right licks]; lick inputs are
(count − 3.139) / 4.159. Licks are re-expressed relative to the fed trial's choice: chosen-side
and other-side counts. "Toward switching" = toward the side not chosen on the fed trial.
Sessions with no licks recorded at all are excluded from sampling and scoring.

  I-L1  2-D update rule: at sampled real states, one step's (chosen, other) counts are set on a
        grid; Δ switch-logit relative to a typical pattern for that outcome (the median chosen
        count after that outcome, 0 other-side licks), median over states. Plus 1-D slices.
  C1-L  impulse: one step's other-side licks +2, chosen-side licks ±3 (floored at 0), or its
        reward flipped; the network runs on 10 more steps with observed inputs.
  I-L2  clamps: full replays with the lick inputs of every non-padding step replaced by
        (a) their mean for that step's choice × reward ("cell mean": removes what licks carry
        beyond choice/reward), (b) the same for the other side only, (c) the chosen side only,
        (d) 0 (population mean), (e) a joint within-session permutation. Scored against C on
        the same trials.

Writes ``fig_I_L1_update_rule<sfx>.png``, ``fig_C1_L_impulse<sfx>.png``, ``fig_I_L2_clamps<sfx>.png``
and ``i_licks_mechanism<sfx>.json`` (sfx empty for L-s0, else ``_<key>``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import i_rt_mechanism as mech  # noqa: E402
from _pk03 import (  # noqa: E402
    ANALYSIS, BASELINE_KEY, CACHE, REWARDED_COLOR, STUDY12, UNREWARDED_COLOR, apply_presentation_style,
    figure_path, write_json,
)
import _pk03  # noqa: E402

PRODUCER = "analysis/i_licks_mechanism.py"
DEFAULT_KEY = "L-s0"
LICK_CENTER, LICK_SCALE = 3.139, 4.159
LEFT, RIGHT = 2, 3
CHOSEN_GRID = np.array([0, 1, 2, 3, 4, 6, 8, 10, 12, 16])
OTHER_GRID = np.array([0, 1, 2, 3, 4, 6, 8])
N_SAMPLE, N_AHEAD, SEED, BATCH = 50_000, 10, 0, 512
LIKELIHOOD_TOLERANCE = 2e-4


def enc(count):
    return (np.asarray(count, dtype=np.float32) - LICK_CENTER) / LICK_SCALE


def trial_index(split, key):
    cols = ["ses_idx", "trial_idx", "subject_id", "action", "reward", "lick_left_count", "lick_right_count",
            "prev_lick_left_input", "prev_lick_right_input"]
    t = pd.read_parquet(CACHE / f"trial_states_{key}.parquet", columns=cols)
    t["col"] = t["ses_idx"].map({s: i for i, s in enumerate(split.session_ids)}).to_numpy()
    t["tau"] = t["trial_idx"].to_numpy() - 1
    t["licks_present"] = (t["lick_left_count"] + t["lick_right_count"]).groupby(t["ses_idx"]).transform("sum") > 0
    c = pd.read_parquet(STUDY12 / "analysis" / "_cache" / f"trial_states_{BASELINE_KEY}.parquet",
                        columns=["ses_idx", "trial_idx", "p_model1_right"])
    return t.merge(c.rename(columns={"p_model1_right": "p_right_C"}), on=["ses_idx", "trial_idx"], validate="1:1")


def side_columns(choice):
    """Input-column index of the chosen and the other side for a fed choice (0 left, 1 right)."""
    chosen = np.where(choice == 1, RIGHT, LEFT)
    return chosen, np.where(choice == 1, LEFT, RIGHT)


def main(key: str = DEFAULT_KEY) -> None:
    _pk03.use_keys(key)
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    apply_presentation_style()
    sfx = "" if key == DEFAULT_KEY else f"_{key}"
    split, step, obs, contexts = mech.build(key, n_obs=4)
    t = trial_index(split, key)
    cols, taus = t["col"].to_numpy(), t["tau"].to_numpy()
    if not (taus < split.n_trials[cols]).all():
        raise AssertionError("Trial table rows fall outside the split's timesteps")
    err = max(float(np.max(np.abs(obs[taus, cols, LEFT] - t["prev_lick_left_input"].to_numpy()))),
              float(np.max(np.abs(obs[taus, cols, RIGHT] - t["prev_lick_right_input"].to_numpy()))))
    if err > 1e-5:
        raise AssertionError(f"xs lick columns differ from the A1 table by {err}")
    action = t["action"].to_numpy()
    present = t["licks_present"].to_numpy()
    intact = mech.replay(step, obs, contexts)[taus, cols]
    logged = _pk03.read_json(ANALYSIS / f"trial_states_{key}.json")["checks"]["logged_eval_likelihood"]
    checks = {"replay_likelihood": mech.likelihood(intact, action), "logged_likelihood": logged, "lick_input_max_abs_error": err}
    if abs(checks["replay_likelihood"] - logged) > LIKELIHOOD_TOLERANCE:
        raise AssertionError(f"Replay does not reproduce the logged likelihood: {checks}")
    print(checks, flush=True)

    # Typical chosen-side count after each outcome (fed trial = row tau−1 of the table).
    chosen_count = np.where(action == 1, t["lick_right_count"], t["lick_left_count"])
    ref_chosen = {r: float(np.median(chosen_count[(t["reward"].to_numpy() == r) & present])) for r in (0, 1)}

    rng = np.random.default_rng(SEED)
    eligible = np.flatnonzero(present & (taus >= 1) & (taus + N_AHEAD < split.n_trials[cols]))
    rows = np.sort(rng.choice(eligible, N_SAMPLE, replace=False))
    c_rows, t_rows = cols[rows], taus[rows]
    h_prev = mech.states_at(step, obs, contexts, c_rows, t_rows - 1)
    base = obs[t_rows, c_rows]
    choice, reward = base[:, 0], base[:, 1]
    ch_col, ot_col = side_columns(choice)
    away = np.where(choice == 1, -1.0, 1.0)
    h, c = jnp.asarray(h_prev), jnp.asarray(contexts[c_rows])
    idx = np.arange(len(rows))

    @jax.jit
    def switch_logit(o):
        return step.logit_diff(step.step(h, o, c)) * jnp.asarray(away)

    def with_licks(chosen, other):
        o = base.copy()
        o[idx, ch_col] = enc(chosen)
        o[idx, ot_col] = enc(other)
        return jnp.asarray(o)

    ref = np.where(reward == 1, ref_chosen[1], ref_chosen[0])
    s_ref = np.asarray(switch_logit(with_licks(ref, np.zeros(len(rows)))))
    grid = np.empty((2, len(CHOSEN_GRID), len(OTHER_GRID)))
    for i, ch in enumerate(CHOSEN_GRID):
        for j, ot in enumerate(OTHER_GRID):
            d = np.asarray(switch_logit(with_licks(np.full(len(rows), ch), np.full(len(rows), ot)))) - s_ref
            for k, r in enumerate((0.0, 1.0)):
                grid[k, i, j] = np.median(d[reward == r])

    fig, axes = plt.subplots(1, 3, figsize=(21, 5.6), gridspec_kw={"width_ratios": [1, 1, 1.1], "wspace": 0.35})
    vmax = np.max(np.abs(grid))
    for k, (r, name) in enumerate(((0.0, "after a loss"), (1.0, "after a win"))):
        im = axes[k].imshow(grid[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")
        axes[k].set_xticks(range(len(OTHER_GRID)), OTHER_GRID)
        axes[k].set_yticks(range(len(CHOSEN_GRID)), CHOSEN_GRID)
        axes[k].set_xlabel("other-side licks on the fed trial")
        axes[k].set_ylabel("chosen-side licks")
        axes[k].set_title(f"{name} (ref: {ref_chosen[int(r)]:.0f} chosen, 0 other)")
        for i in range(len(CHOSEN_GRID)):
            for j in range(len(OTHER_GRID)):
                axes[k].text(j, i, f"{grid[k, i, j]:+.1f}", ha="center", va="center", fontsize=8,
                             color="white" if abs(grid[k, i, j]) > 0.6 * vmax else "#222222")
    cb = fig.colorbar(im, ax=axes[:2], shrink=0.8, pad=0.02)
    cb.set_label("Δ logit toward switching (median over states)")
    for k, (r, color, ls, name) in enumerate(((0.0, UNREWARDED_COLOR, "--", "after a loss"), (1.0, REWARDED_COLOR, "-", "after a win"))):
        i_ref = int(np.argmin(np.abs(CHOSEN_GRID - ref_chosen[int(r)])))
        axes[2].plot(OTHER_GRID, grid[k, i_ref, :], color=color, ls=ls, lw=2, marker="o", ms=5, label=f"other side, {name}")
        axes[2].plot(CHOSEN_GRID, grid[k, :, 0], color=color, ls=ls, lw=1.2, marker="s", ms=4, alpha=0.7,
                     label=f"chosen side (other = 0), {name}")
    axes[2].axhline(0, color="#888888", lw=1)
    axes[2].set_xlabel("lick count on the fed trial")
    axes[2].set_ylabel("Δ logit toward switching")
    axes[2].set_title("slices")
    axes[2].legend(frameon=False, fontsize=10)
    fig.suptitle(f"I-L1  How one trial's licking moves L's next choice ({key}; {N_SAMPLE:,} real states)", y=1.03)
    fig.savefig(figure_path(f"fig_I_L1_update_rule{sfx}.png"), bbox_inches="tight")
    plt.close(fig)

    # C1-L impulse
    window = np.stack([obs[t_rows + k, c_rows] for k in range(N_AHEAD + 1)])
    cw = np.broadcast_to(contexts[c_rows][None], window.shape[:2] + (contexts.shape[1],))
    run = jax.jit(lambda o: step.logit_diff(step.run(jnp.asarray(o), jnp.asarray(cw), h0=h)))
    baseline = np.asarray(run(window))
    floor = float(enc(0))

    def edited(kind):
        w = window.copy()
        first = w[0]
        if kind == "other +2":
            first[idx, ot_col] += 2 / LICK_SCALE
        elif kind in ("chosen +3", "chosen −3"):
            delta = 3 / LICK_SCALE * (1 if kind == "chosen +3" else -1)
            first[idx, ch_col] = np.maximum(first[idx, ch_col] + delta, floor)
        elif kind == "reward flipped":
            first[:, 1] = 1 - first[:, 1]
        return w

    impulse = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    lags = np.arange(N_AHEAD + 1)
    styles = {"other +2": ("#7B3294", "-"), "chosen +3": ("#C44E52", "-"), "chosen −3": ("#4C72B0", "-"),
              "reward flipped": ("#222222", "--")}
    effects = {kind: (np.asarray(run(edited(kind))) - baseline) * away[None] for kind in styles}
    for ax, (r, name) in zip(axes, ((1.0, "pulsed trial rewarded"), (0.0, "pulsed trial unrewarded"))):
        m = reward == r
        for kind, (color, ls) in styles.items():
            mean = effects[kind][:, m].mean(axis=1)
            sem = effects[kind][:, m].std(axis=1) / np.sqrt(m.sum())
            impulse[f"{kind} | {name}"] = {"mean": mean, "sem": sem}
            ax.plot(lags, mean, color=color, ls=ls, lw=2, marker="o", ms=4, label=kind)
            ax.fill_between(lags, mean - 2 * sem, mean + 2 * sem, color=color, alpha=0.15, lw=0)
        ax.axhline(0, color="#888888", lw=1)
        ax.set_xticks(lags[::2])
        ax.set_xlabel("trials after the pulse (0 = next choice)")
        ax.set_title(name)
    axes[0].set_ylabel("Δ logit toward switching\n(away from the pulsed trial's side)")
    axes[0].legend(frameon=False, fontsize=11)
    fig.suptitle(f"C1-L  Impulse response ({key}): one trial's licks changed, vs flipping its reward", y=1.03)
    fig.savefig(figure_path(f"fig_C1_L_impulse{sfx}.png"), bbox_inches="tight")
    plt.close(fig)

    # I-L2 clamps on every non-padding step.
    steps = np.arange(obs.shape[0])[:, None]
    body = (steps < split.n_trials[None, :]) & (steps >= 1)
    cell = np.where(body, 2 * obs[..., 0] + obs[..., 1], -1).astype(int)
    means = {}
    for cc in range(4):
        m = cell == cc
        means[cc] = (float(obs[..., LEFT][m].mean()), float(obs[..., RIGHT][m].mean()))
    lut = np.array([means[cc] for cc in range(4)], dtype=np.float32)  # [cell, (left, right)]
    cell_left = np.where(cell >= 0, lut[cell.clip(0), 0], 0.0).astype(np.float32)
    cell_right = np.where(cell >= 0, lut[cell.clip(0), 1], 0.0).astype(np.float32)
    chosen_is_right = obs[..., 0] == 1
    perm_rng = np.random.default_rng(SEED)
    perm_left, perm_right = obs[..., LEFT].copy(), obs[..., RIGHT].copy()
    for s_ in range(obs.shape[1]):
        ix = np.flatnonzero(body[:, s_])
        p = perm_rng.permutation(len(ix))
        perm_left[ix, s_], perm_right[ix, s_] = perm_left[ix[p], s_], perm_right[ix[p], s_]

    def clamp(new_left, new_right):
        def edit(o, pad):
            b = body[:, pad]
            o[..., LEFT] = np.where(b, new_left[:, pad], o[..., LEFT])
            o[..., RIGHT] = np.where(b, new_right[:, pad], o[..., RIGHT])
            return o
        return edit

    left, right = obs[..., LEFT], obs[..., RIGHT]
    other_cell_left = np.where(chosen_is_right, cell_left, left)   # other side = left when chosen right
    other_cell_right = np.where(chosen_is_right, right, cell_right)
    chosen_cell_left = np.where(chosen_is_right, left, cell_left)
    chosen_cell_right = np.where(chosen_is_right, cell_right, right)
    zeros = np.zeros_like(left)
    conditions = {
        "intact": None,
        "cell mean (both sides)": clamp(cell_left, cell_right),
        "cell mean, other side only": clamp(other_cell_left, other_cell_right),
        "cell mean, chosen side only": clamp(chosen_cell_left, chosen_cell_right),
        "population mean (0)": clamp(zeros, zeros),
        "permuted within session": clamp(perm_left, perm_right),
    }
    p_c = t["p_right_C"].to_numpy()
    lik_c = float(np.exp(np.mean(np.log(np.where(action == 1, p_c, 1 - p_c)[present]))))
    clamps = {}
    for name, edit in conditions.items():
        logit = intact if edit is None else mech.replay(step, obs, contexts, edit)[taus, cols]
        clamps[name] = {"likelihood": mech.likelihood(logit[present], action[present])}
        clamps[name]["minus_C"] = clamps[name]["likelihood"] - lik_c
        print(name, clamps[name], flush=True)
    fig, ax = plt.subplots(figsize=(10.5, 5))
    names = list(clamps)
    vals = [clamps[n]["minus_C"] * 1000 for n in names]
    ax.barh(names[::-1], vals[::-1], color=["#4C72B0" if v > 0 else "#C44E52" for v in vals[::-1]])
    for i, v in enumerate(vals[::-1]):
        ax.annotate(f"{v:+.2f}", (v, i), textcoords="offset points", xytext=(6 if v >= 0 else -6, 0),
                    ha="left" if v >= 0 else "right", va="center", fontsize=12)
    ax.axvline(0, color="#222222", lw=1)
    ax.set_xlim(min(min(vals), 0) * 1.3 - 0.5, max(vals) * 1.25)
    ax.set_xlabel("likelihood − C's (×1000), sessions with licks")
    ax.set_title(f"I-L2  Replays with lick inputs clamped ({key})")
    fig.savefig(figure_path(f"fig_I_L2_clamps{sfx}.png"), bbox_inches="tight")
    plt.close(fig)

    write_json(f"i_licks_mechanism{sfx}.json", {
        "key": key, "checks": checks, "ref_chosen_count": ref_chosen,
        "I_L1": {"chosen_grid": CHOSEN_GRID, "other_grid": OTHER_GRID,
                 "median_delta_switch_logit": {"after_loss": grid[0], "after_win": grid[1]}},
        "C1_L": impulse, "cell_means_encoded": {str(k): v for k, v in means.items()},
        "I_L2": {"likelihood_C": lik_c, "conditions": clamps},
    }, PRODUCER)


if __name__ == "__main__":
    main(*sys.argv[1:2])
