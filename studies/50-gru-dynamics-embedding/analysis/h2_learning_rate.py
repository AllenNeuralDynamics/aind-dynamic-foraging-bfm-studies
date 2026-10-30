"""H2: the effective learning rate by state — reward-flip effects in the GRU vs Q-learning.

    PYTHONPATH=<wrapper>/code python analysis/h2_learning_rate.py     # GPU (JAX)

For N_PROBES random eval trials t (with a later trial in the session), flip trial t's reward
(choice kept) and measure the change of the log-odds of right on trial t+1, signed by trial t's
side: c · (logit | rewarded − logit | unrewarded). This is the model's one-trial learning rate in
log-odds units at that state. GRU: wrapper ``GruStep`` from the cached session contexts (as
I8). Hattori / Bari: exact numpy replay from the cached Q values with each mouse's fitted
parameters (I8's validated ``rl_step``; Bari's choice kernel is not affected by a reward flip).
The effect is binned by the state at trial t:
  confidence |logit(t)| (model's own); P(chosen side) at t (how expected the choice was);
  run length of the chosen side up to t (stays only); stay vs switch on t; trials since the last
  reward before t; reward rate over the 10 trials before t; position in the session; task.
A fixed-α delta rule gives a learning rate that depends on state only through Q (via
α(1 − Q_chosen) …); the question is whether the GRU's depends on state in other ways.
Writes ``h2_learning_rate.json`` and ``fig_H2_learning_rate.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
from i8_v_tests import rl_params, rl_step  # noqa: E402

SEED = 13
N_PROBES = 200_000
BATCH = 512
MODELS = {"gru": "GRU", "hattori": "Hattori (L2F1)", "bari": "Bari (L1F1_CK1)"}
COLORS = {"gru": "#222222", "hattori": "#55A868", "bari": "#4C72B0"}
MIN_N = 300


def main() -> None:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from post_training_analysis.gru_counterfactual import load_gru_split  # noqa: PLC0415
    from post_training_analysis.gru_step import GruStep, split_inputs  # noqa: PLC0415

    s12.apply_presentation_style()
    t = s12.load_trials(["subject_id", "ses_idx", "trial_idx", "task", "action", "reward", "logit_diff", "trial_position"])
    a = t["action"].to_numpy().astype(int)
    r = (t["reward"].to_numpy() > 0).astype(float)
    ses = t["ses_idx"].to_numpy()
    pos = pd.Series(np.arange(len(t))).groupby(ses).cumcount().to_numpy()
    n_in = pd.Series(ses).map(pd.Series(ses).value_counts()).to_numpy()
    probes = np.sort(np.random.default_rng(SEED).choice(np.flatnonzero((pos >= 1) & (pos + 1 < n_in)), size=N_PROBES, replace=False))

    # GRU: state before choice t (full forward pass), then one step with trial t's input.
    split = load_gru_split(s12.CACHE / "models" / s12.MODEL_KEY / "run")
    step = GruStep.from_params(split.params)
    _, _, obs = split_inputs(split.xs, session_conditioning=split.run.session_conditioning_enabled)
    ctx = np.load(s12.CACHE / f"contexts_{s12.MODEL_KEY}.npz")
    if list(ctx["session_ids"]) != split.session_ids:
        raise AssertionError("Cached contexts are not in the split's session order")
    contexts = ctx["contexts"].astype(np.float32)
    col = pd.Series(np.arange(len(split.session_ids)), index=split.session_ids).reindex(ses[probes]).to_numpy()
    tau = t["trial_idx"].to_numpy()[probes] - 1
    run_full = jax.jit(step.run)
    h_t = np.empty((N_PROBES, step.hidden_size), dtype=np.float32)
    for start in range(0, obs.shape[1], BATCH):
        cols = np.arange(start, min(start + BATCH, obs.shape[1]))
        mine = np.flatnonzero(np.isin(col, cols))
        if len(mine) == 0:
            continue
        pad = np.r_[cols, np.full(BATCH - len(cols), cols[-1])]
        c = np.broadcast_to(contexts[pad][None], obs[:, pad].shape[:2] + (contexts.shape[1],))
        states = np.asarray(run_full(jnp.asarray(obs[:, pad]), jnp.asarray(c)))
        h_t[mine] = states[tau[mine], col[mine] - start]
    inp = obs[tau + 1, col].copy()
    if not (np.array_equal(inp[:, 0], a[probes]) and np.array_equal(inp[:, 1], r[probes])):
        raise AssertionError("GRU input row does not match the trial table")
    one = jax.jit(lambda h, o, cc: step.logit_diff(step.step(h, o, cc)))
    flipped = inp.copy()
    flipped[:, 1] = 1 - flipped[:, 1]
    cc = jnp.asarray(contexts[col])
    z_obs = np.asarray(one(jnp.asarray(h_t), jnp.asarray(inp), cc))
    z_flip = np.asarray(one(jnp.asarray(h_t), jnp.asarray(flipped), cc))
    check = float(np.median(np.abs(z_obs - t["logit_diff"].to_numpy()[probes + 1])))
    print("GRU one-step replay vs cached logit(t+1), median |diff|", round(check, 4), flush=True)
    sgn = 2.0 * a[probes] - 1
    effect = {"gru": sgn * np.where(r[probes] > 0, z_obs - z_flip, z_flip - z_obs)}
    logit_t = {"gru": t["logit_diff"].to_numpy()[probes]}

    # RL: exact one-step replay from the cached Q.
    for m in ("hattori", "bari"):
        rl = pd.read_parquet(s12.CACHE / f"rl_trials_{m}_{s12.MODEL_KEY}.parquet")
        ql, qr = rl["q_rl_left"].to_numpy()[probes], rl["q_rl_right"].to_numpy()[probes]
        prm = rl_params(m, t["subject_id"].to_numpy()[probes])
        l1, r1 = rl_step(ql, qr, a[probes], r[probes], prm)
        l0, r0 = rl_step(ql, qr, a[probes], 1 - r[probes], prm)
        d_obs, d_flip = prm["beta"] * (r1 - l1), prm["beta"] * (r0 - l0)
        effect[m] = sgn * np.where(r[probes] > 0, d_obs - d_flip, d_flip - d_obs)
        p = np.clip(rl["p_rl_right"].to_numpy()[probes], 1e-10, 1 - 1e-10)
        logit_t[m] = np.log(p / (1 - p))

    # State variables at trial t.
    first = np.r_[True, ses[1:] != ses[:-1]]
    new_run = first | np.r_[True, a[1:] != a[:-1]]
    run_pos = pd.Series(np.ones(len(t))).groupby(np.cumsum(new_run)).cumsum().to_numpy()
    stay = np.r_[False, a[1:] == a[:-1]] & ~first
    g = t.assign(rew=r).groupby("ses_idx", sort=False)["rew"]
    rate10 = g.transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean()).to_numpy()
    last_rew = pd.Series(np.where(r > 0, np.arange(len(t)), np.nan)).groupby(ses).transform(lambda x: x.shift(1).ffill()).to_numpy()
    since = np.arange(len(t)) - last_rew
    states = {
        "confidence |logit(t)|": {m: np.abs(logit_t[m]) for m in MODELS},
        "P(chosen side) at t": {m: 1 / (1 + np.exp(-sgn * logit_t[m])) for m in MODELS},
        "run length of the chosen side (stays)": np.where(stay[probes], np.minimum(run_pos[probes], 12), np.nan),
        "trials since the last reward": np.minimum(since[probes], 10),
        "reward rate, previous 10 trials": rate10[probes],
        "position in session": t["trial_position"].to_numpy()[probes],
    }
    edges = {"confidence |logit(t)|": np.linspace(0, 6, 13), "P(chosen side) at t": np.linspace(0, 1, 11),
             "run length of the chosen side (stays)": np.arange(1.5, 13.5), "trials since the last reward": np.arange(-0.5, 11),
             # reward rates are discrete (k/10): bins centred on them
             "reward rate, previous 10 trials": np.arange(-0.05, 1.06, 0.1), "position in session": np.linspace(0, 1, 11)}

    def binned(x, y, e):
        b = np.digitize(x, e) - 1
        ok = np.isfinite(x) & (b >= 0) & (b < len(e) - 1)
        df = pd.DataFrame({"b": b[ok], "y": y[ok]}).groupby("b")["y"].agg(["mean", "sem", "size"])
        df = df[df["size"] >= MIN_N]
        centres = (e[:-1] + e[1:]) / 2
        return centres[df.index], df["mean"].to_numpy(), df["sem"].to_numpy()

    out = {}
    rows = list(states) + ["stay vs switch", "task"]
    fig, axes = plt.subplots(len(rows), 3, figsize=(24, 4.3 * len(rows)), sharey="row")
    for i, name in enumerate(rows):
        out[name] = {}
        for j, m in enumerate(MODELS):
            ax = axes[i, j]
            if name == "stay vs switch":
                lab = ["stay", "switch"]
                vals = [effect[m][stay[probes]], effect[m][~stay[probes]]]
                ax.bar(lab, [v.mean() for v in vals], yerr=[v.std() / np.sqrt(len(v)) for v in vals], color=COLORS[m])
                out[name][m] = {k: float(v.mean()) for k, v in zip(lab, vals)}
            elif name == "task":
                task = t["task"].to_numpy()[probes]
                lab = [k for k in s12.TASKS if (task == k).sum() >= MIN_N]
                vals = [effect[m][task == k] for k in lab]
                ax.bar(range(len(lab)), [v.mean() for v in vals], yerr=[v.std() / np.sqrt(len(v)) for v in vals], color=COLORS[m])
                ax.set_xticks(range(len(lab)), [k.replace(" Without", "\nWithout") for k in lab], fontsize=9)
                out[name][m] = {k: float(v.mean()) for k, v in zip(lab, vals)}
            else:
                x = states[name][m] if isinstance(states[name], dict) else states[name]
                cx, mu, se = binned(x, effect[m], edges[name])
                ax.errorbar(cx, mu, yerr=se, color=COLORS[m], marker="o", ms=4, lw=2)
                ax.set_xlabel(name + (" (model's own)" if isinstance(states[name], dict) else ""))
                out[name][m] = {"x": cx, "mean": mu}
            ax.axhline(0, color="0.7", lw=0.8)
            if i == 0:
                ax.set_title(MODELS[m], fontsize=15)
            if j == 0:
                ax.set_ylabel("one-trial reward effect\n(log-odds, signed by side)")
    fig.suptitle(f"Effective learning rate by state: flip one reward, measure the next trial ({N_PROBES:,} probes)", fontsize=17, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.99), h_pad=3.0)
    fig.savefig(s12.figure_path("fig_H2_learning_rate.png"))
    plt.close(fig)

    s12.write_json("h2_learning_rate.json", {
        "n_probes": N_PROBES, "seed": SEED, "gru_replay_median_abs_diff": check,
        "mean_effect": {m: float(v.mean()) for m, v in effect.items()},
        "cv_of_effect_across_probes": {m: float(v.std() / abs(v.mean())) for m, v in effect.items()},
        "by_state": out, "min_per_bin": MIN_N,
        "definition": "c·(logit(t+1) | trial t rewarded − | unrewarded), choice kept, other history as observed",
    }, "analysis/h2_learning_rate.py", extra_groups=s12.rl_groups())
    print({m: (round(float(v.mean()), 3), round(float(v.std() / abs(v.mean())), 3)) for m, v in effect.items()})
    for name in rows:
        print(name, {m: (np.round(d["mean"], 2).tolist() if "mean" in d else {k: round(v, 3) for k, v in d.items()}) for m, d in out[name].items()})


if __name__ == "__main__":
    main()
