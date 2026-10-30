"""Tier 3: how the RT input acts in R's dynamics — update rule, impulse response, clamps.

    PYTHONPATH=<wrapper>/code python analysis/i_rt_mechanism.py [R-s1]   # default R-s0

Stage A (weights + data + JAX; GPU, or JAX_PLATFORMS=cpu). Rebuilds R-s0's network, eval
inputs ``xs`` and per-session contexts, and replays the split with the wrapper's
``GruStep``; the replay must reproduce the A1 likelihood and the RT input column must equal
the A1 table's ``prev_log_rt_input``. RT values are in the model's input units (log RT,
globally standardized: 0 = population mean, 1 = one population sd). "Toward switching"
signs a logit change so that positive = toward the side NOT chosen on the trial whose
outcome is being fed.

  I-RT1  update rule: at sampled real states, the RT input of one step is swept over
         −2.5…+2.5 with choice/reward as observed; Δ switch-logit vs the RT = 0 input, per
         input cluster. Local slope (finite difference at the observed RT) against the state's
         confidence (|logit| with RT input 0, so the RT push itself is not in it) and the outcome.
  C1-RT  impulse: one step's RT input is shifted ±1 sd (or its reward flipped, as a
         reference) and the network runs on with all later inputs as observed; Δ
         switch-logit on that and the next 10 trials.
  I-RT2  clamps: full replays with the RT input permuted within session, set to the session
         mean, the mouse mean, or 0, scored on RT-present sessions against intact R and C. Only
         real RT inputs are replaced (and averaged); inputs for trials with no RT stay 0.

Writes ``fig_I_RT1_update_rule<sfx>.png``, ``fig_C1_RT_impulse<sfx>.png``,
``fig_I_RT2_clamps<sfx>.png`` and ``i_rt_mechanism<sfx>.json`` (sfx empty for R-s0, else
``_<key>``); caches contexts in ``_cache/contexts_<key>.npz``.
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
from _pk03 import (  # noqa: E402
    ANALYSIS, CACHE, MODEL_KEY, REWARDED_COLOR, UNREWARDED_COLOR, STUDY12, BASELINE_KEY,
    apply_presentation_style, figure_path, write_json,
)
import _pk03  # noqa: E402

PRODUCER = "analysis/i_rt_mechanism.py"
BATCH = 512
N_SAMPLE = 50_000
N_AHEAD = 10
SWEEP = np.linspace(-2.5, 2.5, 21)
FD = 0.1  # finite-difference half-step for the local slope (input units)
LIKELIHOOD_TOLERANCE = 2e-4
LOG_RT_CENTER, LOG_RT_SCALE = -1.822, 0.599  # wrapper ``trial_timing_features`` constants
INPUT_COLORS = {"R rewarded": "#0072B2", "R unrewarded": "#56B4E9", "L rewarded": "#D55E00", "L unrewarded": "#E69F00"}
SEED = 0


def seconds(v):
    return np.exp(LOG_RT_CENTER + LOG_RT_SCALE * np.asarray(v))


def build(key: str = MODEL_KEY, n_obs: int = 3):
    import jax  # noqa: PLC0415
    from post_training_analysis.gru_counterfactual import load_gru_split  # noqa: PLC0415
    from post_training_analysis.gru_step import GruStep, split_inputs, subject_context  # noqa: PLC0415

    jax.config.update("jax_default_matmul_precision", "highest")
    split = load_gru_split(CACHE / "models" / key / "run")
    _pk03.register_checkpoint(CACHE / "models" / key / "run")
    subject_idx, session_idx, obs = split_inputs(split.xs, session_conditioning=split.run.session_conditioning_enabled)
    if obs.shape[-1] != n_obs:
        raise AssertionError(f"Expected {n_obs} observation columns, got {obs.shape[-1]}")
    cache = CACHE / f"contexts_{key}.npz"
    if cache.exists() and list(np.load(cache)["session_ids"]) == split.session_ids:
        contexts = np.load(cache)["contexts"]
    else:
        contexts = subject_context(split.make_network, split.params, subject_idx[0], session_idx[0])
        np.savez(cache, contexts=contexts, session_ids=np.array(split.session_ids))
    _pk03._read_cache(cache)
    return split, GruStep.from_params(split.params), obs.astype(np.float32), contexts.astype(np.float32)


def trial_index(split, key: str = MODEL_KEY):
    cols = ["ses_idx", "trial_idx", "subject_id", "action", "reward", "reaction_time", "prev_log_rt_input"]
    t = pd.read_parquet(_pk03._read_cache(CACHE / f"trial_states_{key}.parquet"), columns=cols)
    t["col"] = t["ses_idx"].map({s: i for i, s in enumerate(split.session_ids)}).to_numpy()
    t["tau"] = t["trial_idx"].to_numpy() - 1
    t["rt_present"] = t["reaction_time"].notna().groupby(t["ses_idx"]).transform("all")
    c = pd.read_parquet(_pk03._read_cache(STUDY12 / "analysis" / "_cache" / f"trial_states_{BASELINE_KEY}.parquet"),
                        columns=["ses_idx", "trial_idx", "p_model1_right"])
    t = t.merge(c.rename(columns={"p_model1_right": "p_right_C"}), on=["ses_idx", "trial_idx"], validate="1:1")
    return t


def replay(step, obs, contexts, edit=None):
    """Logit(right) − logit(left) for every timestep, [T, S]; ``edit(obs_batch, idx)`` alters inputs."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    run = jax.jit(lambda o, c: step.logit_diff(step.run(o, c)))
    out = np.empty(obs.shape[:2], dtype=np.float32)
    for start in range(0, obs.shape[1], BATCH):
        idx = np.arange(start, min(start + BATCH, obs.shape[1]))
        pad = np.r_[idx, np.full(BATCH - len(idx), idx[-1])]
        o = obs[:, pad].copy()
        if edit is not None:
            o = edit(o, pad)
        c = np.broadcast_to(contexts[pad][None], o.shape[:2] + (contexts.shape[1],))
        out[:, idx] = np.asarray(run(jnp.asarray(o), jnp.asarray(c)))[:, :len(idx)]
    return out


def states_at(step, obs, contexts, cols, taus):
    """State after step ``tau`` (i.e. the state that predicts trial tau+1) for each (col, tau)."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    run = jax.jit(lambda o, c: step.run(o, c))
    out = np.empty((len(cols), step.hidden_size), dtype=np.float32)
    for start in range(0, obs.shape[1], BATCH):
        idx = np.arange(start, min(start + BATCH, obs.shape[1]))
        want = np.flatnonzero((cols >= start) & (cols < start + BATCH))
        if not len(want):
            continue
        pad = np.r_[idx, np.full(BATCH - len(idx), idx[-1])]
        c = np.broadcast_to(contexts[pad][None], obs[:, pad].shape[:2] + (contexts.shape[1],))
        states = np.asarray(run(jnp.asarray(obs[:, pad]), jnp.asarray(c)))
        out[want] = states[taus[want], cols[want] - start]
    return out


def likelihood(logit, action):
    p_right = 1 / (1 + np.exp(-logit.astype(np.float64)))
    p = np.where(action == 1, p_right, 1 - p_right)
    return float(np.exp(np.mean(np.log(np.clip(p, 1e-10, 1 - 1e-10)))))


def update_rule(step, obs, contexts, t, rows, h_prev):
    """I-RT1. rows: sampled table rows with tau ≥ 1; h_prev: state after step tau−1."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    cols, taus = t["col"].to_numpy()[rows], t["tau"].to_numpy()[rows]
    base = obs[taus, cols]  # [N, 3]: trial tau−1's choice, reward, log RT
    c = contexts[cols]

    @jax.jit
    def logit_with_rt(v):
        o = jnp.asarray(base).at[:, 2].set(v)
        return step.logit_diff(step.step(jnp.asarray(h_prev), o, jnp.asarray(c)))

    sweep = np.stack([np.asarray(logit_with_rt(jnp.full(len(rows), v, dtype=jnp.float32))) for v in SWEEP])
    zero = np.asarray(logit_with_rt(jnp.zeros(len(rows), dtype=jnp.float32)))
    actual_v = jnp.asarray(base[:, 2])
    slope = (np.asarray(logit_with_rt(actual_v + FD)) - np.asarray(logit_with_rt(actual_v - FD))) / (2 * FD)
    away = np.where(base[:, 0] == 1, -1.0, 1.0)  # switch-signed: positive = toward the other side
    frame = pd.DataFrame({
        "input": [f"{'R' if ch == 1 else 'L'} {'rewarded' if r == 1 else 'unrewarded'}" for ch, r in base[:, :2]],
        "reward": base[:, 1], "slope_switch": slope * away, "confidence": np.abs(zero),
        "rt_input": base[:, 2],
    })
    d_switch = (sweep - zero[None]) * away[None]  # [V, N]
    return frame, d_switch


def impulse(step, obs, contexts, t, rows, h_prev):
    """C1-RT. Shift RT (±1 sd) or flip reward at step tau, run N_AHEAD more steps."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    cols, taus = t["col"].to_numpy()[rows], t["tau"].to_numpy()[rows]
    window = np.stack([obs[taus + k, cols] for k in range(N_AHEAD + 1)])  # [K, N, 3]
    c = np.broadcast_to(contexts[cols][None], window.shape[:2] + (contexts.shape[1],))
    run = jax.jit(lambda o: step.logit_diff(step.run(jnp.asarray(o), jnp.asarray(c), h0=jnp.asarray(h_prev))))
    baseline = np.asarray(run(window))
    away = np.where(window[0, :, 0] == 1, -1.0, 1.0)
    out = {}
    for name, edit in (("RT +1 sd", lambda w: w.at[0, :, 2].add(1.0)), ("RT −1 sd", lambda w: w.at[0, :, 2].add(-1.0)),
                       ("reward flipped", lambda w: w.at[0, :, 1].set(1.0 - w[0, :, 1]))):
        out[name] = (np.asarray(run(edit(jnp.asarray(window)))) - baseline) * away[None]  # [K, N]
    return out, window[0, :, 1]


def main(key: str = MODEL_KEY) -> None:
    _pk03.use_keys(key)
    apply_presentation_style()
    sfx = "" if key == MODEL_KEY else f"_{key}"
    split, step, obs, contexts = build(key)
    t = trial_index(split, key)
    if not (t["tau"] < split.n_trials[t["col"]]).all():
        raise AssertionError("Trial table rows fall outside the split's timesteps")
    fed = obs[t["tau"].to_numpy(), t["col"].to_numpy(), 2]
    rt_input_max_err = float(np.max(np.abs(fed - t["prev_log_rt_input"].to_numpy())))
    if rt_input_max_err > 1e-5:
        raise AssertionError(f"xs RT column differs from prev_log_rt_input by {rt_input_max_err}")

    action = t["action"].to_numpy()
    cols, taus = t["col"].to_numpy(), t["tau"].to_numpy()
    present = t["rt_present"].to_numpy()
    intact = replay(step, obs, contexts)[taus, cols]
    logged = _pk03.read_json(ANALYSIS / f"trial_states_{key}.json")["checks"]["logged_eval_likelihood"]
    checks = {"replay_likelihood": likelihood(intact, action), "logged_likelihood": logged,
              "rt_input_max_abs_error": rt_input_max_err}
    if abs(checks["replay_likelihood"] - logged) > LIKELIHOOD_TOLERANCE:
        raise AssertionError(f"Replay does not reproduce the logged likelihood: {checks}")
    print(checks, flush=True)

    # Shared sample: RT-present sessions, tau ≥ 1 (a real previous state), N_AHEAD steps left.
    rng = np.random.default_rng(SEED)
    eligible = np.flatnonzero(present & (taus >= 1) & (taus + N_AHEAD < split.n_trials[cols]))
    rows = np.sort(rng.choice(eligible, N_SAMPLE, replace=False))
    h_prev = states_at(step, obs, contexts, cols[rows], taus[rows] - 1)

    # I-RT1
    frame, d_switch = update_rule(step, obs, contexts, t, rows, h_prev)
    rule = {}
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2), gridspec_kw={"wspace": 0.32})
    for label, color in INPUT_COLORS.items():
        m = frame["input"].to_numpy() == label
        q = np.quantile(d_switch[:, m], [0.25, 0.5, 0.75], axis=1)
        rule[label] = {"median": q[1], "q25": q[0], "q75": q[2]}
        axes[0].plot(SWEEP, q[1], color=color, lw=2, label=f"t−1 {label}")
        axes[0].fill_between(SWEEP, q[0], q[2], color=color, alpha=0.15, lw=0)
    axes[0].axhline(0, color="#888888", lw=1)
    axes[0].axvline(0, color="#888888", lw=1, ls=":")
    axes[0].set_xlabel("RT input (sd; 0 = population mean)")
    axes[0].set_ylabel("Δ logit toward switching\n(vs RT input = 0)")
    axes[0].set_title("RT update rule at real states (median, IQR)")
    axes[0].legend(frameon=False, fontsize=11)
    top = axes[0].secondary_xaxis("top", functions=(lambda v: v, lambda v: v))
    top.set_xticks([-2, -1, 0, 1, 2], [f"{s:.2f}s" for s in seconds([-2, -1, 0, 1, 2])], fontsize=11)
    frame["conf_bin"] = pd.qcut(frame["confidence"], 10, labels=False)
    by_conf = frame.groupby(["conf_bin", "reward"])["slope_switch"].agg(["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
    by_conf.columns = ["median", "q25", "q75"]
    by_conf = by_conf.reset_index()
    conf_mid = frame.groupby("conf_bin")["confidence"].median().to_numpy()
    for reward, color, ls, name in ((1.0, REWARDED_COLOR, "-", "rewarded"), (0.0, UNREWARDED_COLOR, "--", "unrewarded")):
        r = by_conf[by_conf["reward"] == reward].sort_values("conf_bin")
        x = conf_mid[r["conf_bin"].astype(int)]
        axes[1].plot(x, r["median"], color=color, ls=ls, lw=2, marker="o", ms=5, label=f"t−1 {name}")
        axes[1].fill_between(x, r["q25"], r["q75"], color=color, alpha=0.15, lw=0)
    axes[1].axhline(0, color="#888888", lw=1)
    axes[1].set_xlabel("confidence: |logit| at RT input 0")
    axes[1].set_ylabel("local slope, Δ switch-logit per sd")
    axes[1].set_title("additive (flat) or a gain (varies)?")
    axes[1].legend(frameon=False, fontsize=11)
    frame["rt_bin"] = pd.qcut(frame["rt_input"], 10, labels=False, duplicates="drop")
    by_rt = frame.groupby(["rt_bin", "reward"])["slope_switch"].median().reset_index()
    rt_mid = frame.groupby("rt_bin")["rt_input"].median().to_numpy()
    for reward, color, ls, name in ((1.0, REWARDED_COLOR, "-", "rewarded"), (0.0, UNREWARDED_COLOR, "--", "unrewarded")):
        r = by_rt[by_rt["reward"] == reward].sort_values("rt_bin")
        axes[2].plot(rt_mid[r["rt_bin"].astype(int)], r["slope_switch"], color=color, ls=ls, lw=2, marker="o", ms=5, label=f"t−1 {name}")
    axes[2].axhline(0, color="#888888", lw=1)
    axes[2].set_xlabel("observed RT input on t−1 (sd)")
    axes[2].set_ylabel("local slope (median)")
    axes[2].set_title("is the rule linear in log RT?")
    fig.suptitle(f"I-RT1  How one RT input moves R's choice ({key}; {N_SAMPLE:,} real states, RT-present sessions)", y=1.04)
    fig.savefig(figure_path(f"fig_I_RT1_update_rule{sfx}.png"), bbox_inches="tight")
    plt.close(fig)
    slope_summary = {
        "median_slope_by_reward": frame.groupby("reward")["slope_switch"].median().to_dict(),
        "median_slope_by_input": frame.groupby("input")["slope_switch"].median().to_dict(),
        "slope_iqr": [float(frame["slope_switch"].quantile(0.25)), float(frame["slope_switch"].quantile(0.75))],
        "fraction_positive_slope": float((frame["slope_switch"] > 0).mean()),
        "spearman_slope_vs_confidence": float(frame[["slope_switch", "confidence"]].corr("spearman").iloc[0, 1]),
        "by_confidence": by_conf.to_dict("records"), "by_rt_input": by_rt.to_dict("records"),
    }

    # C1-RT
    effects, reward0 = impulse(step, obs, contexts, t, rows, h_prev)
    lags = np.arange(N_AHEAD + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    imp = {}
    for ax, (reward, name) in zip(axes, ((1.0, "pulsed trial rewarded"), (0.0, "pulsed trial unrewarded"))):
        m = reward0 == reward
        for effect, color, ls in (("RT +1 sd", "#C44E52", "-"), ("RT −1 sd", "#4C72B0", "-"), ("reward flipped", "#222222", "--")):
            mean = effects[effect][:, m].mean(axis=1)
            sem = effects[effect][:, m].std(axis=1) / np.sqrt(m.sum())
            imp[f"{effect} | {name}"] = {"mean": mean, "sem": sem}
            ax.plot(lags, mean, color=color, ls=ls, lw=2, marker="o", ms=5, label=effect)
            ax.fill_between(lags, mean - 2 * sem, mean + 2 * sem, color=color, alpha=0.15, lw=0)
        ax.axhline(0, color="#888888", lw=1)
        ax.set_xlabel("trials after the pulse (0 = next choice)")
        ax.set_title(name)
        ax.set_xticks(lags[::2])
    axes[0].set_ylabel("Δ logit toward switching\n(away from the pulsed trial's side)")
    axes[0].legend(frameon=False, fontsize=12)
    fig.suptitle(f"C1-RT  Impulse response ({key}): one trial's RT shifted ±1 sd, vs flipping its reward", y=1.03)
    fig.savefig(figure_path(f"fig_C1_RT_impulse{sfx}.png"), bbox_inches="tight")
    plt.close(fig)

    # I-RT2 clamps (rows tau ≥ 1 only; tau = 0 keeps the −1 padding).
    subj = t.drop_duplicates("col").set_index("col")["subject_id"]
    col_subject = subj.reindex(range(obs.shape[1])).to_numpy()
    valid = np.arange(obs.shape[0])[:, None] < split.n_trials[None, :]
    # Inputs that carry a real RT: not the padding row, and the fed trial had an RT (row τ+1
    # carries the RT of the trial on row τ).
    has_rt = np.zeros(obs.shape[:2], dtype=bool)
    nxt = taus + 1 < split.n_trials[cols]
    has_rt[taus[nxt] + 1, cols[nxt]] = t["reaction_time"].notna().to_numpy()[nxt]
    body = valid & (np.arange(obs.shape[0])[:, None] >= 1) & has_rt
    rt = np.where(body, obs[..., 2], np.nan)
    with np.errstate(invalid="ignore"):
        session_mean = np.nan_to_num(np.nanmean(rt, axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        mouse_mean = np.nan_to_num(pd.Series(np.nansum(rt, axis=0)).groupby(col_subject).transform("sum").to_numpy() /
                                   pd.Series(body.sum(axis=0)).groupby(col_subject).transform("sum").to_numpy())
    perm_rng = np.random.default_rng(SEED)
    permuted = obs[..., 2].copy()
    for s in range(obs.shape[1]):
        idx = np.flatnonzero(body[:, s])
        permuted[idx, s] = permuted[idx, s][perm_rng.permutation(len(idx))]

    def clamp(values):
        def edit(o, pad):
            v = values[:, pad] if values.ndim == 2 else np.broadcast_to(values[pad][None], o.shape[:2])
            o[..., 2] = np.where(body[:, pad], v, o[..., 2])
            return o
        return edit

    conditions = {"intact": None, "permuted within session": clamp(permuted), "session mean": clamp(session_mean),
                  "mouse mean": clamp(mouse_mean), "population mean (0)": clamp(np.zeros(obs.shape[1], dtype=np.float32))}
    p_c = t["p_right_C"].to_numpy()
    lik_c = float(np.exp(np.mean(np.log(np.where(action == 1, p_c, 1 - p_c)[present]))))
    clamps = {}
    for name, edit in conditions.items():
        logit = intact if edit is None else replay(step, obs, contexts, edit)[taus, cols]
        clamps[name] = {"likelihood_rt_present": likelihood(logit[present], action[present]),
                        "likelihood_all": likelihood(logit, action)}
        print(name, clamps[name], flush=True)
    for v in clamps.values():
        v["minus_C_rt_present"] = v["likelihood_rt_present"] - lik_c
    fig, ax = plt.subplots(figsize=(10, 4.8))
    names = list(clamps)
    vals = [clamps[n]["minus_C_rt_present"] * 1000 for n in names]
    ax.barh(names[::-1], vals[::-1], color=["#4C72B0" if v > 0 else "#C44E52" for v in vals[::-1]])
    for i, v in enumerate(vals[::-1]):
        ax.annotate(f"{v:+.2f}", (v, i), textcoords="offset points", xytext=(6 if v >= 0 else -6, 0),
                    ha="left" if v >= 0 else "right", va="center", fontsize=13)
    ax.axvline(0, color="#222222", lw=1)
    ax.set_xlim(min(vals) * 1.35, max(vals) * 1.35)
    ax.set_xlabel("likelihood − C's (×1000), RT-present sessions")
    ax.set_title(f"I-RT2  Replays with RT clamped ({key})")
    fig.savefig(figure_path(f"fig_I_RT2_clamps{sfx}.png"), bbox_inches="tight")
    plt.close(fig)

    write_json(f"i_rt_mechanism{sfx}.json", {"key": key,
        "checks": checks, "sample": {"n_states": int(N_SAMPLE), "seed": SEED},
        "I_RT1_update_rule": {"sweep": SWEEP, "sweep_seconds": seconds(SWEEP), "by_input": rule, "slopes": slope_summary},
        "C1_RT_impulse": imp,
        "I_RT2_clamps": {"likelihood_C_rt_present": lik_c, "conditions": clamps},
    }, PRODUCER)
    print(json.dumps({k: slope_summary[k] for k in ("median_slope_by_reward", "median_slope_by_input", "fraction_positive_slope",
                                                    "spearman_slope_vs_confidence")}, indent=1, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:2])
