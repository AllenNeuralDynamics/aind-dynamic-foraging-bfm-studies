"""I3c: causal test of the carried value — nudge the state along the TDR value axis and watch.

    python analysis/i3c_value_causal.py

I3 showed that the projection of the state on I2's value axis is linearly predictable from
step to step (v(t+1) ≈ s·v(t) + c, R² 0.90–0.99). That is correlational. Here, for N_STARTS
real steps t (equal numbers per input received at t), the stored state h_t is replaced by
h_t ± δ·q (q a unit direction) and the GRU (wrapper ``gru_step``, the session's own context)
is re-run for K steps on the *observed* inputs; the response is the difference from the
unperturbed rollout, averaged over ±δ (the symmetric, linear part). Directions q:
  * ``value`` — I2's orthonormal value axis;
  * ``engage`` — I2's orthonormal engage axis (does a change in engagement leak into v?);
  * ``random ⊥ value`` — N_RANDOM random unit directions orthogonal to the value axis.
Measured per step k: Δv(t+k)/δ (value coordinate), Δside and Δengage readouts per unit δ.
Predictions from I3: Δv(t+1)/δ ≈ the I3 slope of that input × source cell; for k > 1 the
product of the cells' slopes along the observed sequence (a linear-chain prediction).
Parity first: the unperturbed rollout must reproduce the stored states.
Writes ``i3c_value_causal.json`` and ``fig_I3c_value_causal.png``.
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
import _pk02 as s13  # noqa: E402

N_PER_INPUT = 4000
K = 10
DELTA = 0.5
N_RANDOM = 5
SEED = 7
INPUTS = ["R+", "R−", "L+", "L−", "ignore"]
SOURCES = {"R": "R lobe", "L": "L lobe", "ignore": "post-ignore"}
PARITY_TOLERANCE = 1e-3


def labels(action: np.ndarray, reward: np.ndarray) -> np.ndarray:
    return np.where(action == s13.IGNORE, "ignore",
                    np.where(action == 1, "R", "L").astype(object) + np.where(reward > 0, "+", "−"))


def main() -> None:
    import jax  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp  # noqa: PLC0415

    from post_training_analysis.gru_step import GruStep  # noqa: PLC0415

    s13.apply_presentation_style()
    axes_file = np.load(s13.CACHE / f"i2_tdr_axes_{s13.MODEL_KEY}.npz")
    basis, terms = axes_file["basis"], [str(t) for t in axes_file["terms"]]
    q_value, q_engage = basis[:, terms.index("value")], basis[:, terms.index("engage")]
    i3 = json.loads((s13.ANALYSIS / "i3_tdr_dynamics.json").read_text())["transitions"]
    cell_slope = {}
    for key, r in i3.items():
        inp, src = key.split(" | from ")
        cell_slope[(inp, next(k for k, v in SOURCES.items() if src.startswith(v)))] = r["slope"]

    t = s13.load_trials(["ses_idx", "action", "reward", *s13.STATE_COLUMNS])
    ses = t["ses_idx"].to_numpy()
    action, reward = t["action"].to_numpy(), (t["reward"].to_numpy() > 0).astype(float)
    states = t[s13.STATE_COLUMNS].to_numpy(np.float64)
    del t
    lab = labels(action, reward)
    prev = np.r_[-1, action[:-1]]
    src = np.where(prev == s13.IGNORE, "ignore", np.where(prev == 1, "R", "L"))
    # Start t needs a previous step (for its source) and K following steps in the session.
    n = len(ses)
    same_ahead = np.ones(n, bool)
    for k in range(1, K + 1):
        same_ahead &= np.r_[ses[k:] == ses[:-k], np.zeros(k, bool)]
    eligible = same_ahead & np.r_[False, ses[1:] == ses[:-1]]
    rng = np.random.default_rng(SEED)
    starts = np.sort(np.concatenate([rng.choice(np.flatnonzero(eligible & (lab == i)), N_PER_INPUT, replace=False)
                                     for i in INPUTS]))
    ctx = np.load(s13.CACHE / f"contexts_{s13.MODEL_KEY}.npz")
    ctx_row = pd.Series(np.arange(len(ctx["session_ids"])), index=ctx["session_ids"]).reindex(ses[starts]).to_numpy()
    c = jnp.asarray(ctx["contexts"][ctx_row], dtype=jnp.float64)
    obs_seq = np.stack([np.stack([np.where(action[starts + k] == s13.IGNORE, 2.0, action[starts + k]),
                                  np.where(action[starts + k] == s13.IGNORE, 0.0, reward[starts + k])], axis=-1)
                        for k in range(K)])  # obs fed at step k = outcome of row t+k
    step = GruStep.from_params({k: {nm: np.asarray(a, np.float64) for nm, a in v.items()}
                                for k, v in s13.best_params().items() if isinstance(v, dict)})
    w, b = np.asarray(step.readout_w, np.float64), np.asarray(step.readout_b, np.float64)

    @jax.jit
    def rollout(h0, obs):
        def body(h, u):
            h = step.step(h, u, c)
            return h, h
        return jax.lax.scan(body, h0, obs)[1]

    h0 = states[starts]
    base = np.asarray(rollout(jnp.asarray(h0), jnp.asarray(obs_seq)))  # [K, N, H] = states t+1..t+K
    target = np.stack([states[starts + k] for k in range(1, K + 1)])
    parity = float(np.median(np.abs(base - target)))
    if parity > PARITY_TOLERANCE:
        raise AssertionError(f"Unperturbed rollout does not reproduce stored states: median |Δh| {parity}")

    def readouts(h):
        z = h @ w + b
        return z[..., 1] - z[..., 0], np.logaddexp(z[..., 0], z[..., 1]) - z[..., 2]

    s_base, e_base = readouts(base)
    random_dirs = []
    for _ in range(N_RANDOM):
        r = rng.normal(size=s13.HIDDEN)
        r -= (r @ q_value) * q_value
        random_dirs.append(r / np.linalg.norm(r))
    directions = {"value": q_value, "engage": q_engage, **{f"random ⊥ value {i + 1}": d for i, d in enumerate(random_dirs)}}
    response = {}
    for name, q in directions.items():
        outs = []
        for sign in (1.0, -1.0):
            h = np.asarray(rollout(jnp.asarray(h0 + sign * DELTA * q), jnp.asarray(obs_seq)))
            s_, e_ = readouts(h)
            outs.append(((h - base) @ q_value, s_ - s_base, e_ - e_base))
        response[name] = {k: (outs[0][i] - outs[1][i]) / (2 * DELTA) for i, k in enumerate(("value", "side", "engage"))}
    # The perturbation itself, at step 0, on the readouts (linear part).
    s0p, e0p = readouts(h0 + DELTA * q_value)
    s0m, e0m = readouts(h0 - DELTA * q_value)

    # Predictions from I3's cell slopes along the observed sequence.
    seq_lab = np.stack([lab[starts + k] for k in range(K)])
    seq_src = np.stack([src[starts + k] for k in range(K)])
    slope_seq = np.vectorize(lambda i, s_: cell_slope.get((i, s_), np.nan))(seq_lab, seq_src)
    predicted = np.cumprod(slope_seq, axis=0)  # Δv(t+k)/δ for k = 1..K under the linear chain
    observed = response["value"]["value"]

    by_cell = {}
    for inp in INPUTS:
        for s_key, s_name in SOURCES.items():
            m = (lab[starts] == inp) & (src[starts] == s_key)
            if m.sum() < 200:
                continue
            by_cell[f"{inp} | from {s_name}"] = {
                "n": int(m.sum()), "causal_k1_mean": float(observed[0, m].mean()),
                "causal_k1_sd": float(observed[0, m].std()), "i3_slope": cell_slope.get((inp, s_key)),
                "side_k1_per_delta": float(response["value"]["side"][0, m].mean()),
                "engage_k1_per_delta": float(response["value"]["engage"][0, m].mean())}

    summary = {name: {"value_per_delta_mean": r["value"].mean(axis=1), "side_per_delta_mean": r["side"].mean(axis=1),
                      "engage_per_delta_mean": r["engage"].mean(axis=1)} for name, r in response.items()}
    ok = np.isfinite(predicted).all(axis=0)
    chain = {"observed_mean": observed[:, ok].mean(axis=1), "predicted_mean": predicted[:, ok].mean(axis=1),
             "n_with_prediction": int(ok.sum()),
             "corr_per_k": [float(np.corrcoef(observed[k, ok], predicted[k, ok])[0, 1]) for k in range(K)]}

    fig, axes = plt.subplots(1, 3, figsize=(23, 6.6))
    ks = np.arange(1, K + 1)
    ax = axes[0]
    xs = [v["i3_slope"] for v in by_cell.values()]
    ys = [v["causal_k1_mean"] for v in by_cell.values()]
    ax.scatter(xs, ys, s=70, color=s13.ENGAGE_COLOR, edgecolors="k", zorder=3)
    for name, x_, y_ in zip(by_cell, xs, ys):
        ax.annotate(name.replace(" | from ", "\n"), (x_, y_), fontsize=7.5, xytext=(4, 3), textcoords="offset points")
    lo, hi = min(xs + ys) - 0.05, max(xs + ys) + 0.05
    ax.plot([lo, hi], [lo, hi], color="0.6", ls=":", lw=1)
    ax.set(xlabel="I3 regression slope (correlational)", ylabel="causal Δv(t+1) / δ",
           title="a. one step: causal carry vs I3 slope, per input × source")
    ax = axes[1]
    ax.plot(ks, chain["observed_mean"], color=s13.ENGAGE_COLOR, marker="o", lw=2.4, label="value nudge: Δv/δ (causal)")
    ax.plot(ks, chain["predicted_mean"], color="k", ls="--", lw=1.8, label="product of I3 slopes along the sequence")
    ax.plot(ks, summary["engage"]["value_per_delta_mean"], color=s13.ENGAGE_COLOR, ls=":", lw=1.8,
            label="engage nudge: Δv/δ")
    rnd = np.array([summary[k]["value_per_delta_mean"] for k in summary if k.startswith("random")])
    ax.fill_between(ks, rnd.min(axis=0), rnd.max(axis=0), color="0.7", alpha=0.5, label=f"random ⊥ value nudges ({N_RANDOM}): Δv/δ")
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set(xlabel="steps after the nudge (k)", ylabel="Δ value coordinate per unit nudge", title="b. how long the nudge stays on the value axis")
    ax.legend(frameon=False, fontsize=10)
    ax = axes[2]
    side0 = float(np.mean((s0p - s0m) / (2 * DELTA)))
    ax.plot(np.r_[0, ks], np.r_[side0, summary["value"]["side_per_delta_mean"]], color=s13.MODEL_COLOR, marker="o", lw=2,
            label="value nudge → side readout")
    ax.plot(np.r_[0, ks], np.r_[float(np.mean((e0p - e0m) / (2 * DELTA))), summary["value"]["engage_per_delta_mean"]],
            color=s13.ENGAGE_COLOR, marker="o", lw=2, label="value nudge → engage readout")
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set(xlabel="steps after the nudge (k; 0 = the nudged state itself)", ylabel="Δ readout per unit nudge",
           title="c. what the value nudge does to behaviour")
    ax.legend(frameon=False)
    fig.suptitle(f"Causal test of the carried value: nudge ±{DELTA} along the TDR value axis, re-run on the observed inputs "
                 f"({len(starts):,} starts)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(s13.figure_path("fig_I3c_value_causal.png"))
    plt.close(fig)

    s13.write_json("i3c_value_causal.json", {
        "n_starts": int(len(starts)), "n_per_input": N_PER_INPUT, "k": ks, "delta": DELTA, "seed": SEED,
        "parity_median_abs_state_err": parity, "response_linear_part": "(Δ at +δ − Δ at −δ) / 2δ",
        "nudge_to_readouts_at_k0": {"side": side0, "engage": float(np.mean((e0p - e0m) / (2 * DELTA)))},
        "by_cell_k1": by_cell, "summary": summary, "linear_chain": chain,
    }, "analysis/i3c_value_causal.py")
    for name, v in by_cell.items():
        print(f"{name:32s} n={v['n']:5d} causal k1 {v['causal_k1_mean']:.3f} (sd {v['causal_k1_sd']:.3f})  I3 slope {v['i3_slope']:.3f}")
    print("chain observed", np.round(chain["observed_mean"], 3), "\npredicted", np.round(chain["predicted_mean"], 3))
    print("engage nudge Δv", np.round(summary["engage"]["value_per_delta_mean"], 3))
    print("random Δv k1", np.round(rnd[:, 0], 3), "parity", parity)


if __name__ == "__main__":
    main()
