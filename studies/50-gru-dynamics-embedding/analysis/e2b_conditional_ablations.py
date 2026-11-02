"""E2b (stage A): ablations that stay near the manifold — conditional-mean replacement.

    JAX_PLATFORMS=cpu python analysis/e2b_conditional_ablations.py

E2 replaced a subspace's component by the *global* mean state. That leaves the other 127
coordinates at their values and puts the state off the visited manifold: random typical-
variance directions then cost about as much as meaningful ones, and removing 127 dimensions
(keeping only the readout coordinate) cost less than removing one. Here the component is
replaced by the mean of the state's *own last-input cluster* (wrapper ``run_ablated`` with
per-input means), removing within-cluster variation along the axis — the integration found
in phase I — while keeping the state on its cluster. Memory-only mode throughout (the choice
is read from the intact state; only the carried state is ablated). Controls: 10 random 1-d
and 10 random 2-d subspaces drawn from the state covariance, ablated the same way.
Scores: pooled eval likelihood and the Su2022 reward kernel (soft labels) of every
condition. Writes ``e2b_conditional_ablations.json`` and ``fig_E2b_conditional_ablations.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

BATCH = 512
N_RANDOM = 10
SEED = 6
NAMED = ["value axis", "lobe value axes (2-d)", "choice axis", "reward axis", "readout direction", "PC5 (time)"]


def orthonormal(*vectors) -> np.ndarray:
    q, _ = np.linalg.qr(np.column_stack(vectors))
    return q


def main() -> None:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from c1_kernels import as_kernels, design, fit_rows  # noqa: PLC0415
    from post_training_analysis.gru_counterfactual import load_gru_split  # noqa: PLC0415
    from post_training_analysis.gru_step import GruStep, split_inputs  # noqa: PLC0415

    s12.apply_presentation_style()
    split = load_gru_split(s12.CACHE / "models" / s12.MODEL_KEY / "run")
    step = GruStep.from_params(split.params)
    _, _, obs = split_inputs(split.xs, session_conditioning=split.run.session_conditioning_enabled)
    ctx = np.load(s12.CACHE / f"contexts_{s12.MODEL_KEY}.npz")
    if list(ctx["session_ids"]) != split.session_ids:
        raise AssertionError("Cached contexts are not in the split's session order")
    contexts = ctx["contexts"].astype(np.float32)
    t = s12.load_trials(["ses_idx", "trial_idx", "action", "reward"])
    col = t["ses_idx"].map({s: i for i, s in enumerate(split.session_ids)}).to_numpy()
    tau = t["trial_idx"].to_numpy() - 1
    action = t["action"].to_numpy()

    # Cluster means: the state after each input (previous choice x outcome), and first steps.
    states = s12.load_trials(s12.STATE_COLUMNS).to_numpy(dtype=np.float32)
    g = t.groupby("ses_idx", sort=False)
    prev_a, prev_r = g["action"].shift(1).to_numpy(), (g["reward"].shift(1) > 0).to_numpy()
    index = np.where(np.isnan(prev_a), 4, 2 * np.nan_to_num(prev_a) + prev_r).astype(int)
    means = np.stack([states[index == k].mean(axis=0) for k in range(5)]).astype(np.float32)
    del states

    pca = s12.load_pca()
    components = pca["components"]
    tdr = np.load(s12.keyed_cache("i2_tdr_axes.npz"))
    terms = [str(x) for x in tdr["terms"]]
    q = tdr["basis"]
    r = np.asarray(step.readout_w[:, 1] - step.readout_w[:, 0], dtype=np.float64)
    r /= np.linalg.norm(r)
    cov = (components.T * np.clip(pca["eigvals"], 0, None)) @ components
    rng = np.random.default_rng(SEED)
    subspaces = {
        "none": np.zeros((len(r), 0)),
        "value axis": q[:, [terms.index("value")]],
        "lobe value axes (2-d)": orthonormal(tdr["lobe_r"], tdr["lobe_l"]),
        "choice axis": q[:, [terms.index("choice")]],
        "reward axis": q[:, [terms.index("reward")]],
        "readout direction": r[:, None],
        "PC5 (time)": components[4][:, None],
        **{f"random 1-d #{i}": orthonormal(rng.multivariate_normal(np.zeros(len(r)), cov)) for i in range(N_RANDOM)},
        **{f"random 2-d #{i}": orthonormal(*rng.multivariate_normal(np.zeros(len(r)), cov, size=2)) for i in range(N_RANDOM)},
    }

    run = jax.jit(lambda o, c, b, m: step.run_ablated(o, c, b, m, memory_only=True))
    n_sessions = obs.shape[1]

    def p_right_for(basis: np.ndarray) -> np.ndarray:
        out = np.empty(obs.shape[:2], dtype=np.float32)
        for start in range(0, n_sessions, BATCH):
            idx = np.arange(start, min(start + BATCH, n_sessions))
            pad = np.r_[idx, np.full(BATCH - len(idx), idx[-1])]
            c = np.broadcast_to(contexts[pad][None], obs[:, pad].shape[:2] + (contexts.shape[1],))
            res = np.asarray(run(jnp.asarray(obs[:, pad]), jnp.asarray(c), jnp.asarray(basis, dtype=jnp.float32),
                                 jnp.asarray(means)))
            out[:, idx] = res[:, :len(idx)]
        return (1 / (1 + np.exp(-out[tau, col]))).astype(np.float32)

    def likelihood(p: np.ndarray) -> float:
        p = np.where(action == 1, p, 1 - p)
        return float(np.exp(np.mean(np.log(np.clip(p, 1e-10, 1 - 1e-10)))))

    x, rows_t = design()
    position = pd.Series(np.arange(len(t)), index=pd.MultiIndex.from_arrays([t["ses_idx"], t["trial_idx"]]))
    design_rows = position.loc[list(zip(rows_t["ses_idx"], rows_t["trial_idx"]))].to_numpy()
    results, probs = {}, {}
    for name, basis in subspaces.items():
        probs[name] = p_right_for(basis)
        results[name] = {"likelihood": likelihood(probs[name]), "k": int(basis.shape[1])}
        print(name, round(results[name]["likelihood"], 5), flush=True)
    names = list(subspaces)
    coefs = Parallel(n_jobs=8)(delayed(fit_rows)(x, probs[n][design_rows].astype(np.float64), None, True) for n in names)
    base = results["none"]["likelihood"]
    for name, coef in zip(names, coefs):
        results[name]["delta_vs_none"] = results[name]["likelihood"] - base
        results[name]["reward_effect"] = as_kernels(coef)["reward_effect"]
    summary = {}
    for dim in ("1-d", "2-d"):
        group = [results[f"random {dim} #{i}"] for i in range(N_RANDOM)]
        deltas = [g_["delta_vs_none"] for g_ in group]
        kern = np.array([g_["reward_effect"] for g_ in group])
        summary[f"random {dim}"] = {"mean_delta": float(np.mean(deltas)), "min_delta": float(np.min(deltas)),
                                    "max_delta": float(np.max(deltas)), "kernel_mean": kern.mean(axis=0),
                                    "kernel_min": kern.min(axis=0), "kernel_max": kern.max(axis=0)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), gridspec_kw={"width_ratios": [1, 1.2]})
    labels = NAMED + ["random 1-d", "random 2-d"]
    vals = [results[n]["delta_vs_none"] for n in NAMED] + [summary[k]["mean_delta"] for k in ("random 1-d", "random 2-d")]
    y = np.arange(len(labels))
    ax1.barh(y, vals, color=["#4C72B0"] * len(NAMED) + ["#999999"] * 2)
    for k, key in enumerate(("random 1-d", "random 2-d")):
        ax1.plot([summary[key]["min_delta"], summary[key]["max_delta"]], [y[len(NAMED) + k]] * 2, color="k", lw=2)
    ax1.set_yticks(y, labels, fontsize=11)
    ax1.invert_yaxis()
    ax1.axvline(0, color="0.6", lw=0.8)
    ax1.set_xlabel("Δ eval likelihood vs intact (memory-only, conditional-mean replacement)")
    ax1.set_title(f"Ablation cost; random: mean and range of {N_RANDOM} (intact {base:.4f})", fontsize=13)
    lags = np.arange(1, len(results["none"]["reward_effect"]) + 1)
    ax2.plot(lags, results["none"]["reward_effect"], color="k", lw=3, marker="o", ms=3, label="intact")
    for n in NAMED:
        ax2.plot(lags, results[n]["reward_effect"], lw=1.8, marker="o", ms=3, label=n)
    for key, color in (("random 1-d", "#bbbbbb"), ("random 2-d", "#777777")):
        ax2.fill_between(lags, summary[key]["kernel_min"], summary[key]["kernel_max"], color=color, alpha=0.35,
                         label=f"{key} range")
    ax2.axhline(0, color="0.7", lw=0.8)
    ax2.set_xlabel("trials back")
    ax2.set_ylabel("reward effect (Su2022, soft labels)")
    ax2.set_title("Reward kernel after ablation", fontsize=13)
    ax2.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_E2b_conditional_ablations.png"))
    plt.close(fig)
    s12.write_json("e2b_conditional_ablations.json", {
        "intact_likelihood": base, "mode": "memory-only", "replacement": "mean state of the last-input cluster (5 means)",
        "results": results, "random_summary": summary, "n_random": N_RANDOM, "random_seed": SEED,
    }, "analysis/e2b_conditional_ablations.py")


if __name__ == "__main__":
    main()
