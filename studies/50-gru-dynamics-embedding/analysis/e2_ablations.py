"""E2 (stage A): what each subspace of the state is needed for — mean-replacement ablations.

    JAX_PLATFORMS=cpu python analysis/e2_ablations.py

At every step the state's component in a subspace S is replaced by the mean state's, and the
network runs on (wrapper ``GruStep.run_ablated``). Two modes:
  * full        — the ablated state is read out and carried: is S needed at all?
  * memory-only — the choice is read from the intact state, only the carried state is
                  ablated: does S carry information to later trials?
Subspaces: I2's value, choice and reward axes and both lobe value axes; PC5 (time drift);
the readout direction; the readout-null space (everything but the readout); and random
1-d control directions drawn from the state covariance (typical-variance directions).
Scores: pooled eval likelihood, and the Su2022 reward kernel of the ablated model (C1's
design, soft labels) for the main conditions. An un-ablated replay must reproduce the A1
likelihood first. Writes ``e2_ablations.json`` and ``fig_E2_ablations.png``.
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

BATCH = 512
N_RANDOM = 10
SEED = 5
LIKELIHOOD_TOLERANCE = 2e-3  # CPU float32 replay vs the A1 GPU run
KERNEL_CONDITIONS = ["none", "value axis", "lobe value axes (2-d)", "choice axis", "reward axis", "PC5 (time)",
                     "readout direction", "random 1-d (mean)"]


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
    t = s12.load_trials(["ses_idx", "trial_idx", "action"])
    col = t["ses_idx"].map({s: i for i, s in enumerate(split.session_ids)}).to_numpy()
    tau = t["trial_idx"].to_numpy() - 1
    action = t["action"].to_numpy()

    pca = s12.load_pca()
    mean, components = pca["mean"].astype(np.float32), pca["components"]
    tdr = np.load(s12.keyed_cache("i2_tdr_axes.npz"))
    terms = [str(x) for x in tdr["terms"]]
    q = tdr["basis"]
    r = np.asarray(step.readout_w[:, 1] - step.readout_w[:, 0], dtype=np.float64)
    r /= np.linalg.norm(r)
    null = np.linalg.svd(np.eye(len(r)) - np.outer(r, r))[0][:, :len(r) - 1]
    cov = (components.T * np.clip(pca["eigvals"], 0, None)) @ components
    rng = np.random.default_rng(SEED)
    random_dirs = rng.multivariate_normal(np.zeros(len(r)), cov, size=N_RANDOM)
    subspaces = {
        "none": np.zeros((len(r), 0)),
        "value axis": q[:, [terms.index("value")]],
        "lobe value axes (2-d)": orthonormal(tdr["lobe_r"], tdr["lobe_l"]),
        "choice axis": q[:, [terms.index("choice")]],
        "reward axis": q[:, [terms.index("reward")]],
        "PC5 (time)": components[4][:, None],
        "readout direction": r[:, None],
        "readout-null space (127-d)": null,
        **{f"random 1-d #{i}": (d / np.linalg.norm(d))[:, None] for i, d in enumerate(random_dirs)},
    }

    run = jax.jit(lambda o, c, b, m, memory: step.run_ablated(o, c, b, m, memory_only=memory), static_argnums=4)
    n_sessions = obs.shape[1]

    def logits_for(basis: np.ndarray, memory: bool) -> np.ndarray:
        out = np.empty(obs.shape[:2], dtype=np.float32)
        for start in range(0, n_sessions, BATCH):
            idx = np.arange(start, min(start + BATCH, n_sessions))
            pad = np.r_[idx, np.full(BATCH - len(idx), idx[-1])]  # fixed shape: one compilation
            c = np.broadcast_to(contexts[pad][None], obs[:, pad].shape[:2] + (contexts.shape[1],))
            res = np.asarray(run(jnp.asarray(obs[:, pad]), jnp.asarray(c), jnp.asarray(basis, dtype=jnp.float32),
                                 jnp.asarray(mean), memory))
            out[:, idx] = res[:, :len(idx)]
        return out

    def likelihood(p_right: np.ndarray) -> float:
        p = np.where(action == 1, p_right, 1 - p_right)
        return float(np.exp(np.mean(np.log(np.clip(p, 1e-10, 1 - 1e-10)))))

    p_by_condition, results = {}, {}
    for name, basis in subspaces.items():
        for memory in (False, True):
            if name == "none" and memory:
                continue
            logit = logits_for(basis, memory)[tau, col]
            p_right = 1 / (1 + np.exp(-logit))
            key = f"{name} | {'memory-only' if memory else 'full'}"
            results[key] = {"likelihood": likelihood(p_right), "k": int(basis.shape[1])}
            if not name.startswith("random"):
                p_by_condition[key] = p_right.astype(np.float32)
            else:
                p_by_condition.setdefault(f"random 1-d (mean) | {'memory-only' if memory else 'full'}", []).append(p_right)
            print(key, round(results[key]["likelihood"], 5), flush=True)
    base = results["none | full"]["likelihood"]
    logged = json_load(s12.STUDY / "analysis" / f"trial_states_{s12.MODEL_KEY}.json")["checks"]["logged_eval_likelihood"]
    if abs(base - logged) > LIKELIHOOD_TOLERANCE:
        raise AssertionError(f"Un-ablated replay likelihood {base} vs logged {logged}")
    for key in results:
        results[key]["delta_vs_none"] = results[key]["likelihood"] - base
    for mode in ("full", "memory-only"):
        vals = [results[f"random 1-d #{i} | {mode}"]["delta_vs_none"] for i in range(N_RANDOM)]
        results[f"random 1-d (summary) | {mode}"] = {"mean_delta": float(np.mean(vals)), "min_delta": float(np.min(vals)),
                                                     "max_delta": float(np.max(vals))}
        p_by_condition[f"random 1-d (mean) | {mode}"] = np.mean(p_by_condition[f"random 1-d (mean) | {mode}"], axis=0)

    # Reward kernels of the ablated models (C1 design, soft labels, pooled).
    x, rows_t = design()
    position = pd.Series(np.arange(len(t)), index=pd.MultiIndex.from_arrays([t["ses_idx"], t["trial_idx"]]))
    design_rows = position.loc[list(zip(rows_t["ses_idx"], rows_t["trial_idx"]))].to_numpy()
    kernels = {}
    for key, p in p_by_condition.items():
        if key.split(" | ")[0] in KERNEL_CONDITIONS:
            kernels[key] = as_kernels(fit_rows(x, np.asarray(p)[design_rows].astype(np.float64), None, True))
            print("kernel", key, np.round(kernels[key]["reward_effect"][:5], 3), flush=True)

    labels = [k for k in subspaces if not k.startswith("random") and k != "none"] + ["random 1-d (summary)"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), gridspec_kw={"width_ratios": [1.2, 1]})
    y = np.arange(len(labels))
    for off, mode, color in ((-0.2, "full", "#4C72B0"), (0.2, "memory-only", "#E69F00")):
        vals = [results[f"{lab} | {mode}"]["mean_delta" if lab.startswith("random") else "delta_vs_none"] for lab in labels]
        ax1.barh(y + off, vals, 0.4, color=color, label=mode)
        lo = results[f"random 1-d (summary) | {mode}"]
        ax1.plot([lo["min_delta"], lo["max_delta"]], [y[-1] + off] * 2, color="k", lw=2)
    ax1.set_yticks(y, labels, fontsize=11)
    ax1.invert_yaxis()
    ax1.axvline(0, color="0.6", lw=0.8)
    ax1.set_xscale("symlog", linthresh=1e-3)
    ax1.set_xlabel("Δ eval likelihood vs intact model (symlog)")
    ax1.set_title(f"Ablation cost (intact {base:.4f}); random: mean and range of {N_RANDOM}", fontsize=13)
    ax1.legend(frameon=False)
    lags = np.arange(1, len(kernels["none | full"]["reward_effect"]) + 1)
    for key, kern in kernels.items():
        name, mode = key.split(" | ")
        if mode == "memory-only" or name == "none":
            ax2.plot(lags, kern["reward_effect"], lw=3 if name == "none" else 1.6, color="k" if name == "none" else None,
                     ls="-" if name == "none" else "--" if name.startswith("random") else "-",
                     label="intact" if name == "none" else name, marker="o", ms=3)
    ax2.axhline(0, color="0.7", lw=0.8)
    ax2.set_xlabel("trials back")
    ax2.set_ylabel("reward effect (Su2022, soft labels)")
    ax2.set_title("Reward kernel after memory-only ablation", fontsize=13)
    ax2.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_E2_ablations.png"))
    plt.close(fig)
    s12.write_json("e2_ablations.json", {
        "intact_likelihood": base, "logged_likelihood": logged, "modes": ["full", "memory-only"],
        "replacement": "component in S set to the mean state's (PCA mean) at every step", "results": results,
        "kernels": kernels, "n_random": N_RANDOM, "random_seed": SEED,
        "random_directions": "drawn from N(0, Σ_state), normalised",
    }, "analysis/e2_ablations.py")


def json_load(path: Path) -> dict:
    import json  # noqa: PLC0415

    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
