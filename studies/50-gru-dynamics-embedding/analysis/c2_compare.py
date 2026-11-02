"""C2: the GRU against the fitted RL baselines, trial by trial, on the same 4.3M trials.

    python analysis/c2_compare.py

Reads the A1 table, the B3 PCA and the C2 rollouts (all cached). Per-trial advantage =
log p_GRU(choice) − log p_RL(choice); CTT (the strongest RL model) is the reference in the
figures, all three are in the JSON. "Where the GRU wins" is split only by quantities known
before the choice (state, previous trial, each model's predicted P(switch)); splitting by
whether the animal then switched conditions on the outcome being scored and favours any
better-calibrated model on rare outcomes, so that split is kept in the JSON only, labelled
as outcome-conditioned. Previous choice/outcome are taken from the table's own shift (the
add_candidate_variables columns are NaN for each session's first 10 trials). Writes ``fig_C2a_likelihood.png``,
``fig_C2b_agreement.png``, ``fig_C2c_advantage.png`` and ``c2_compare.json``.
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

RL = ("bari", "ctt", "hattori")
REF = "ctt"
MODEL_COLORS = {"gru": "#222222", "bari": "#4C72B0", "ctt": "#C44E52", "hattori": "#55A868"}
MIN_PER_BIN = 200


def log_p(action: np.ndarray, p_right: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.where(action == 1, p_right, 1 - p_right), 1e-10, 1 - 1e-10))


def binned(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    which = np.digitize(x, edges) - 1
    n = np.bincount(which[(which >= 0) & (which < len(edges) - 1)], minlength=len(edges) - 1)
    sums = np.bincount(which[(which >= 0) & (which < len(edges) - 1)], weights=y[(which >= 0) & (which < len(edges) - 1)],
                       minlength=len(edges) - 1)
    mean = np.where(n >= MIN_PER_BIN, sums / np.maximum(n, 1), np.nan)
    sq = np.bincount(which[(which >= 0) & (which < len(edges) - 1)], weights=y[(which >= 0) & (which < len(edges) - 1)] ** 2,
                     minlength=len(edges) - 1)
    sem = np.sqrt(np.maximum(sq / np.maximum(n, 1) - (sums / np.maximum(n, 1)) ** 2, 0) / np.maximum(n - 1, 1))
    return mean, np.where(n >= MIN_PER_BIN, sem, np.nan), n


def main() -> None:
    s12.apply_presentation_style()
    t = s12.load_trials(["subject_id", "ses_idx", "task", "action", "reward", "p_model1_right", "logit_diff",
                         "trial_position"])
    action = t["action"].to_numpy()
    ll = {"gru": log_p(action, t["p_model1_right"].to_numpy())}
    p_rl = {}
    for name in RL:
        p_rl[name] = pd.read_parquet(s12.CACHE / f"rl_trials_{name}_{s12.MODEL_KEY}.parquet", columns=["p_rl_right"])["p_rl_right"].to_numpy()
        ll[name] = log_p(action, p_rl[name])
    scores = s12.load_pca()["scores"][:, :2]
    adv = {name: ll["gru"] - ll[name] for name in RL}

    # (a) likelihood by task and per mouse.
    groups = {"all": np.ones(len(t), bool), **{task: (t["task"] == task).to_numpy() for task in s12.TASKS}}
    lik = {g: {m: float(np.exp(ll[m][mask].mean())) for m in ll} for g, mask in groups.items()}
    per_mouse = pd.DataFrame({m: pd.Series(ll[m]).groupby(t["subject_id"].to_numpy()).mean() for m in ll}).apply(np.exp)
    frac_better = {name: float(np.mean(per_mouse["gru"] > per_mouse[name])) for name in RL}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.2), gridspec_kw={"width_ratios": [1.6, 1]})
    width = 0.2
    for i, m in enumerate(ll):
        ax1.bar(np.arange(len(groups)) + (i - 1.5) * width, [lik[g][m] for g in groups], width,
                color=MODEL_COLORS[m], label=m.upper() if m == "gru" else m)
    ax1.set_xticks(range(len(groups)), [g.replace(" Without", "\nWithout") for g in groups], fontsize=11)
    lo = min(min(v.values()) for v in lik.values())
    ax1.set_ylim(lo - 0.01, max(max(v.values()) for v in lik.values()) + 0.005)
    ax1.set_ylabel("eval likelihood (exp mean log p)")
    ax1.legend(frameon=False, ncol=4, fontsize=11)
    ax2.scatter(per_mouse[REF], per_mouse["gru"], s=8, alpha=0.6, color="#555555")
    lims = [per_mouse.min().min() - 0.01, per_mouse.max().max() + 0.01]
    ax2.plot(lims, lims, color="0.6", lw=1, ls=":")
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel(f"{REF.upper()} likelihood (per mouse)")
    ax2.set_ylabel("GRU likelihood (per mouse)")
    ax2.set_title(f"GRU better in {100 * frac_better[REF]:.1f}% of {len(per_mouse)} mice", fontsize=13)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_C2a_likelihood.png"))
    plt.close(fig)

    # (b) trial-by-trial agreement of P(right).
    logit_rl = {name: np.log(np.clip(p_rl[name], 1e-6, 1 - 1e-6) / np.clip(1 - p_rl[name], 1e-6, 1)) for name in RL}
    agreement = {name: {"corr_p_right": float(np.corrcoef(t["p_model1_right"], p_rl[name])[0, 1]),
                        "corr_logit": float(np.corrcoef(t["logit_diff"], logit_rl[name])[0, 1]),
                        "mean_abs_diff_p_right": float(np.mean(np.abs(t["p_model1_right"] - p_rl[name])))} for name in RL}
    fig, axes = plt.subplots(1, len(RL), figsize=(18, 5.6), sharey=True)
    for ax, name in zip(axes, RL):
        ax.hist2d(p_rl[name], t["p_model1_right"], bins=60, cmap="Greys", norm=matplotlib.colors.LogNorm(), rasterized=True)
        ax.plot([0, 1], [0, 1], color=MODEL_COLORS[name], lw=1.2, ls="--")
        ax.set_xlabel(f"{name} P(right)")
        ax.set_title(f"r = {agreement[name]['corr_p_right']:.3f}, mean |Δp| = {agreement[name]['mean_abs_diff_p_right']:.3f}",
                     fontsize=12)
    axes[0].set_ylabel("GRU P(right)")
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_C2b_agreement.png"))
    plt.close(fig)

    # (c) where the GRU wins: PC1-PC2 map and by condition.
    a = adv[REF]
    edges = [np.linspace(-4, 4, 41), np.linspace(-4, 3, 36)]
    ix = np.digitize(scores[:, 0], edges[0]) - 1
    iy = np.digitize(scores[:, 1], edges[1]) - 1
    ok = (ix >= 0) & (ix < 40) & (iy >= 0) & (iy < 35)
    flat = ix[ok] * 35 + iy[ok]
    n = np.bincount(flat, minlength=40 * 35)
    mean_map = np.where(n >= MIN_PER_BIN, np.bincount(flat, weights=a[ok], minlength=40 * 35) / np.maximum(n, 1), np.nan)
    total_map = np.bincount(flat, weights=a[ok], minlength=40 * 35)
    mean_map, total_map, n = (v.reshape(40, 35) for v in (mean_map, total_map, n))

    g = t.groupby("ses_idx", sort=False)
    prev_action = g["action"].shift(1)
    prev_reward = (g["reward"].shift(1) > 0).where(prev_action.notna())
    prev = (prev_action.map({0.0: "L", 1.0: "R", 0: "L", 1: "R"}) + " "
            + prev_reward.map({False: "unrew.", True: "rew."})).to_numpy()
    by_prev = {c: {"mean": float(a[prev == c].mean()), "sem": float(a[prev == c].std(ddof=1) / np.sqrt((prev == c).sum())),
                   "n": int((prev == c).sum())} for c in ("R rew.", "R unrew.", "L rew.", "L unrew.")}
    switch = (action != prev_action.to_numpy()).astype(float)
    switch[prev_action.isna().to_numpy()] = np.nan
    by_switch = {lab: {"mean": float(a[switch == v].mean()), "n": int((switch == v).sum())}
                 for lab, v in (("stay", 0.0), ("switch", 1.0))}
    # Pre-choice: each model's predicted probability of switching away from the previous choice.
    pa = prev_action.to_numpy()
    p_switch = {"gru": np.where(pa == 1, 1 - t["p_model1_right"].to_numpy(), t["p_model1_right"].to_numpy()),
                REF: np.where(pa == 1, 1 - p_rl[REF], p_rl[REF])}
    ps_edges = np.linspace(0, 1, 21)
    by_p_switch = {}
    for name, ps in p_switch.items():
        ok = np.isfinite(pa)
        mean_, sem_, n_ = binned(ps[ok], a[ok], ps_edges)
        which = np.digitize(ps[ok], ps_edges) - 1
        share = np.bincount(which[(which >= 0) & (which < 20)], weights=a[ok][(which >= 0) & (which < 20)], minlength=20)
        by_p_switch[name] = {"bin_edges": ps_edges, "mean": mean_, "sem": sem_, "n": n_,
                             "share_of_total_advantage": share / a[ok].sum()}
    conf_edges = np.linspace(0, 6, 25)
    conf_mean, conf_sem, conf_n = binned(np.abs(t["logit_diff"].to_numpy()), a, conf_edges)
    pos_edges = np.linspace(0, 1, 21)
    pos_mean, pos_sem, _ = binned(t["trial_position"].to_numpy(), a, pos_edges)

    fig, axes = plt.subplots(1, 4, figsize=(26, 6), gridspec_kw={"width_ratios": [1.25, 1.25, 0.9, 1]})
    lim = np.nanpercentile(np.abs(mean_map), 98)
    im = axes[0].imshow(mean_map.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto",
                        extent=(edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]))
    fig.colorbar(im, ax=axes[0], fraction=0.046, label=f"mean log p_GRU − log p_{REF.upper()}")
    axes[0].set_title("Mean advantage per trial", fontsize=13)
    tlim = np.nanpercentile(np.abs(total_map), 99.5)
    im = axes[1].imshow(total_map.T, origin="lower", cmap="RdBu_r", vmin=-tlim, vmax=tlim, aspect="auto",
                        extent=(edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]))
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="summed advantage (nats)")
    axes[1].set_title("Where the total advantage comes from", fontsize=13)
    for ax in axes[:2]:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    ps_centres = (ps_edges[:-1] + ps_edges[1:]) / 2
    for name, color in (("gru", "#222222"), (REF, "#C44E52")):
        axes[2].errorbar(ps_centres, by_p_switch[name]["mean"], yerr=by_p_switch[name]["sem"], color=color, marker="o", ms=4,
                         label=f"binned by {name.upper()}'s predicted P(switch)")
    axes[2].axhline(0, color="0.6", lw=0.8)
    axes[2].set_xlabel("predicted P(switch) before the choice")
    axes[2].set_ylabel("mean advantage (nats / trial)")
    axes[2].set_title("Advantage vs predicted switch probability", fontsize=13)
    axes[2].legend(frameon=False, fontsize=10)
    centres = (conf_edges[:-1] + conf_edges[1:]) / 2
    axes[3].errorbar(centres, conf_mean, yerr=conf_sem, color="#222222", marker="o", ms=4, label="vs GRU |logit|")
    axes[3].axhline(0, color="0.6", lw=0.8)
    axes[3].set_xlabel("GRU confidence |logit(right)|")
    axes[3].set_ylabel("mean advantage (nats / trial)")
    axes[3].set_title("Advantage vs the GRU's confidence", fontsize=13)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_C2c_advantage.png"))
    plt.close(fig)

    s12.write_json("c2_compare.json", {
        "reference_rl": REF, "likelihood": lik, "per_mouse_frac_gru_better": frac_better,
        "per_mouse_median_gain": {name: float(np.median(per_mouse["gru"] - per_mouse[name])) for name in RL},
        "agreement": agreement,
        "advantage_mean_nats_per_trial": {name: float(adv[name].mean()) for name in RL},
        "advantage_by_prev_choice_outcome": by_prev,
        "advantage_by_predicted_p_switch": by_p_switch,
        "advantage_stay_vs_switch_OUTCOME_CONDITIONED": by_switch,
        "advantage_by_confidence": {"bin_edges": conf_edges, "mean": conf_mean, "sem": conf_sem, "n": conf_n},
        "advantage_by_trial_position": {"bin_edges": pos_edges, "mean": pos_mean, "sem": pos_sem},
        "pc_map": {"pc1_edges": edges[0], "pc2_edges": edges[1], "min_per_bin": MIN_PER_BIN},
    }, "analysis/c2_compare.py", extra_groups=s12.rl_groups())


if __name__ == "__main__":
    main()
