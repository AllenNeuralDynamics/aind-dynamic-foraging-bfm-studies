"""C4: how the GRU's behaviour differs from the RL models', judged against the animals.

    python analysis/c4_behaviour.py

Models: the GRU, Hattori (L2F1), Bari (L1F1_CK1), CTT; all per-trial predictions on the same
4.34M eval trials. Row t's prediction is made before choice t; the transition t → t+1 is
grouped by trial t's choice × reward and whether it repeated trial t−1's side
(``_transitions``). Four figures, models always in side-by-side panels:

  C4a  logit-change maps: Δlogit = logit(t+1) − logit(t) against logit(t), per input (rows)
       and model (columns); stay solid, switch dashed; binned median and IQR.
  C4b  the same, anchored to the animals: within each group, trials are binned by the
       model's own logit(t); per bin, the model's mean predicted P(right at t+1) against the
       animals' empirical rate, both as log-odds. On the diagonal = the model updates the way
       the animals do. Summary: n-weighted RMS deviation (log-odds) per model and group.
  C4c  calibration by three-trial history: for each of the 64 patterns of the inputs at
       t−1..t−3, the animals' P(right at t) and each model's mean prediction minus it, laid out
       like fig_B3c_prev_outcome_t3.
  C4d  where the GRU and CTT disagree: trials binned by p_GRU(stay) − p_CTT(stay); the
       animals' P(stay) − p_CTT(stay) against it (the GRU is right on the diagonal, CTT on
       zero). Then the 5% of trials with the largest |disagreement| profiled against all
       trials.

Writes ``fig_C4a_logit_updates.png``, ``fig_C4b_update_calibration.png``,
``fig_C4c_history_calibration.png``, ``fig_C4d_disagreement.png`` and ``c4_behaviour.json``.
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
import _transitions as tr  # noqa: E402
from b3c_history import INPUT_LONG, INPUTS, history_codes  # noqa: E402

MODELS = {"gru": "GRU", "hattori": "Hattori (L2F1)", "bari": "Bari (L1F1_CK1)", "ctt": "CTT"}
COLORS = {"gru": "#222222", "hattori": "#55A868", "bari": "#4C72B0", "ctt": "#C44E52"}
INPUT_ORDER = ["R rewarded", "R unrewarded", "L rewarded", "L unrewarded"]  # = INPUTS (R+, R−, L+, L−)
N_BINS = 24
MIN_PER_BIN = 200
P_CLIP = 1e-4
TOP_FRACTION = 0.05


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, P_CLIP, 1 - P_CLIP)
    return np.log(p / (1 - p))


def transition_rows(lay: dict, name: str, kind: str) -> np.ndarray:
    idx = np.flatnonzero(lay["has_next"])
    return idx[lay["group"][idx] == tr.GROUP_NAMES.index(f"{name} ({kind})")]


def bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(x, edges) - 1, -1, len(edges) - 1)


def update_maps(z: dict, lay: dict, edges: dict) -> dict:
    """C4a: binned median / IQR of Δlogit against logit(t) per model and group."""
    out = {}
    for m, x in z.items():
        e = edges[m]
        centres = (e[:-1] + e[1:]) / 2
        for name in INPUT_ORDER:
            for kind in ("stay", "switch"):
                rows = transition_rows(lay, name, kind)
                d = x[rows + 1] - x[rows]
                b = bin_index(x[rows], e)
                ok = (b >= 0) & (b < N_BINS)
                q = pd.Series(d[ok]).groupby(b[ok]).quantile([0.25, 0.5, 0.75]).unstack()
                n = pd.Series(b[ok]).value_counts()
                q = q[n.reindex(q.index).fillna(0).to_numpy() >= MIN_PER_BIN]
                out[(m, name, kind)] = {"x": centres[q.index], "q25": q[0.25].to_numpy(), "median": q[0.5].to_numpy(),
                                        "q75": q[0.75].to_numpy()}
    return out


def update_calibration(p: dict, z: dict, action: np.ndarray, lay: dict, edges: dict) -> tuple[dict, dict]:
    """C4b: per bin of the model's logit(t), predicted vs empirical log-odds of right at t+1."""
    curves, rms = {}, {}
    for m in p:
        rms[m] = {}
        for name in INPUT_ORDER:
            for kind in ("stay", "switch"):
                rows = transition_rows(lay, name, kind)
                b = bin_index(z[m][rows], edges[m])
                ok = (b >= 0) & (b < N_BINS)
                df = pd.DataFrame({"b": b[ok], "pred": p[m][rows + 1][ok], "obs": action[rows + 1][ok]})
                g = df.groupby("b").agg(pred=("pred", "mean"), obs=("obs", "mean"), n=("obs", "size"))
                g = g[g["n"] >= MIN_PER_BIN]
                obs = (g["obs"] * g["n"] + 0.5) / (g["n"] + 1)  # keeps empty-tail bins finite
                pred_lo, obs_lo = logit(g["pred"].to_numpy()), logit(obs.to_numpy())
                curves[(m, name, kind)] = {"pred": pred_lo, "obs": obs_lo, "n": g["n"].to_numpy()}
                w = g["n"].to_numpy() / g["n"].sum()
                rms[m][f"{name} ({kind})"] = float(np.sqrt(np.sum(w * (pred_lo - obs_lo) ** 2)))
    return curves, rms


def history_calibration(p: dict, action: np.ndarray, codes: np.ndarray) -> dict:
    """C4c: animals' P(right) and each model's mean prediction for each 3-trial history."""
    ok = (codes >= 0).all(axis=1)
    key = codes[ok] @ np.array([16, 4, 1])
    out = {"n": np.bincount(key, minlength=64), "animal": np.bincount(key, weights=action[ok], minlength=64)}
    out["animal"] = out["animal"] / np.maximum(out["n"], 1)
    for m in p:
        out[m] = np.bincount(key, weights=p[m][ok], minlength=64) / np.maximum(out["n"], 1)
    return out


def grid(values: np.ndarray) -> np.ndarray:
    """64 history values (index 16·a + 4·b + c) → 8 × 8 B3c layout (block a; row b, column c)."""
    g = np.empty((8, 8))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                g[(a // 2) * 4 + b, (a % 2) * 4 + c] = values[16 * a + 4 * b + c]
    return g


def draw_grid(ax, values, title, cmap, vmin, vmax, fmt):
    g = grid(values)
    im = ax.imshow(g, cmap=cmap, vmin=vmin, vmax=vmax)
    for r in range(8):
        for c in range(8):
            ax.text(c, r, fmt(g[r, c]), ha="center", va="center", fontsize=10)
    ax.axhline(3.5, color="k", lw=2)
    ax.axvline(3.5, color="k", lw=2)
    ax.set_xticks(range(8), INPUTS * 2, fontsize=10)
    ax.set_yticks(range(8), INPUTS * 2, fontsize=10)
    ax.set_xlabel("t−3 (within each block)")
    ax.set_ylabel("t−2 (within each block)")
    for a in range(4):  # top blocks labelled above the grid, bottom blocks on the dividing line
        y, va = (-0.62, "bottom") if a // 2 == 0 else (3.5, "center")
        ax.text((a % 2) * 4 + 1.5, y, f"t−1 {INPUTS[a]}", ha="center", va=va, fontsize=11, weight="bold",
                bbox=None if a // 2 == 0 else {"facecolor": "white", "edgecolor": "k", "pad": 1.5})
    ax.set_title(title, fontsize=13, pad=26)
    return im


def main() -> None:
    s12.apply_presentation_style()
    t = s12.load_trials(["subject_id", "ses_idx", "trial_idx", "task", "action", "reward", "p_model1_right",
                         "trial_position", "trials_since_reward", "reward_probabilityL", "reward_probabilityR"])
    action = t["action"].to_numpy().astype(float)
    p = {"gru": t["p_model1_right"].to_numpy(dtype=np.float64)}
    for m in ("hattori", "bari", "ctt"):
        rl = pd.read_parquet(s12.CACHE / f"rl_trials_{m}_{s12.MODEL_KEY}.parquet")
        if not (rl["ses_idx"].to_numpy() == t["ses_idx"].to_numpy()).all():
            raise AssertionError(f"{m} rollout rows are not aligned with the trial table")
        p[m] = rl["p_rl_right"].to_numpy(dtype=np.float64)
    z = {m: logit(v) for m, v in p.items()}
    lay = tr.session_layout(t)
    edges = {m: np.linspace(*np.percentile(v, [1, 99]), N_BINS + 1) for m, v in z.items()}

    # C4a: logit-change maps, rows = input, columns = model.
    maps = update_maps(z, lay, edges)
    lim = max(np.nanmax(np.abs(np.r_[v["q25"], v["q75"]])) for v in maps.values())
    fig, axes = plt.subplots(4, 4, figsize=(24, 22), sharey=True)
    for r, name in enumerate(INPUT_ORDER):
        for c, m in enumerate(MODELS):
            ax = axes[r, c]
            for kind, ls in (("stay", "-"), ("switch", "--")):
                v = maps[(m, name, kind)]
                ax.plot(v["x"], v["median"], color=COLORS[m], ls=ls, lw=2, marker="o", ms=3, label=f"{kind} (median)")
                ax.fill_between(v["x"], v["q25"], v["q75"], color=COLORS[m], alpha=0.15 if kind == "stay" else 0.08)
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_ylim(-lim, lim)
            if r == 0:
                ax.set_title(MODELS[m], fontsize=15)
            if c == 0:
                ax.set_ylabel(f"trial t: {name}\nΔlogit = logit(t+1) − logit(t)")
            if r == 3:
                ax.set_xlabel("logit P(right) before trial t")
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle("How each model's policy changes after each input (band: interquartile range)", fontsize=17)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_C4a_logit_updates.png"))
    plt.close(fig)

    # C4b: the same, anchored to the animals' next choice.
    curves, rms = update_calibration(p, z, action, lay, edges)
    fig, axes = plt.subplots(4, 4, figsize=(24, 22), sharex=True, sharey=True)
    for r, name in enumerate(INPUT_ORDER):
        for c, m in enumerate(MODELS):
            ax = axes[r, c]
            for kind, marker in (("stay", "o"), ("switch", "s")):
                v = curves[(m, name, kind)]
                ax.plot(v["pred"], v["obs"], color=COLORS[m], ls="-" if kind == "stay" else "--", marker=marker, ms=4,
                        lw=1.6, label=f"{kind}: RMS {rms[m][f'{name} ({kind})']:.2f}")
            ax.plot([-6, 6], [-6, 6], color="0.6", ls=":", lw=1)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            if r == 0:
                ax.set_title(MODELS[m], fontsize=15)
            if c == 0:
                ax.set_ylabel(f"trial t: {name}\nanimals' log-odds of right at t+1")
            if r == 3:
                ax.set_xlabel("model's predicted log-odds at t+1")
            ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.suptitle("Does each model's update match the animals'? Bins of the model's own logit(t); "
                 "diagonal = matches (RMS: n-weighted log-odds deviation)", fontsize=16)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_C4b_update_calibration.png"))
    plt.close(fig)

    # C4c: calibration by three-trial history.
    codes = history_codes(t, 3)
    hist = history_calibration(p, action, codes)
    resid = {m: 100 * (hist[m] - hist["animal"]) for m in MODELS}
    rlim = np.ceil(max(np.abs(v).max() for v in resid.values()))
    fig, axes = plt.subplots(2, 3, figsize=(27, 19), layout="constrained")
    im = draw_grid(axes[0, 0], 100 * hist["animal"], "Animals: P(right) at t (%)", "coolwarm", 0, 100, lambda x: f"{x:.0f}")
    fig.colorbar(im, ax=axes[0, 0], location="left", fraction=0.046, pad=0.08, label="animals' P(right) (%)")
    mae = {m: float(np.average(np.abs(resid[m]), weights=hist["n"])) for m in MODELS}
    for ax, m in zip((axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]), MODELS):
        im = draw_grid(ax, resid[m], f"{MODELS[m]} − animals (pp)", "PuOr_r", -rlim, rlim, lambda x: f"{x:+.1f}")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, label="predicted − observed P(right) (pp; all residual panels)")
    axes[1, 2].barh(range(len(MODELS)), [mae[m] for m in MODELS], color=[COLORS[m] for m in MODELS])
    axes[1, 2].set_yticks(range(len(MODELS)), list(MODELS.values()), fontsize=13)
    axes[1, 2].invert_yaxis()
    for k, m in enumerate(MODELS):
        axes[1, 2].text(mae[m] + 0.03, k, f"{mae[m]:.2f}", va="center", fontsize=12)
    axes[1, 2].set_xlabel("trial-weighted mean |predicted − observed| (pp)")
    axes[1, 2].set_title("Calibration error over the 64 histories", fontsize=13)
    fig.suptitle("Calibration by the last three inputs (+ rewarded, − unrewarded); every pattern has ≥ 3,500 trials",
                 fontsize=17)
    fig.savefig(s12.figure_path("fig_C4c_history_calibration.png"))
    plt.close(fig)

    # C4d: where the GRU and CTT disagree.
    prev = np.r_[np.nan, action[:-1]]
    prev[lay["first"]] = np.nan
    ok = ~np.isnan(prev)
    stay = {m: np.where(prev == 1, v, 1 - v) for m, v in p.items()}
    obs_stay = (action == prev).astype(float)
    d = {m: stay["gru"] - stay[m] for m in ("hattori", "bari", "ctt")}
    d_edges = np.linspace(-0.5, 0.5, 26)
    calib = {}
    for m, dm in d.items():
        b = bin_index(dm[ok], d_edges)
        keep = (b >= 0) & (b < len(d_edges) - 1)
        g = pd.DataFrame({"b": b[keep], "d": dm[ok][keep], "y": (obs_stay - stay[m])[ok][keep]}).groupby("b").agg(
            d=("d", "mean"), y=("y", "mean"), sem=("y", "sem"), n=("y", "size"))
        calib[m] = g[g["n"] >= MIN_PER_BIN]
    a = d["ctt"][ok]
    cut = np.quantile(np.abs(a), 1 - TOP_FRACTION)
    top = np.zeros(len(t), bool)
    top[np.flatnonzero(ok)[np.abs(a) >= cut]] = True
    # Length of the run of same-side choices ending at trial t−1 (0 for a session's first trial).
    new_run = lay["first"] | np.r_[True, action[1:] != action[:-1]]
    run_pos = pd.Series(np.ones(len(t))).groupby(np.cumsum(new_run)).cumsum().to_numpy()
    run_len = np.where(lay["first"], 0, np.r_[0, run_pos[:-1]])
    profile = {
        "run length of previous side (capped 10)": np.minimum(run_len, 10),
        "trials since reward (capped 10)": np.minimum(t["trials_since_reward"].to_numpy(), 10),
        "|P(rew|R) − P(rew|L)|": np.abs(t["reward_probabilityR"].to_numpy() - t["reward_probabilityL"].to_numpy()),
        "trial position in session": t["trial_position"].to_numpy(),
    }
    llg = np.log(np.clip(np.where(action == 1, p["gru"], 1 - p["gru"]), 1e-10, 1))
    llc = np.log(np.clip(np.where(action == 1, p["ctt"], 1 - p["ctt"]), 1e-10, 1))
    gru_says_stay_more = top & (d["ctt"] > 0)
    top_summary = {
        "threshold_abs_d": float(cut), "n_top": int(top.sum()),
        "share_of_total_advantage": float((llg - llc)[top].sum() / (llg - llc)[ok].sum()),
        "advantage_top_nats_per_trial": float((llg - llc)[top].mean()),
        "frac_gru_predicts_more_staying": float(gru_says_stay_more.sum() / top.sum()),
        "animal_stay_rate_top_gru_more_stay": float(obs_stay[gru_says_stay_more].mean()),
        "gru_stay_top_gru_more_stay": float(stay["gru"][gru_says_stay_more].mean()),
        "ctt_stay_top_gru_more_stay": float(stay["ctt"][gru_says_stay_more].mean()),
        "animal_stay_rate_top_gru_less_stay": float(obs_stay[top & ~gru_says_stay_more].mean()),
        "gru_stay_top_gru_less_stay": float(stay["gru"][top & ~gru_says_stay_more].mean()),
        "ctt_stay_top_gru_less_stay": float(stay["ctt"][top & ~gru_says_stay_more].mean()),
    }
    last = np.where(codes[:, 0] >= 0, codes[:, 0], -1)
    last_share = {INPUT_LONG[INPUTS[k]]: {"top": float(np.mean(last[top] == k)), "all": float(np.mean(last[ok] == k))}
                  for k in range(4)}
    task = t["task"].fillna("missing").to_numpy()
    task_share = {k: {"top": float(np.mean(task[top] == k)), "all": float(np.mean(task[ok] == k))} for k in s12.TASKS}

    fig, axes = plt.subplots(2, 4, figsize=(30, 13))
    for ax, m in zip(axes[0, :3], ("ctt", "hattori", "bari")):
        g = calib[m]
        ax.errorbar(g["d"], g["y"], yerr=g["sem"], color=COLORS[m], marker="o", ms=4, lw=1.6, label="animals")
        ax.plot([-0.5, 0.5], [-0.5, 0.5], color="k", ls="--", lw=1, label="GRU is right")
        ax.axhline(0, color=COLORS[m], ls=":", lw=1.2, label=f"{MODELS[m]} is right")
        ax.set_xlabel(f"p_GRU(stay) − p_{m.upper()}(stay)")
        ax.set_ylabel(f"animals' P(stay) − p_{m.upper()}(stay)")
        ax.set_title(f"GRU vs {MODELS[m]}: who is right where they disagree?", fontsize=13)
        ax.legend(frameon=False, fontsize=10)
    ax = axes[0, 3]
    labels = list(last_share)
    xx = np.arange(len(labels))
    ax.bar(xx - 0.2, [last_share[k]["all"] for k in labels], 0.4, color="0.7", label="all trials")
    ax.bar(xx + 0.2, [last_share[k]["top"] for k in labels], 0.4, color="#E69F00",
           label=f"top {100 * TOP_FRACTION:.0f}% |GRU − CTT|")
    ax.set_xticks(xx, labels, fontsize=11)
    ax.set_ylabel("fraction of trials")
    ax.set_title("Previous input of the top-disagreement trials", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    for ax, (label, x) in zip(axes[1], profile.items()):
        lo, hi = np.nanpercentile(x, [0, 100])
        e = np.linspace(lo, hi, 21) if "position" in label or "|P" in label else np.arange(-0.5, 11.5)
        ax.hist(x[ok & ~np.isnan(x)], bins=e, density=True, color="0.7", alpha=0.8, label="all trials")
        ax.hist(x[top & ~np.isnan(x)], bins=e, density=True, histtype="step", lw=2.2, color="#E69F00",
                label=f"top {100 * TOP_FRACTION:.0f}% |GRU − CTT|")
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=10)
    axes[1, 0].set_title("Profile of the top-disagreement trials", fontsize=13)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_C4d_disagreement.png"))
    plt.close(fig)

    s12.write_json("c4_behaviour.json", {
        "models": MODELS, "groups": [f"{n} ({k})" for n in INPUT_ORDER for k in ("stay", "switch")],
        "update_calibration_rms_logodds": rms,
        "history_calibration": {"patterns": [f"{INPUTS[i // 16]},{INPUTS[(i // 4) % 4]},{INPUTS[i % 4]}" for i in range(64)],
                                "pattern_order": "t−1, t−2, t−3", "n": hist["n"], "animal_p_right": hist["animal"],
                                **{f"{m}_minus_animal_pp": resid[m] for m in MODELS},
                                "weighted_mean_abs_error_pp": {m: float(np.average(np.abs(resid[m]), weights=hist["n"]))
                                                               for m in MODELS}},
        "disagreement_calibration": {m: {"d": g["d"].to_numpy(), "animal_minus_rl": g["y"].to_numpy(), "n": g["n"].to_numpy()}
                                     for m, g in calib.items()},
        "top_disagreement_vs_ctt": {**top_summary, "previous_input_share": last_share, "task_share": task_share,
                                    "profile_medians": {k: {"top": float(np.nanmedian(v[top])), "all": float(np.nanmedian(v[ok]))}
                                                        for k, v in profile.items()},
                                    "profile_means": {k: {"top": float(np.nanmean(v[top])), "all": float(np.nanmean(v[ok]))}
                                                      for k, v in profile.items()}},
        "n_bins": N_BINS, "min_per_bin": MIN_PER_BIN, "top_fraction": TOP_FRACTION, "p_clip": P_CLIP,
    }, "analysis/c4_behaviour.py", extra_groups=s12.rl_groups())
    for m in MODELS:
        print(m, "history |err| pp", round(float(np.average(np.abs(resid[m]), weights=hist["n"])), 3),
              "update RMS", {k: round(v, 2) for k, v in rms[m].items()})
    print(top_summary)
    print(last_share)
    print({k: (round(float(np.nanmean(v[top])), 3), round(float(np.nanmean(v[ok])), 3)) for k, v in profile.items()})


if __name__ == "__main__":
    main()
