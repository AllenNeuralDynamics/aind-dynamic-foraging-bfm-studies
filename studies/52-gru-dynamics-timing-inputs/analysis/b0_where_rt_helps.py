"""B0: where does the RT model (R) beat the choice/reward model (C), and what did it learn?

    python analysis/b0_where_rt_helps.py

Same eval trials for R-s0 and C-s0, restricted to sessions where every trial has an RT
(A2: R's whole gain lives there). ``dlog`` = log P_R(chosen) − log P_C(chosen) on trial t;
"prev RT" is trial t−1's RT, the input R received on row t. RT is binned two ways:
pooled deciles (mixes between-mouse and within-mouse variation) and within-mouse deciles
(each trial ranked among its own mouse's trials), which isolate trial-level RT. Error bands
are 95% CIs from resampling mice.

  B0a  mean dlog by prev-RT decile, split by prev reward.
  B0b  per-trial disagreement: the R − C dlog distribution beside the R seed 0 − seed 1
       distribution (how much two trainings of the same model disagree), gross vs net.
  B0c  P(switch) vs prev RT for mice, R and C (side by side), and each model minus mice.
  B0d  the same for RT at lags 1–3 (within-mouse deciles): mice, R − mice, C − mice. These are
       MARGINAL: each lag is binned alone, and RT is autocorrelated (≈0.3) and drifts within
       sessions, so a lag-2/3 trend can be lag-1 RT or session state. B0g is the conditional test.
  B0e  is RT redundant with C's value? RT_t vs C's confidence; dlog by RT and confidence.
  B0f  per mouse: gain vs the mouse's median RT and RT variability.
  B0g  conditional lag effects: logistic regression of switching on RT 1–3 back jointly
       (within-session z of log RT, so each lag is controlled for the others and for slow
       session drift), rewards 1–3 back and RT×reward 1 back. The same estimator is fitted to
       the mice (binary switch) and to each model's P(switch) as soft labels; 95% CIs from
       refitting on resampled mice. Reward is coded 0/1, so the "RT 1 back" coefficient is the
       slope after an UNREWARDED trial; after a rewarded one it is that plus RT×reward. Both
       lag-1 slopes are plotted.

Writes ``fig_B0[a-f]_*.png`` and ``b0_where_rt_helps.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pk03 import (  # noqa: E402
    ANIMAL_COLOR, MODEL_LABELS, N_BINS, REWARDED_COLOR, UNREWARDED_COLOR,
    apply_presentation_style, binned_means, figure_path, load_paired_trials, rt_bins, rt_log_axis, write_json,
)

PRODUCER = "analysis/b0_where_rt_helps.py"
MILLI = 1000.0  # dlog is shown in millinats per trial
REWARD_STYLE = {1.0: ("rewarded", REWARDED_COLOR, "-"), 0.0: ("unrewarded", UNREWARDED_COLOR, "--")}
CONF_COLORS = ["#9ECAE1", "#4292C6", "#08519C"]  # one hue, light → dark: low → high confidence


def _band(ax, x, row, value, color, ls, label, scale=1.0):
    ax.plot(x, row[value] * scale, color=color, ls=ls, lw=2, marker="o", ms=5, label=label)
    ax.fill_between(x, row[f"{value}_lo"] * scale, row[f"{value}_hi"] * scale, color=color, alpha=0.18, lw=0)


def _x_for(binning: str, table: pd.DataFrame, centres: np.ndarray) -> np.ndarray:
    idx = table["bin"].astype(int).to_numpy()
    return centres[idx] if binning == "pooled" else idx + 1


def _style_x(ax, binning: str, lag: int = 1) -> None:
    if binning == "pooled":
        rt_log_axis(ax)
        ax.set_xlabel(f"RT on trial t−{lag} (s, decile median)")
    else:
        ax.set_xlabel(f"RT on t−{lag}, within-mouse decile")
        ax.set_xticks([1, 5, 10])


def fig_gain_by_rt(df, centres, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    tables = {}
    for ax, binning in zip(axes, ("pooled", "within")):
        t = binned_means(df.assign(bin=df[f"bin_{binning}_lag1"]), ["bin", "reward_lag1"], ["dlog"])
        tables[binning] = t
        for reward, (name, color, ls) in REWARD_STYLE.items():
            row = t[t["reward_lag1"] == reward].sort_values("bin")
            _band(ax, _x_for(binning, row, centres), row, "dlog", color, ls, f"t−1 {name}", MILLI)
        ax.axhline(0, color="#888888", lw=1)
        ax.axhline(df["dlog"].mean() * MILLI, color="#888888", lw=1, ls=":", label="mean over all trials")
        _style_x(ax, binning)
        ax.set_title("pooled RT deciles" if binning == "pooled" else "within-mouse RT deciles")
    axes[0].set_ylabel("log P_R − log P_C (millinats/trial)")
    axes[0].legend(frameon=False, fontsize=12)
    fig.suptitle("B0a  Where R beats C: per-trial gain by the previous trial's RT", y=1.02)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return {k: v.to_dict("records") for k, v in tables.items()}


def _disagreement(d: np.ndarray) -> dict:
    return {"mean": float(d.mean()), "sd": float(d.std()), "fraction_positive": float(np.mean(d > 0)),
            "gross_positive": float(d[d > 0].sum()), "gross_negative": float(-d[d < 0].sum()),
            "net": float(d.sum()), "mean_abs": float(np.abs(d).mean())}


def fig_disagreement(df, out):
    pairs = {"R − C": df["dlog"].to_numpy(), "R seed 0 − R seed 1": (df["log_p"] - df["log_p_seed"]).to_numpy()}
    summary = {name: _disagreement(d) for name, d in pairs.items()}
    lim = max(np.abs(d).max() for d in pairs.values()) * MILLI
    bins = np.linspace(-lim, lim, 201)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, (name, d) in zip(axes[:2], pairs.items()):
        ax.hist(d * MILLI, bins=bins, color="#4C72B0")
        ax.set_yscale("log")
        ax.set_xlabel("Δ log P(chosen) per trial (millinats)")
        st = summary[name]
        ax.set_title(f"{name}: sd {st['sd'] * MILLI:.0f}, mean {st['mean'] * MILLI:+.2f}")
    axes[0].set_ylabel("trials")
    axes[1].sharey(axes[0])
    names = list(pairs)
    x = np.arange(len(names))
    for offset, key, color, label in ((-0.2, "gross_positive", "#0072B2", "favours 1st model"),
                                      (0.2, "gross_negative", "#C44E52", "favours 2nd model")):
        axes[2].bar(x + offset, [summary[n][key] for n in names], width=0.38, color=color, label=label)
    for i, n in enumerate(names):
        axes[2].annotate(f"net {summary[n]['net']:+.0f}", (i, max(summary[n]["gross_positive"], summary[n]["gross_negative"])),
                         textcoords="offset points", xytext=(0, 6), ha="center", fontsize=13)
    axes[2].set_xticks(x, names, fontsize=13)
    axes[2].set_ylabel("summed |difference| (nats)")
    axes[2].legend(frameon=False, fontsize=11, loc="lower left")
    fig.suptitle(f"B0b  How much do the models disagree trial by trial? ({len(df):,} trials)", y=1.02)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return summary


def _switch_tables(df, bin_column):
    return binned_means(df.assign(bin=df[bin_column]), ["bin", "reward_lag1"], ["switch", "p_switch", "p_switch_seed", "p_switch_C"])


def fig_switch(df, centres, out, out_resid):
    sources = [("mice", "switch"), ("R", "p_switch"), ("C", "p_switch_C")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    fig_r, axes_r = plt.subplots(2, 3, figsize=(17, 9), sharey=True)
    tables = {}
    for i, binning in enumerate(("pooled", "within")):
        t = _switch_tables(df, f"bin_{binning}_lag1")
        tables[binning] = t
        for j, (name, col) in enumerate(sources):
            ax = axes[i, j]
            for reward, (rname, color, ls) in REWARD_STYLE.items():
                row = t[t["reward_lag1"] == reward].sort_values("bin")
                _band(ax, _x_for(binning, row, centres), row, col, color, ls, f"t−1 {rname}")
            _style_x(ax, binning)
            if i == 0:
                ax.set_title(MODEL_LABELS[name])
        axes[i, 0].set_ylabel(("pooled" if binning == "pooled" else "within-mouse") + " deciles\nP(switch on t)")
        for j, (name, col) in enumerate([("R", "p_switch"), ("R seed 1", "p_switch_seed"), ("C", "p_switch_C")]):
            ax = axes_r[i, j]
            for reward, (rname, color, ls) in REWARD_STYLE.items():
                row = t[t["reward_lag1"] == reward].sort_values("bin")
                ax.plot(_x_for(binning, row, centres), row[col] - row["switch"], color=color, ls=ls, lw=2,
                        marker="o", ms=5, label=f"t−1 {rname}")
            ax.axhline(0, color="#888888", lw=1)
            _style_x(ax, binning)
            if i == 0:
                ax.set_title(f"{MODEL_LABELS.get(name, 'RT model, seed 1')} − mice")
        axes_r[i, 0].set_ylabel(("pooled" if binning == "pooled" else "within-mouse") + " deciles\nP(switch): model − mice")
    axes[0, 0].legend(frameon=False, fontsize=12)
    axes_r[0, 0].legend(frameon=False, fontsize=12)
    fig.suptitle("B0c  Switching after fast vs slow responses: mice and both models", y=1.0)
    fig_r.suptitle("B0c  Each model's error in P(switch), by previous RT", y=1.0)
    fig.savefig(out, bbox_inches="tight")
    fig_r.savefig(out_resid, bbox_inches="tight")
    plt.close(fig)
    plt.close(fig_r)
    return {k: v.to_dict("records") for k, v in tables.items()}


def fig_lags(df, out):
    rows = [("Mice", None), ("R − mice", "p_switch"), ("C − mice", "p_switch_C")]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    tables = {}
    for j, lag in enumerate((1, 2, 3)):
        t = _switch_tables(df, f"bin_within_lag{lag}")
        tables[f"lag{lag}"] = t
        for i, (label, col) in enumerate(rows):
            ax = axes[i, j]
            for reward, (rname, color, ls) in REWARD_STYLE.items():
                row = t[t["reward_lag1"] == reward].sort_values("bin")
                x = row["bin"].to_numpy() + 1
                y = row["switch"] if col is None else row[col] - row["switch"]
                ax.plot(x, y, color=color, ls=ls, lw=2, marker="o", ms=5, label=f"t−1 {rname}")
                if col is None:
                    ax.fill_between(x, row["switch_lo"], row["switch_hi"], color=color, alpha=0.18, lw=0)
            if col is not None:
                ax.axhline(0, color="#888888", lw=1)
            if i == 0:
                ax.set_title(f"RT on trial t−{lag}")
            if i == 2:
                _style_x(ax, "within", lag)
            if j == 0:
                ax.set_ylabel(f"{label}\nP(switch on t)" if col is None else f"{label}\nP(switch)")
    for i in (1, 2):
        lo = min(axes[i, j].get_ylim()[0] for j in range(3) for i in (1, 2))
        hi = max(axes[i, j].get_ylim()[1] for j in range(3) for i in (1, 2))
        for j in range(3):
            axes[i, j].set_ylim(lo, hi)
    lo = min(axes[0, j].get_ylim()[0] for j in range(3))
    hi = max(axes[0, j].get_ylim()[1] for j in range(3))
    for j in range(3):
        axes[0, j].set_ylim(lo, hi)
    axes[0, 0].legend(frameon=False, fontsize=12)
    fig.suptitle("B0d  Does an older RT still predict switching? (within-mouse deciles; reward on t−1)", y=1.0)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return {k: v.to_dict("records") for k, v in tables.items()}


def _switch_design(df):
    log_rt = np.log(df["rt"])
    z = (log_rt - log_rt.groupby(df["ses_idx"]).transform("mean")) / log_rt.groupby(df["ses_idx"]).transform("std")
    z = z.clip(-4, 4)
    lagged = {k: z.groupby(df["ses_idx"]).shift(k) for k in (1, 2, 3)}
    x = pd.DataFrame({"RT 1 back": lagged[1], "RT 2 back": lagged[2], "RT 3 back": lagged[3],
                      "reward 1 back": df["reward_lag1"], "reward 2 back": df["reward_lag2"], "reward 3 back": df["reward_lag3"]})
    x["RT×reward 1 back"] = x["RT 1 back"] * x["reward 1 back"]
    return x


def fig_conditional_lags(df, out, n_trials=600_000, n_boot=100):
    from sklearn.linear_model import LogisticRegression

    x = _switch_design(df)
    keep = x.notna().all(axis=1) & df["switch"].notna()
    sub = df[keep].sample(n_trials, random_state=0)
    xs = x.loc[sub.index]
    targets = {"mice": sub["switch"].to_numpy(), "R": sub["p_switch"].to_numpy(),
               "R seed 1": sub["p_switch_seed"].to_numpy(), "C": sub["p_switch_C"].to_numpy()}

    def fit(rows):
        out = {}
        xv = xs.to_numpy()[rows]
        for name, p in targets.items():
            pv = p[rows]
            m = LogisticRegression(C=1e6, max_iter=500).fit(np.vstack([xv, xv]), np.r_[np.ones(len(pv)), np.zeros(len(pv))],
                                                             sample_weight=np.r_[pv, 1 - pv])
            out[name] = m.coef_[0]
        return out

    point = fit(np.arange(len(sub)))
    mice = sub["subject_id"].to_numpy()
    uniq = np.unique(mice)
    by_mouse = pd.Series(np.arange(len(sub))).groupby(mice).apply(np.array)
    rng = np.random.default_rng(0)
    boots = [fit(np.concatenate(by_mouse.loc[rng.choice(uniq, len(uniq))].to_numpy())) for _ in range(n_boot)]
    names = list(xs.columns)
    i_rt1, i_int = names.index("RT 1 back"), names.index("RT×reward 1 back")

    def slopes(c):  # RT 1 back after a loss, after a win; RT 2 and 3 back
        return np.array([c[i_rt1], c[i_rt1] + c[i_int], c[names.index("RT 2 back")], c[names.index("RT 3 back")]])

    labels = ["1 back\nafter loss", "1 back\nafter win", "2 back", "3 back"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    result = {}
    for ax, src in zip(axes, targets):
        coef = point[src]
        s_point = slopes(coef)
        s_boot = np.stack([slopes(b[src]) for b in boots])
        s_ci = np.quantile(s_boot, [0.025, 0.975], axis=0)
        ax.bar(range(4), s_point, color=["#C44E52", "#0072B2", "#4C72B0", "#4C72B0"])
        ax.errorbar(range(4), s_point, yerr=[s_point - s_ci[0], s_ci[1] - s_point], fmt="none", color="#222222", capsize=4)
        ax.axhline(0, color="#888888", lw=1)
        ax.set_xticks(range(4), labels, fontsize=12)
        ax.set_xlabel("RT, trials back")
        ax.set_title({"mice": "Mice", "R": "RT model (R)", "R seed 1": "RT model, seed 1", "C": "Choice/reward model (C)"}[src])
        ci = np.quantile(np.stack([b[src] for b in boots]), [0.025, 0.975], axis=0)
        result[src] = {**{n: {"coef": float(coef[i]), "ci": ci[:, i].tolist()} for i, n in enumerate(names)},
                       "slopes": {lab.replace("\n", " "): {"value": float(s_point[k]), "ci": s_ci[:, k].tolist()}
                                  for k, lab in enumerate(labels)}}
    axes[0].set_ylabel("log-odds of switching per sd of RT\n(within-session z; joint fit)")
    fig.suptitle("B0g  Does an older RT matter once the more recent RT is controlled for?", y=1.03)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return {"n_trials": int(n_trials), "n_boot": n_boot, "coefficients": result}


def fig_redundancy(df, out):
    df = df.assign(conf_C=np.abs(df["logit_C"]))
    df["conf_bin"] = pd.qcut(df["conf_C"], N_BINS, labels=False)
    df["conf_tercile"] = pd.qcut(df["conf_C"], 3, labels=False)
    log_rt = np.log(df["rt"])
    df["rt_z"] = (log_rt - log_rt.groupby(df["subject_id"]).transform("mean")) / log_rt.groupby(df["subject_id"]).transform("std")
    t_rt = binned_means(df, ["conf_bin", "reward_lag1"], ["rt_z"])
    t_gain = binned_means(df.assign(bin=df["bin_within_lag1"]), ["bin", "conf_tercile"], ["dlog"])
    conf_mid = df.groupby("conf_bin")["conf_C"].median().to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for reward, (name, color, ls) in REWARD_STYLE.items():
        row = t_rt[t_rt["reward_lag1"] == reward].sort_values("conf_bin")
        _band(axes[0], conf_mid[row["conf_bin"].astype(int)], row, "rt_z", color, ls, f"t−1 {name}")
    axes[0].axhline(0, color="#888888", lw=1)
    axes[0].set_xlabel("C's confidence on trial t, |logit| (decile median)")
    axes[0].set_ylabel("RT on trial t (within-mouse z, log s)")
    axes[0].set_title("are slow responses the ones C was unsure about?")
    axes[0].legend(frameon=False, fontsize=12)
    for k, name in enumerate(("low", "mid", "high")):
        row = t_gain[t_gain["conf_tercile"] == k].sort_values("bin")
        _band(axes[1], row["bin"].to_numpy() + 1, row, "dlog", CONF_COLORS[k], "-", f"C confidence {name}", MILLI)
    axes[1].axhline(0, color="#888888", lw=1)
    _style_x(axes[1], "within")
    axes[1].set_ylabel("log P_R − log P_C (millinats/trial)")
    axes[1].set_title("gain by prev RT, split by C's confidence on t")
    axes[1].legend(frameon=False, fontsize=12)
    fig.suptitle("B0e  Is RT redundant with what C already knows?", y=1.02)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    valid = df[["conf_C", "rt_z"]].dropna()  # nan_policy="omit" is very slow at this size
    rho = stats.spearmanr(valid["conf_C"], valid["rt_z"])
    return {"rt_vs_C_confidence": t_rt.to_dict("records"), "gain_by_rt_and_confidence": t_gain.to_dict("records"),
            "spearman_conf_C_vs_rt_z": {"rho": float(rho.statistic), "p": float(rho.pvalue)}}


def fig_per_mouse(df, out):
    log_rt = np.log(df["rt"])
    mice = pd.DataFrame({
        "gain": df.groupby("subject_id")["dlog"].mean() * MILLI,
        "median_rt": df.groupby("subject_id")["rt"].median(),
        "sd_log_rt": log_rt.groupby(df["subject_id"]).std(),
        "n": df.groupby("subject_id").size(),
    }).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    out_stats = {}
    for ax, col, label in ((axes[0], "median_rt", "mouse's median RT (s)"), (axes[1], "sd_log_rt", "mouse's RT variability (sd of log RT)")):
        ax.scatter(mice[col], mice["gain"], s=14, color="#4C72B0", alpha=0.6, lw=0)
        ax.axhline(0, color="#888888", lw=1)
        rho = stats.spearmanr(mice[col], mice["gain"])
        out_stats[col] = {"spearman_rho": float(rho.statistic), "p": float(rho.pvalue)}
        ax.set_xlabel(label)
        ax.set_title(f"Spearman ρ = {rho.statistic:+.2f} (p = {rho.pvalue:.1g})")
        if col == "median_rt":
            rt_log_axis(ax)
    axes[0].set_ylabel("mean gain, R − C\n(millinats/trial)")
    fig.suptitle(f"B0f  Which mice gain? ({len(mice)} mice with RT)", y=1.02)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    out_stats["fraction_mice_gaining"] = float((mice["gain"] > 0).mean())
    out_stats["n_mice"] = int(len(mice))
    return {"summary": out_stats, "mice": mice.to_dict("records")}


def main() -> None:
    apply_presentation_style()
    df = load_paired_trials(rt_present_only=True, with_seed=True)
    for lag in (1, 2, 3):
        df[f"bin_within_lag{lag}"] = rt_bins(df, f"rt_lag{lag}", within_mouse=True)
    df["bin_pooled_lag1"] = rt_bins(df, "rt_lag1", within_mouse=False)
    centres = df.groupby("bin_pooled_lag1")["rt_lag1"].median().to_numpy()

    payload = {
        "trials": {"n": int(len(df)), "n_sessions": int(df["ses_idx"].nunique()), "n_mice": int(df["subject_id"].nunique()),
                   "mean_dlog": float(df["dlog"].mean()),
                   "likelihood_R": float(np.exp(df["log_p"].mean())), "likelihood_C": float(np.exp(df["log_p_C"].mean()))},
        "pooled_bin_median_rt_s": centres,
    }
    payload["B0a_gain_by_prev_rt"] = fig_gain_by_rt(df, centres, figure_path("fig_B0a_gain_by_prev_rt.png"))
    payload["B0b_disagreement"] = fig_disagreement(df, figure_path("fig_B0b_disagreement.png"))
    payload["B0c_switch_by_prev_rt"] = fig_switch(df, centres, figure_path("fig_B0c_switch_by_prev_rt.png"),
                                                  figure_path("fig_B0c_switch_residuals.png"))
    payload["B0d_switch_by_lagged_rt"] = fig_lags(df, figure_path("fig_B0d_switch_by_lagged_rt.png"))
    payload["B0e_redundancy_with_C"] = fig_redundancy(df, figure_path("fig_B0e_redundancy_with_C.png"))
    payload["B0f_per_mouse"] = fig_per_mouse(df, figure_path("fig_B0f_per_mouse.png"))
    payload["B0g_conditional_lags"] = fig_conditional_lags(df, figure_path("fig_B0g_conditional_lags.png"))
    write_json("b0_where_rt_helps.json", payload, PRODUCER)
    print({k: payload[k] for k in ("trials",)}, payload["B0b_disagreement"], payload["B0e_redundancy_with_C"]["spearman_conf_C_vs_rt_z"], payload["B0f_per_mouse"]["summary"])


if __name__ == "__main__":
    main()
