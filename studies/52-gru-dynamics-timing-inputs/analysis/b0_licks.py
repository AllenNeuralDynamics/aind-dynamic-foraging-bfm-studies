"""B0-L: where does the licks model (L) beat C, and what in the animals' licking does it use?

    python analysis/b0_licks.py

Same eval trials for L-s0 and C-s0 (Study 50), all sessions with any lick recorded (L-s1 is
the seed-noise baseline and a replicate). Lick counts are on trial t−1 (the input L received
on row t), re-expressed relative to that trial's choice:
  * chosen-side licks — licking the port the mouse chose (includes consumption after a reward);
  * other-side licks  — licking the unchosen port.
dlog = log P_L(chosen) − log P_C(chosen). Error bands: 95% CIs from resampling mice.

  B0L-a  gain by other-side licks (0,1,2,3,4–5,6+) and by chosen-side licks (within-mouse
         deciles, ranked separately after rewarded and unrewarded trials, ties broken at
         random), per outcome.
  B0L-b  P(switch) by the same bins for mice, L and C (side by side), and each model − mice
         (L, L seed 1, C).
  B0L-c  trial-by-trial disagreement: L − C vs L seed 0 − seed 1.
  B0L-d  joint logistic fit of switching on log(1 + licks), chosen and other side, at lags 1–3
         (z-scored within session, clipped ±4), rewards 1–3 back and lick × reward at lag 1.
         Fitted identically to the mice (binary) and to each model's P(switch) (soft labels).
         Reward is coded 0/1, so lag-1 slopes are reported after a loss (main effect) and
         after a win (main + interaction).
  B0L-e  per mouse: gain vs the mouse's mean other-side and chosen-side licks.

Writes ``fig_B0L[a-e]_*.png`` and ``b0_licks.json``.
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
    N_BINS, REWARDED_COLOR, UNREWARDED_COLOR, apply_presentation_style, binned_means, figure_path,
    load_paired_trials, write_json,
)

PRODUCER = "analysis/b0_licks.py"
KEY, SEED_KEY = "L-s0", "L-s1"
MILLI = 1000.0
OTHER_EDGES = [0, 1, 2, 3, 4, 6, np.inf]  # bins [a, b)
OTHER_LABELS = ["0", "1", "2", "3", "4–5", "6+"]
REWARD_STYLE = {1.0: ("rewarded", REWARDED_COLOR, "-"), 0.0: ("unrewarded", UNREWARDED_COLOR, "--")}
LABELS = {"mice": "Mice", "L": "Licks model (L)", "L seed 1": "Licks model, seed 1", "C": "Choice/reward model (C)"}


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    df["other_bin"] = pd.cut(df["licks_other_lag1"], OTHER_EDGES, right=False, labels=False)
    # Chosen-side deciles within each mouse and outcome (rewarded trials include consumption).
    # Counts are small integers with many ties; a uniform jitter in [0, 1) breaks ties at random
    # (order between different counts is preserved), so the bins are true deciles. Ranking with
    # tied ranks instead piles whole blocks of equal counts into one bin.
    jitter = np.random.default_rng(0).random(len(df))
    rank = (df["licks_chosen_lag1"] + jitter).groupby([df["subject_id"], df["reward_lag1"]]).rank(pct=True)
    df["chosen_bin"] = np.minimum((rank * N_BINS).fillna(-1).astype(int), N_BINS - 1).where(df["licks_chosen_lag1"].notna())
    return df


def _x(binning, table):
    return table["bin"].astype(int).to_numpy() + (0 if binning == "other" else 1)


def _style(ax, binning):
    if binning == "other":
        ax.set_xticks(range(len(OTHER_LABELS)), OTHER_LABELS)
        ax.set_xlabel("other-side licks on t−1")
    else:
        ax.set_xticks([1, 5, 10])
        ax.set_xlabel("chosen-side licks on t−1\n(decile in mouse × outcome)")


def fig_gain(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    tables = {}
    for ax, binning in zip(axes, ("other", "chosen")):
        t = binned_means(df.assign(bin=df[f"{binning}_bin"]), ["bin", "reward_lag1"], ["dlog"])
        tables[binning] = t.to_dict("records")
        for reward, (name, color, ls) in REWARD_STYLE.items():
            row = t[t["reward_lag1"] == reward].sort_values("bin")
            x = _x(binning, row)
            ax.plot(x, row["dlog"] * MILLI, color=color, ls=ls, lw=2, marker="o", ms=5, label=f"t−1 {name}")
            ax.fill_between(x, row["dlog_lo"] * MILLI, row["dlog_hi"] * MILLI, color=color, alpha=0.18, lw=0)
        ax.axhline(0, color="#888888", lw=1)
        ax.axhline(df["dlog"].mean() * MILLI, color="#888888", lw=1, ls=":", label="mean over all trials")
        _style(ax, binning)
    axes[0].set_ylabel("log P_L − log P_C (millinats/trial)")
    axes[0].legend(frameon=False, fontsize=12)
    fig.suptitle("B0L-a  Where L beats C: gain by the previous trial's licking", y=1.02)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return tables


def fig_switch(df, out, out_resid):
    cols = ["switch", "p_switch", "p_switch_seed", "p_switch_C"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), sharey="row")
    fig_r, axes_r = plt.subplots(2, 3, figsize=(17, 9.5), sharey="row")
    tables = {}
    for i, binning in enumerate(("other", "chosen")):
        t = binned_means(df.assign(bin=df[f"{binning}_bin"]), ["bin", "reward_lag1"], cols)
        tables[binning] = t.to_dict("records")
        for j, (name, col) in enumerate((("mice", "switch"), ("L", "p_switch"), ("C", "p_switch_C"))):
            ax = axes[i, j]
            for reward, (rname, color, ls) in REWARD_STYLE.items():
                row = t[t["reward_lag1"] == reward].sort_values("bin")
                x = _x(binning, row)
                ax.plot(x, row[col], color=color, ls=ls, lw=2, marker="o", ms=5, label=f"t−1 {rname}")
                ax.fill_between(x, row[f"{col}_lo"], row[f"{col}_hi"], color=color, alpha=0.18, lw=0)
            _style(ax, binning)
            if i == 0:
                ax.set_title(LABELS[name])
        for j, (name, col) in enumerate((("L", "p_switch"), ("L seed 1", "p_switch_seed"), ("C", "p_switch_C"))):
            ax = axes_r[i, j]
            for reward, (rname, color, ls) in REWARD_STYLE.items():
                row = t[t["reward_lag1"] == reward].sort_values("bin")
                ax.plot(_x(binning, row), row[col] - row["switch"], color=color, ls=ls, lw=2, marker="o", ms=5,
                        label=f"t−1 {rname}")
            ax.axhline(0, color="#888888", lw=1)
            _style(ax, binning)
            if i == 0:
                ax.set_title(f"{LABELS[name]} − mice")
        axes[i, 0].set_ylabel("P(switch on t)")
        axes_r[i, 0].set_ylabel("P(switch): model − mice")
    axes[0, 0].legend(frameon=False, fontsize=12)
    axes_r[0, 0].legend(frameon=False, fontsize=12)
    for f in (fig, fig_r):
        f.subplots_adjust(hspace=0.45)
    fig.suptitle("B0L-b  Switching after licking the other port / licking the chosen port", y=0.98)
    fig_r.suptitle("B0L-b  Each model's error in P(switch), by the previous trial's licking", y=0.98)
    fig.savefig(out, bbox_inches="tight")
    fig_r.savefig(out_resid, bbox_inches="tight")
    plt.close(fig)
    plt.close(fig_r)
    return tables


def _disagreement(d):
    return {"mean": float(d.mean()), "sd": float(d.std()), "fraction_positive": float(np.mean(d > 0)),
            "gross_positive": float(d[d > 0].sum()), "gross_negative": float(-d[d < 0].sum()), "net": float(d.sum())}


def fig_disagreement(df, out):
    pairs = {"L − C": df["dlog"].to_numpy(), "L seed 0 − L seed 1": (df["log_p"] - df["log_p_seed"]).to_numpy()}
    summary = {n: _disagreement(d) for n, d in pairs.items()}
    lim = max(np.abs(d).max() for d in pairs.values()) * MILLI
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, (n, d) in zip(axes[:2], pairs.items()):
        ax.hist(d * MILLI, bins=np.linspace(-lim, lim, 201), color="#4C72B0")
        ax.set_yscale("log")
        ax.set_xlabel("Δ log P(chosen) per trial (millinats)")
        ax.set_title(f"{n}: sd {summary[n]['sd'] * MILLI:.0f}, mean {summary[n]['mean'] * MILLI:+.2f}")
    axes[0].set_ylabel("trials")
    axes[1].sharey(axes[0])
    names = list(pairs)
    x = np.arange(len(names))
    for off, key, color, lab in ((-0.2, "gross_positive", "#0072B2", "favours 1st model"),
                                 (0.2, "gross_negative", "#C44E52", "favours 2nd model")):
        axes[2].bar(x + off, [summary[n][key] for n in names], width=0.38, color=color, label=lab)
    for i, n in enumerate(names):
        axes[2].annotate(f"net {summary[n]['net']:+.0f}", (i, max(summary[n]["gross_positive"], summary[n]["gross_negative"])),
                         textcoords="offset points", xytext=(0, 6), ha="center", fontsize=13)
    axes[2].set_xticks(x, names, fontsize=13)
    axes[2].set_ylabel("summed |difference| (nats)")
    axes[2].legend(frameon=False, fontsize=11, loc="lower left")
    fig.suptitle(f"B0L-c  Trial-by-trial disagreement vs seed noise ({len(df):,} trials)", y=1.02)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return summary


def _design(df):
    x = {}
    for side in ("chosen", "other"):
        for k in (1, 2, 3):
            v = np.log1p(df[f"licks_{side}_lag{k}"])
            z = (v - v.groupby(df["ses_idx"]).transform("mean")) / v.groupby(df["ses_idx"]).transform("std")
            x[f"{side} {k} back"] = z.clip(-4, 4)
    for k in (1, 2, 3):
        x[f"reward {k} back"] = df[f"reward_lag{k}"]
    x = pd.DataFrame(x)
    x["chosen×reward 1 back"] = x["chosen 1 back"] * x["reward 1 back"]
    x["other×reward 1 back"] = x["other 1 back"] * x["reward 1 back"]
    return x


def fig_conditional(df, out, n_trials=600_000, n_boot=100):
    from sklearn.linear_model import LogisticRegression

    x = _design(df)
    keep = x.notna().all(axis=1) & df["switch"].notna()
    sub = df[keep].sample(n_trials, random_state=0)
    xs = x.loc[sub.index]
    names = list(xs.columns)
    targets = {"mice": sub["switch"].to_numpy(), "L": sub["p_switch"].to_numpy(),
               "L seed 1": sub["p_switch_seed"].to_numpy(), "C": sub["p_switch_C"].to_numpy()}

    def fit(rows):
        out = {}
        xv = xs.to_numpy()[rows]
        for name, p in targets.items():
            pv = p[rows]
            m = LogisticRegression(C=1e6, max_iter=500).fit(np.vstack([xv, xv]), np.r_[np.ones(len(pv)), np.zeros(len(pv))],
                                                             sample_weight=np.r_[pv, 1 - pv])
            out[name] = m.coef_[0]
        return out

    def slopes(c):
        ix = names.index
        return np.array([c[ix("other 1 back")], c[ix("other 1 back")] + c[ix("other×reward 1 back")],
                         c[ix("chosen 1 back")], c[ix("chosen 1 back")] + c[ix("chosen×reward 1 back")],
                         c[ix("other 2 back")], c[ix("other 3 back")], c[ix("chosen 2 back")], c[ix("chosen 3 back")]])

    labels = ["other 1\nafter loss", "other 1\nafter win", "chosen 1\nafter loss", "chosen 1\nafter win",
              "other 2", "other 3", "chosen 2", "chosen 3"]
    point = fit(np.arange(len(sub)))
    by_mouse = pd.Series(np.arange(len(sub))).groupby(sub["subject_id"].to_numpy()).apply(np.array)
    uniq = by_mouse.index.to_numpy()
    rng = np.random.default_rng(0)
    boots = [fit(np.concatenate(by_mouse.loc[rng.choice(uniq, len(uniq))].to_numpy())) for _ in range(n_boot)]
    colors = ["#C44E52", "#0072B2", "#C44E52", "#0072B2", "#888888", "#888888", "#888888", "#888888"]
    hatches = ["", "", "//", "//", "", "", "//", "//"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), sharey=True)
    result = {}
    for ax, src in zip(axes, targets):
        s_point = slopes(point[src])
        s_ci = np.quantile(np.stack([slopes(b[src]) for b in boots]), [0.025, 0.975], axis=0)
        ax.bar(range(len(labels)), s_point, color=colors, hatch=hatches, edgecolor="white")
        ax.errorbar(range(len(labels)), s_point, yerr=[s_point - s_ci[0], s_ci[1] - s_point], fmt="none",
                    color="#222222", capsize=3)
        ax.axhline(0, color="#888888", lw=1)
        ax.set_xticks(range(len(labels)), labels, fontsize=10, rotation=90)
        ax.set_title(LABELS[src])
        ci = np.quantile(np.stack([b[src] for b in boots]), [0.025, 0.975], axis=0)
        result[src] = {"coefficients": {n: {"coef": float(point[src][i]), "ci": ci[:, i].tolist()} for i, n in enumerate(names)},
                       "slopes": {lab.replace("\n", " "): {"value": float(s_point[k]), "ci": s_ci[:, k].tolist()}
                                  for k, lab in enumerate(labels)}}
    axes[0].set_ylabel("log-odds of switching per sd of log(1 + licks)\n(within-session z; joint fit)")
    fig.suptitle("B0L-d  Conditional effects of the previous trials' licking on switching (hatched = chosen side)", y=1.03)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return {"n_trials": int(n_trials), "n_boot": n_boot, "by_source": result}


def fig_per_mouse(df, out):
    g = df.groupby("subject_id")
    mice = pd.DataFrame({"gain": g["dlog"].mean() * MILLI, "other": g["licks_other"].mean(),
                         "chosen": g["licks_chosen"].mean()}).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    stats_out = {}
    for ax, col, label in ((axes[0], "other", "mouse's mean other-side licks"), (axes[1], "chosen", "mouse's mean chosen-side licks")):
        ax.scatter(mice[col], mice["gain"], s=14, color="#4C72B0", alpha=0.6, lw=0)
        ax.axhline(0, color="#888888", lw=1)
        rho = stats.spearmanr(mice[col], mice["gain"])
        stats_out[col] = {"spearman_rho": float(rho.statistic), "p": float(rho.pvalue)}
        ax.set_xlabel(label)
        ax.set_title(f"Spearman ρ = {rho.statistic:+.2f} (p = {rho.pvalue:.1g})")
    axes[0].set_ylabel("mean gain, L − C\n(millinats/trial)")
    fig.suptitle(f"B0L-e  Which mice gain? ({len(mice)} mice)", y=1.02)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    stats_out["fraction_mice_gaining"] = float((mice["gain"] > 0).mean())
    stats_out["n_mice"] = int(len(mice))
    return {"summary": stats_out, "mice": mice.to_dict("records")}


def main() -> None:
    apply_presentation_style()
    df = load_paired_trials(rt_present_only=False, with_seed=True, key=KEY, seed_key=SEED_KEY, licks_present_only=True)
    df = add_bins(df)
    payload = {"trials": {"n": int(len(df)), "n_sessions": int(df["ses_idx"].nunique()), "n_mice": int(df["subject_id"].nunique()),
                          "mean_dlog": float(df["dlog"].mean()), "likelihood_L": float(np.exp(df["log_p"].mean())),
                          "likelihood_C": float(np.exp(df["log_p_C"].mean())),
                          "trials_by_other_bin": df["other_bin"].value_counts().sort_index().to_dict()}}
    payload["B0L_a_gain"] = fig_gain(df, figure_path("fig_B0La_gain_by_licks.png"))
    payload["B0L_b_switch"] = fig_switch(df, figure_path("fig_B0Lb_switch_by_licks.png"), figure_path("fig_B0Lb_switch_residuals.png"))
    payload["B0L_c_disagreement"] = fig_disagreement(df, figure_path("fig_B0Lc_disagreement.png"))
    payload["B0L_e_per_mouse"] = fig_per_mouse(df, figure_path("fig_B0Le_per_mouse.png"))
    payload["B0L_d_conditional"] = fig_conditional(df, figure_path("fig_B0Ld_conditional_lags.png"))
    write_json("b0_licks.json", payload, PRODUCER)
    print(payload["trials"], payload["B0L_c_disagreement"], payload["B0L_e_per_mouse"]["summary"])
    for src, r in payload["B0L_d_conditional"]["by_source"].items():
        print(src, {k: round(v["value"], 3) for k, v in r["slopes"].items()})


if __name__ == "__main__":
    main()
