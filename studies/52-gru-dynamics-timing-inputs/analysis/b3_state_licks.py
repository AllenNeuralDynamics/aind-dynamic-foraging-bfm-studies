"""B3-L: how the licks model's hidden state represents licking, beside C on the same trials.

    python analysis/b3_state_licks.py

A random sample of eval sessions with licks (≈400k trials). Lick variables are on trial t−k,
relative to that trial's choice: other-side and chosen-side counts, as log(1 + count) z-scored
within mouse and winsorized at ±4. Row t's state has seen trials up to t−1, so the lag-1 counts
are L's most recent input (decodable by construction); C sees licking only through correlates.

  B3L-a  PCA (each model's own fit), PCs 1–2 and 3–4, coloured by other-side licks 1 back.
  B3L-b  ridge decoding (5-fold CV grouped by session) of other- and chosen-side licks 1–5 back:
         raw; beyond choice/reward/choice×reward 1–6 back; and "unique" (also beyond more recent
         lags of both lick variables).
  B3L-c  within each previous choice × reward cluster, the state's joint OLS slopes on other- and
         chosen-side licks 1 back (OBSERVATIONAL — includes correlates; C has slopes too).
         Cosines between directions, the readout and a reward axis.

Writes ``fig_B3L[a-c]_*.png`` and ``b3_state_licks.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3_state_rt import INPUT_COLORS, INPUTS, STATES, _cv_r2, pca  # noqa: E402
from _pk03 import (  # noqa: E402
    BASELINE_KEY, CACHE, STUDY12, apply_presentation_style, figure_path, load_paired_trials, write_json,
)

PRODUCER = "analysis/b3_state_licks.py"
KEY = "L-s0"
N_TRIALS, N_SCATTER, N_LAGS, Z_CLIP = 400_000, 20_000, 5, 4.0
TABLES = {"L": CACHE / f"trial_states_{KEY}.parquet",
          "C": STUDY12 / "analysis" / "_cache" / f"trial_states_{BASELINE_KEY}.parquet"}
LABELS = {"L": "Licks model (L)", "C": "Choice/reward model (C)"}
VARS = ("other", "chosen")


def sample(seed: int = 0):
    df = load_paired_trials(rt_present_only=False, key=KEY, licks_present_only=True)
    g_s = df.groupby("ses_idx", sort=False)
    for v in VARS:
        x = np.log1p(df[f"licks_{v}"])
        z = (x - x.groupby(df["subject_id"]).transform("mean")) / x.groupby(df["subject_id"]).transform("std")
        df[f"{v}_z"] = z.clip(-Z_CLIP, Z_CLIP)
    g_s = df.groupby("ses_idx", sort=False)
    for k in range(1, N_LAGS + 2):
        for v in VARS:
            df[f"{v}_z_lag{k}"] = g_s[f"{v}_z"].shift(k)
        df[f"choice_lag{k}"] = 2.0 * g_s["action"].shift(k) - 1.0
        df[f"reward_lag{k}"] = g_s["reward"].shift(k).astype(float)
    sizes = df.groupby("ses_idx").size()
    rng = np.random.default_rng(seed)
    order = rng.permutation(sizes.index.to_numpy())
    keep = order[: np.searchsorted(np.cumsum(sizes.loc[order].to_numpy()), N_TRIALS) + 1]
    df = df[df["ses_idx"].isin(set(keep))].dropna(subset=[f"other_z_lag{N_LAGS + 1}"]).reset_index(drop=True)
    states = {}
    for name, path in TABLES.items():
        t = pq.read_table(path, columns=["ses_idx", "trial_idx", *STATES], filters=[("ses_idx", "in", list(keep))]).to_pandas()
        t = df[["ses_idx", "trial_idx"]].merge(t, on=["ses_idx", "trial_idx"], how="left", validate="1:1")
        states[name] = t[STATES].to_numpy(dtype=np.float64)
    df["input"] = [INPUTS[(int(a), int(r))] for a, r in zip((df["choice_lag1"] > 0), df["reward_lag1"])]
    return df, states


def fig_pca(df, states, fits, out):
    rng = np.random.default_rng(1)
    idx = rng.choice(len(df), N_SCATTER, replace=False)
    color = np.clip(df["other_z_lag1"].to_numpy()[idx], -2, 3)
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 11.5), gridspec_kw={"hspace": 0.4})
    for i, name in enumerate(("L", "C")):
        fit = fits[name]
        scores = (states[name][idx] - fit["mean"]) @ fit["components"][:4].T
        for j, (a, b) in enumerate(((0, 1), (2, 3))):
            ax = axes[i, j]
            order = np.argsort(color)  # draw high other-side licking on top
            sc = ax.scatter(scores[order, a], scores[order, b], s=3, c=color[order], cmap="viridis", vmin=-2, vmax=3, alpha=0.6, lw=0)
            ax.set_xlabel(f"PC{a + 1} ({fit['explained'][a]:.0%})")
            ax.set_ylabel(f"PC{b + 1} ({fit['explained'][b]:.0%})")
            if j == 0:
                ax.set_title(f"{LABELS[name]}   PR = {fit['participation_ratio']:.1f}", loc="left")
    cbar = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("other-side licks on t−1 (within-mouse z of log(1+n), clipped)")
    fig.suptitle("B3L-a  State space coloured by other-side licking on the previous trial", y=0.98)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def pc_correlations(df, x, fit):
    scores = (x - fit["mean"]) @ fit["components"][:4].T
    cols = {"other_z_lag1": "other licks 1 back", "chosen_z_lag1": "chosen licks 1 back",
            "reward_lag1": "reward 1 back", "choice_lag1": "choice 1 back"}
    return {f"pc{k + 1}": {lab: float(np.corrcoef(scores[:, k], df[c])[0, 1]) for c, lab in cols.items()} for k in range(4)}


def decoding(df, states, out):
    history = df[[f"{v}_lag{k}" for k in range(1, N_LAGS + 2) for v in ("choice", "reward")]].to_numpy()
    history = np.hstack([history, history[:, 0::2] * history[:, 1::2]])
    groups = df["ses_idx"].to_numpy()
    result = {}
    for name in ("L", "C"):
        result[name] = {}
        for v in VARS:
            r = {"raw": [], "beyond_history": [], "unique": []}
            for k in range(1, N_LAGS + 1):
                y = df[f"{v}_z_lag{k}"].to_numpy()
                resid = y - LinearRegression().fit(history, y).predict(history)
                recent = [f"{w}_z_lag{j}" for j in range(1, k) for w in VARS] + ([f"{w}_z_lag{k}" for w in VARS if w != v])
                nuis = np.hstack([history, df[recent].to_numpy()])
                unique = y - LinearRegression().fit(nuis, y).predict(nuis)
                r["raw"].append(_cv_r2(states[name], y, groups))
                r["beyond_history"].append(_cv_r2(states[name], resid, groups))
                r["unique"].append(_cv_r2(states[name], unique, groups))
            result[name][v] = r
    lags = np.arange(1, N_LAGS + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True, sharex=True)
    for j, name in enumerate(("L", "C")):
        for i, v in enumerate(VARS):
            ax = axes[i, j]
            r = result[name][v]
            ax.plot(lags, r["raw"], color="#222222", lw=2, marker="o", label="raw")
            ax.plot(lags, r["beyond_history"], color="#4C72B0", lw=2, ls="--", marker="o", label="beyond choice/reward history")
            ax.plot(lags, r["unique"], color="#C44E52", lw=2, ls=":", marker="o", label="beyond history and more recent licks")
            ax.axhline(0, color="#888888", lw=1)
            ax.set_xticks(lags)
            if i == 0:
                ax.set_title(LABELS[name])
            if j == 0:
                ax.set_ylabel(f"{v}-side licks\ncross-validated R²")
            if i == 1:
                ax.set_xlabel("trials back (1 = the input L just received)")
    axes[0, 0].legend(frameon=False, fontsize=11)
    fig.suptitle("B3L-b  How much of the previous trials' licking the state carries", y=1.0)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return result


def directions(df, states, out):
    labels = list(INPUT_COLORS)
    result = {}
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.6), gridspec_kw={"wspace": 0.5})
    for ax, name in zip(axes, ("L", "C")):
        x = states[name]
        readout = np.linalg.lstsq(np.hstack([x, np.ones((len(x), 1))]), df["logit" if name == "L" else "logit_C"].to_numpy(), rcond=None)[0][:-1]
        reward_axis = np.mean([x[df["input"].to_numpy() == f"{s} rewarded"].mean(0) - x[df["input"].to_numpy() == f"{s} unrewarded"].mean(0)
                               for s in ("R", "L")], axis=0)
        vecs, logit = {}, {}
        for label in labels:
            m = df["input"].to_numpy() == label
            z = df.loc[m, ["other_z_lag1", "chosen_z_lag1"]].to_numpy()
            z = z - z.mean(0)
            coef = np.linalg.lstsq(z, x[m] - x[m].mean(0), rcond=None)[0]  # [2, H]
            for v, row in zip(VARS, coef):
                vecs[f"{v} | {label}"] = row
                logit[f"{v} | {label}"] = float(row @ readout)
        all_vecs = {**vecs, "readout": readout, "reward axis": reward_axis}
        names = list(all_vecs)
        mat = np.array([[a @ b / np.linalg.norm(a) / np.linalg.norm(b) for b in all_vecs.values()] for a in all_vecs.values()])
        ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(mat[i, j]) > 0.6 else "#222222")
        ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=9)
        ax.set_yticks(range(len(names)), names, fontsize=9)
        ax.set_title(LABELS[name], fontsize=13)
        result[name] = {"cosines": {"names": names, "matrix": mat}, "logit_change_per_sd": logit,
                        "norm_per_sd": {k: float(np.linalg.norm(v)) for k, v in vecs.items()}}
    fig.suptitle("B3L-c  State covariation with licks 1 back (observational; joint fit within each t−1 input cluster)", y=1.0)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return result


def main() -> None:
    apply_presentation_style()
    df, states = sample()
    fits = {name: pca(x) for name, x in states.items()}
    fig_pca(df, states, fits, figure_path("fig_B3La_pca_other_licks.png"))
    payload = {
        "sample": {"n_trials": int(len(df)), "n_sessions": int(df["ses_idx"].nunique()), "n_mice": int(df["subject_id"].nunique())},
        "pca": {n: {"explained_first10": f["explained"][:10], "participation_ratio": f["participation_ratio"],
                    "pc_correlations": pc_correlations(df, states[n], f)} for n, f in fits.items()},
        "lick_autocorrelation_lag1_vs_lagk": {v: [float(df[f"{v}_z_lag1"].corr(df[f"{v}_z_lag{k}"])) for k in range(2, N_LAGS + 1)] for v in VARS},
        "corr_other_chosen_lag1": float(df["other_z_lag1"].corr(df["chosen_z_lag1"])),
        "B3L_b_decoding": decoding(df, states, figure_path("fig_B3Lb_lick_decoding.png")),
        "B3L_c_directions": directions(df, states, figure_path("fig_B3Lc_lick_directions.png")),
    }
    write_json("b3_state_licks.json", payload, PRODUCER)
    print({n: (round(v["participation_ratio"], 2), {k: {a: round(b, 2) for a, b in c.items()} for k, c in v["pc_correlations"].items()})
           for n, v in payload["pca"].items()})
    print({n: {v: {k: [round(x, 3) for x in r] for k, r in d.items()} for v, d in dd.items()} for n, dd in payload["B3L_b_decoding"].items()})
    print({n: {k: round(v, 3) for k, v in payload["B3L_c_directions"][n]["logit_change_per_sd"].items()} for n in ("L", "C")})


if __name__ == "__main__":
    main()
