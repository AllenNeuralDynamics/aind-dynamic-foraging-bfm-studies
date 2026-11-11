"""B3: how R's hidden state represents reaction time, beside C on the same trials.

    python analysis/b3_state_rt.py

A random sample of RT-present eval sessions (≈400k trials). ``rt_z`` is a trial's log RT
z-scored within its mouse; lag k is trial t−k. Row t's state has seen trials up to t−1, so
RT 1 back is R's most recent input (decodable by construction), and anything C encodes
about RT comes through correlates (choices, rewards, the mouse, the session).

  B3a  PCA (each model its own fit), PCs 1–2 and 3–4, coloured by the previous trial's
       choice × reward (Study 50's four input clusters). Variance and participation ratio.
  B3b  the same projections coloured by RT 1 back (within-mouse z).
  B3c  ridge decoding of RT 1–5 back from all 128 units (5-fold CV grouped by session):
       raw; after regressing out choice/reward/choice×reward 1–6 back (lag 6 included because
       RT k back is driven by the outcome k+1 back); and "unique", also regressing out RT
       1..k−1 back (RT is autocorrelated, so lag k is partly lag 1 again). "Unique" can still
       contain the part of RT k back predicted by RTs older than k (a running average).
  B3d  the RT direction: within each input cluster, the state's regression slope on RT
       1 back. Cosines between clusters' directions, the readout and the reward axis. This
       is OBSERVATIONAL: it includes everything that covaries with RT (C, which never sees RT,
       has a non-zero "RT direction" too). The causal effect of the RT input is I-RT1.

RT z-scores are winsorized at ±RT_Z_CLIP: raw RTs ≤ 0 (1,909 trials) are clipped to 1 ms by
the model's encoder and land near −15 within-mouse sd, where they would dominate the OLS fits.

Writes ``fig_B3[a-d]_*.png`` and ``b3_state_rt.json``.
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
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pk03 import (  # noqa: E402
    BASELINE_KEY, CACHE, MODEL_KEY, MODEL_LABELS, STUDY12, apply_presentation_style, figure_path,
    load_paired_trials, write_json,
)

PRODUCER = "analysis/b3_state_rt.py"
N_TRIALS = 400_000
N_SCATTER = 20_000
N_LAGS = 5
RT_Z_CLIP = 4.0
ALPHA = 1.0
HIDDEN = 128
STATES = [f"rnn_state_{i}" for i in range(HIDDEN)]
# Study 50's input-cluster colours (index 2·choice + reward on t−1).
INPUTS = {(1, 1): "R rewarded", (1, 0): "R unrewarded", (0, 1): "L rewarded", (0, 0): "L unrewarded"}
INPUT_COLORS = {"R rewarded": "#0072B2", "R unrewarded": "#56B4E9", "L rewarded": "#D55E00", "L unrewarded": "#E69F00"}
TABLES = {"R": CACHE / f"trial_states_{MODEL_KEY}.parquet",
          "C": STUDY12 / "analysis" / "_cache" / f"trial_states_{BASELINE_KEY}.parquet"}


def sample(seed: int = 0) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    df = load_paired_trials(rt_present_only=True)
    log_rt = np.log(df["rt"])
    rt_z = (log_rt - log_rt.groupby(df["subject_id"]).transform("mean")) / log_rt.groupby(df["subject_id"]).transform("std")
    df["rt_z"] = rt_z.clip(-RT_Z_CLIP, RT_Z_CLIP)
    grouped = df.groupby("ses_idx", sort=False)
    for k in range(1, N_LAGS + 2):
        df[f"rt_z_lag{k}"] = grouped["rt_z"].shift(k)
        df[f"choice_lag{k}"] = 2.0 * grouped["action"].shift(k) - 1.0
        df[f"reward_lag{k}"] = grouped["reward"].shift(k).astype(float)
    sizes = df.groupby("ses_idx").size()
    rng = np.random.default_rng(seed)
    order = rng.permutation(sizes.index.to_numpy())
    keep = order[: np.searchsorted(np.cumsum(sizes.loc[order].to_numpy()), N_TRIALS) + 1]
    df = df[df["ses_idx"].isin(set(keep))].dropna(subset=[f"rt_z_lag{N_LAGS + 1}"]).reset_index(drop=True)
    states = {}
    for name, path in TABLES.items():
        t = pq.read_table(path, columns=["ses_idx", "trial_idx", *STATES], filters=[("ses_idx", "in", list(keep))]).to_pandas()
        t = df[["ses_idx", "trial_idx"]].merge(t, on=["ses_idx", "trial_idx"], how="left", validate="1:1")
        states[name] = t[STATES].to_numpy(dtype=np.float64)
    df["input"] = [INPUTS[(int(a), int(r))] for a, r in zip((df["choice_lag1"] > 0), df["reward_lag1"])]
    return df, states


def pca(x: np.ndarray) -> dict:
    mean = x.mean(0)
    _, sv, vt = np.linalg.svd(x - mean, full_matrices=False)
    var = sv**2 / (len(x) - 1)
    return {"mean": mean, "components": vt, "variance": var,
            "explained": var / var.sum(), "participation_ratio": float(var.sum() ** 2 / (var**2).sum())}


def pc_correlations(df, x, fit) -> dict:
    scores = (x - fit["mean"]) @ fit["components"][:4].T
    return {f"corr_pc{k + 1}": {"rt_z_lag1": float(np.corrcoef(scores[:, k], df["rt_z_lag1"])[0, 1]),
                                "reward_lag1": float(np.corrcoef(scores[:, k], df["reward_lag1"])[0, 1]),
                                "choice_lag1": float(np.corrcoef(scores[:, k], df["choice_lag1"])[0, 1])} for k in range(4)}


def fig_pca(df, states, fits, out_inputs, out_rt):
    rng = np.random.default_rng(1)
    idx = rng.choice(len(df), N_SCATTER, replace=False)
    pairs = [(0, 1), (2, 3)]
    fig_a, axes_a = plt.subplots(2, 2, figsize=(12, 11.5), gridspec_kw={"hspace": 0.4})
    fig_b, axes_b = plt.subplots(2, 2, figsize=(12.6, 11.5), gridspec_kw={"hspace": 0.4})
    rt = np.clip(df["rt_z_lag1"].to_numpy()[idx], -2, 2)
    for i, name in enumerate(("R", "C")):
        fit = fits[name]
        scores = (states[name][idx] - fit["mean"]) @ fit["components"][:4].T
        for j, (a, b) in enumerate(pairs):
            ax = axes_a[i, j]
            for label, color in INPUT_COLORS.items():
                m = df["input"].to_numpy()[idx] == label
                ax.scatter(scores[m, a], scores[m, b], s=3, color=color, alpha=0.35, lw=0, label=label)
            ax_b = axes_b[i, j]
            sc = ax_b.scatter(scores[:, a], scores[:, b], s=3, c=rt, cmap="coolwarm", vmin=-2, vmax=2, alpha=0.5, lw=0)
            for axx in (ax, ax_b):
                axx.set_xlabel(f"PC{a + 1} ({fit['explained'][a]:.0%})")
                axx.set_ylabel(f"PC{b + 1} ({fit['explained'][b]:.0%})")
                if j == 0:
                    axx.set_title(f"{MODEL_LABELS[name]}   PR = {fit['participation_ratio']:.1f}", loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=8, label=f"t−1 {l}") for l, c in INPUT_COLORS.items()]
    axes_a[0, 1].legend(handles=handles, frameon=False, fontsize=12)
    fig_a.suptitle("B3a  State space, each model's own PCs, coloured by the last input", y=1.0)
    cbar = fig_b.colorbar(sc, ax=axes_b, shrink=0.6, pad=0.02)
    cbar.set_label("RT on t−1 (within-mouse z, clipped ±2)")
    fig_b.suptitle("B3b  The same projections coloured by the previous trial's RT", y=0.98)
    fig_a.savefig(out_inputs, bbox_inches="tight")
    fig_b.savefig(out_rt, bbox_inches="tight")
    plt.close(fig_a)
    plt.close(fig_b)


def _cv_r2(x, y, groups) -> float:
    pred = np.empty_like(y)
    for train, test in GroupKFold(5).split(x, y, groups):
        pred[test] = Ridge(alpha=ALPHA).fit(x[train], y[train]).predict(x[test])
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def decoding(df, states, out):
    history = df[[f"{v}_lag{k}" for k in range(1, N_LAGS + 2) for v in ("choice", "reward")]].to_numpy()
    history = np.hstack([history, history[:, 0::2] * history[:, 1::2]])
    groups = df["ses_idx"].to_numpy()
    result = {}
    for name in ("R", "C"):
        result[name] = {"raw": [], "beyond_history": [], "unique": []}
        for k in range(1, N_LAGS + 1):
            y = df[f"rt_z_lag{k}"].to_numpy()
            resid = y - LinearRegression().fit(history, y).predict(history)
            result[name]["raw"].append(_cv_r2(states[name], y, groups))
            result[name]["beyond_history"].append(_cv_r2(states[name], resid, groups))
            nuisance = np.hstack([history, df[[f"rt_z_lag{j}" for j in range(1, k)]].to_numpy()]) if k > 1 else history
            unique = y - LinearRegression().fit(nuisance, y).predict(nuisance)
            result[name]["unique"].append(_cv_r2(states[name], unique, groups))
    history_r2 = [float(LinearRegression().fit(history, df[f"rt_z_lag{k}"]).score(history, df[f"rt_z_lag{k}"]))
                  for k in range(1, N_LAGS + 1)]
    lags = np.arange(1, N_LAGS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, name in zip(axes, ("R", "C")):
        ax.plot(lags, result[name]["raw"], color="#222222", lw=2, marker="o", label="RT")
        ax.plot(lags, result[name]["beyond_history"], color="#4C72B0", lw=2, ls="--", marker="o",
                label="RT beyond choice/reward history")
        ax.plot(lags, result[name]["unique"], color="#C44E52", lw=2, ls=":", marker="o",
                label="RT beyond history and more recent RTs")
        ax.axhline(0, color="#888888", lw=1)
        ax.set_xticks(lags)
        ax.set_xlabel("trials back (1 = the input R just received)")
        ax.set_title(MODEL_LABELS[name])
    axes[0].set_ylabel("cross-validated R² (state → RT)")
    axes[0].legend(frameon=False, fontsize=12)
    fig.suptitle("B3c  How much RT the state carries, and for how long", y=1.02)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    rt_autocorr = [float(df["rt_z_lag1"].corr(df[f"rt_z_lag{k}"])) for k in range(2, N_LAGS + 1)]
    return {**result, "history_only_r2": history_r2, "rt_autocorrelation_lag1_vs_lagk": rt_autocorr}


def directions(df, states, out):
    """Per input cluster, the state's OLS slope on RT 1 back; compared by cosine."""
    labels = list(INPUT_COLORS)
    result = {}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), gridspec_kw={"wspace": 0.45})
    for ax, name in zip(axes, ("R", "C")):
        x = states[name]
        vecs, logit_per_sd = {}, {}
        readout = np.linalg.lstsq(np.hstack([x, np.ones((len(x), 1))]), df["logit" if name == "R" else "logit_C"].to_numpy(), rcond=None)[0][:-1]
        for label in labels:
            m = df["input"].to_numpy() == label
            rt = df["rt_z_lag1"].to_numpy()[m]
            rt = rt - rt.mean()
            slope = (x[m] - x[m].mean(0)).T @ rt / (rt @ rt)
            vecs[label] = slope
            logit_per_sd[label] = float(slope @ readout)
        reward_axis = np.mean([x[df["input"].to_numpy() == f"{s} rewarded"].mean(0) - x[df["input"].to_numpy() == f"{s} unrewarded"].mean(0)
                               for s in ("R", "L")], axis=0)
        all_vecs = {**{f"RT dir | {l}": v for l, v in vecs.items()}, "readout": readout, "reward axis": reward_axis}
        names = list(all_vecs)
        mat = np.array([[a @ b / np.linalg.norm(a) / np.linalg.norm(b) for b in all_vecs.values()] for a in all_vecs.values()])
        ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=10,
                        color="white" if abs(mat[i, j]) > 0.6 else "#222222")
        ax.set_xticks(range(len(names)), [n.replace("RT dir | ", "") for n in names], rotation=45, ha="right", fontsize=11)
        ax.set_yticks(range(len(names)), names, fontsize=11)
        ax.set_title(f"{MODEL_LABELS[name]}\n|RT dir| per sd: " + ", ".join(f"{np.linalg.norm(v):.2f}" for v in vecs.values()), fontsize=12)
        result[name] = {"cosines": {"names": names, "matrix": mat}, "logit_change_per_sd_rt": logit_per_sd,
                        "rt_direction_norm_per_sd": {l: float(np.linalg.norm(v)) for l, v in vecs.items()}}
    fig.suptitle("B3d  State covariation with RT 1 back (observational; rows 1–4: within each t−1 input cluster)", y=1.0)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return result


def main() -> None:
    apply_presentation_style()
    df, states = sample()
    fits = {name: pca(x) for name, x in states.items()}
    fig_pca(df, states, fits, figure_path("fig_B3a_pca_inputs.png"), figure_path("fig_B3b_pca_prev_rt.png"))
    payload = {
        "sample": {"n_trials": int(len(df)), "n_sessions": int(df["ses_idx"].nunique()), "n_mice": int(df["subject_id"].nunique())},
        "pca": {name: {"explained_first10": f["explained"][:10], "participation_ratio": f["participation_ratio"],
                       **pc_correlations(df, states[name], f)} for name, f in fits.items()},
        "rt_z_winsorized_fraction": float((np.abs(np.log(df["rt"]).groupby(df["subject_id"]).transform(lambda x: (x - x.mean()) / x.std())) > RT_Z_CLIP).mean()),
        "B3c_decoding": decoding(df, states, figure_path("fig_B3c_rt_decoding.png")),
        "B3d_directions": directions(df, states, figure_path("fig_B3d_rt_direction.png")),
    }
    print(payload["B3c_decoding"])
    payload["B3d_directions"]["R_minus_C_logit_change_per_sd_rt"] = {
        k: payload["B3d_directions"]["R"]["logit_change_per_sd_rt"][k] - payload["B3d_directions"]["C"]["logit_change_per_sd_rt"][k]
        for k in INPUT_COLORS}
    write_json("b3_state_rt.json", payload, PRODUCER)
    print({n: payload["pca"][n] for n in ("R", "C")})
    print({n: payload["B3d_directions"][n]["logit_change_per_sd_rt"] for n in ("R", "C")})


if __name__ == "__main__":
    main()
