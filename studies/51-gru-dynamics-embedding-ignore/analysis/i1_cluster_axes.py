"""I1: is the side value carried on one shared axis — including through an ignore?

    python analysis/i1_cluster_axes.py

The state falls into clusters by the input just received (r1 B3c): L+, L−, R+, R−, and for
C3 a fifth, post-ignore cluster. Within each cluster c, the encoding axis of the side readout
is u_c = cov_c(h, v) / var_c(v), v = logit_diff (the direction along which the state moves
as v changes within the cluster). Cosines between the u_c say whether one shared axis
carries v across clusters. Same for the engage readout (C3). C (Study 50's table) gives the
four-cluster reference on the shared sessions. Also: how much side variance each cluster
holds (std of v) — the post-ignore cluster should hold less if ignores shrink v.
Writes ``i1_cluster_axes.json`` and ``fig_I1_cluster_axes.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk02 as s13  # noqa: E402

CLUSTERS = ["L+", "L−", "R+", "R−", "ignore"]
N_SAMPLE = 600_000


def prev_input(t) -> np.ndarray:
    action, reward = t["action"].to_numpy(), t["reward"].to_numpy() > 0
    label = np.where(action == s13.IGNORE, "ignore", np.where(action == 1, "R", "L").astype(object) + np.where(reward, "+", "−"))
    prev = np.r_[[None], label[:-1]].astype(object)
    ses = t["ses_idx"].to_numpy()
    prev[np.r_[True, ses[1:] != ses[:-1]]] = None
    return prev


def axes_and_stats(h: np.ndarray, readouts: dict[str, np.ndarray], cluster: np.ndarray, names: list[str]) -> dict:
    out = {}
    for rname, v in readouts.items():
        axes, stats = {}, {}
        for c in names:
            m = cluster == c
            if m.sum() < 1000:
                continue
            hc, vc = h[m] - h[m].mean(axis=0), v[m] - v[m].mean()
            u = hc.T @ vc / np.sum(vc ** 2)
            axes[c] = u / np.linalg.norm(u)
            stats[c] = {"n": int(m.sum()), "readout_std": float(v[m].std()), "readout_mean": float(v[m].mean())}
        cos = np.array([[axes[a] @ axes[b] for b in axes] for a in axes])
        out[rname] = {"clusters": list(axes), "cosine": cos.round(4), "stats": stats, "_axes": axes}
    return out


def main() -> None:
    s13.apply_presentation_style()
    drop = s13.unshared_sessions()
    cols = ["ses_idx", "session_id", "action", "reward", "logit_diff", "choice_logit_0", "choice_logit_1", "choice_logit_2"]
    c3 = s13.load_trials(cols + s13.STATE_COLUMNS)
    c3 = c3[~c3["session_id"].isin(drop["C3"])].reset_index(drop=True)
    c3["cluster"] = prev_input(c3)
    sub = s13.subsample_rows(len(c3), N_SAMPLE, seed=3)
    r3 = axes_and_stats(c3[s13.STATE_COLUMNS].to_numpy(np.float64)[sub],
                        {"side": c3["logit_diff"].to_numpy(np.float64)[sub], "engage": c3["engage_logit"].to_numpy()[sub]},
                        c3["cluster"].to_numpy()[sub], CLUSTERS)
    del c3
    c = s13.load_twin(["ses_idx", "session_id", "action", "reward", "logit_diff"] + s13.STATE_COLUMNS)
    c = c[~c["session_id"].isin(drop["C"])].reset_index(drop=True)
    c["cluster"] = prev_input(c)
    sub = s13.subsample_rows(len(c), N_SAMPLE, seed=3)
    r2 = axes_and_stats(c[s13.STATE_COLUMNS].to_numpy(np.float64)[sub], {"side": c["logit_diff"].to_numpy(np.float64)[sub]},
                        c["cluster"].to_numpy()[sub], CLUSTERS[:4])
    del c

    # Readout directions for reference: how well each cluster axis aligns with the readout itself.
    directions = s13.readout_directions()
    readout_cos = {name: {cl: float(abs(ax @ directions[name]) / np.linalg.norm(directions[name]))
                          for cl, ax in r3[name]["_axes"].items()} for name in ("side", "engage")}

    fig, axes = plt.subplots(1, 4, figsize=(25, 6), gridspec_kw={"width_ratios": [1, 1, 1, 1.1]})
    for ax, (title, res) in zip(axes[:3], (("C3 side axis", r3["side"]), ("C side axis (Study 50)", r2["side"]),
                                           ("C3 engage axis", r3["engage"]))):
        im = ax.imshow(res["cosine"], cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(res["clusters"])), res["clusters"])
        ax.set_yticks(range(len(res["clusters"])), res["clusters"])
        for i in range(len(res["clusters"])):
            for j in range(len(res["clusters"])):
                ax.text(j, i, f"{res['cosine'][i, j]:.2f}", ha="center", va="center", fontsize=11)
        ax.set_title(f"{title}: cosines between clusters", fontsize=13)
        ax.set_xlabel("input just received")
    fig.colorbar(im, ax=axes[:3], fraction=0.015)
    ax = axes[3]
    x = np.arange(len(CLUSTERS))
    ax.bar(x - 0.2, [r3["side"]["stats"][c]["readout_std"] for c in CLUSTERS], 0.4, color=s13.ENGAGE_COLOR, label="C3")
    ax.bar(x[:4] + 0.2, [r2["side"]["stats"][c]["readout_std"] for c in CLUSTERS[:4]], 0.4, color=s13.MODEL_COLOR, label="C")
    ax.set_xticks(x, CLUSTERS)
    ax.set(ylabel="std of side logit within the cluster", title="How much side value each cluster holds")
    ax.legend(frameon=False)
    fig.suptitle("Is the side value carried on one axis across the last-input clusters? (shared sessions)", fontsize=15)
    fig.savefig(s13.figure_path("fig_I1_cluster_axes.png"), bbox_inches="tight")
    plt.close(fig)

    def clean(res):
        return {k: {kk: vv for kk, vv in v.items() if kk != "_axes"} for k, v in res.items()}

    s13.write_json("i1_cluster_axes.json", {
        "n_sample_each": N_SAMPLE, "definition": "u_c = cov_c(h, readout) / var_c(readout), unit-normalised",
        "C3": clean(r3), "C": clean(r2), "cosine_cluster_axis_with_readout_direction_C3": readout_cos,
    }, "analysis/i1_cluster_axes.py", extra_groups=s13.twin_wandb_groups())


if __name__ == "__main__":
    main()
