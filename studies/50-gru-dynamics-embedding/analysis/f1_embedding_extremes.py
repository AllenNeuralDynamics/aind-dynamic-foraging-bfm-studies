"""F1: mice at the two ends of the embedding's gain and bias directions, side by side.

    python analysis/f1_embedding_extremes.py

Directions come from F3 (the embedding direction predicting the GRU's per-mouse reward gain,
and the one predicting its bias). Mice are projected onto each and the bottom / top FRAC
taken. Row 1 (gain): update maps folded by choice side — x = logit toward the side chosen on
trial t, y = the same after its outcome — and per-mouse reward-effect kernels (animal, GRU).
Row 2 (bias): P(right) distribution and where each group sits in PC1–PC2.
Reads cached / committed inputs. Writes ``fig_F1_embedding_extremes.png`` and
``f1_embedding_extremes.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

FRAC = 0.10
AXES = {"gain": "gru reward gain (1 back)", "bias": "gru bias"}
EDGES = np.linspace(-4, 4, 25)
MIN_PER_BIN = 200
LOW, HIGH = "#56B4E9", "#D55E00"


def binned_mean(x, y):
    which = np.digitize(x, EDGES) - 1
    ok = (which >= 0) & (which < len(EDGES) - 1)
    n = np.bincount(which[ok], minlength=len(EDGES) - 1)
    s = np.bincount(which[ok], weights=y[ok], minlength=len(EDGES) - 1)
    return np.where(n >= MIN_PER_BIN, s / np.maximum(n, 1), np.nan)


def main() -> None:
    s12.apply_presentation_style()
    f3 = json.loads((s12.ANALYSIS / "f3_embedding_features.json").read_text())
    kernels = json.loads((s12.ANALYSIS / "c1_kernels.json").read_text())
    emb = s12.subject_embeddings()
    groups = {}
    for axis, feature in AXES.items():
        projection = emb.to_numpy() @ np.asarray(f3["decoding"][feature]["direction"])
        order = emb.index[np.argsort(projection)]
        n = int(round(FRAC * len(order)))
        groups[axis] = {"low": list(order[:n]), "high": list(order[-n:])}

    t = s12.load_trials(["subject_id", "ses_idx", "action", "reward", "logit_diff", "p_model1_right"])
    t["logit_next"] = t.groupby("ses_idx", sort=False)["logit_diff"].shift(-1)
    side = np.where(t["action"].to_numpy() == 1, 1.0, -1.0)
    folded_now, folded_next = side * t["logit_diff"].to_numpy(), side * t["logit_next"].to_numpy()
    rewarded = t["reward"].to_numpy() > 0
    scores = s12.load_pca()["scores"][:, :2]
    centres = (EDGES[:-1] + EDGES[1:]) / 2
    lags = np.arange(1, kernels["n_back"] + 1)

    def member(subjects):
        return t["subject_id"].isin(subjects).to_numpy()

    def kernel(source, subjects):
        nb = kernels["n_back"]
        effects = [np.asarray(c)[:nb][::-1] - np.asarray(c)[nb:2 * nb][::-1]
                   for s, c in kernels["per_mouse_coef"][source].items() if s in set(subjects)]
        return np.median(effects, axis=0)

    summary = {"frac": FRAC, "directions": AXES, "groups": groups, "gain": {}, "bias": {}}
    fig, axes = plt.subplots(2, 3, figsize=(21, 12.5))
    for ax, is_rew, title in ((axes[0, 0], True, "after a rewarded choice"), (axes[0, 1], False, "after an unrewarded choice")):
        for level, color in (("low", LOW), ("high", HIGH)):
            m = member(groups["gain"][level]) & (rewarded == is_rew) & np.isfinite(folded_next)
            curve = binned_mean(folded_now[m], folded_next[m])
            ax.plot(centres, curve, color=color, lw=2.4, label=f"{level} gain ({len(groups['gain'][level])} mice)")
            summary["gain"][f"update_map_{'rewarded' if is_rew else 'unrewarded'}_{level}"] = curve
        ax.plot(EDGES, EDGES, color="0.6", ls=":", lw=1)
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        ax.set_xlabel("logit toward the chosen side, trial t")
        ax.set_ylabel("same, trial t+1")
        ax.set_title(f"Gain axis: {title}", fontsize=13)
        ax.legend(frameon=False, fontsize=10)
    for level, color in (("low", LOW), ("high", HIGH)):
        for source, ls in (("animal", "--"), ("gru", "-")):
            k = kernel(source, groups["gain"][level])
            axes[0, 2].plot(lags, k, color=color, ls=ls, marker="o", ms=3, lw=2, label=f"{level} gain, {source}")
            summary["gain"][f"reward_kernel_{source}_{level}"] = k
    axes[0, 2].axhline(0, color="0.8", lw=0.8)
    axes[0, 2].set_xlabel("trials back")
    axes[0, 2].set_ylabel("reward effect (median over mice)")
    axes[0, 2].set_title("Gain axis: reward-effect kernels", fontsize=13)
    axes[0, 2].legend(frameon=False, fontsize=9)

    bins = np.linspace(0, 1, 41)
    for level, color in (("low", LOW), ("high", HIGH)):
        m = member(groups["bias"][level])
        axes[1, 0].hist(t["p_model1_right"].to_numpy()[m], bins=bins, density=True, histtype="step", lw=2.2, color=color,
                        label=f"{level} bias projection: mean P(right) {t['p_model1_right'].to_numpy()[m].mean():.3f}")
        summary["bias"][f"mean_p_right_{level}"] = float(t["p_model1_right"].to_numpy()[m].mean())
        summary["bias"][f"mean_animal_p_right_{level}"] = float(t["action"].to_numpy()[m].mean())
    axes[1, 0].set_xlabel("model P(right)")
    axes[1, 0].set_title("Bias axis: P(right) per trial", fontsize=13)
    axes[1, 0].legend(frameon=False, fontsize=10)
    sub = s12.subsample_rows(len(t), 150_000)
    for ax, level in ((axes[1, 1], "low"), (axes[1, 2], "high")):
        m = member(groups["bias"][level])
        ax.scatter(scores[sub, 0], scores[sub, 1], s=1.5, color="0.88", linewidths=0, rasterized=True)
        ax.hist2d(scores[m, 0], scores[m, 1], bins=[np.linspace(-4, 4, 61), np.linspace(-4, 3, 53)], cmap="magma_r",
                  cmin=5, rasterized=True)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Bias axis, {level}: where its trials sit", fontsize=13)
        summary["bias"][f"mean_pc1_{level}"] = float(scores[m, 0].mean())
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_F1_embedding_extremes.png"))
    plt.close(fig)
    s12.write_json("f1_embedding_extremes.json", {**summary, "bin_edges": EDGES}, "analysis/f1_embedding_extremes.py")
    print({k: v for k, v in summary["bias"].items()})
    print("gain kernels lag1 low/high gru", summary["gain"]["reward_kernel_gru_low"][0], summary["gain"]["reward_kernel_gru_high"][0],
          "animal", summary["gain"]["reward_kernel_animal_low"][0], summary["gain"]["reward_kernel_animal_high"][0])


if __name__ == "__main__":
    main()
