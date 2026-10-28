"""F13: the six F11 example mice in the GRU's hidden-state space (B3f's view).

    python analysis/f13_prototype_state_space.py

For each F11 pair (rows: switch rate, learning rate, bias; columns: low, high), all of the
mouse's eval trials in hidden-state PC1–PC2 (A1's PCA, ``_cache/pca_<key>.npz``), coloured by
the GRU's P(right), over the same grey background as B3f (N_SCATTER trials of all mice,
``s12.subsample_rows`` seed 0). The mice come from ``f11_prototype_examples.json``.
Writes ``fig_F13_prototype_state_space.png``.
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

N_SCATTER = 150_000  # as in B3
ROWS = ["switch rate", "learning rate (CTT)", "bias"]
COL = {"low": "#2166ac", "high": "#b2182b"}


def main() -> None:
    s12.apply_presentation_style()
    pairs = json.loads((s12.ANALYSIS / "f11_prototype_examples.json").read_text())["pairs"]
    trials = s12.load_trials(["subject_id", "p_model1_right"])
    scores = s12.load_pca()["scores"][:, :2]
    sub = s12.subsample_rows(len(trials), N_SCATTER)
    subj = trials["subject_id"].to_numpy()
    p_right = trials["p_model1_right"].to_numpy()
    fig, axes = plt.subplots(3, 2, figsize=(15, 19), sharex=True, sharey=True, layout="constrained")
    for i, name in enumerate(ROWS):
        label = name.replace(" (CTT)", "")
        for j, tag in enumerate(("low", "high")):
            ax = axes[i, j]
            m = pairs[name][tag]["subject_id"]
            rows = np.flatnonzero(subj == m)
            ax.scatter(scores[sub, 0], scores[sub, 1], s=2, color="0.85", linewidths=0, rasterized=True)
            sc = ax.scatter(scores[rows, 0], scores[rows, 1], s=3, c=p_right[rows], cmap="coolwarm", vmin=0, vmax=1,
                            alpha=0.5, linewidths=0, rasterized=True)
            ax.set_title(f"{tag} {label}: mouse {m}\n(n = {len(rows):,} trials)", fontsize=21, color=COL[tag])
            ax.tick_params(labelsize=17)
    for ax in axes[-1]:
        ax.set_xlabel("hidden-state PC1", fontsize=21)
    for ax in axes[:, 0]:
        ax.set_ylabel("hidden-state PC2", fontsize=21)
    cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("GRU P(right)", fontsize=20)
    cb.ax.tick_params(labelsize=17)
    cb.solids.set_alpha(1)
    fig.suptitle("The F11 example mice in the GRU's state space", fontsize=26)
    fig.savefig(s12.figure_path("fig_F13_prototype_state_space.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
