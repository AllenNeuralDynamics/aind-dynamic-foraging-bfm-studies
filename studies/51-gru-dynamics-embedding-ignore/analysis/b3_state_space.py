"""B3b–c: where the two readouts live in the hidden state, and the state in PCs 1–4.

    python analysis/b3_state_space.py

Uses ``b3_pca.py``'s fit (all timesteps). Writes
  * ``b3_state_space.json`` — per-PC correlation with each readout; share of each readout
    direction inside PCs 1–4; cosine between the two directions; R² of each readout from
    PCs 1–4; state counts per previous input;
  * ``fig_B3b_state_space_readouts.png`` — the six PC pairs of PCs 1–4, coloured by
    P(right | engaged) (top row) and P(ignore) (bottom row);
  * ``fig_B3c_prev_input.png`` — PC1 vs the PC most correlated with engagement, one panel per
    previous input (L+, L−, R+, R−, ignore), coloured by each readout.

By construction: logit_diff is linear in the state (side·h + b), so its full-state R² is 1;
the engage readout's linear part is (w_L + w_R)/2 − w_ignore. What is a finding is how much
of each direction lies in the high-variance PCs and how the states arrange by input.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk02 as s13  # noqa: E402

N_PLOT_PCS = 4
N_CORR_PCS = 10
N_SCATTER = 150_000
COLUMNS = ["ses_idx", "action", "reward", "choice_logit_0", "choice_logit_1", "choice_logit_2",
           "p_model1_left", "p_model1_right", "p_model1_ignore", "logit_diff"]
INPUT_ORDER = ["L+", "L−", "R+", "R−", "ignore"]


def previous_input(trials: pd.DataFrame) -> np.ndarray:
    """What the GRU received before each timestep: last timestep's choice × reward, or ignore."""
    action, reward = trials["action"].to_numpy(), trials["reward"].to_numpy() > 0
    label = np.where(action == s13.IGNORE, "ignore",
                     np.where(action == 1, "R", "L").astype(object) + np.where(reward, "+", "−"))
    prev = np.r_[[None], label[:-1]].astype(object)
    ses = trials["ses_idx"].to_numpy()
    prev[np.r_[True, ses[1:] != ses[:-1]]] = None
    return prev


def _r2(y: np.ndarray, x: np.ndarray) -> float:
    design = np.column_stack([x, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(1 - np.var(y - design @ coef) / np.var(y))


def _scatter(ax, x, y, c, cmap, label):
    return ax.scatter(x, y, s=2, c=c, cmap=cmap, vmin=0, vmax=1, linewidths=0, rasterized=True)


def main() -> None:
    s13.apply_presentation_style()
    trials = s13.load_trials(COLUMNS)
    pca = s13.load_pca()
    scores, components = pca["scores"], pca["components"]
    eig = np.clip(pca["eigvals"], 0, None)
    trials["prev_input"] = previous_input(trials)

    # Where each readout lives.
    directions = s13.readout_directions()
    readouts = {"side": trials["logit_diff"].to_numpy(np.float64), "engage": trials["engage_logit"].to_numpy()}
    geometry = {}
    for name, d in directions.items():
        unit = d / np.linalg.norm(d)
        loadings = components @ unit  # cosine of the direction with each PC
        geometry[name] = {
            "share_of_direction_in_pcs_1_4": float(np.sum(loadings[:N_PLOT_PCS] ** 2)),
            "cosine_with_pc": loadings[:N_CORR_PCS].round(4).tolist(),
            # Variance of the readout carried by each PC (projection of the state onto the
            # direction, decomposed over PCs).
            "readout_variance_share_by_pc": (
                (loadings ** 2 * eig) / np.sum(loadings ** 2 * eig))[:N_CORR_PCS].round(4).tolist(),
            "corr_with_pc": [float(np.corrcoef(scores[:, k], readouts[name])[0, 1]) for k in range(N_CORR_PCS)],
            "r2_from_pcs_1_4": _r2(readouts[name], scores[:, :N_PLOT_PCS].astype(np.float64)),
        }
    cos_side_engage = float(directions["side"] @ directions["engage"]
                            / np.linalg.norm(directions["side"]) / np.linalg.norm(directions["engage"]))
    engage_pc = int(np.argmax(np.abs(geometry["engage"]["corr_with_pc"][1:N_PLOT_PCS])) + 1)

    sub = s13.subsample_rows(len(trials), N_SCATTER)
    pr = trials["p_right_engaged"].to_numpy()[sub]
    pi = trials["p_model1_ignore"].to_numpy()[sub]
    xs = scores[sub]

    pairs = list(combinations(range(N_PLOT_PCS), 2))
    fig, axes = plt.subplots(2, len(pairs), figsize=(4.6 * len(pairs), 9.6))
    for col, (i, j) in enumerate(pairs):
        for row, (values, cmap) in enumerate(((pr, "coolwarm"), (pi, "magma_r"))):
            ax = axes[row, col]
            sc = _scatter(ax, xs[:, i], xs[:, j], values, cmap, None)
            ax.set_xlabel(f"PC{i + 1}")
            ax.set_ylabel(f"PC{j + 1}")
            if row == 0:
                ax.set_title(f"PC{i + 1} vs PC{j + 1}", fontsize=14)
            if col == len(pairs) - 1:
                fig.colorbar(sc, ax=axes[row, :], fraction=0.012, pad=0.01,
                             label="P(right | engaged)" if row == 0 else "P(ignore)")
    fig.suptitle(f"C3 hidden state, PCs 1–4 ({N_SCATTER:,} random timesteps of {len(trials):,})", fontsize=15)
    fig.savefig(s13.figure_path("fig_B3b_state_space_readouts.png"), bbox_inches="tight")
    plt.close(fig)

    prev = trials["prev_input"].to_numpy()[sub]
    fig, axes = plt.subplots(2, len(INPUT_ORDER), figsize=(4.8 * len(INPUT_ORDER), 9.6), sharex=True, sharey=True)
    counts = trials["prev_input"].value_counts().reindex(INPUT_ORDER).fillna(0).astype(int).to_dict()
    for col, value in enumerate(INPUT_ORDER):
        mask = prev == value
        for row, (values, cmap, label) in enumerate(((pr, "coolwarm", "P(right | engaged)"),
                                                     (pi, "magma_r", "P(ignore)"))):
            ax = axes[row, col]
            ax.scatter(xs[:, 0], xs[:, engage_pc], s=2, color="0.88", linewidths=0, rasterized=True)
            sc = _scatter(ax, xs[mask, 0], xs[mask, engage_pc], values[mask], cmap, label)
            if row == 0:
                ax.set_title(f"previous trial: {value}\n(n={counts[value]:,})", fontsize=13)
            ax.set_xlabel("PC1")
            if col == 0:
                ax.set_ylabel(f"PC{engage_pc + 1}")
            if col == len(INPUT_ORDER) - 1:
                fig.colorbar(sc, ax=axes[row, :], fraction=0.012, pad=0.01, label=label)
    fig.suptitle("C3 state by the input it just received (others grey)", fontsize=15)
    fig.savefig(s13.figure_path("fig_B3c_prev_input.png"), bbox_inches="tight")
    plt.close(fig)

    by_input = trials.groupby("prev_input")[["p_right_engaged", "p_model1_ignore"]].mean().reindex(INPUT_ORDER)
    s13.write_json("b3_state_space.json", {
        "n_timesteps": int(len(trials)),
        "n_scatter": N_SCATTER,
        "by_construction": "logit_diff = side·h + b exactly; engage direction = linear part of engage_logit",
        "readout_geometry": geometry,
        "cosine_side_vs_engage_direction": cos_side_engage,
        "engage_pc_plotted": engage_pc + 1,
        "prev_input_counts": counts,
        "mean_by_prev_input": by_input.round(4).to_dict(orient="index"),
    }, "analysis/b3_state_space.py")


if __name__ == "__main__":
    main()
