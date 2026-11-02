"""C2c in every PC pair: where the GRU beats CTT, for all six pairs of PCs 1–4.

    python analysis/c2c_pc_pairs.py

Same quantity as the first two panels of ``fig_C2c_advantage``: the per-trial advantage
log p_GRU(choice) − log p_CTT(choice), binned by the trial's state on a 48 × 48 grid per PC
pair (range: 0.01th–99.99th percentile of each PC plus a 5% margin). Top row: mean advantage per trial (cells
with < MIN_PER_BIN trials blank). Bottom row: summed advantage (mean × count) as a percentage
of the total over all trials, so every pair shares one scale (all grids are 48 × 48). Writes ``fig_C2c_advantage_pc_pairs.png``.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
from c2_compare import MIN_PER_BIN, REF, log_p  # noqa: E402

N_PCS = 4
N_BINS = 48
RANGE_PCT, MARGIN = (0.01, 99.99), 0.05


def padded_range(v: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(v, RANGE_PCT)
    return lo - MARGIN * (hi - lo), hi + MARGIN * (hi - lo)


def maps(x: np.ndarray, y: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple]:
    ex, ey = (np.linspace(*padded_range(v), N_BINS + 1) for v in (x, y))
    ix, iy = np.digitize(x, ex) - 1, np.digitize(y, ey) - 1
    ok = (ix >= 0) & (ix < N_BINS) & (iy >= 0) & (iy < N_BINS)
    flat = ix[ok] * N_BINS + iy[ok]
    n = np.bincount(flat, minlength=N_BINS ** 2)
    total = np.bincount(flat, weights=a[ok], minlength=N_BINS ** 2)
    mean = np.where(n >= MIN_PER_BIN, total / np.maximum(n, 1), np.nan)
    return mean.reshape(N_BINS, N_BINS), total.reshape(N_BINS, N_BINS), (ex[0], ex[-1], ey[0], ey[-1])


def main() -> None:
    s12.apply_presentation_style()
    t = s12.load_trials(["action", "p_model1_right"])
    action = t["action"].to_numpy()
    p_ref = pd.read_parquet(s12.CACHE / f"rl_trials_{REF}_{s12.MODEL_KEY}.parquet", columns=["p_rl_right"])["p_rl_right"].to_numpy()
    a = log_p(action, t["p_model1_right"].to_numpy()) - log_p(action, p_ref)
    scores = s12.load_pca()["scores"][:, :N_PCS]
    pairs = list(itertools.combinations(range(N_PCS), 2))
    results = [maps(scores[:, i], scores[:, j], a) for i, j in pairs]
    lim = np.nanpercentile(np.abs(np.concatenate([m[0].ravel() for m in results])), 98)
    shares = [100 * total / a.sum() for _, total, _ in results]
    slim = np.nanpercentile(np.abs(np.concatenate([x.ravel() for x in shares])), 99.5)

    fig, axes = plt.subplots(2, len(pairs), figsize=(6 * len(pairs), 11), layout="constrained")
    for k, ((i, j), (mean, total, extent)) in enumerate(zip(pairs, results)):
        im = axes[0, k].imshow(mean.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", extent=extent)
        im2 = axes[1, k].imshow(shares[k].T, origin="lower", cmap="RdBu_r", vmin=-slim, vmax=slim, aspect="auto",
                                extent=extent)
        for ax in axes[:, k]:
            ax.set_xlabel(f"PC{i + 1}")
            ax.set_ylabel(f"PC{j + 1}")
        axes[0, k].set_title(f"PC{i + 1} vs PC{j + 1}: mean per trial", fontsize=13)
        axes[1, k].set_title(f"PC{i + 1} vs PC{j + 1}: summed", fontsize=13)
    fig.colorbar(im, ax=axes[0, :], fraction=0.012, pad=0.01, label=f"mean log p_GRU − log p_{REF.upper()}\n(nats / trial)")
    fig.colorbar(im2, ax=axes[1, :], fraction=0.012, pad=0.01, label="share of the total advantage\n(% per cell)")
    fig.suptitle(f"Where the GRU beats {REF.upper()}, in every pair of PCs 1–{N_PCS} (top: per trial; bottom: total)",
                 fontsize=17)
    fig.savefig(s12.figure_path("fig_C2c_advantage_pc_pairs.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
