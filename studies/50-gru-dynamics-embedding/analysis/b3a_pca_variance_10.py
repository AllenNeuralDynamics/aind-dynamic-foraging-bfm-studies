"""B3a, truncated to the first N_SHOWN PCs: the variance spectrum from the cached PCA.

    python analysis/b3a_pca_variance_10.py

Plot-only: reads the eigenvalues saved by ``b3_pca.py`` (``_cache/pca_<key>.npz``) and draws
B3a's bars + cumulative curve for PCs 1–N_SHOWN; percentages are of the total variance (all 128
PCs). Writes ``fig_B3a_pca_variance_10.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

N_SHOWN = 10


def main() -> None:
    s12.apply_presentation_style()
    pca = s12.load_pca()
    eigvals = np.clip(pca["eigvals"], 0, None)
    ratio = eigvals / eigvals.sum()
    cumulative = np.cumsum(ratio)
    participation_ratio = float(eigvals.sum() ** 2 / np.sum(eigvals ** 2))
    n_for = {f"{int(q * 100)}%": int(np.searchsorted(cumulative, q) + 1) for q in (0.9, 0.95, 0.99)}

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    pcs = np.arange(1, N_SHOWN + 1)
    ax.bar(pcs, ratio[:N_SHOWN] * 100, color="#4C72B0", label="per PC")
    ax.set_xlabel("PC")
    ax.set_ylabel("variance explained (%)")
    ax.set_xticks(pcs)
    twin = ax.twinx()
    twin.plot(pcs, cumulative[:N_SHOWN] * 100, color="#222222", marker="o", ms=4, label="cumulative")
    twin.set_ylim(0, 101)
    twin.set_ylabel("cumulative (%)")
    twin.spines["right"].set_visible(True)
    ax.set_title(f"Hidden-state PCA, {int(pca['n_trials']):,} trials\nparticipation ratio "
                 f"{participation_ratio:.2f}; PCs for 90/95/99%: "
                 f"{n_for['90%']}/{n_for['95%']}/{n_for['99%']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_B3a_pca_variance_10.png"))
    plt.close(fig)
    print("cumulative at 10 PCs", round(100 * cumulative[N_SHOWN - 1], 1), flush=True)


if __name__ == "__main__":
    main()
