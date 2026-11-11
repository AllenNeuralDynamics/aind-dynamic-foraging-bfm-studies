"""B3a: one PCA of the hidden state over every trial of the split, and its variance spectrum.

    python analysis/b3_pca.py

Fits on all timesteps, ignored trials included (streamed; float64 covariance), so every later state-space figure uses
the same axes. Writes
  * ``_cache/pca_<key>.npz`` — mean, components, eigenvalues, top-20 scores per trial
    (row order = the A1 table's order); gitignored;
  * ``b3_pca.json`` — variance explained, participation ratio, PCs needed for 90/95/99%;
  * ``fig_B3a_pca_variance.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk02 as s13  # noqa: E402

BATCH = 500_000
N_SHOWN = 20


def _batches():
    parquet = pq.ParquetFile(s13._read_cache(s13.trial_table_path()))
    for batch in parquet.iter_batches(batch_size=BATCH, columns=s13.STATE_COLUMNS):
        yield np.column_stack([batch.column(c).to_numpy() for c in s13.STATE_COLUMNS]).astype(np.float64)


def fit() -> dict[str, np.ndarray]:
    n, total, cross = 0, np.zeros(s13.HIDDEN), np.zeros((s13.HIDDEN, s13.HIDDEN))
    for x in _batches():
        n += len(x)
        total += x.sum(axis=0)
        cross += x.T @ x
    mean = total / n
    cov = (cross - n * np.outer(mean, mean)) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, components = eigvals[order], eigvecs[:, order].T  # components: PCs x units
    # Fix each PC's sign so that it is reproducible: largest-magnitude loading positive.
    signs = np.sign(components[np.arange(len(components)), np.argmax(np.abs(components), axis=1)])
    components *= signs[:, None]

    projection = components[: s13.N_PCS_SAVED].T
    scores = np.concatenate([(x - mean) @ projection for x in _batches()]).astype(np.float32)
    return {"mean": mean, "components": components, "eigvals": eigvals, "scores": scores, "n_trials": np.array(n)}


def main() -> None:
    s13.apply_presentation_style()
    pca = fit()
    np.savez(s13.pca_path(), **pca)

    eigvals = np.clip(pca["eigvals"], 0, None)
    ratio = eigvals / eigvals.sum()
    cumulative = np.cumsum(ratio)
    participation_ratio = float(eigvals.sum() ** 2 / np.sum(eigvals ** 2))
    n_for = {f"{int(q * 100)}%": int(np.searchsorted(cumulative, q) + 1) for q in (0.9, 0.95, 0.99)}

    fig, ax = plt.subplots(figsize=(9, 5.5))
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
    ax.set_title(f"Hidden-state PCA, {int(pca['n_trials']):,} trials — participation ratio "
                 f"{participation_ratio:.2f}; PCs for 90/95/99%: "
                 f"{n_for['90%']}/{n_for['95%']}/{n_for['99%']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(s13.figure_path("fig_B3a_pca_variance.png"))
    plt.close(fig)

    s13.write_json("b3_pca.json", {
        "n_trials": int(pca["n_trials"]),
        "fit": "all trials of the split; float64 covariance; eigh; PC sign = largest |loading| positive",
        "variance_explained_ratio": ratio[:N_SHOWN].round(6).tolist(),
        "cumulative_variance_explained": cumulative[:N_SHOWN].round(6).tolist(),
        "participation_ratio": round(participation_ratio, 4),
        "n_pcs_for_variance": n_for,
    }, "analysis/b3_pca.py")


if __name__ == "__main__":
    main()
