"""I1: is the decision variable carried along one shared axis in all four last-input clusters?

    python analysis/i1_cluster_axes.py

Clusters = previous choice × outcome (from the table's own within-session shift; each
session's first trial has no previous trial and is excluded). For a value variable v the
cluster's encoding axis is u_c = cov_c(h, v) / var_c(v) — the direction states move as v
changes within the cluster. For v = the model's own logit (= r·h, r the readout) this is
u_c ∝ Σ_c r, which is NOT r itself; testing r would be circular. Two values from outside the
GRU are checked as well: Hattori ΔQ (C3) and the true P(rew|R) − P(rew|L).

Everything follows from per-cluster joint covariances of [h (128), values (3), nuisances
(4-d session context, trial position)], accumulated in one streaming pass. Reported: 4×4
cosines of u_c across clusters (raw, and with the nuisances partialled out), cosines with r
and PCs 1–5, the share of within-cluster state variance along u_c, and local axes per
logit tercile within each cluster (curvature). Writes ``i1_cluster_axes.json`` and
``fig_I1_cluster_axes.png``.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

CLUSTERS = ["R rewarded", "R unrewarded", "L rewarded", "L unrewarded"]
VALUES = ["logit(right)", "Hattori ΔQ", "P(rew|R) − P(rew|L)"]
BATCH = 400_000


def covariance(acc: dict) -> np.ndarray:
    n, s, ss = acc["n"], acc["s"], acc["ss"]
    mean = s / n
    return (ss - n * np.outer(mean, mean)) / (n - 1)


def axes_from(cov: np.ndarray, n_h: int, n_v: int, partial: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encoding axes (n_v × n_h), value variances, and the (partial) state covariance."""
    h, v, z = slice(0, n_h), slice(n_h, n_h + n_v), slice(n_h + n_v, None)
    if partial:
        zz_inv = np.linalg.pinv(cov[z, z])
        cov = cov - cov[:, z] @ zz_inv @ cov[z, :]
    c_hv, var_v = cov[h, v], np.diag(cov[v, v])
    return (c_hv / var_v).T, var_v, cov[h, h]


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def main() -> None:
    s12.apply_presentation_style()
    t = s12.load_trials(["ses_idx", "trial_idx", "action", "reward", "logit_diff", "trial_position",
                         "reward_probabilityL", "reward_probabilityR"])
    g = t.groupby("ses_idx", sort=False)
    prev_a, prev_r = g["action"].shift(1), g["reward"].shift(1)
    label = np.full(len(t), -1)
    for k, (a, r) in enumerate(((1, True), (1, False), (0, True), (0, False))):
        label[((prev_a == a) & ((prev_r > 0) == r)).to_numpy()] = k
    hattori = pd.read_parquet(s12.CACHE / f"rl_trials_hattori_{s12.MODEL_KEY}.parquet", columns=["q_rl_left", "q_rl_right"])
    values = np.column_stack([t["logit_diff"].to_numpy(), (hattori["q_rl_right"] - hattori["q_rl_left"]).to_numpy(),
                              (t["reward_probabilityR"] - t["reward_probabilityL"]).to_numpy()])
    ctx = np.load(s12.CACHE / f"contexts_{s12.MODEL_KEY}.npz")
    ctx_rows = pd.Series(np.arange(len(ctx["session_ids"])), index=ctx["session_ids"]).reindex(t["ses_idx"]).to_numpy()
    if np.isnan(ctx_rows.astype(float)).any():
        raise AssertionError("Some sessions have no cached context (run d_dynamics.py first)")
    nuisance = np.column_stack([ctx["contexts"][ctx_rows], t["trial_position"].to_numpy()])
    tercile = np.full(len(t), -1)
    for k in range(4):
        m = label == k
        edges = np.quantile(values[m, 0], [1 / 3, 2 / 3])
        tercile[m] = np.digitize(values[m, 0], edges)

    n_h, n_v = s12.HIDDEN, values.shape[1]
    dim = n_h + n_v + nuisance.shape[1]
    acc = {(k, q): {"n": 0, "s": np.zeros(dim), "ss": np.zeros((dim, dim))} for k in range(4) for q in (None, 0, 1, 2)}
    parquet = pq.ParquetFile(s12.trial_table_path())
    start = 0
    for batch in parquet.iter_batches(batch_size=BATCH, columns=s12.STATE_COLUMNS):
        h = np.column_stack([batch.column(c).to_numpy() for c in s12.STATE_COLUMNS]).astype(np.float64)
        rows = slice(start, start + len(h))
        x = np.column_stack([h, values[rows], nuisance[rows]])
        for k in range(4):
            for q in (None, 0, 1, 2):
                m = (label[rows] == k) & ((tercile[rows] == q) if q is not None else True)
                if m.any():
                    a = acc[(k, q)]
                    a["n"] += int(m.sum())
                    a["s"] += x[m].sum(axis=0)
                    a["ss"] += x[m].T @ x[m]
        start += len(h)

    pca = s12.load_pca()
    readout = np.asarray(s12.best_params()["multisubject_gru/~/readout"]["w"], dtype=np.float64)
    r = unit(readout[:, 1] - readout[:, 0])
    pcs = pca["components"][:5]
    out = {"clusters": CLUSTERS, "values": VALUES, "n": [acc[(k, None)]["n"] for k in range(4)]}
    figure = {}
    for partial in (False, True):
        key = "partial" if partial else "raw"
        per_cluster = [axes_from(covariance(acc[(k, None)]), n_h, n_v, partial) for k in range(4)]
        enc = np.stack([unit(a) for a, _, _ in per_cluster])  # cluster × value × unit
        entry = {}
        for vi, vname in enumerate(VALUES):
            u = enc[:, vi]
            share = [float(np.linalg.norm(a[vi]) ** 2 * var_v[vi] / np.trace(c_hh)) for a, var_v, c_hh in per_cluster]
            entry[vname] = {"cosine_across_clusters": u @ u.T, "cosine_with_readout": u @ r,
                            "cosine_with_pcs": u @ pcs.T, "share_of_cluster_variance": share}
            figure[(key, vname)] = u @ u.T
        cross = {f"{VALUES[0]} vs {VALUES[j]}": [float(enc[k, 0] @ enc[k, j]) for k in range(4)] for j in (1, 2)}
        out[key] = {"by_value": entry, "logit_axis_vs_external_value_axes_per_cluster": cross}
    # Curvature: local logit axes per tercile vs the cluster's axis (partial).
    curvature = {}
    for k, cname in enumerate(CLUSTERS):
        whole = unit(axes_from(covariance(acc[(k, None)]), n_h, n_v, True)[0][0])
        local = [unit(axes_from(covariance(acc[(k, q)]), n_h, n_v, True)[0][0]) for q in range(3)]
        curvature[cname] = {"tercile_vs_cluster_axis": [float(v @ whole) for v in local],
                            "low_vs_high_tercile": float(local[0] @ local[2])}
    out["curvature_partial_logit"] = curvature

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.8))
    for ax, vname in zip(axes[:3], VALUES):
        m = figure[("partial", vname)]
        im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(4), CLUSTERS, rotation=30, ha="right", fontsize=10)
        ax.set_yticks(range(4), CLUSTERS, fontsize=10)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{m[i, j]:+.2f}", ha="center", va="center", fontsize=11)
        ax.set_title(f"encoding axis of {vname}\n(cosine across clusters, partial)", fontsize=12)
    fig.colorbar(im, ax=axes[:3], fraction=0.015)
    width = 0.26
    for vi, vname in enumerate(VALUES):
        e = out["partial"]["by_value"][vname]
        axes[3].bar(np.arange(4) + (vi - 1) * width, e["cosine_with_readout"], width, label=vname)
    axes[3].set_xticks(range(4), CLUSTERS, rotation=30, ha="right", fontsize=10)
    axes[3].set_ylabel("cosine with readout r")
    axes[3].axhline(0, color="0.7", lw=0.8)
    axes[3].set_title("encoding axis vs readout direction", fontsize=12)
    axes[3].legend(frameon=False, fontsize=9)
    fig.savefig(s12.figure_path("fig_I1_cluster_axes.png"), bbox_inches="tight")
    plt.close(fig)
    s12.write_json("i1_cluster_axes.json", out, "analysis/i1_cluster_axes.py", extra_groups=s12.rl_groups())

    for key in ("raw", "partial"):
        for vname in VALUES:
            e = out[key]["by_value"][vname]
            c = np.asarray(e["cosine_across_clusters"])
            print(f"{key:7s} {vname:22s} min cos across clusters {c[np.triu_indices(4, 1)].min():+.2f} "
                  f"mean {c[np.triu_indices(4, 1)].mean():+.2f} | cos w/ readout {np.round(e['cosine_with_readout'], 2)} "
                  f"| share {np.round(e['share_of_cluster_variance'], 3)}")
    print("logit vs external axes per cluster", {k: np.round(v, 2).tolist() for k, v in out["partial"]["logit_axis_vs_external_value_axes_per_cluster"].items()})
    print("curvature", {k: np.round(v["tercile_vs_cluster_axis"], 2).tolist() for k, v in curvature.items()})


if __name__ == "__main__":
    main()
