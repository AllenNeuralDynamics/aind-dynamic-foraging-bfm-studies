"""I2: targeted dimensionality reduction (TDR) of the state around the decision variable.

    python analysis/i2_tdr.py

Regress the centred 128-d state on task variables (Mante et al. 2013 style), on a fixed
subsample of trials with a previous trial:
    choice c = ±1 (previous choice R/L), reward ρ = ±1 (previous trial rewarded / not),
    c·ρ, value v (the model's logit, standardised), v·c,
    nuisances: 4-d session context, trial position.
v·c is included because I1 found two value axes — one per previous-choice lobe — so a single
pooled value axis would average them; the lobe axes are β_v ± β_vc. The axes β (128-d) are
orthogonalised in the order [v, c, ρ, v·c, c·ρ] and the states projected onto them.
Writes ``_cache/i2_tdr_axes.npz`` (axes for I3), ``i2_tdr.json`` and ``fig_I2_tdr.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

N_TRIALS = 1_000_000
SEED = 3
TERMS = ["value", "choice", "reward", "value×choice", "choice×reward"]
CLUSTERS = {"R rewarded": (1, 1), "R unrewarded": (1, -1), "L rewarded": (-1, 1), "L unrewarded": (-1, -1)}
COLORS = {"R rewarded": "#0072B2", "R unrewarded": "#56B4E9", "L rewarded": "#D55E00", "L unrewarded": "#E69F00"}


def regressors(t: pd.DataFrame) -> pd.DataFrame:
    """Task regressors for every row (NaN where the session has no previous trial)."""
    g = t.groupby("ses_idx", sort=False)
    c = 2.0 * g["action"].shift(1) - 1.0
    rho = np.where(g["reward"].shift(1) > 0, 1.0, -1.0)
    rho = pd.Series(rho, index=t.index).where(c.notna())
    v = t["logit_diff"]
    v = (v - v.mean()) / v.std()
    return pd.DataFrame({"value": v, "choice": c, "reward": rho, "value×choice": v * c, "choice×reward": c * rho})


def main() -> None:
    s12.apply_presentation_style()
    t = s12.load_trials(["ses_idx", "action", "reward", "logit_diff", "trial_position"])
    x_task = regressors(t)
    ctx = np.load(s12.CACHE / f"contexts_{s12.MODEL_KEY}.npz")
    ctx_rows = pd.Series(np.arange(len(ctx["session_ids"])), index=ctx["session_ids"]).reindex(t["ses_idx"]).to_numpy()
    nuisance = np.column_stack([ctx["contexts"][ctx_rows], t["trial_position"].to_numpy()])
    eligible = np.flatnonzero(x_task["choice"].notna().to_numpy())
    rows = np.sort(np.random.default_rng(SEED).choice(eligible, size=N_TRIALS, replace=False))

    pca = s12.load_pca()
    h = s12.load_trials(s12.STATE_COLUMNS).to_numpy(dtype=np.float32)[rows].astype(np.float64) - pca["mean"]
    design = np.column_stack([x_task.to_numpy()[rows], nuisance[rows], np.ones(len(rows))])
    beta, *_ = np.linalg.lstsq(design, h, rcond=None)  # (n_regressors, 128)
    raw_axes = beta[:len(TERMS)]
    q, _ = np.linalg.qr(raw_axes.T)  # orthonormal basis in TERMS order
    q = q * np.sign(np.sum(q * raw_axes.T, axis=0))  # keep each axis pointing along its β
    proj = h @ q

    readout = np.asarray(s12.best_params()["multisubject_gru/~/readout"]["w"], dtype=np.float64)
    r = readout[:, 1] - readout[:, 0]
    r /= np.linalg.norm(r)
    unit_raw = raw_axes / np.linalg.norm(raw_axes, axis=1, keepdims=True)
    lobe = {"R lobe (β_v + β_vc)": raw_axes[0] + raw_axes[3], "L lobe (β_v − β_vc)": raw_axes[0] - raw_axes[3]}
    lobe_unit = {k: v / np.linalg.norm(v) for k, v in lobe.items()}
    total_var = float(np.trace(np.cov(h, rowvar=False)))
    tdr_var = float(np.sum(np.var(proj, axis=0, ddof=1)))
    pc_var = float(np.sum(pca["eigvals"][:len(TERMS)]))
    fitted = design[:, :len(TERMS)] @ raw_axes
    task_r2 = float(1 - np.sum((h - fitted - (design[:, len(TERMS):] @ beta[len(TERMS):])) ** 2) / np.sum((h - h.mean(0)) ** 2))

    np.savez(s12.keyed_cache("i2_tdr_axes.npz"), basis=q, raw_axes=raw_axes, terms=np.array(TERMS),
             lobe_r=lobe["R lobe (β_v + β_vc)"], lobe_l=lobe["L lobe (β_v − β_vc)"])

    c, rho = design[:, 1], design[:, 2]
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.6))
    order = np.random.default_rng(0).permutation(len(rows))[:150_000]
    for name, (cc, rr) in CLUSTERS.items():
        m = np.zeros(len(rows), bool)
        m[order] = True
        m &= (c == cc) & (rho == rr)
        axes[0].scatter(proj[m, 1], proj[m, 0], s=1.5, alpha=0.3, color=COLORS[name], linewidths=0, rasterized=True)
        axes[1].scatter(proj[m, 2], proj[m, 0], s=1.5, alpha=0.3, color=COLORS[name], linewidths=0, rasterized=True)
        axes[2].scatter(proj[m, 1], proj[m, 3], s=1.5, alpha=0.3, color=COLORS[name], linewidths=0, rasterized=True)
        for ax in axes:
            ax.scatter([], [], s=30, color=COLORS[name], label=f"previous {name}")
    labels = [("choice axis", "value axis"), ("reward axis", "value axis"), ("choice axis", "value×choice axis")]
    for ax, (xl, yl) in zip(axes, labels):
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    axes[0].legend(frameon=False, fontsize=10, markerscale=1)
    axes[0].set_title("Value vs previous choice", fontsize=13)
    axes[1].set_title("Value vs previous outcome", fontsize=13)
    axes[2].set_title("Lobe-specific tilt of the value axis", fontsize=13)
    fig.suptitle(f"TDR projection ({len(order):,} of {len(rows):,} fitted trials); axes orthogonalised in the order "
                 f"{', '.join(TERMS)}", fontsize=12)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_I2_tdr.png"))
    plt.close(fig)

    s12.write_json("i2_tdr.json", {
        "n_trials": int(len(rows)), "seed": SEED, "terms": TERMS, "value": "model logit(right), standardised",
        "nuisances": ["context e0..e3", "trial_position"],
        "raw_axis_cosines": unit_raw @ unit_raw.T, "raw_axis_cosine_with_readout": unit_raw @ r,
        "raw_axis_norm": np.linalg.norm(raw_axes, axis=1),
        "lobe_axes_cosine": float(lobe_unit["R lobe (β_v + β_vc)"] @ lobe_unit["L lobe (β_v − β_vc)"]),
        "lobe_axes_cosine_with_readout": {k: float(v @ r) for k, v in lobe_unit.items()},
        "variance_share_tdr_subspace": tdr_var / total_var, "variance_share_top_pcs_same_dim": pc_var / total_var,
        "task_plus_nuisance_r2": task_r2,
        "cluster_centroids": {n: proj[(c == cc) & (rho == rr)].mean(axis=0) for n, (cc, rr) in CLUSTERS.items()},
        "cluster_value_sd": {n: float(proj[(c == cc) & (rho == rr), 0].std()) for n, (cc, rr) in CLUSTERS.items()},
    }, "analysis/i2_tdr.py")
    print("raw axis cosines\n", np.round(unit_raw @ unit_raw.T, 2))
    print("cos with readout", np.round(unit_raw @ r, 2), "| lobe axes cos", round(float(lobe_unit["R lobe (β_v + β_vc)"] @ lobe_unit["L lobe (β_v − β_vc)"]), 2))
    print(f"TDR subspace variance share {tdr_var / total_var:.3f} vs top-{len(TERMS)} PCs {pc_var / total_var:.3f}; R2 {task_r2:.3f}")


if __name__ == "__main__":
    main()
