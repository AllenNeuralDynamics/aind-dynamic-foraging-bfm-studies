"""I2: targeted dimensionality reduction (TDR) of C3's state, with ignored trials as regressors.

    python analysis/i2_tdr.py

Study 50's design (Mante et al. 2013 style) on a fixed subsample of steps with a previous
step, extended for the third action:
    choice c = ±1 (previous step R / L; 0 if it was ignored), reward ρ = ±1 (rewarded / not;
    0 if ignored), c·ρ, value v (side readout logit_diff, standardised), v·c,
    ignored ι = 1 if the previous step was ignored, engage e (engage readout, standardised),
    nuisances: 4-d session context, trial position (fraction of the session).
Axes orthogonalised in the order [v, c, ρ, v·c, c·ρ, ι, e]; states projected onto them.
Also refits Study 50's design exactly (first five terms, steps after an engaged step only)
to ask whether the value axis is the same with and without ignored trials in the fit.
Writes ``_cache/i2_tdr_axes_C3-s0.npz`` (for I3), ``i2_tdr.json`` and ``fig_I2_tdr.png``.
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
import _pk02 as s13  # noqa: E402

N_STEPS = 1_000_000
SEED = 3
TERMS = ["value", "choice", "reward", "value×choice", "choice×reward", "ignored", "engage"]
STUDY12_TERMS = TERMS[:5]
COLORS = {"R+": "#0072B2", "R−": "#56B4E9", "L+": "#D55E00", "L−": "#E69F00", "ignore": s13.IGNORE_COLOR}


def regressors(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby("ses_idx", sort=False)
    prev = g["action"].shift(1)
    ignored = (prev == s13.IGNORE).astype(float).where(prev.notna())
    c = np.where(prev == s13.IGNORE, 0.0, 2.0 * prev - 1.0)
    c = pd.Series(c, index=t.index).where(prev.notna())
    rho = np.where(prev == s13.IGNORE, 0.0, np.where(g["reward"].shift(1) > 0, 1.0, -1.0))
    rho = pd.Series(rho, index=t.index).where(prev.notna())
    v = (t["logit_diff"] - t["logit_diff"].mean()) / t["logit_diff"].std()
    e = (t["engage_logit"] - t["engage_logit"].mean()) / t["engage_logit"].std()
    return pd.DataFrame({"value": v, "choice": c, "reward": rho, "value×choice": v * c, "choice×reward": c * rho,
                         "ignored": ignored, "engage": e})


def fit(h: np.ndarray, x: np.ndarray, nuisance: np.ndarray, n_terms: int):
    design = np.column_stack([x, nuisance, np.ones(len(x))])
    beta, *_ = np.linalg.lstsq(design, h, rcond=None)
    raw = beta[:n_terms]
    q, _ = np.linalg.qr(raw.T)
    q = q * np.sign(np.sum(q * raw.T, axis=0))
    r2 = float(1 - np.sum((h - design @ beta) ** 2) / np.sum((h - h.mean(0)) ** 2))
    return raw, q, r2


def main() -> None:
    s13.apply_presentation_style()
    t = s13.load_trials(["ses_idx", "trial_idx", "action", "reward", "logit_diff",
                         "choice_logit_0", "choice_logit_1", "choice_logit_2"])
    x_task = regressors(t)
    ctx = np.load(s13.CACHE / f"contexts_{s13.MODEL_KEY}.npz")
    ctx_rows = pd.Series(np.arange(len(ctx["session_ids"])), index=ctx["session_ids"]).reindex(t["ses_idx"]).to_numpy()
    position = (t["trial_idx"] / t.groupby("ses_idx", sort=False)["trial_idx"].transform("max")).to_numpy()
    nuisance = np.column_stack([ctx["contexts"][ctx_rows], position])
    has_prev = x_task["choice"].notna().to_numpy()
    rng = np.random.default_rng(SEED)
    rows = np.sort(rng.choice(np.flatnonzero(has_prev), size=N_STEPS, replace=False))
    prev_engaged = has_prev & (x_task["ignored"].to_numpy() == 0)
    rows12 = np.sort(rng.choice(np.flatnonzero(prev_engaged), size=N_STEPS, replace=False))

    pca = s13.load_pca()
    states = s13.load_trials(s13.STATE_COLUMNS).to_numpy(dtype=np.float32)
    h = states[rows].astype(np.float64) - pca["mean"]
    raw, q, r2 = fit(h, x_task.to_numpy()[rows], nuisance[rows], len(TERMS))
    h12 = states[rows12].astype(np.float64) - pca["mean"]
    raw12, q12, r2_12 = fit(h12, x_task[STUDY12_TERMS].to_numpy()[rows12], nuisance[rows12], len(STUDY12_TERMS))
    del states, h12
    proj = h @ q

    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    unit12 = raw12 / np.linalg.norm(raw12, axis=1, keepdims=True)
    directions = {k: v / np.linalg.norm(v) for k, v in s13.readout_directions().items()}
    lobe = {"R lobe (β_v + β_vc)": raw[0] + raw[3], "L lobe (β_v − β_vc)": raw[0] - raw[3]}
    lobe_unit = {k: v / np.linalg.norm(v) for k, v in lobe.items()}
    np.savez(s13.CACHE / f"i2_tdr_axes_{s13.MODEL_KEY}.npz", basis=q, raw_axes=raw, terms=np.array(TERMS),
             basis_study12_design=q12, raw_axes_study12_design=raw12)

    prev_label = np.where(x_task["ignored"].to_numpy()[rows] == 1, "ignore",
                          np.where(x_task["choice"].to_numpy()[rows] > 0, "R", "L").astype(object)
                          + np.where(x_task["reward"].to_numpy()[rows] > 0, "+", "−"))
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.6))
    sub = np.zeros(len(rows), bool)
    sub[rng.permutation(len(rows))[:150_000]] = True
    i_v, i_c, i_rho, i_ign, i_e = (TERMS.index(n) for n in ("value", "choice", "reward", "ignored", "engage"))
    for name, color in COLORS.items():
        m = sub & (prev_label == name)
        for ax, (xi, yi) in zip(axes, ((i_c, i_v), (i_ign, i_v), (i_e, i_v))):
            ax.scatter(proj[m, xi], proj[m, yi], s=1.5, alpha=0.3, color=color, linewidths=0, rasterized=True)
            ax.scatter([], [], s=30, color=color, label=f"previous {name}")
    for ax, (xl, title) in zip(axes, (("choice axis", "value vs previous choice"),
                                      ("ignored axis", "value vs previous ignored"),
                                      ("engage axis", "value vs engagement"))):
        ax.set(xlabel=xl, ylabel="value axis", title=title)
    axes[0].legend(frameon=False, fontsize=10)
    fig.suptitle(f"C3 TDR projection ({sub.sum():,} of {len(rows):,} fitted steps); axes orthogonalised in the order "
                 f"{', '.join(TERMS)}", fontsize=12)
    fig.tight_layout()
    fig.savefig(s13.figure_path("fig_I2_tdr.png"))
    plt.close(fig)

    s13.write_json("i2_tdr.json", {
        "n_steps": int(len(rows)), "seed": SEED, "terms": TERMS,
        "coding": "choice ±1 and reward ±1 after an engaged step, both 0 after an ignored one; value, engage standardised",
        "nuisances": ["context e0..e3", "trial position (fraction of session)"],
        "raw_axis_cosines": unit @ unit.T,
        "raw_axis_cosine_with_side_readout": unit @ directions["side"],
        "raw_axis_cosine_with_engage_direction": unit @ directions["engage"],
        "lobe_axes_cosine": float(lobe_unit["R lobe (β_v + β_vc)"] @ lobe_unit["L lobe (β_v − β_vc)"]),
        "task_plus_nuisance_r2": r2,
        "study12_design": {"terms": STUDY12_TERMS, "rows": "steps after an engaged step only", "r2": r2_12,
                           "value_axis_cosine_with_full_fit": float(unit12[0] @ unit[0]),
                           "orthonormal_value_axis_cosine_with_full_fit": float(q12[:, 0] @ q[:, 0])},
        "cluster_value_mean_sd": {n: {"mean": float(proj[prev_label == n, i_v].mean()), "sd": float(proj[prev_label == n, i_v].std())}
                                  for n in COLORS},
    }, "analysis/i2_tdr.py")
    print("cos raw axes\n", np.round(unit @ unit.T, 2))
    print("value axis: full fit vs Study 50 design cos", round(float(unit12[0] @ unit[0]), 3),
          "| orthonormal", round(float(q12[:, 0] @ q[:, 0]), 3), "| R2", round(r2, 3), round(r2_12, 3))


if __name__ == "__main__":
    main()
