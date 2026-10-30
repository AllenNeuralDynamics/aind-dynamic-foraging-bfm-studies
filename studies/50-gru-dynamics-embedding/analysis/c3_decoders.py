"""C3: are the RL models' latent variables in the GRU's hidden state, and where?

    python analysis/c3_decoders.py

Same method as B7 (cross-validated ridge from all 128 units, folds grouped by session,
location by activation pattern) for the Q-learning baselines' Q values (Bari, Hattori;
CTT exposes none) and every baseline's log-odds. The converse — how much of the GRU's own
log-odds a linear read of the RL variables explains — is reported alongside.
Writes ``fig_C3_q_decoders.png`` and ``c3_decoders.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
from b7_decoders import ALPHA, N_PCS_SHOWN, N_TRIALS, cv_r2  # noqa: E402

RL = ("bari", "ctt", "hattori")
Q_MODELS = ("bari", "hattori")


def targets() -> pd.DataFrame:
    out = {}
    for name in RL:
        rl = pd.read_parquet(s12.CACHE / f"rl_trials_{name}_{s12.MODEL_KEY}.parquet")
        p = np.clip(rl["p_rl_right"].to_numpy(), 1e-6, 1 - 1e-6)
        out[f"{name} logit(right)"] = np.log(p / (1 - p))
        if name in Q_MODELS:
            out[f"{name} Q_L"] = rl["q_rl_left"].to_numpy()
            out[f"{name} Q_R"] = rl["q_rl_right"].to_numpy()
            out[f"{name} Q_R − Q_L"] = out[f"{name} Q_R"] - out[f"{name} Q_L"]
            out[f"{name} Q_R + Q_L"] = out[f"{name} Q_R"] + out[f"{name} Q_L"]
    return pd.DataFrame(out)


def main() -> None:
    s12.apply_presentation_style()
    d = s12.load_trials(["ses_idx", "logit_diff"])
    y_all = targets()
    sub = s12.subsample_rows(len(d), N_TRIALS, seed=1)  # same trials as B7
    pca = s12.load_pca()
    x = s12.load_trials(s12.STATE_COLUMNS).to_numpy(dtype=np.float32)[sub].astype(np.float64) - pca["mean"]
    groups = d["ses_idx"].to_numpy()[sub]
    eigvals, components = np.clip(pca["eigvals"], 0, None), pca["components"]

    rows = []
    for name in y_all.columns:
        y = y_all[name].to_numpy(dtype=float)[sub]
        w = Ridge(alpha=ALPHA).fit(x, y).coef_
        per_pc = eigvals * (components @ w) ** 2
        share = per_pc / per_pc.sum()
        rows.append({"variable": name, "cv_r2": cv_r2(x, y, groups),
                     "signal_share_by_pc": share[:N_PCS_SHOWN].round(4).tolist(),
                     "signal_share_rest": float(share[N_PCS_SHOWN:].sum())})

    # Converse: the GRU log-odds from the RL variables (all of them, then per model).
    gru = d["logit_diff"].to_numpy(dtype=float)[sub]
    converse = {}
    for label, cols in [("all RL variables", list(y_all.columns))] + [
            (name, [c for c in y_all.columns if c.startswith(name)]) for name in RL]:
        converse[label] = cv_r2(y_all[cols].to_numpy(dtype=float)[sub], gru, groups)

    names = [r["variable"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5), gridspec_kw={"width_ratios": [1, 1.2, 0.7]})
    axes[0].barh(range(len(names)), [r["cv_r2"] for r in rows], color="#4C72B0")
    axes[0].set_yticks(range(len(names)), names, fontsize=11)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("cross-validated R² (from the GRU hidden state)")
    axes[0].set_title("RL latent variables decodable from the GRU?", fontsize=13)
    shares = np.array([r["signal_share_by_pc"] + [r["signal_share_rest"]] for r in rows])
    im = axes[1].imshow(shares, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    axes[1].set_xticks(range(N_PCS_SHOWN + 1), [f"PC{i + 1}" for i in range(N_PCS_SHOWN)] + ["rest"], fontsize=10)
    axes[1].set_yticks(range(len(names)), [""] * len(names))
    axes[1].set_title("Share of the decoded signal carried by each PC", fontsize=13)
    fig.colorbar(im, ax=axes[1], fraction=0.04)
    labels = list(converse)
    axes[2].barh(range(len(labels)), [converse[k] for k in labels], color="#E69F00")
    axes[2].set_yticks(range(len(labels)), labels, fontsize=11)
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("cross-validated R²")
    axes[2].set_title("GRU logit(right) from RL variables", fontsize=13)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_C3_q_decoders.png"))
    plt.close(fig)

    s12.write_json("c3_decoders.json", {
        "n_trials": int(len(sub)), "subsample_seed": 1, "ridge_alpha": ALPHA, "cv": "5-fold GroupKFold by session",
        "location": "activation pattern Σw; per-PC share of decoded-signal variance",
        "decoders": rows, "gru_logit_from_rl_cv_r2": converse,
    }, "analysis/c3_decoders.py", extra_groups=s12.rl_groups())
    for r in rows:
        print(f"{r['variable']:24s} R2={r['cv_r2']:.3f} PC1={r['signal_share_by_pc'][0]:.2f} PC2={r['signal_share_by_pc'][1]:.2f}")
    print("converse", {k: round(v, 3) for k, v in converse.items()})


if __name__ == "__main__":
    main()
