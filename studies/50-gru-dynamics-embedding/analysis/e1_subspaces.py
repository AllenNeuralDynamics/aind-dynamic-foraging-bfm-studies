"""E1 (without slow modes): the geometry of the GRU state in its full 128 dimensions.

    python analysis/e1_subspaces.py

Axes compared (all unit vectors in unit space):
  * readout   — w_right − w_left of the linear readout: the only direction that moves the
                current choice; its orthogonal complement is the readout-null space;
  * input     — mean state change h(t+1) − h(t) after each choice x reward of trial t,
                combined into choice, reward and choice×reward directions;
  * variables — activation patterns Σw of ridge decoders (B7 method) for key variables;
  * PCs 1–6   — from B3.
For each: share of state variance along it and cos² with the readout — the part of a
variable's signal direction that moves the current choice; the rest lies in the
readout-null space. (Decoding with the readout direction projected out is not a useful
test: the other 127 dimensions carry every variable redundantly, the previous choice
included, so a static one-dimensional removal changes nothing. Whether the null space
matters is a dynamic question — E2.) Slow modes (D3) join later.
Reads cached inputs. Writes ``fig_E1_subspaces.png`` and ``e1_subspaces.json``.
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
from b7_decoders import ALPHA, N_TRIALS, targets  # noqa: E402

N_PCS = 6
VARIABLES = ["choice 1 back", "reward 1 back", "choice×reward 1 back", "reward 2 back", "reward 3 back",
             "choice 3 back", "P(rew|R) − P(rew|L)", "hattori Q_R − Q_L", "hattori Q_R + Q_L", "trial position"]


def main() -> None:
    s12.apply_presentation_style()
    d = s12.load_trials(["ses_idx", "trial_idx", "action", "reward", "logit_diff", "trial_position",
                         "reward_probabilityL", "reward_probabilityR"])
    states = s12.load_trials(s12.STATE_COLUMNS).to_numpy(dtype=np.float32)
    pca = s12.load_pca()
    mean, components, eigvals = pca["mean"], pca["components"], np.clip(pca["eigvals"], 0, None)
    cov = (components.T * eigvals) @ components
    total_var = float(np.trace(cov))

    w = np.asarray(s12.best_params()["multisubject_gru/~/readout"]["w"])
    readout = (w[:, 1] - w[:, 0]) / np.linalg.norm(w[:, 1] - w[:, 0])

    # Input-driven directions: mean Δh after each choice x reward of trial t (within session).
    nxt = np.r_[np.arange(1, len(d)), -1]
    same = np.r_[d["ses_idx"].to_numpy()[1:] == d["ses_idx"].to_numpy()[:-1], False]
    delta = states[nxt[same]].astype(np.float64) - states[same]
    c, r = d["action"].to_numpy()[same], (d["reward"].to_numpy()[same] > 0)
    mu = {(ci, ri): delta[(c == ci) & (r == ri)].mean(axis=0) for ci in (0, 1) for ri in (False, True)}
    input_dirs = {"input: choice (R − L)": (mu[1, True] + mu[1, False] - mu[0, True] - mu[0, False]) / 2,
                  "input: reward (+ − −)": (mu[1, True] + mu[0, True] - mu[1, False] - mu[0, False]) / 2,
                  "input: choice × reward": (mu[1, True] - mu[1, False]) - (mu[0, True] - mu[0, False])}

    y_all = targets(d)
    hattori = pd.read_parquet(s12.CACHE / f"rl_trials_hattori_{s12.MODEL_KEY}.parquet")
    y_all["hattori Q_R − Q_L"] = (hattori["q_rl_right"] - hattori["q_rl_left"]).to_numpy()
    y_all["hattori Q_R + Q_L"] = (hattori["q_rl_right"] + hattori["q_rl_left"]).to_numpy()
    y_all["trial position"] = d["trial_position"].to_numpy()
    sub = s12.subsample_rows(len(d), N_TRIALS, seed=1)
    x = states[sub].astype(np.float64) - mean
    patterns = {}
    for name in VARIABLES:
        y = y_all[name].to_numpy(dtype=float)[sub]
        ok = np.isfinite(y)
        patterns[f"var: {name}"] = cov @ Ridge(alpha=ALPHA).fit(x[ok], y[ok]).coef_

    axes_dict = {"readout (R − L)": readout, **input_dirs, **patterns,
                 **{f"PC{i + 1}": components[i] for i in range(N_PCS)}}
    names = list(axes_dict)
    unit = np.array([v / np.linalg.norm(v) for v in axes_dict.values()])
    cosine = unit @ unit.T
    variance_share = {n: float(u @ cov @ u / total_var) for n, u in zip(names, unit)}
    readout_share = variance_share["readout (R − L)"]

    fig, axes = plt.subplots(1, 2, figsize=(24, 11), gridspec_kw={"width_ratios": [1.35, 1]})
    im = axes[0].imshow(np.abs(cosine), cmap="Greys", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(names)), names, rotation=70, ha="right", fontsize=10)
    axes[0].set_yticks(range(len(names)), names, fontsize=10)
    for i in range(len(names)):
        for j in range(len(names)):
            axes[0].text(j, i, f"{cosine[i, j]:+.2f}".replace("+0.", ".").replace("-0.", "-."), ha="center", va="center",
                         fontsize=6.5, color="white" if abs(cosine[i, j]) > 0.6 else "black")
    axes[0].set_title("|cosine| between axes (signed value printed)", fontsize=13)
    fig.colorbar(im, ax=axes[0], fraction=0.035)
    labels = [n for n in names if not n.startswith("readout")]
    yy = np.arange(len(labels))
    potent = [cosine[0, names.index(n)] ** 2 for n in labels]
    axes[1].barh(yy, potent, color="#E69F00", label="cos² with readout (drives the current choice)")
    axes[1].barh(yy, [variance_share[n] for n in labels], height=0.3, color="#4C72B0", label="share of state variance")
    axes[1].set_yticks(yy, labels, fontsize=11)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_title(f"Choice-driving vs held (readout carries {100 * readout_share:.1f}% of state variance)", fontsize=13)
    axes[1].legend(frameon=False, fontsize=11, loc="lower right")
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_E1_subspaces.png"))
    plt.close(fig)

    s12.write_json("e1_subspaces.json", {
        "axes": names, "cosine": cosine, "variance_share": variance_share,
        "readout_variance_share": readout_share, "readout_null_variance_share": 1 - readout_share,
        "cosine_with_readout": {n: float(cosine[0, i]) for i, n in enumerate(names)},
        "readout_cos2": {n: float(cosine[0, i] ** 2) for i, n in enumerate(names)},
        "n_trials_decoding": int(len(sub)), "n_transitions_input_dirs": int(same.sum()),
    }, "analysis/e1_subspaces.py", extra_groups=s12.rl_groups())
    print(f"readout variance share {readout_share:.4f}")
    for n in names:
        print(f"{n:32s} var_share={variance_share[n]:.3f} cos_readout={cosine[0, names.index(n)]:+.3f}")


if __name__ == "__main__":
    main()
