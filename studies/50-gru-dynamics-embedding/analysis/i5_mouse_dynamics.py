"""I5: which parameter of the distilled one-variable model carries each behavioural axis?

    python analysis/i5_mouse_dynamics.py

From I4's per-mouse fits (``_cache/i4_mouse_params.npz``), per mouse:
  λ stay / λ switch      — mean retention over the four stay / switch groups;
  reward push            — rewarded − unrewarded stay target, averaged over sides;
  choice push            — how far an unrewarded stay pulls toward the chosen side;
  target asymmetry       — mean of the four stay targets (> 0: rightward);
  switch target          — where a switch sends v, signed toward the new side;
  readout slope a, readout offset — mean b over the four last-input clusters.
Compared with F3's per-mouse GRU kernel features (bias, reward gain, persistence, decay) by
Spearman correlation across mice, and decoded from the subject embedding with F3's method
(rank R², 5-fold CV, shuffle null). Writes ``i5_mouse_dynamics.json`` and
``fig_I5_mouse_dynamics.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
import _transitions as tr  # noqa: E402
from f3_embedding_features import decode, kernel_features  # noqa: E402

SEED = 0
F3_FEATURES = ["bias", "reward gain (1 back)", "persistence (1 back)", "reward decay (2/1 back)"]


def group(name: str, kind: str) -> int:
    return tr.GROUP_NAMES.index(f"{name} ({kind})")


def dynamics_features(lam: np.ndarray, target: np.ndarray, a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    g = {f"{n}|{k}": group(n, k) for n in tr.INPUT_NAMES for k in ("stay", "switch")}
    stay = [g[f"{n}|stay"] for n in tr.INPUT_NAMES]
    switch = [g[f"{n}|switch"] for n in tr.INPUT_NAMES]
    t = lambda n, k: target[:, g[f"{n}|{k}"]]  # noqa: E731
    return pd.DataFrame({
        "λ stay": lam[:, stay].mean(axis=1),
        "λ switch": lam[:, switch].mean(axis=1),
        "reward push": ((t("R rewarded", "stay") - t("R unrewarded", "stay"))
                        + (t("L unrewarded", "stay") - t("L rewarded", "stay"))) / 2,
        "choice push": (t("R unrewarded", "stay") - t("L unrewarded", "stay")) / 2,
        "target asymmetry": target[:, stay].mean(axis=1),
        "switch target": (t("R rewarded", "switch") + t("R unrewarded", "switch")
                          - t("L rewarded", "switch") - t("L unrewarded", "switch")) / 4,
        "readout slope a": a,
        "readout offset": b[:, :4].mean(axis=1),
    })


def main() -> None:
    s12.apply_presentation_style()
    p = np.load(s12.keyed_cache("i4_mouse_params.npz"), allow_pickle=False)
    subjects = [str(s) for s in p["subjects"]]
    dyn = dynamics_features(p["lam"], p["target"], p["a"], p["b"]).set_index(pd.Index(subjects, name="subject_id"))
    kernels = json.loads((s12.ANALYSIS / "c1_kernels.json").read_text())
    beh = kernel_features(kernels["per_mouse_coef"]["gru"], kernels["n_back"])[F3_FEATURES]
    emb = s12.subject_embeddings()
    mice = dyn.index.intersection(beh.index).intersection(emb.index)
    dyn, beh, x = dyn.loc[mice], beh.loc[mice], emb.loc[mice].to_numpy()

    rho = pd.DataFrame({f: [float(pd.Series(dyn[d]).corr(beh[f], method="spearman")) for d in dyn.columns]
                        for f in F3_FEATURES}, index=dyn.columns)
    rng = np.random.default_rng(SEED)
    decoding = {d: decode(x, dyn[d].to_numpy(dtype=float), rng) for d in dyn.columns}

    fig, axes = plt.subplots(1, 4, figsize=(27, 6.6), gridspec_kw={"width_ratios": [1.1, 0.9, 0.9, 0.9]})
    im = axes[0].imshow(rho.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_xticks(range(len(F3_FEATURES)), [f"GRU {f}" for f in F3_FEATURES], rotation=25, ha="right", fontsize=10)
    axes[0].set_yticks(range(len(dyn.columns)), dyn.columns, fontsize=11)
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            axes[0].text(j, i, f"{rho.iat[i, j]:+.2f}", ha="center", va="center", fontsize=10)
    axes[0].set_title("Spearman across mice: dynamics × behaviour", fontsize=13)
    fig.colorbar(im, ax=axes[0], fraction=0.04)
    pairs = [("reward gain (1 back)", ["switch target", "λ switch", "reward push"]),
             ("bias", ["target asymmetry", "readout offset"])]
    for ax, (f, ds) in zip(axes[1:3], pairs):
        for d, marker in zip(ds, ("o", "s", "^")):
            ax.scatter(rankdata(beh[f]) / len(beh), rankdata(dyn[d]) / len(dyn), s=9, alpha=0.5, marker=marker,
                       label=f"{d}: ρ {rho.at[d, f]:+.2f}")
        ax.set_xlabel(f"GRU {f} (rank across mice)")
        ax.set_ylabel("dynamics parameter (rank)")
        ax.legend(frameon=False, fontsize=10)
        ax.set_title(f"What carries {f.split(' (')[0]}?", fontsize=13)
    names = list(decoding)
    axes[3].barh(range(len(names)), [decoding[n]["rank_r2"] for n in names], color="#E69F00")
    axes[3].scatter([decoding[n]["null_95"] for n in names], range(len(names)), marker="|", s=120, color="#C44E52",
                    label="shuffle 95th pct")
    axes[3].set_yticks(range(len(names)), names, fontsize=11)
    axes[3].invert_yaxis()
    axes[3].set_xlabel("rank R² from the 4-d embedding (5-fold CV)")
    axes[3].set_title("Is each parameter set by the embedding?", fontsize=13)
    axes[3].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_I5_mouse_dynamics.png"))
    plt.close(fig)

    s12.write_json("i5_mouse_dynamics.json", {
        "n_mice": int(len(mice)), "dynamics_features": list(dyn.columns), "behaviour_features": F3_FEATURES,
        "spearman": rho.to_dict(), "embedding_decoding": decoding,
        "per_mouse_summary": {d: np.percentile(dyn[d], [10, 50, 90]).round(4).tolist() for d in dyn.columns},
    }, "analysis/i5_mouse_dynamics.py")
    print(rho.round(2).to_string())
    print({d: round(v["rank_r2"], 3) for d, v in decoding.items()})
    print({d: np.percentile(dyn[d], [10, 50, 90]).round(3).tolist() for d in dyn.columns})


if __name__ == "__main__":
    main()
