"""F3 (behavioural half): which per-mouse behaviours does the 4-d subject embedding carry, and
along which directions?

    python analysis/f3_embedding_features.py

Per mouse: features of its history kernel (C1 per-mouse Su2022 fits — for the GRU's
predictions and for the animal's choices) and the fitted RL parameters of Study 01's three
baselines. Each is decoded from the mouse's embedding (4-d, raw coordinates) by linear
regression, 5-fold cross-validated across mice, scored as rank R² (R² on rank-transformed
targets, robust to fits pinned at a bound) against a 200-permutation label-shuffle null —
Study 01 r10's method, here on model C. The fitted coefficient vector is the feature's
direction in embedding space; cosines between directions test whether, e.g., bias and
reward gain are carried along separate embedding axes.

Task-mix control: mice ran different mixes of tasks and kernels differ by task, so each
feature is also decoded from the mouse's per-task trial fractions alone, and from the
embedding after regressing those fractions out of the feature ("beyond task mix").

Reads committed/cached inputs only. Writes ``fig_F3_embedding_features.png`` and
``f3_embedding_features.json``.
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
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import KFold, cross_val_predict  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

N_PERM = 200
SEED = 0
RL = {"bari": "bg3nzqz9", "ctt": "lmg1i9yd", "hattori": "unhmbrk4"}
DIRECTIONS = ["gru bias", "gru reward gain (1 back)", "gru reward decay (2/1 back)", "gru persistence (1 back)",
              "ctt biasL", "ctt learn_rate", "bari learn_rate", "hattori learn_rate_rew", "hattori learn_rate_unrew"]


def kernel_features(coefs: dict[str, list[float]], n_back: int) -> pd.DataFrame:
    rows = {}
    for subject, c in coefs.items():
        c = np.asarray(c)
        rew, unr = (c[i * n_back:(i + 1) * n_back][::-1] for i in range(2))  # Su2022 terms, lag 1..n_back
        effect = rew - unr
        rows[subject] = {"bias": c[-1], "reward gain (1 back)": effect[0],
                         "reward effect (sum 1–5)": effect[:5].sum(),
                         "reward decay (2/1 back)": effect[1] / effect[0] if effect[0] > 0.05 else np.nan,
                         "persistence (1 back)": (rew[0] + unr[0]) / 2,
                         "rewarded choice (1 back)": rew[0], "unrewarded choice (1 back)": unr[0]}
    return pd.DataFrame.from_dict(rows, orient="index")


def rl_parameters() -> pd.DataFrame:
    frames = []
    for name, run in RL.items():
        out = json.loads((s12.CACHE / "models" / "rl" / run / "baseline_rl_output.json").read_text())
        params = pd.DataFrame({s: f["fitted_params"] for s, f in out["fitted_params_per_subject"].items()}).T
        params = params.loc[:, params.std() > 0]  # drop parameters fixed for every mouse
        frames.append(params.add_prefix(f"{name} "))
    return pd.concat(frames, axis=1)


def task_mix() -> pd.DataFrame:
    t = s12.load_trials(["subject_id", "task"])
    return pd.crosstab(t["subject_id"], t["task"].fillna("<missing>"), normalize="index")


def cv_r2(x: np.ndarray, y: np.ndarray) -> float:
    pred = cross_val_predict(LinearRegression(), x, y, cv=KFold(5, shuffle=True, random_state=SEED))
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def task_control(emb: np.ndarray, mix: np.ndarray, y: np.ndarray) -> dict:
    ok = np.isfinite(y)
    yr = rankdata(y[ok]) / ok.sum()
    residual = yr - LinearRegression().fit(mix[ok], yr).predict(mix[ok])
    return {"task_mix_only_rank_r2": cv_r2(mix[ok], yr),
            "embedding_beyond_task_mix_r2": cv_r2(emb[ok], residual)}


def decode(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    ok = np.isfinite(y)
    x, y = x[ok], rankdata(y[ok]) / ok.sum()
    folds = KFold(5, shuffle=True, random_state=SEED)

    def r2(target):
        pred = cross_val_predict(LinearRegression(), x, target, cv=folds)
        return 1 - np.sum((target - pred) ** 2) / np.sum((target - target.mean()) ** 2)

    observed = r2(y)
    null = np.array([r2(rng.permutation(y)) for _ in range(N_PERM)])
    coef = LinearRegression().fit(x, y).coef_
    return {"rank_r2": float(observed), "null_95": float(np.percentile(null, 95)),
            "p_shuffle": float((1 + np.sum(null >= observed)) / (N_PERM + 1)), "n": int(ok.sum()),
            "direction": coef / np.linalg.norm(coef)}


def main() -> None:
    s12.apply_presentation_style()
    kernels = json.loads((s12.ANALYSIS / "c1_kernels.json").read_text())
    emb = s12.subject_embeddings()
    features = pd.concat([kernel_features(kernels["per_mouse_coef"]["gru"], kernels["n_back"]).add_prefix("gru "),
                          kernel_features(kernels["per_mouse_coef"]["animal"], kernels["n_back"]).add_prefix("animal "),
                          rl_parameters()], axis=1)
    features = features.loc[emb.index.intersection(features.index)]
    x = emb.loc[features.index].to_numpy()
    rng = np.random.default_rng(SEED)
    results = {name: decode(x, features[name].to_numpy(dtype=float), rng) for name in features.columns}
    mix = task_mix().reindex(features.index).fillna(0.0)
    mix_x = mix.drop(columns=[c for c in ("Random Walk", "<missing>") if c in mix.columns]).to_numpy()
    for name in features.columns:
        results[name].update(task_control(x, mix_x, features[name].to_numpy(dtype=float)))

    # Does the GRU's per-mouse kernel follow the animal's?
    shared = [c.removeprefix("gru ") for c in features.columns if c.startswith("gru ")]
    gru_vs_animal = {f: float(pd.Series(features[f"gru {f}"]).corr(features[f"animal {f}"], method="spearman"))
                     for f in shared}
    directions = np.array([results[d]["direction"] for d in DIRECTIONS])
    cosine = directions @ directions.T
    # Coordinate-free versions: raw-coordinate cosines depend on how the (correlated, unequally
    # scaled) embedding dimensions happen to be laid out. Under the embedding covariance Σ the
    # cosine equals the correlation of the linearly predicted features; Spearman of the
    # features themselves needs no regression at all.
    sigma = np.cov(x, rowvar=False)
    gram = directions @ sigma @ directions.T
    cosine_sigma = gram / np.sqrt(np.outer(np.diag(gram), np.diag(gram)))
    feature_spearman = features[DIRECTIONS].rank().corr().to_numpy()

    names = list(features.columns)
    fig, axes = plt.subplots(1, 3, figsize=(26, 9), gridspec_kw={"width_ratios": [1.1, 1, 0.9]})
    colors = ["#E69F00" if n.startswith("gru") else "#222222" if n.startswith("animal") else "#4C72B0" for n in names]
    axes[0].barh(range(len(names)), [results[n]["rank_r2"] for n in names], color=colors)
    axes[0].scatter([results[n]["null_95"] for n in names], range(len(names)), marker="|", s=120, color="#C44E52",
                    label="shuffle 95th pct")
    axes[0].scatter([results[n]["task_mix_only_rank_r2"] for n in names], range(len(names)), marker="x", s=40,
                    color="#555555", label="task mix only")
    axes[0].scatter([results[n]["embedding_beyond_task_mix_r2"] for n in names], range(len(names)), marker="D", s=22,
                    color="white", edgecolors="#222222", label="embedding, beyond task mix")
    axes[0].set_yticks(range(len(names)), names, fontsize=10)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("cross-validated rank R² from the 4-d embedding")
    axes[0].set_title("What the subject embedding predicts (614 mice)", fontsize=13)
    axes[0].legend(frameon=False, fontsize=10)
    im = axes[1].imshow(cosine_sigma, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(DIRECTIONS)), DIRECTIONS, rotation=60, ha="right", fontsize=10)
    axes[1].set_yticks(range(len(DIRECTIONS)), DIRECTIONS, fontsize=10)
    for i in range(len(DIRECTIONS)):
        for j in range(len(DIRECTIONS)):
            axes[1].text(j, i, f"{cosine_sigma[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    axes[1].set_title("Cosine between embedding directions (Σ metric = corr. of predictions)", fontsize=13)
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    for f, marker in (("bias", "o"), ("reward gain (1 back)", "s"), ("persistence (1 back)", "^")):
        a, g = features[f"animal {f}"], features[f"gru {f}"]
        axes[2].scatter(rankdata(a) / len(a), rankdata(g) / len(g), s=10, alpha=0.5, marker=marker,
                        label=f"{f}: Spearman {gru_vs_animal[f]:.2f}")
    axes[2].plot([0, 1], [0, 1], color="0.6", ls=":", lw=1)
    axes[2].set_xlabel("animal (rank across mice)")
    axes[2].set_ylabel("GRU (rank across mice)")
    axes[2].set_title("Per-mouse kernel features: GRU vs animal", fontsize=13)
    axes[2].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_F3_embedding_features.png"))
    plt.close(fig)

    s12.write_json("f3_embedding_features.json", {
        "n_mice": int(len(features)), "embedding": "4-d subject embedding table, raw coordinates (best_eval)",
        "method": "5-fold CV (shuffled, seed 0) linear regression across mice; rank R²; 200-permutation shuffle null",
        "kernel_source": "c1_kernels.json per_mouse_coef (Su2022, 10 back; GRU soft labels)",
        "decoding": results, "direction_names": DIRECTIONS, "direction_cosine_raw": cosine,
        "direction_cosine_sigma": cosine_sigma, "feature_spearman": feature_spearman,
        "gru_vs_animal_spearman": gru_vs_animal,
        "task_mix_columns": [c for c in mix.columns if c not in ("Random Walk", "<missing>")],
    }, "analysis/f3_embedding_features.py", extra_groups=s12.rl_groups())
    for n in names:
        r = results[n]
        print(f"{n:42s} rankR2={r['rank_r2']:+.3f} task-only={r['task_mix_only_rank_r2']:+.3f} "
              f"beyond-task={r['embedding_beyond_task_mix_r2']:+.3f} p={r['p_shuffle']:.3f}")
    print("gru vs animal", {k: round(v, 3) for k, v in gru_vs_animal.items()})


if __name__ == "__main__":
    main()
