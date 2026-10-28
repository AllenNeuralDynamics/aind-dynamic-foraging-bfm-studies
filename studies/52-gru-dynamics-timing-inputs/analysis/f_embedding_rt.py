"""F-RT: does the subject embedding carry each mouse's typical RT, and does adding RT change it?

    python analysis/f_embedding_rt.py

The 4-d subject embeddings (best_eval checkpoint) of C-s0 (Study 50), R-s0 and R-s1, all over
the same 614 training mice. Per mouse (≥ 500 RT trials in RT-present eval sessions): median
and sd of log RT. Each is decoded from a model's embedding by linear regression, 5-fold CV
across mice, scored as rank R² against a 200-permutation shuffle null (Study 50 F3's method),
raw and "beyond task mix" (the feature after regressing out the mouse's per-task trial
fractions). C never sees RT, so what C's embedding carries about RT comes from behaviour
that covaries with it.

Alignment: how well each model's embedding linearly predicts another's (5-fold CV R², pooled
over the 4 dimensions). R-s0 ↔ R-s1 is the same-model reference.

Writes ``fig_F_RT_embedding.png`` and ``f_embedding_rt.json``.
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
from _pk03 import CACHE, RT_CLIP_S, STUDY12, apply_presentation_style, figure_path, write_json  # noqa: E402
import _pk03  # noqa: E402

PRODUCER = "analysis/f_embedding_rt.py"
MODELS = {"C": STUDY12 / "analysis" / "_cache" / "models" / "C-s0", "R-s0": CACHE / "models" / "R-s0",
          "R-s1": CACHE / "models" / "R-s1"}
MIN_TRIALS = 500
N_PERM = 200
SEED = 0
TASKS = ["Uncoupled Baiting", "Coupled Baiting", "Uncoupled Without Baiting", "Coupled Without Baiting"]


def embedding(root: Path) -> pd.DataFrame:
    outputs = root / "run" / "outputs"
    index_map = json.loads((outputs / "subject_index_map.json").read_text())["subject_id_to_index"]
    index = json.loads((outputs / "checkpoints" / "index.json").read_text())
    best = max(index["checkpoints"], key=lambda c: c["eval_likelihood"])
    params = json.loads((outputs / "checkpoints" / f"step_{best['step']}" / "params.json").read_text())
    table = np.asarray(params["multisubject_gru"]["subject_embeddings"])
    subjects = sorted(index_map, key=index_map.get)
    return pd.DataFrame(table[[index_map[s] for s in subjects]], index=pd.Index(subjects, name="subject_id"),
                        columns=[f"e{i}" for i in range(table.shape[1])])


def mouse_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    t = pd.read_parquet(CACHE / "trial_states_R-s0.parquet", columns=["subject_id", "ses_idx", "task", "reaction_time"])
    present = t["reaction_time"].notna().groupby(t["ses_idx"]).transform("all")
    log_rt = np.log(t.loc[present, "reaction_time"].clip(*RT_CLIP_S))
    g = log_rt.groupby(t.loc[present, "subject_id"])
    feats = pd.DataFrame({"median log RT": g.median(), "sd log RT": g.std(), "n": g.size()})
    feats = feats[feats["n"] >= MIN_TRIALS].drop(columns="n")
    mix = pd.crosstab(t["subject_id"], t["task"], normalize="index").reindex(columns=TASKS, fill_value=0.0)
    return feats, mix


def rank_r2(x: np.ndarray, y: np.ndarray, seed: int = SEED) -> float:
    ry = rankdata(y)
    pred = cross_val_predict(LinearRegression(), x, ry, cv=KFold(5, shuffle=True, random_state=seed))
    return float(1 - np.sum((ry - pred) ** 2) / np.sum((ry - ry.mean()) ** 2))


def decode(x, y) -> dict:
    rng = np.random.default_rng(SEED)
    null = np.array([rank_r2(x, rng.permutation(y)) for _ in range(N_PERM)])
    r2 = rank_r2(x, y)
    return {"rank_r2": r2, "null_95": float(np.quantile(null, 0.95)), "p": float((np.sum(null >= r2) + 1) / (N_PERM + 1))}


def cv_r2(a: np.ndarray, b: np.ndarray) -> float:
    pred = cross_val_predict(LinearRegression(), a, b, cv=KFold(5, shuffle=True, random_state=SEED))
    return float(1 - np.sum((b - pred) ** 2) / np.sum((b - b.mean(0)) ** 2))


def main() -> None:
    _pk03.use_keys("C-s0", "R-s0", "R-s1")
    apply_presentation_style()
    emb = {name: embedding(root) for name, root in MODELS.items()}
    feats, mix = mouse_features()
    mice = feats.index.intersection(emb["C"].index)
    decoding = {}
    for name, e in emb.items():
        x = e.loc[mice].to_numpy()
        decoding[name] = {}
        for feat in feats.columns:
            y = feats.loc[mice, feat].to_numpy()
            m = mix.loc[mice].to_numpy()
            resid = y - LinearRegression().fit(m, y).predict(m)
            decoding[name][feat] = {"raw": decode(x, y), "beyond_task_mix": decode(x, resid)}
    names = list(emb)
    align = {a: {b: cv_r2(emb[a].to_numpy(), emb[b].to_numpy()) for b in names if b != a} for a in names}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.6, 1]})
    colors = {"C": "#888888", "R-s0": "#4C72B0", "R-s1": "#8DA9D6"}
    groups = [(f, k) for f in feats.columns for k in ("raw", "beyond_task_mix")]
    width = 0.26
    for i, name in enumerate(names):
        vals = [decoding[name][f][k]["rank_r2"] for f, k in groups]
        nulls = [decoding[name][f][k]["null_95"] for f, k in groups]
        xs = np.arange(len(groups)) + (i - 1) * width
        axes[0].bar(xs, vals, width=width, color=colors[name], label=name)
        axes[0].scatter(xs, nulls, marker="_", s=200, color="#222222", zorder=3, label="shuffle null, 95th pct" if i == 0 else None)
    axes[0].set_xticks(np.arange(len(groups)), [f"{f}\n{k.replace('_', ' ')}" for f, k in groups], fontsize=12)
    axes[0].axhline(0, color="#888888", lw=1)
    axes[0].set_ylabel("cross-validated rank R²\n(embedding → mouse feature)")
    axes[0].set_title(f"Does the embedding carry the mouse's RT? ({len(mice)} mice)")
    axes[0].legend(frameon=False, fontsize=11)
    mat = np.array([[np.nan if a == b else align[a][b] for b in names] for a in names])
    axes[1].imshow(mat, cmap="Greys", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                axes[1].text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=14,
                             color="white" if mat[i, j] > 0.6 else "#222222")
    axes[1].set_xticks(range(len(names)), names)
    axes[1].set_yticks(range(len(names)), names)
    axes[1].set_xlabel("predicted embedding")
    axes[1].set_ylabel("from embedding")
    axes[1].set_title("linear alignment between embeddings (CV R²)")
    fig.suptitle("F-RT  Subject embeddings: mouse-level RT and how R's embedding relates to C's", y=1.03)
    fig.savefig(figure_path("fig_F_RT_embedding.png"), bbox_inches="tight")
    plt.close(fig)
    write_json("f_embedding_rt.json", {"n_mice": int(len(mice)), "decoding": decoding, "alignment_cv_r2": align}, PRODUCER)
    print(json.dumps({n: {f: {k: round(v["rank_r2"], 3) for k, v in d.items()} for f, d in dd.items()} for n, dd in decoding.items()}, indent=1))
    print(json.dumps(align, indent=1))


if __name__ == "__main__":
    main()
