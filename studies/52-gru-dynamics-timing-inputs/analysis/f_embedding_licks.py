"""F-L: does the subject embedding carry each mouse's typical licking, and does adding licks change it?

    python analysis/f_embedding_licks.py

Same method as ``f_embedding_rt.py`` (CV rank R² from the 4-d embedding against a 200-shuffle
null, raw and beyond task mix; CV linear alignment between embeddings), for C-s0, L-s0 and L-s1.
Mouse features (all eval sessions with licks): mean log(1 + other-side licks), and mean
log(1 + chosen-side licks) after rewarded and after unrewarded trials (consumption vigour and
non-rewarded licking are separate). Mice with ≥ 500 trials, as in ``f_embedding_rt.py``. Writes ``fig_F_L_embedding.png`` and ``f_embedding_licks.json``.
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
from sklearn.linear_model import LinearRegression  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f_embedding_rt import TASKS, cv_r2, decode, embedding  # noqa: E402
from _pk03 import CACHE, STUDY12, apply_presentation_style, figure_path, write_json  # noqa: E402
import _pk03  # noqa: E402

PRODUCER = "analysis/f_embedding_licks.py"
MODELS = {"C": STUDY12 / "analysis" / "_cache" / "models" / "C-s0", "L-s0": CACHE / "models" / "L-s0",
          "L-s1": CACHE / "models" / "L-s1"}


def mouse_features():
    t = pd.read_parquet(CACHE / "trial_states_L-s0.parquet",
                        columns=["subject_id", "ses_idx", "task", "action", "reward", "lick_left_count", "lick_right_count"])
    t = t[(t["lick_left_count"] + t["lick_right_count"]).groupby(t["ses_idx"]).transform("sum") > 0]
    right = t["action"] == 1
    chosen = np.log1p(np.where(right, t["lick_right_count"], t["lick_left_count"]))
    other = np.log1p(np.where(right, t["lick_left_count"], t["lick_right_count"]))
    d = pd.DataFrame({"subject_id": t["subject_id"], "reward": t["reward"], "chosen": chosen, "other": other})
    feats = pd.DataFrame({
        "other-side licks": d.groupby("subject_id")["other"].mean(),
        "chosen licks after win": d[d["reward"] == 1].groupby("subject_id")["chosen"].mean(),
        "chosen licks after loss": d[d["reward"] == 0].groupby("subject_id")["chosen"].mean(),
        "n": d.groupby("subject_id").size(),
    }).dropna()
    feats = feats[feats["n"] >= 500].drop(columns="n")
    mix = pd.crosstab(t["subject_id"], t["task"], normalize="index").reindex(columns=TASKS, fill_value=0.0)
    return feats, mix


def main() -> None:
    _pk03.use_keys("C-s0", "L-s0", "L-s1")
    apply_presentation_style()
    emb = {n: embedding(r) for n, r in MODELS.items()}
    feats, mix = mouse_features()
    mice = feats.index.intersection(emb["C"].index)
    decoding = {}
    for name, e in emb.items():
        x = e.loc[mice].to_numpy()
        decoding[name] = {}
        for f in feats.columns:
            y = feats.loc[mice, f].to_numpy()
            m = mix.loc[mice].to_numpy()
            resid = y - LinearRegression().fit(m, y).predict(m)
            decoding[name][f] = {"raw": decode(x, y), "beyond_task_mix": decode(x, resid)}
    names = list(emb)
    align = {a: {b: cv_r2(emb[a].to_numpy(), emb[b].to_numpy()) for b in names if b != a} for a in names}
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.4), gridspec_kw={"width_ratios": [1.9, 1]})
    colors = {"C": "#888888", "L-s0": "#4C72B0", "L-s1": "#8DA9D6"}
    groups = [(f, k) for f in feats.columns for k in ("raw", "beyond_task_mix")]
    width = 0.26
    for i, name in enumerate(names):
        xs = np.arange(len(groups)) + (i - 1) * width
        axes[0].bar(xs, [decoding[name][f][k]["rank_r2"] for f, k in groups], width=width, color=colors[name], label=name)
        axes[0].scatter(xs, [decoding[name][f][k]["null_95"] for f, k in groups], marker="_", s=200, color="#222222",
                        zorder=3, label="shuffle null, 95th pct" if i == 0 else None)
    short = {"other-side licks": "other side", "chosen licks after win": "chosen, win", "chosen licks after loss": "chosen, loss"}
    axes[0].set_xticks(np.arange(len(groups)), [f"{short[f]}\n{'raw' if k == 'raw' else 'beyond mix'}" for f, k in groups], fontsize=11)
    axes[0].axhline(0, color="#888888", lw=1)
    axes[0].set_ylabel("cross-validated rank R²\n(embedding → mouse feature)")
    axes[0].set_title(f"Does the embedding carry the mouse's licking? ({len(mice)} mice)")
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
    fig.suptitle("F-L  Subject embeddings: mouse-level licking and how L's embedding relates to C's", y=1.03)
    fig.savefig(figure_path("fig_F_L_embedding.png"), bbox_inches="tight")
    plt.close(fig)
    write_json("f_embedding_licks.json", {"n_mice": int(len(mice)), "decoding": decoding, "alignment_cv_r2": align}, PRODUCER)
    for n, dd in decoding.items():
        print(n, {f: {k: (round(v["rank_r2"], 3), round(v["p"], 3)) for k, v in d.items()} for f, d in dd.items()})
    print(json.dumps(align, indent=1))


if __name__ == "__main__":
    main()
