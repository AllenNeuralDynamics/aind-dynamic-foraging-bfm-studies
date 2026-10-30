"""F5: what the subject embedding encodes — an atlas, its directions, and batch effects.

    python analysis/f5_embedding_atlas.py

One row per mouse (614), aggregated over the mouse's eval sessions:
  RL parameters      Hattori, Bari, CTT per-mouse fits (F3's ``rl_parameters``);
  GRU / animal       F3's kernel features (bias, reward gain, persistence, decay);
  dynamics           F2 (attractor asymmetry / depth, v retention), I5 (λ stay, switch target);
  model fit          GRU likelihood and its per-trial advantage over CTT (nats);
  metadata           from the pinned snapshot's session table (cached on first run):
                     categorical (mode over sessions) institute, data source, rig type, room,
                     trainer, PI, curriculum, first-session year; continuous (mean) foraging
                     efficiency, reaction time, base weight, daily water, reward contrast, block
                     length, finished / ignore rate, trials and run time per session, number of
                     eval sessions, task mix.
Analyses:
  atlas       — the embedding's own PCs (PC1–PC2) coloured by every variable (rank colour);
  directions  — every continuous variable decoded from the 4-d embedding (F3's rank R² with a
                shuffle null) and the cosines between the decodable variables' directions in the
                embedding-covariance metric (= correlation of their linear predictions);
  batch       — categorical metadata (balanced accuracy, 5-fold CV, standardised features,
                class-balanced logistic regression) decoded from: the embedding; the task mix;
                the animals' own kernel features (measured from their choices, independent of
                the model); animal behaviour + task mix (the control); and control + embedding.
                The incremental gain (control + embedding) − control is tested against a null
                that shuffles the embedding across mice. A gain above the null means the
                embedding carries setup / cohort information that the animals' behaviour and
                task exposure do not explain. (Revised 2026-10-24: the first version used the
                GRU's own kernels — circular — folded task mix into "behaviour", and did not
                standardise features.)
Writes ``f5_embedding_atlas.json``, ``fig_F5a_embedding_atlas[_pc3_pc4].png``, ``fig_F5b_embedding_directions.png``,
``fig_F5c_embedding_batch.png`` and ``_cache/f5_mouse_table_<key>.parquet``.
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
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
from f3_embedding_features import decode, kernel_features, rl_parameters, task_mix  # noqa: E402

SNAPSHOT = "aind-scratch-data/aind-dynamic-foraging-cache/snapshots/20260603/session_table.parquet"
CATEGORICAL = ["institute", "data_source", "rig_type", "room", "trainer", "PI", "curriculum_name", "first_year"]
CONTINUOUS_META = ["foraging_eff", "reaction_time_median", "base_weight", "water_day_total", "p_reward_contrast_mean",
                   "effective_block_length_mean", "finished_rate", "ignore_rate", "total_trials", "session_run_time_in_min"]
MIN_CLASS = 20
N_PERM_CAT = 100
SEED = 12
ATLAS = ["hattori learn_rate_rew", "hattori learn_rate_unrew", "hattori forget_rate_unchosen", "hattori softmax_inverse_temperature",
         "hattori biasL", "bari learn_rate", "bari choice_kernel_relative_weight", "ctt threshold",
         "gru bias", "gru reward gain (1 back)", "gru persistence (1 back)", "gru advantage over CTT",
         "attractor asymmetry", "v retention (e_vᵀJe_v)", "foraging_eff", "reaction_time_median",
         "base_weight", "p_reward_contrast_mean", "ignore_rate", "n eval sessions",
         "institute", "data_source", "rig_type", "first_year"]


def session_metadata() -> pd.DataFrame:
    path = s12.CACHE / "session_metadata_20260603.parquet"
    if not path.exists():
        import pyarrow.parquet as pq  # noqa: PLC0415
        import s3fs  # noqa: PLC0415

        cols = ["_session_id", "subject_id", "session_date"] + [c for c in CATEGORICAL if c != "first_year"] + CONTINUOUS_META
        table = pq.read_table(s3fs.S3FileSystem(anon=True).open(SNAPSHOT), columns=cols).to_pandas()
        table.to_parquet(path)
    return pd.read_parquet(path)


def mouse_table() -> pd.DataFrame:
    t = s12.load_trials(["subject_id", "ses_idx", "session_id", "action", "p_model1_right"])
    sessions = t.drop_duplicates("ses_idx")[["subject_id", "ses_idx", "session_id"]]
    meta = session_metadata().rename(columns={"subject_id": "meta_subject"})
    meta["first_year"] = pd.to_datetime(meta["session_date"]).dt.year.astype("Int64").astype(str)
    s = sessions.merge(meta, left_on="session_id", right_on="_session_id", how="left", validate="1:1")
    g = s.groupby("subject_id")
    cat = g[[c for c in CATEGORICAL if c != "first_year"]].agg(lambda x: x.mode().iloc[0] if x.notna().any() else np.nan)
    cat["first_year"] = s.assign(d=pd.to_datetime(s["session_date"])).groupby("subject_id")["d"].min().dt.year.astype(str)
    cont = g[CONTINUOUS_META].mean()
    cont["n eval sessions"] = g.size()
    cont["session metadata matched"] = g["_session_id"].apply(lambda x: x.notna().mean())

    kernels = json.loads((s12.ANALYSIS / "c1_kernels.json").read_text())
    beh = pd.concat([kernel_features(kernels["per_mouse_coef"]["gru"], kernels["n_back"]).add_prefix("gru "),
                     kernel_features(kernels["per_mouse_coef"]["animal"], kernels["n_back"]).add_prefix("animal ")], axis=1)
    f2 = pd.read_parquet(s12.CACHE / f"f2_fixed_points_{s12.MODEL_KEY}.parquet")
    att = f2.dropna(subset=["logit_right"]).query("context != 'all-sessions mean'")
    dom = att.sort_values("n_hits", ascending=False).drop_duplicates(["context", "input"]).set_index(["context", "input"])
    logit = dom["logit_right"].unstack().dropna()
    dyn = pd.DataFrame({"attractor asymmetry": logit.mean(axis=1), "attractor depth": logit.abs().mean(axis=1),
                        "v retention (e_vᵀJe_v)": dom["v_retention"].unstack().reindex(logit.index).mean(axis=1)})
    from i5_mouse_dynamics import dynamics_features  # noqa: PLC0415

    p = np.load(s12.keyed_cache("i4_mouse_params.npz"), allow_pickle=False)
    i5 = dynamics_features(p["lam"], p["target"], p["a"], p["b"]).set_index(pd.Index([str(x) for x in p["subjects"]]))
    i5 = i5[["λ stay", "switch target"]].add_prefix("I5 ")
    a = t["action"].to_numpy()
    p_ctt = pd.read_parquet(s12.CACHE / f"rl_trials_ctt_{s12.MODEL_KEY}.parquet", columns=["p_rl_right"])["p_rl_right"].to_numpy()
    lp = lambda q: np.log(np.clip(np.where(a == 1, q, 1 - q), 1e-10, 1))  # noqa: E731
    fit = pd.DataFrame({"gru likelihood": pd.Series(lp(t["p_model1_right"].to_numpy())).groupby(t["subject_id"].to_numpy()).mean().apply(np.exp),
                        "gru advantage over CTT": pd.Series(lp(t["p_model1_right"].to_numpy()) - lp(p_ctt)).groupby(t["subject_id"].to_numpy()).mean()})
    mix = task_mix().add_prefix("task share: ")
    table = pd.concat([rl_parameters(), beh, dyn, i5, fit, cont, mix, cat], axis=1)
    table.index.name = "subject_id"
    return table


def _classes(labels: pd.Series) -> pd.Series | None:
    counts = labels.value_counts()
    keep = labels.isin(counts[counts >= MIN_CLASS].index)
    if keep.sum() < 2 * MIN_CLASS or (counts >= MIN_CLASS).sum() < 2:
        return None
    return keep


def _cv_balanced_accuracy(x: np.ndarray, y: np.ndarray) -> float:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return float(balanced_accuracy_score(y, cross_val_predict(clf, x, y, cv=folds)))


def categorical_decoding(sets: dict[str, np.ndarray], labels: pd.Series, rng) -> dict | None:
    """Balanced accuracy from each feature set (label-shuffle null), plus the embedding's
    incremental gain over the control (embedding-shuffle null)."""
    keep = _classes(labels)
    if keep is None:
        return None
    k = keep.to_numpy()
    y = labels[keep].to_numpy()
    out = {"classes": {str(c): int(n) for c, n in labels[keep].value_counts().items()}, "chance": 1 / len(set(y)), "n": int(k.sum())}
    for name, x in sets.items():
        observed = _cv_balanced_accuracy(x[k], y)
        null = [_cv_balanced_accuracy(x[k], rng.permutation(y)) for _ in range(N_PERM_CAT)]
        out[name] = {"balanced_accuracy": observed, "null_95": float(np.percentile(null, 95))}
    control, emb = sets["animal behaviour + task mix"][k], sets["embedding"][k]
    gain = _cv_balanced_accuracy(np.column_stack([control, emb]), y) - out["animal behaviour + task mix"]["balanced_accuracy"]
    null_gain = [_cv_balanced_accuracy(np.column_stack([control, emb[rng.permutation(len(emb))]]), y)
                 - out["animal behaviour + task mix"]["balanced_accuracy"] for _ in range(N_PERM_CAT)]
    out["embedding gain over control"] = {"gain": float(gain), "null_95": float(np.percentile(null_gain, 95)),
                                          "p": float((1 + np.sum(np.array(null_gain) >= gain)) / (N_PERM_CAT + 1))}
    return out


def main() -> None:
    s12.apply_presentation_style()
    table = mouse_table()
    emb = s12.subject_embeddings()
    mice = emb.index.intersection(table.index)
    table, x = table.loc[mice], emb.loc[mice].to_numpy()
    table.to_parquet(s12.CACHE / f"f5_mouse_table_{s12.MODEL_KEY}.parquet")
    print("mice", len(mice), "| session metadata matched (median share)", table["session metadata matched"].median(), flush=True)

    rng = np.random.default_rng(SEED)
    numeric = [c for c in table.columns if c not in CATEGORICAL and c != "session metadata matched"
               and pd.api.types.is_numeric_dtype(table[c]) and table[c].notna().sum() >= 100 and table[c].nunique() > 5]
    dec = {c: decode(x, table[c].to_numpy(dtype=float), rng) for c in numeric}
    sigma = np.cov(x, rowvar=False)
    good = [c for c in numeric if dec[c]["rank_r2"] > max(0.1, dec[c]["null_95"])]
    dirs = np.array([dec[c]["direction"] for c in good])
    gram = dirs @ sigma @ dirs.T
    cos_sigma = gram / np.sqrt(np.outer(np.diag(gram), np.diag(gram)))

    animal = table[[c for c in table.columns if c.startswith("animal ")]]
    mix = table[[c for c in table.columns if c.startswith("task share: ")]]
    animal, mix = animal.fillna(animal.median()), mix.fillna(0.0)
    sets = {"embedding": x, "task mix": mix.to_numpy(), "animal behaviour": animal.to_numpy(),
            "animal behaviour + task mix": np.column_stack([animal.to_numpy(), mix.to_numpy()])}
    cat = {}
    for c in CATEGORICAL:
        labels = table[c].astype(str).where(table[c].notna())
        if c == "trainer":  # the same people appear as "Bowen Tan" and "bowen.tan"
            labels = labels.str.lower().str.replace(".", " ", regex=False).str.strip()
        labels = labels.where(~labels.isin(["None", "none", "nan", "<missing>", ""]))  # missing is not a class
        out = categorical_decoding(sets, labels, rng)
        if out is not None:
            cat[c] = out
    print({c: {k: (round(v["balanced_accuracy"], 3), round(v["null_95"], 3)) for k, v in d.items() if k in sets}
           | {"gain": round(d["embedding gain over control"]["gain"], 3), "gain_null95": round(d["embedding gain over control"]["null_95"], 3)}
           for c, d in cat.items()}, flush=True)

    # (a) Atlas in the embedding's own PCs.
    xc = x - x.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(xc, rowvar=False))
    order = np.argsort(evals)[::-1]
    pcs = xc @ evecs[:, order]
    share = evals[order] / evals.sum()
    for (pa, pb), fname in (((0, 1), "fig_F5a_embedding_atlas.png"), ((2, 3), "fig_F5a_embedding_atlas_pc3_pc4.png")):
        n = len(ATLAS)
        ncol = 6
        fig, axes = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(6 * ncol, 5.6 * int(np.ceil(n / ncol))),
                                 layout="constrained")
        for ax, c in zip(axes.ravel(), ATLAS):
            v = table[c]
            if c in CATEGORICAL:
                labels = v.astype(str)
                top = labels.value_counts().index[:7]
                palette = plt.get_cmap("tab10")
                for k, lab in enumerate(top):
                    m = (labels == lab).to_numpy()
                    ax.scatter(pcs[m, pa], pcs[m, pb], s=9, color=palette(k), label=f"{lab} ({m.sum()})", alpha=0.8)
                other = ~labels.isin(top).to_numpy()
                ax.scatter(pcs[other, pa], pcs[other, pb], s=6, color="0.8", label=f"other ({other.sum()})")
                ax.legend(frameon=False, fontsize=7, loc="best")
                title = c
                if c in cat:
                    e = cat[c]["embedding"]
                    title += f"  (bal. acc {e['balanced_accuracy']:.2f}, chance {cat[c]['chance']:.2f})"
            else:
                sc = ax.scatter(pcs[:, pa], pcs[:, pb], s=9, c=v.rank(pct=True), cmap="viridis")
                title = c + (f"  (rank R² {dec[c]['rank_r2']:.2f})" if c in dec else "")
            ax.set_title(title.replace("  (", "\n("), fontsize=11)
            ax.set_xlabel(f"embedding PC{pa + 1} ({100 * share[pa]:.0f}%)")
            ax.set_ylabel(f"embedding PC{pb + 1} ({100 * share[pb]:.0f}%)")
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.colorbar(sc, ax=axes, fraction=0.01, pad=0.01, label="rank across mice (continuous variables)")
        fig.suptitle(f"Subject embedding atlas: each mouse in the embedding's PC{pa + 1}–PC{pb + 1}, coloured by one variable", fontsize=18)
        fig.savefig(s12.figure_path(fname), bbox_inches="tight")
        plt.close(fig)

    # (b) Directions: loadings on the embedding PCs (coordinate-free given the PCs).
    loadings = pd.DataFrame({k: [float(pd.Series(pcs[:, k]).corr(table[c].reset_index(drop=True), method="spearman"))
                                 for c in good] for k in range(4)}, index=good)
    fam = lambda c: ("#55A868" if c.startswith(("hattori", "bari", "ctt")) else "#E69F00" if c.startswith("gru")  # noqa: E731
                     else "#222222" if c.startswith("animal") else "#4C72B0" if c.startswith(("attractor", "v ret", "I5"))
                     else "#999999")
    names = sorted(numeric, key=lambda c: -dec[c]["rank_r2"])
    fig, axes = plt.subplots(1, 2, figsize=(30, max(10, 0.32 * len(names))), gridspec_kw={"width_ratios": [0.8, 1.2]})
    axes[0].barh(range(len(names)), [dec[c]["rank_r2"] for c in names], color=[fam(c) for c in names])
    axes[0].scatter([dec[c]["null_95"] for c in names], range(len(names)), marker="|", s=100, color="#C44E52")
    axes[0].set_yticks(range(len(names)), names, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("rank R² from the 4-d embedding (5-fold CV; red tick: shuffle 95th pct)")
    axes[0].set_title("What is decodable from the embedding?  green RL, orange GRU, black animal, blue dynamics, grey metadata",
                      fontsize=12)
    load_order = sorted(good, key=lambda c: (int(np.argmax(np.abs(loadings.loc[c]))), -float(np.max(np.abs(loadings.loc[c])))))
    im = axes[1].imshow(loadings.loc[load_order].to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[1].set_xticks(range(4), [f"embedding PC{k + 1} ({100 * share[k]:.0f}%)" for k in range(4)])
    axes[1].set_yticks(range(len(load_order)), load_order, fontsize=9)
    for i_, c in enumerate(load_order):
        for k in range(4):
            axes[1].text(k, i_, f"{loadings.at[c, k]:+.2f}", ha="center", va="center", fontsize=7)
    axes[1].set_title("Dictionary: Spearman correlation of each decodable variable (rank R² > 0.1) with the embedding PCs",
                      fontsize=13)
    fig.colorbar(im, ax=axes[1], fraction=0.03)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_F5b_embedding_directions.png"))
    plt.close(fig)

    # (c) Batch effects.
    cats = list(cat)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(26, 0.9 * len(cats) + 3), gridspec_kw={"width_ratios": [1.4, 1]})
    y = np.arange(len(cats))
    names_ = list(sets)
    palette_ = ("#E69F00", "#999999", "#222222", "#4C72B0")
    for j, (k, color) in enumerate(zip(names_, palette_)):
        off = (j - 1.5) * 0.2
        ax.barh(y + off, [cat[c][k]["balanced_accuracy"] for c in cats], height=0.2, color=color, label=k)
        ax.scatter([cat[c][k]["null_95"] for c in cats], y + off, marker="|", s=60, color="#C44E52")
    ax.set_yticks(y, [f"{c} ({len(cat[c]['classes'])} classes)" for c in cats])
    ax.invert_yaxis()
    ax.set_xlabel("balanced accuracy, 5-fold CV (red tick: label-shuffle 95th pct)")
    ax.set_title("Decoding setup / cohort labels from each feature set", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    gains = [cat[c]["embedding gain over control"] for c in cats]
    ax2.barh(y, [g["gain"] for g in gains], color="#C44E52")
    ax2.scatter([g["null_95"] for g in gains], y, marker="|", s=120, color="k", label="embedding-shuffle 95th pct")
    ax2.axvline(0, color="0.6", lw=0.8)
    ax2.set_yticks(y, cats)
    ax2.invert_yaxis()
    ax2.set_xlabel("gain in balanced accuracy from adding\nthe embedding to animal behaviour + task mix")
    ax2.set_title("Does the embedding add setup information?", fontsize=13)
    ax2.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_F5c_embedding_batch.png"))
    plt.close(fig)

    s12.write_json("f5_embedding_atlas.json", {
        "n_mice": int(len(mice)), "embedding_pc_variance_share": share, "numeric_decoding": dec,
        "direction_cosines_sigma": {"variables": good, "matrix": cos_sigma},
        "loadings_on_embedding_pcs_spearman": loadings.to_dict(orient="index"), "categorical_decoding": cat,
        "batch_control_features": {"animal behaviour": list(animal.columns), "task mix": list(mix.columns)}, "snapshot": SNAPSHOT,
        "metadata_matched_share_median": float(table["session metadata matched"].median()),
    }, "analysis/f5_embedding_atlas.py", extra_groups=s12.rl_groups())
    print({c: round(dec[c]["rank_r2"], 3) for c in names})


if __name__ == "__main__":
    main()
