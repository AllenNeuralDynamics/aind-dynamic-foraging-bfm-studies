"""F: does the subject embedding carry each mouse's engagement style — in C3, and in C?

    python analysis/f_engagement_embedding.py

Per mouse, from its eval steps (C3 table): ignore rate, mean bout length, P(ignore | the
previous step was ignored) (staying disengaged), and the rise in ignore rate from the first
to the last third of sessions. Then:
  a. decode each feature from the 4-d subject embedding (5-fold CV linear regression), for
     C3's embedding and C's (Study 50's model). C never saw ignores, so its embedding can only
     carry engagement through behaviour that correlates with it;
  b. C3's own per-mouse ignore dynamics: with the mouse's mean session context (cached by
     ``d_dynamics.py``), iterate the ignore input from the mouse's mean engaged state:
     P(ignore) after 1 and 3 ignores and at the ignore attractor; then iterate unrewarded
     choices from that attractor: P(ignore) after 3 choices. Correlated with the observed
     features across mice.
Writes ``f_engagement_embedding.json`` and ``fig_F_engagement_embedding.png``.
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
from sklearn.model_selection import KFold, cross_val_predict  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk02 as s13  # noqa: E402

FEATURES = ["ignore rate", "mean bout length", "P(stay disengaged)", "late − early ignore rate"]
N_ITER_ATTRACTOR = 300


def embeddings(model_dir: Path) -> pd.DataFrame:
    outputs = model_dir / "outputs"
    index_map = json.loads((outputs / "subject_index_map.json").read_text())["subject_id_to_index"]
    ckpts = json.loads((outputs / "checkpoints" / "index.json").read_text())["checkpoints"]
    best = max(ckpts, key=lambda c: c["eval_likelihood"])
    params = json.loads((outputs / "checkpoints" / f"step_{best['step']}" / "params.json").read_text())
    table = np.asarray(params["multisubject_gru"]["subject_embeddings"])
    subjects = sorted(index_map, key=index_map.get)
    return pd.DataFrame(table[[index_map[s] for s in subjects]], index=pd.Index([str(s) for s in subjects], name="subject_id"))


def mouse_features(t: pd.DataFrame) -> pd.DataFrame:
    ign = (t["action"] == s13.IGNORE).to_numpy()
    ses = t["ses_idx"].to_numpy()
    prev_ign = np.r_[False, ign[:-1]] & np.r_[False, ses[1:] == ses[:-1]]
    runs = s13.ignore_runs(t["action"].to_numpy(), ses)
    runs["subject_id"] = t["subject_id"].to_numpy()[runs["start"]]
    third = t.groupby("ses_idx", sort=False)["trial_idx"].transform(lambda s: np.ceil(3 * s / s.max()).clip(1, 3)).to_numpy()
    df = pd.DataFrame({"subject_id": t["subject_id"].to_numpy(), "ign": ign, "prev_ign": prev_ign, "third": third})
    out = pd.DataFrame({
        "ignore rate": df.groupby("subject_id")["ign"].mean(),
        "mean bout length": runs.groupby("subject_id")["length"].mean(),
        "P(stay disengaged)": df[df["prev_ign"]].groupby("subject_id")["ign"].mean(),
        "late − early ignore rate": df[df["third"] == 3].groupby("subject_id")["ign"].mean()
        - df[df["third"] == 1].groupby("subject_id")["ign"].mean(),
        "n_steps": df.groupby("subject_id").size(),
        "n_ignored": df.groupby("subject_id")["ign"].sum(),
    })
    out.index = out.index.astype(str)
    return out


def cv_r2(x: np.ndarray, y: np.ndarray) -> float:
    pred = cross_val_predict(LinearRegression(), x, y, cv=KFold(5, shuffle=True, random_state=0))
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def main() -> None:
    import jax  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp  # noqa: PLC0415

    from post_training_analysis.gru_step import GruStep  # noqa: PLC0415

    s13.apply_presentation_style()
    t = s13.load_trials(["subject_id", "ses_idx", "trial_idx", "action"])
    feats = mouse_features(t)
    feats = feats[(feats["n_ignored"] >= 20)]
    emb3 = embeddings(s13.CACHE / "models" / s13.MODEL_KEY / "run")
    emb2 = embeddings(s13.STUDY12 / "analysis" / "_cache" / "models" / s13.TWIN_KEY / "run")
    mice = feats.index.intersection(emb3.index).intersection(emb2.index)
    feats = feats.loc[mice]
    decode = {name: {"C3": cv_r2(emb3.loc[mice].to_numpy(), feats[name].to_numpy()),
                     "C": cv_r2(emb2.loc[mice].to_numpy(), feats[name].to_numpy())} for name in FEATURES}

    # C3's per-mouse ignore dynamics at the mouse's mean session context.
    ctx = np.load(s13.CACHE / f"contexts_{s13.MODEL_KEY}.npz")
    ses_subject = t.drop_duplicates("ses_idx").set_index("ses_idx")["subject_id"].astype(str)
    ctx_subject = ses_subject.reindex(ctx["session_ids"]).to_numpy()
    c_mouse = np.stack([ctx["contexts"][ctx_subject == m].mean(axis=0) for m in mice])
    states = s13.load_trials(["subject_id", "action", *s13.STATE_COLUMNS])
    eng = states[states["action"] != s13.IGNORE]
    h_mouse = eng.groupby(eng["subject_id"].astype(str))[s13.STATE_COLUMNS].mean().loc[mice].to_numpy(np.float64)
    del states, eng
    step = GruStep.from_params(s13.best_params())
    w, b = np.asarray(step.readout_w, np.float64), np.asarray(step.readout_b, np.float64)

    def p_ignore(h):
        z = np.asarray(h) @ w + b
        z = z - z.max(axis=-1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
        return p[..., 2]

    c = jnp.asarray(c_mouse)
    u_ign = jnp.broadcast_to(jnp.asarray([2.0, 0.0]), (len(mice), 2))
    h = jnp.asarray(h_mouse)
    curve = []
    for _ in range(N_ITER_ATTRACTOR):
        h = step.step(h, u_ign, c)
        curve.append(p_ignore(h))
    curve = np.stack(curve)
    attractor = np.asarray(h)
    residual = np.abs(np.asarray(step.step(h, u_ign, c)) - attractor).max(axis=1)
    side = np.where((attractor @ w + b)[:, 1] >= (attractor @ w + b)[:, 0], 1.0, 0.0)
    h = jnp.asarray(attractor)
    rec = []
    for _ in range(3):
        h = step.step(h, jnp.stack([jnp.asarray(side), jnp.zeros(len(mice))], axis=-1), c)
        rec.append(p_ignore(h))
    dyn = pd.DataFrame({"P(ignore) after 1 ignore": curve[0], "P(ignore) after 3 ignores": curve[2],
                        "P(ignore) at the ignore attractor": curve[-1], "P(ignore) 3 choices after the attractor": rec[2]},
                       index=mice)
    corr = {d: {f: float(np.corrcoef(dyn[d], feats[f])[0, 1]) for f in FEATURES} for d in dyn.columns}

    fig, axes = plt.subplots(1, 3, figsize=(23, 6.8), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    x = np.arange(len(FEATURES))
    axes[0].bar(x - 0.2, [decode[f]["C3"] for f in FEATURES], 0.4, color=s13.ENGAGE_COLOR, label="C3 embedding")
    axes[0].bar(x + 0.2, [decode[f]["C"] for f in FEATURES], 0.4, color=s13.MODEL_COLOR, label="C embedding")
    axes[0].set_xticks(x, [f.replace(" ", "\n", 1) for f in FEATURES], fontsize=11)
    axes[0].axhline(0, color="0.7", lw=0.8)
    axes[0].set(ylabel="cross-validated R² (5-fold)", title=f"a. decoding engagement from the 4-d embedding ({len(mice)} mice)")
    axes[0].legend(frameon=False)
    for ax, (dname, fname) in zip(axes[1:], (("P(ignore) after 1 ignore", "ignore rate"),
                                             ("P(ignore) after 3 ignores", "P(stay disengaged)"))):
        ax.scatter(dyn[dname], feats[fname], s=12, color=s13.ENGAGE_COLOR, alpha=0.6, linewidths=0)
        ax.set(xlabel=f"C3 (mouse's context): {dname}", ylabel=f"observed {fname}",
               title=f"r = {corr[dname][fname]:.2f}")
    fig.suptitle("Engagement style across mice: in the embedding, and in each mouse's own ignore dynamics", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(s13.figure_path("fig_F_engagement_embedding.png"))
    plt.close(fig)

    s13.write_json("f_engagement_embedding.json", {
        "n_mice": int(len(mice)), "min_ignored_steps_per_mouse": 20,
        "feature_summary": feats[FEATURES].describe().round(4).to_dict(),
        "decode_cv_r2": decode,
        "mouse_dynamics_summary": dyn.describe().round(4).to_dict(),
        "corr_mouse_dynamics_vs_observed": corr,
        "attractor_max_residual": float(residual.max()), "n_iter_attractor": N_ITER_ATTRACTOR,
        "dynamics": "mouse's mean session context; start = mouse's mean engaged state; ignore input (2, 0) repeated; "
                    "recovery = unrewarded choices of the side favoured at the attractor",
    }, "analysis/f_engagement_embedding.py", extra_groups=s13.twin_wandb_groups())
    print("decode", {f: {k: round(v, 3) for k, v in d.items()} for f, d in decode.items()})
    print("corr", {d: {f: round(v, 2) for f, v in r.items()} for d, r in corr.items()})
    print("attractor residual", residual.max())


if __name__ == "__main__":
    main()
