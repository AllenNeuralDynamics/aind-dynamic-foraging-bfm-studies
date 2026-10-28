"""F4c: each mouse's context across all its training sessions — trajectories and their size.

    PYTHONPATH=<wrapper>/code python analysis/f4c_context_trajectories.py

The GRU's context for mouse m in its session s is c(m, s) = embedding(m) + δ(embedding(m),
phase(s)), with phase = s / (m's number of sessions) computed by the network itself
(``compute_subject_context`` via the wrapper's ``subject_context``). Here it is evaluated for
every session index 1…N_m of every mouse — train and eval sessions alike — and checked against
the cached eval-session contexts. Shown:
  * trajectories of 12 example mice (spread over the embedding, ≥ 40 sessions) on the
    embedding's PCs (F5's dictionary: PC2 attractor depth, PC3 learning rate / gain, PC4 bias),
    over the cloud of all mice's mean contexts — at the population scale, then zoomed;
  * for every mouse, the net displacement (first → last session) and path length, in units of
    the between-mouse SD along each PC, and the median trajectory along training phase (change
    from each mouse's first decile of sessions) — the drift is non-monotonic and front-loaded;
  * the trajectories translated into behaviour: linear maps from a mouse's mean context to its
    GRU kernel features (bias, reward gain, persistence; fitted across mice, 5-fold CV R²) applied
    to every session's context, next to the between-mouse 10–90% range of those features.
Writes ``f4c_context_trajectories.json`` and ``fig_F4c_context_trajectories.png``.
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
from matplotlib.collections import LineCollection  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import KFold, cross_val_predict  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

N_EXAMPLES = 12
MIN_SESSIONS = 40
FEATURES = ["bias", "reward gain (1 back)", "persistence (1 back)"]
PC_NAMES = ["PC1 (mixed)", "PC2 (attractor depth)", "PC3 (learning rate / gain)", "PC4 (bias)"]


def farthest_points(x: np.ndarray, k: int) -> list[int]:
    chosen = [int(np.argmax(np.linalg.norm(x - x.mean(0), axis=1)))]
    for _ in range(k - 1):
        d = np.min(np.linalg.norm(x[:, None] - x[chosen][None], axis=2), axis=1)
        chosen.append(int(np.argmax(d)))
    return chosen


def main() -> None:
    from f3_embedding_features import kernel_features  # noqa: PLC0415
    from post_training_analysis.gru_counterfactual import load_gru_split  # noqa: PLC0415
    from post_training_analysis.gru_step import subject_context  # noqa: PLC0415

    s12.apply_presentation_style()
    outputs = s12.CACHE / "models" / s12.MODEL_KEY / "run" / "outputs"
    cmap = json.loads((outputs / "session_context_map.json").read_text())
    if cmap["indexing"] != "1_based":
        raise AssertionError("expected 1-based session indices")
    split = load_gru_split(s12.CACHE / "models" / s12.MODEL_KEY / "run")
    rows = [(e["subject_id"], e["subject_index"], s + 1, sid) for e in cmap["per_subject"]
            for s, sid in enumerate(e["ordered_session_ids"])]
    traj = pd.DataFrame(rows, columns=["subject_id", "subject_index", "session_index", "ses_idx"])
    traj["phase"] = traj["session_index"] / traj.groupby("subject_id")["session_index"].transform("max")
    ctx = subject_context(split.make_network, split.params, traj["subject_index"].to_numpy(), traj["session_index"].to_numpy())
    dims = [f"c{k}" for k in range(ctx.shape[1])]
    traj[dims] = ctx

    # Check against the contexts the forward pass used for the eval sessions.
    cached = np.load(s12.CACHE / f"contexts_{s12.MODEL_KEY}.npz")
    ref = pd.DataFrame(cached["contexts"], index=cached["session_ids"], columns=dims)
    both = traj.set_index("ses_idx").loc[ref.index, dims]
    parity = float(np.abs(both.to_numpy() - ref.to_numpy()).max())
    print("max |recomputed − cached context| over eval sessions", parity, flush=True)

    emb = s12.subject_embeddings()
    e = emb.to_numpy()
    evals, evecs = np.linalg.eigh(np.cov(e - e.mean(0), rowvar=False))
    order = np.argsort(evals)[::-1]
    evecs, sd = evecs[:, order], np.sqrt(evals[order])
    pcs = (traj[dims].to_numpy() - e.mean(0)) @ evecs
    for k in range(4):
        traj[f"pc{k + 1}"] = pcs[:, k]
    mouse_mean = traj.groupby("subject_id")[[f"pc{k + 1}" for k in range(4)]].mean()
    n_sess = traj.groupby("subject_id").size()

    # Size of each mouse's trajectory, in between-mouse SDs.
    size = {}
    for k in range(4):
        col = f"pc{k + 1}"
        g = traj.sort_values(["subject_id", "session_index"]).groupby("subject_id")[col]
        net = (g.last() - g.first()) / sd[k]
        path = g.apply(lambda s: np.abs(np.diff(s.to_numpy())).sum()) / sd[k]
        rng = (g.max() - g.min()) / sd[k]
        size[PC_NAMES[k]] = {"net_displacement_median": float(np.median(net)), "net_displacement_abs_median": float(np.median(np.abs(net))),
                             "path_length_median": float(np.median(path)), "range_median": float(np.median(rng)),
                             "range_90th": float(np.percentile(rng, 90)), "share_mice_net_positive": float(np.mean(net > 0))}
    total_var = float(((traj[dims] - traj[dims].mean()) ** 2).to_numpy().sum())
    within_var = float(((traj[dims] - traj.groupby("subject_id")[dims].transform("mean")) ** 2).to_numpy().sum())
    print("share within", round(within_var / total_var, 4), {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in size.items()}, flush=True)

    # Behavioural translation: mean context → GRU kernel features, across mice.
    kernels = json.loads((s12.ANALYSIS / "c1_kernels.json").read_text())
    beh = kernel_features(kernels["per_mouse_coef"]["gru"], kernels["n_back"])[FEATURES]
    mice = mouse_mean.index.intersection(beh.index)
    xm = traj.groupby("subject_id")[dims].mean().loc[mice].to_numpy()
    maps, cv_r2 = {}, {}
    for f in FEATURES:
        y = beh.loc[mice, f].to_numpy()
        pred = cross_val_predict(LinearRegression(), xm, y, cv=KFold(5, shuffle=True, random_state=0))
        cv_r2[f] = float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
        maps[f] = LinearRegression().fit(xm, y)
        traj[f"pred {f}"] = maps[f].predict(traj[dims].to_numpy())
    beh_range = {f: np.percentile(beh.loc[mice, f], [10, 90]).tolist() for f in FEATURES}
    pred_change = {f: float(np.median(traj.sort_values("session_index").groupby("subject_id")[f"pred {f}"].agg(lambda s: s.iloc[-1] - s.iloc[0])))
                   for f in FEATURES}

    # Example mice: spread over the embedding, with enough sessions.
    eligible = mouse_mean.index[n_sess.reindex(mouse_mean.index) >= MIN_SESSIONS]
    pick = farthest_points(mouse_mean.loc[eligible].to_numpy() / sd, N_EXAMPLES)
    examples = [eligible[i] for i in pick]
    palette = plt.get_cmap("tab20")

    fig = plt.figure(figsize=(32, 15))
    gs = fig.add_gridspec(2, 4)
    ax_planes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    for ax, (i, j) in zip(ax_planes, ((0, 1), (2, 3))):
        ax.scatter(mouse_mean[f"pc{i + 1}"], mouse_mean[f"pc{j + 1}"], s=8, color="0.82", label="all 614 mice (mean context)")
        for n, m in enumerate(examples):
            d = traj[traj["subject_id"] == m].sort_values("session_index")
            ax.plot(d[f"pc{i + 1}"], d[f"pc{j + 1}"], color=palette(n), lw=2)
            ax.scatter(d[f"pc{i + 1}"].iloc[0], d[f"pc{j + 1}"].iloc[0], color=palette(n), s=40, marker="o", edgecolor="k", zorder=3)
            ax.scatter(d[f"pc{i + 1}"].iloc[-1], d[f"pc{j + 1}"].iloc[-1], color=palette(n), s=70, marker=">", edgecolor="k", zorder=3)
        ax.set_xlabel(f"embedding {PC_NAMES[i]}")
        ax.set_ylabel(f"embedding {PC_NAMES[j]}")
        ax.set_title(f"12 example mice across training (o first session, ▶ last), full scale", fontsize=13)
        ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax = fig.add_subplot(gs[0, 2])
    for n, m in enumerate(examples):
        d = traj[traj["subject_id"] == m].sort_values("session_index")
        x, y = (d["pc3"] - d["pc3"].iloc[0]) / sd[2], (d["pc4"] - d["pc4"].iloc[0]) / sd[3]
        pts = np.column_stack([x, y])
        lc = LineCollection(np.stack([pts[:-1], pts[1:]], axis=1), cmap="viridis", norm=plt.Normalize(0, 1), lw=2)
        lc.set_array(d["phase"].to_numpy()[:-1])
        ax.add_collection(lc)
        ax.scatter(x.iloc[-1], y.iloc[-1], color=palette(n), s=60, marker=">", edgecolor="k", zorder=3)
    ax.axhline(0, color="0.8", lw=0.8)
    ax.axvline(0, color="0.8", lw=0.8)
    lim = 1.6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.add_patch(plt.Circle((0, 0), 0.5, fill=False, ls=":", color="0.5"))
    ax.set_xlabel("PC3 change from the first session (between-mouse SDs)")
    ax.set_ylabel("PC4 change (between-mouse SDs)")
    ax.set_title("Zoomed: each trajectory from its own start (colour = training phase;\ndotted circle = 0.5 between-mouse SD)", fontsize=13)
    fig.colorbar(lc, ax=ax, fraction=0.046, label="training phase (session / number of sessions)")
    ax = fig.add_subplot(gs[0, 3])
    bins = np.linspace(0, 1, 11)
    centres = (bins[:-1] + bins[1:]) / 2
    profiles = {}
    for k, color in zip(range(4), ("#999999", "#4C72B0", "#E69F00", "#C44E52")):
        col = f"pc{k + 1}"
        b = np.clip(np.digitize(traj["phase"], bins) - 1, 0, len(centres) - 1)
        per = traj.assign(b=b).groupby(["subject_id", "b"])[col].mean().unstack()
        per = per.sub(per[0], axis=0) / sd[k]  # change from each mouse's first phase decile
        q = per.quantile([0.25, 0.5, 0.75])
        ax.fill_between(centres, q.loc[0.25], q.loc[0.75], color=color, alpha=0.15)
        ax.plot(centres, q.loc[0.5], color=color, lw=2.5, marker="o", ms=4, label=PC_NAMES[k])
        profiles[PC_NAMES[k]] = {"phase": centres, "median": q.loc[0.5].to_numpy(), "q25": q.loc[0.25].to_numpy(), "q75": q.loc[0.75].to_numpy()}
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("training phase (session / number of sessions)")
    ax.set_ylabel("change from the first decile of training (between-mouse SDs)")
    ax.set_title(f"All mice: median trajectory per PC (band: IQR)\nwithin-mouse share of context variance {100 * within_var / total_var:.1f}%",
                 fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    for ax, f in zip([fig.add_subplot(gs[1, k]) for k in range(3)], FEATURES):
        lo, hi = beh_range[f]
        ax.axhspan(lo, hi, color="0.9", label="10–90% range across mice")
        for n, m in enumerate(examples):
            d = traj[traj["subject_id"] == m].sort_values("session_index")
            ax.plot(d["phase"], d[f"pred {f}"], color=palette(n), lw=2)
        ax.set_xlabel("training phase (session / number of sessions)")
        ax.set_ylabel(f"predicted GRU {f}")
        ax.set_title(f"{f}: what the context drift implies (map CV R² {cv_r2[f]:.2f})", fontsize=13)
        ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax = fig.add_subplot(gs[1, 3])
    names = list(size)
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, [size[n]["net_displacement_abs_median"] for n in names], 0.4, color="#4C72B0", label="median |net change|")
    ax.bar(xs + 0.2, [size[n]["range_90th"] for n in names], 0.4, color="#E69F00", label="90th pct of per-mouse range")
    ax.axhline(1, color="k", ls="--", lw=1, label="1 between-mouse SD")
    ax.set_xticks(xs, [n.replace(" (", "\n(") for n in names], fontsize=10)
    ax.set_ylabel("between-mouse SDs")
    ax.set_title("Trajectory size vs the spread between mice", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_F4c_context_trajectories.png"))
    plt.close(fig)

    s12.write_json("f4c_context_trajectories.json", {
        "n_mice": int(traj["subject_id"].nunique()), "n_session_contexts": int(len(traj)), "parity_max_abs_vs_cached": parity,
        "share_of_context_variance_within_mice_all_sessions": within_var / total_var, "trajectory_size_in_between_mouse_sd": size,
        "median_profile_by_phase": profiles, "behaviour_map_cv_r2": cv_r2, "behaviour_between_mouse_10_90": beh_range, "median_predicted_change_first_to_last": pred_change,
        "examples": examples, "example_n_sessions": {m: int(n_sess[m]) for m in examples}, "min_sessions_for_examples": MIN_SESSIONS,
    }, "analysis/f4c_context_trajectories.py")
    print("cv R2", cv_r2, "| predicted change", pred_change, "| range", beh_range)


if __name__ == "__main__":
    main()
