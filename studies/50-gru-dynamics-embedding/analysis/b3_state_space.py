"""B3b–B3f: the hidden state in PCs 1–4, under many colourings, conditions, tasks and mice.

    python analysis/b3_state_space.py

Uses ``b3_pca.py``'s fit (all trials) and draws a fixed random subsample of trials. This is
the exploration step of PLAN.md B3: every colouring and condition is written to
``_cache/explore/``; the views worth keeping are then promoted to committed ``fig_B3*.png``
by listing them in ``PROMOTED`` below. Pairwise grids reuse the wrapper's state-space
plotting helpers.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
from evaluation.plotting import plot_latents_in_space  # noqa: E402
from post_training_analysis import likelihood_advantage_analysis as laa  # noqa: E402

N_PLOT_PCS = 4
N_SCATTER = 150_000
CAP = s12.CAP
COLUMNS = ["subject_id", "ses_idx", "trial_idx", "task", "action", "reward",
           "p_model1_left", "p_model1_right", "prev_action", "prev_outcome",
           "history_pattern_1", "history_pattern_2", "recent_reward_rate_5",
           "recent_reward_rate_10", "trials_since_reward", "trial_position",
           "reward_probabilityL", "reward_probabilityR"]
CONTINUOUS = ["p_model1_right", "p_model1_right_minus_left", "choice_confidence", "p_model1_switch",
              "recent_reward_rate_5", "recent_reward_rate_10", "trial_position",
              "trials_since_reward_cap", "p_reward_right_minus_left",
              "trials_since_left_chosen_cap", "trials_since_right_chosen_cap"]
CONDITIONS = ["prev_action_x_outcome", "history_pattern_1", "history_pattern_2"]
# Exploration outputs to commit as figures, chosen after inspecting _cache/explore/.
PROMOTED = {
    "B3b_p_model1_right.png": "fig_B3b_state_space_p_right.png",
    "B3f_mice_PC1_PC2.png": "fig_B3f_mice_over_embedding.png",
    # PC5 tracks time within the session (the >5% rule in PLAN.md B3 drew PCs 5-6).
    "B3_pc56_trial_position.png": "fig_B3_pc56_trial_position.png",
}
SNIPPET = 60  # trials shown per example trajectory


def add_derived_columns(trials: pd.DataFrame) -> pd.DataFrame:
    """Columns that need the full, ordered table (previous trial, gaps since a choice)."""
    trials = laa._add_rnn_state_space_overview_columns(trials)
    trials["p_reward_right_minus_left"] = trials["reward_probabilityR"] - trials["reward_probabilityL"]
    trials["trials_since_reward_cap"] = trials["trials_since_reward"].clip(upper=CAP)
    position = trials.groupby("ses_idx", sort=False).cumcount().to_numpy()
    for side, name in ((0, "left"), (1, "right")):
        # As of the decision on trial t: trials since this side was last chosen (before t).
        chosen_at = pd.Series(np.where(trials["action"].to_numpy() == side, position, np.nan))
        last = chosen_at.groupby(trials["ses_idx"].to_numpy(), sort=False).transform(
            lambda s: s.shift(1).ffill()).to_numpy()
        trials[f"trials_since_{name}_chosen_cap"] = np.clip(position - last, None, CAP)
    action = trials["prev_action"].map({0.0: "L", 1.0: "R"})
    outcome = trials["prev_outcome"].map({0.0: "unrewarded", 1.0: "rewarded"})
    trials["prev_action_x_outcome"] = (action + " " + outcome).where(action.notna() & outcome.notna())
    return trials


def _safe(value) -> str:
    """File-name form of a condition value; history letters become case-free (L+ / L-)."""
    text = str(value)
    if text and all(c in "LlRr" for c in text):
        return "".join(c.upper() + ("+" if c.isupper() else "-") for c in text)
    return text.replace(" ", "_")


def prev_outcome_row(result: dict, path: Path, pcs: tuple[int, int] = (0, 1)) -> None:
    """One PC pair (default PC1 vs PC2) for each previous choice x outcome, coloured by P(right), others grey."""
    i, j = pcs
    df, scores = result["trial_df"], result["scores"]
    order = ["R rewarded", "R unrewarded", "L rewarded", "L unrewarded"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.6), sharex=True, sharey=True)
    for ax, value in zip(axes, order):
        mask = (df["prev_action_x_outcome"] == value).to_numpy()
        ax.scatter(scores[:, i], scores[:, j], s=2, color="0.88", linewidths=0, rasterized=True)
        sc = ax.scatter(scores[mask, i], scores[mask, j], s=2, c=df["p_model1_right"].to_numpy()[mask],
                        cmap="coolwarm", vmin=0, vmax=1, linewidths=0, rasterized=True)
        ax.set_title(f"previous trial: {value}\n(n={mask.sum():,})", fontsize=13)
        ax.set_xlabel(f"PC{i + 1}")
    axes[0].set_ylabel(f"PC{j + 1}")
    fig.colorbar(sc, ax=axes, fraction=0.015, pad=0.01, label="P(right)")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def trajectory_snippets(scores_all: np.ndarray, trials: pd.DataFrame, sub: np.ndarray,
                        sessions: dict[str, str], path: Path) -> dict:
    """SNIPPET consecutive trials per example session in PC1-PC2, centred on its first
    switch of choice after trial 30 so each snippet crosses between the two sides."""
    fig, axes = plt.subplots(1, len(sessions), figsize=(5.6 * len(sessions), 5.6), sharex=True, sharey=True)
    shown = {}
    for ax, (task, ses) in zip(axes, sessions.items()):
        rows = np.flatnonzero((trials["ses_idx"] == ses).to_numpy())
        action = trials["action"].to_numpy()[rows]
        switches = np.flatnonzero(np.diff(action) != 0) + 1
        centre = int(next((k for k in switches if k >= 30), len(rows) // 2))
        start = max(0, min(centre - SNIPPET // 2, len(rows) - SNIPPET))
        seg = rows[start:start + SNIPPET]
        rewarded = trials["reward"].to_numpy()[seg] > 0
        ax.scatter(scores_all[sub, 0], scores_all[sub, 1], s=2, color="0.9", linewidths=0, rasterized=True)
        ax.plot(scores_all[seg, 0], scores_all[seg, 1], color="0.35", lw=0.8, zorder=2)
        sc = ax.scatter(scores_all[seg, 0], scores_all[seg, 1], c=np.arange(len(seg)), cmap="viridis",
                        s=np.where(rewarded, 60, 22), edgecolors=np.where(rewarded, "k", "none"),
                        linewidths=0.6, zorder=3)
        ax.set_title(f"{task}\ntrials {start + 1}–{start + len(seg)}", fontsize=12)
        ax.set_xlabel("PC1")
        shown[task] = {"ses_idx": ses, "first_trial": start + 1, "n": int(len(seg))}
    axes[0].set_ylabel("PC2")
    fig.colorbar(sc, ax=axes, fraction=0.015, pad=0.01, label="trial within snippet")
    fig.suptitle("Example trajectories (large outlined dots = rewarded trial; each dot is the state "
                 "the model chooses from)", fontsize=12, y=1.02)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return shown


def higher_pcs(scores_all: np.ndarray, trials: pd.DataFrame, sub: np.ndarray, column: str, path: Path,
               label: str, cmap: str, vmin: float, vmax: float) -> None:
    """PCs 5-6 against PCs 1-2 (PLAN.md B3: drawn because PC5 carries >5% variance)."""
    pairs = [(0, 4), (0, 5), (1, 4), (1, 5), (4, 5)]
    colors = trials[column].to_numpy(dtype=float)[sub]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 4.8))
    for ax, (i, j) in zip(axes, pairs):
        sc = ax.scatter(scores_all[sub, i], scores_all[sub, j], s=2, c=colors, cmap=cmap, vmin=vmin, vmax=vmax,
                        alpha=0.5, linewidths=0, rasterized=True)
        ax.set_xlabel(f"PC{i + 1}")
        ax.set_ylabel(f"PC{j + 1}")
    fig.colorbar(sc, ax=axes, fraction=0.015, pad=0.01, label=label)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def pca_result(trials: pd.DataFrame, scores: np.ndarray) -> dict:
    """The dict shape the wrapper's state-space plotting helpers read."""
    return {"trial_df": trials.reset_index(drop=True), "scores": scores.astype(float),
            "finite_mask": np.isfinite(scores).all(axis=1)}


def categorical_grid(result: dict, column: str, order: list[str], colors: dict[str, str], path: Path,
                     title: str) -> None:
    df, scores = result["trial_df"], result["scores"]
    pairs = list(combinations(range(N_PLOT_PCS), 2))
    keep = df[column].isin(order).to_numpy()
    # Draw in one shuffled pass so no category is systematically painted over the others.
    rows = np.random.default_rng(0).permutation(np.flatnonzero(keep))
    point_colors = df[column].map(colors).to_numpy()[rows]
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.2), dpi=120)
    for ax, (i, j) in zip(axes.flat, pairs):
        ax.scatter(scores[rows, i], scores[rows, j], s=3, alpha=0.25, linewidths=0, c=list(point_colors))
        for value in order:
            ax.scatter([], [], s=12, color=colors[value], label=f"{value} (n={(df[column] == value).sum():,})")
        ax.set_xlabel(f"PC{i + 1}")
        ax.set_ylabel(f"PC{j + 1}")
    axes.flat[0].legend(markerscale=5, fontsize=9, frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def example_trajectories(scores_all: np.ndarray, trials: pd.DataFrame, sub: np.ndarray,
                         sessions: dict[str, str]) -> None:
    background = scores_all[sub, :N_PLOT_PCS]
    colors = trials["p_model1_right"].to_numpy()[sub]
    for task, ses in sessions.items():
        rows = np.flatnonzero((trials["ses_idx"] == ses).to_numpy())
        fig = plot_latents_in_space(background, color_values=colors, color_label="P(right)",
                                    example_run=scores_all[rows, :N_PLOT_PCS], axis_label_prefix="PC")
        for ax in fig.axes:
            if ax.get_xlabel().startswith("PC "):
                ax.set_xlabel(f"PC{int(ax.get_xlabel().split()[-1]) + 1}")
                ax.set_ylabel(f"PC{int(ax.get_ylabel().split()[-1]) + 1}")
                ax.set_title(f"{task}: {ses.split('__')[-1]}", fontsize=12)
        fig.savefig(s12.explore_path(f"B3e_trajectory_{task.replace(' ', '_')}.png"))
        plt.close(fig)


def embedding_spread_mice(trials: pd.DataFrame) -> list[str]:
    """Nine mice spread over the subject embedding: nearest to the 10/50/90% grid of its PCs."""
    outputs = s12.CACHE / "models" / s12.MODEL_KEY / "run" / "outputs"
    index_map = json.loads((outputs / "subject_index_map.json").read_text())["subject_id_to_index"]
    params = json.loads((outputs / "checkpoints" / "step_90000" / "params.json").read_text())
    table = np.asarray(params["multisubject_gru"]["subject_embeddings"])
    present = sorted(set(trials["subject_id"]) & set(index_map))
    emb = table[[index_map[m] for m in present]]
    centred = emb - emb.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    coords = centred @ vt[:2].T
    chosen = []
    for q2 in (90, 50, 10):
        for q1 in (10, 50, 90):
            target = np.array([np.percentile(coords[:, 0], q1), np.percentile(coords[:, 1], q2)])
            for k in np.argsort(np.linalg.norm(coords - target, axis=1)):
                if present[k] not in chosen:
                    chosen.append(present[k])
                    break
    return chosen


def mice_panels(scores_all: np.ndarray, trials: pd.DataFrame, sub: np.ndarray, mice: list[str]) -> None:
    for i, j in ((0, 1), (0, 2), (1, 2)):
        fig, axes = plt.subplots(3, 3, figsize=(13, 12), dpi=120, sharex=True, sharey=True)
        for ax, mouse in zip(axes.flat, mice):
            rows = np.flatnonzero((trials["subject_id"] == mouse).to_numpy())
            ax.scatter(scores_all[sub, i], scores_all[sub, j], s=2, color="0.85", linewidths=0)
            sc = ax.scatter(scores_all[rows, i], scores_all[rows, j], s=2, c=trials["p_model1_right"].to_numpy()[rows],
                            cmap="coolwarm", vmin=0, vmax=1, alpha=0.5, linewidths=0)
            ax.set_title(f"mouse {mouse} (n={len(rows):,})", fontsize=11)
        for ax in axes[-1]:
            ax.set_xlabel(f"PC{i + 1}")
        for ax in axes[:, 0]:
            ax.set_ylabel(f"PC{j + 1}")
        fig.colorbar(sc, ax=axes, fraction=0.02, label="P(right)")
        fig.suptitle("Mice spread over the subject embedding (rows: embedding PC2 high→low; "
                     "columns: embedding PC1 low→high)", fontsize=12)
        fig.savefig(s12.explore_path(f"B3f_mice_PC{i + 1}_PC{j + 1}.png"))
        plt.close(fig)


def main() -> None:
    s12.apply_presentation_style()
    trials = add_derived_columns(s12.load_trials(COLUMNS))
    scores_all = s12.load_pca()["scores"]
    if len(scores_all) != len(trials):
        raise AssertionError("PCA scores and trial table are not row-aligned")
    sub = s12.subsample_rows(len(trials), N_SCATTER)
    result = pca_result(trials.iloc[sub], scores_all[sub, :N_PLOT_PCS])

    # B3b: one grid per continuous colouring, plus task.
    for column in CONTINUOUS:
        laa._plot_rnn_state_space_overview_figure(
            result, color_column=column, output_path=s12.explore_path(f"B3b_{column}.png"), n_plot_pcs=N_PLOT_PCS)
    categorical_grid(result, "task", s12.TASKS, s12.TASK_COLORS, s12.explore_path("B3b_task.png"),
                     "Hidden state coloured by task")

    # B3c: one grid per condition value, coloured by P(right), other trials grey.
    for column in CONDITIONS:
        for value in sorted(result["trial_df"][column].dropna().unique()):
            mask = (result["trial_df"][column] == value).to_numpy()
            laa._plot_rnn_state_condition_figure(
                result, condition_column=column, condition_label=str(value), condition_mask=mask,
                output_path=s12.explore_path(f"B3c_{column}_{_safe(value)}.png"),
                n_plot_pcs=N_PLOT_PCS, color_column="p_model1_right", colorbar_label="P(right)",
                cmap="coolwarm", color_range=(0.0, 1.0))

    # B3d: each task on the shared axes, coloured by P(right).
    for task in s12.TASKS:
        df = result["trial_df"].copy()
        column = f"p_right__{task.replace(' ', '_')}"
        df[column] = df["p_model1_right"].where(df["task"] == task)
        laa._plot_rnn_state_space_overview_figure(
            {**result, "trial_df": df}, color_column=column,
            output_path=s12.explore_path(f"B3d_{task.replace(' ', '_')}.png"), n_plot_pcs=N_PLOT_PCS)

    # B3c, committed compact view: previous choice x outcome in PC1-PC2.
    prev_outcome_row(result, s12.figure_path("fig_B3c_prev_outcome.png"))
    prev_outcome_row(result, s12.figure_path("fig_B3c_prev_outcome_pc1_pc4.png"), pcs=(0, 3))

    # B3e: one example session per task over the P(right)-coloured cloud (full sessions to
    # explore; a committed snippet view).
    sessions = s12.example_sessions(trials)
    example_trajectories(scores_all, trials, sub, sessions)
    snippets = trajectory_snippets(scores_all, trials, sub, sessions, s12.figure_path("fig_B3e_trajectories.png"))

    # PCs 5-6 (PC5 explains >5%): several colourings to explore, P(right) committed.
    for column, label, cmap, lo, hi in (("p_model1_right", "P(right)", "coolwarm", 0, 1),
                                        ("prev_outcome", "previous trial rewarded", "coolwarm", 0, 1),
                                        ("trial_position", "trial position in session", "viridis", 0, 1),
                                        ("p_reward_right_minus_left", "P(rew|R) − P(rew|L)", "coolwarm", -1, 1)):
        higher_pcs(scores_all, trials, sub, column, s12.explore_path(f"B3_pc56_{column}.png"), label, cmap, lo, hi)

    # B3f: nine mice spread over the subject embedding.
    mice = embedding_spread_mice(trials)
    mice_panels(scores_all, trials, sub, mice)

    for source, target in PROMOTED.items():
        (s12.ANALYSIS / target).write_bytes(s12.explore_path(source).read_bytes())
    s12.write_json("b3_state_space.json", {
        "n_scatter_trials": int(len(sub)), "scatter_seed": 0, "n_plot_pcs": N_PLOT_PCS,
        "continuous_colourings": CONTINUOUS, "conditions": CONDITIONS,
        "example_sessions": sessions, "trajectory_snippets": snippets, "embedding_spread_mice": mice,
        "promoted": PROMOTED,
    }, "analysis/b3_state_space.py")


if __name__ == "__main__":
    main()
