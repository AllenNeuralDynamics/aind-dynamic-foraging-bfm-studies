"""B3g: interactive 3-D viewer of the hidden state — any three of PCs 1–6, several colourings.

    python analysis/b3g_viewer.py

Exploration only: writes a self-contained ``_cache/explore/B3g_viewer.html`` (plotly inlined,
opens offline) over a fixed random subsample of trials. Pick the axes and the colouring
from the menus; views worth keeping become committed figures in ``b3_state_space.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
from b3_state_space import COLUMNS, add_derived_columns  # noqa: E402

N_POINTS = 40_000
N_PCS = 6
AXES = [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 3, 4), (1, 2, 4), (0, 4, 5)]


def main() -> None:
    trials = add_derived_columns(s12.load_trials(COLUMNS))
    scores = s12.load_pca()["scores"][:, :N_PCS]
    sub = s12.subsample_rows(len(trials), N_POINTS, seed=2)
    t, x = trials.iloc[sub].reset_index(drop=True), scores[sub]

    outcome_code = t["prev_action_x_outcome"].map(
        {"R rewarded": 0, "R unrewarded": 1, "L rewarded": 2, "L unrewarded": 3}).to_numpy(dtype=float)
    task_code = t["task"].map({task: i for i, task in enumerate(s12.TASKS)}).to_numpy(dtype=float)
    colourings = {
        "P(right)": (t["p_model1_right"].to_numpy(), "RdBu_r", 0, 1),
        "previous trial (R+ / R− / L+ / L−)": (outcome_code, "Turbo", 0, 3),
        "trial position in session": (t["trial_position"].to_numpy(), "Viridis", 0, 1),
        "P(rew|R) − P(rew|L)": (t["p_reward_right_minus_left"].to_numpy(), "RdBu_r", -1, 1),
        "task (UB / CB / UWB / CWB)": (task_code, "Turbo", 0, 3),
        f"trials since reward (≤{s12.CAP})": (t["trials_since_reward_cap"].to_numpy(), "Viridis", 0, s12.CAP),
    }
    hover = [f"mouse {m}<br>{task}<br>trial {k}<br>P(right) {p:.2f}<br>prev: {o}"
             for m, task, k, p, o in zip(t["subject_id"], t["task"], t["trial_idx"], t["p_model1_right"],
                                         t["prev_action_x_outcome"].fillna("—"))]
    first_colour = next(iter(colourings.values()))
    i, j, k = AXES[0]
    fig = go.Figure(go.Scatter3d(
        x=x[:, i], y=x[:, j], z=x[:, k], mode="markers", text=hover, hoverinfo="text",
        marker={"size": 1.6, "opacity": 0.6, "color": first_colour[0], "colorscale": first_colour[1],
                "cmin": first_colour[2], "cmax": first_colour[3], "colorbar": {"thickness": 14}}))

    axis_buttons = [
        {"label": f"PC{a + 1} / PC{b + 1} / PC{c + 1}", "method": "update",
         "args": [{"x": [x[:, a]], "y": [x[:, b]], "z": [x[:, c]]},
                  {"scene.xaxis.title.text": f"PC{a + 1}", "scene.yaxis.title.text": f"PC{b + 1}",
                   "scene.zaxis.title.text": f"PC{c + 1}"}]}
        for a, b, c in AXES]
    colour_buttons = [
        {"label": name, "method": "restyle",
         "args": [{"marker.color": [values], "marker.colorscale": scale, "marker.cmin": lo, "marker.cmax": hi}]}
        for name, (values, scale, lo, hi) in colourings.items()]
    fig.update_layout(
        title=f"Model C seed 0 hidden state — {len(sub):,} random trials of {len(trials):,}",
        scene={"xaxis": {"title": {"text": f"PC{i + 1}"}}, "yaxis": {"title": {"text": f"PC{j + 1}"}},
               "zaxis": {"title": {"text": f"PC{k + 1}"}}},
        updatemenus=[{"buttons": axis_buttons, "x": 0.0, "y": 1.08, "xanchor": "left"},
                     {"buttons": colour_buttons, "x": 0.32, "y": 1.08, "xanchor": "left"}],
        margin={"l": 0, "r": 0, "t": 70, "b": 0}, height=820)
    out = s12.explore_path("B3g_viewer.html")
    fig.write_html(out, include_plotlyjs=True)
    print(out)


if __name__ == "__main__":
    main()
