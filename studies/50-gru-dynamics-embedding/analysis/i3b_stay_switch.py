"""I3b: I3's transitions regrouped — one panel for stays, one for switches, inputs by colour.

    python analysis/i3b_stay_switch.py

Same transitions as I3 (seed 4, 1M transitions with trials t−1, t, t+1 in one session) and the
same value coordinate v (I2's value axis; ``_cache/value_coord_<key>.npy``, identical to I3's
projection). x = v of the state after trial t−1 (before choice t); y = v after trial t. The
input is trial t's choice × outcome; stay = choice t on the same side as choice t−1, switch
otherwise. Lines: least-squares fit per input. Dashed: each input's attractor on v from I3 (all-
sessions mean context; shown in the stay panel, where a repeated input converges). Writes
``fig_I3b_stay_switch.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
from i3_tdr_dynamics import INPUTS, N_TRANSITIONS, SEED  # noqa: E402
from i4_distilled_model import value_coordinate  # noqa: E402

COLORS = {"R rewarded": "#0072B2", "R unrewarded": "#56B4E9", "L rewarded": "#D55E00", "L unrewarded": "#E69F00"}
N_PLOT = 15_000
LIM = (-3.5, 4.5)


def main() -> None:
    s12.apply_presentation_style()
    t = s12.load_trials(["ses_idx", "action", "reward"])
    ses = t["ses_idx"].to_numpy()
    action, reward = t["action"].to_numpy(), (t["reward"].to_numpy() > 0).astype(int)
    ok = np.flatnonzero((ses[1:-1] == ses[2:]) & (ses[1:-1] == ses[:-2])) + 1
    rows = np.sort(np.random.default_rng(SEED).choice(ok, size=min(N_TRANSITIONS, len(ok)), replace=False))
    v = value_coordinate()
    x_all, y_all = v[rows], v[rows + 1]
    stay = action[rows] == action[rows - 1]
    i3 = json.loads((s12.ANALYSIS / "i3_tdr_dynamics.json").read_text())
    v_idx = i3["axes"].index("value")
    attract = {k: [p[v_idx] for p in d["tdr"]] for k, d in i3["attractors_all_sessions_context"].items()}

    fig, axes = plt.subplots(1, 2, figsize=(17, 8), sharex=True, sharey=True)
    rng = np.random.default_rng(1)
    for ax, (kind, mask_kind) in zip(axes, (("stay", stay), ("switch", ~stay))):
        ax.plot(LIM, LIM, color="0.35", lw=3, ls="-", zorder=1, label="v unchanged (diagonal)")
        for name, (ch, rw) in INPUTS.items():
            m = mask_kind & (action[rows] == ch) & (reward[rows] == rw)
            x, y = x_all[m], y_all[m]
            sub = rng.permutation(len(x))[:N_PLOT]
            slope, intercept = np.polyfit(x, y, 1)
            r2 = np.corrcoef(x, y)[0, 1] ** 2
            ax.scatter(x[sub], y[sub], s=2, alpha=0.25, color=COLORS[name], linewidths=0, rasterized=True, zorder=2)
            xs = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 2)
            ax.plot(xs, slope * xs + intercept, color=COLORS[name], lw=2.5, zorder=3,
                    label=f"{name}: slope {slope:.2f}, R² {r2:.2f}, n={m.sum():,}")
            if kind == "stay":
                for a in attract[name]:
                    ax.axhline(a, color=COLORS[name], lw=1.3, ls="--", zorder=1)
        ax.set_xlim(LIM)
        ax.set_ylim(LIM)
        ax.set_aspect("equal")
        ax.set_xlabel("v after trial t−1 (before choice t)")
        ax.set_title(f"{kind}: choice t {'same side as' if kind == 'stay' else 'other side from'} choice t−1", fontsize=14)
        ax.legend(frameon=False, fontsize=10, markerscale=6, loc="upper left")
    axes[0].set_ylabel("v after trial t (before choice t+1)")
    fig.suptitle("Value coordinate across one trial, by trial t's input (dashed: that input's attractor, stay panel)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_I3b_stay_switch.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
