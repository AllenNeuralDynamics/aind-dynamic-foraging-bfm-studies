"""I11d: one figure for v's carry-over — I11's example session in the (c, v) plane, then I3b's
stay and switch transition maps.

    python analysis/i11d_carry_summary.py

Panel 1 is I11's top-left panel (same session, window and axes); panels 2–3 are I3b (same 1M
transitions, seed 4; least-squares line per input; dashed = that input's attractor on v, stay
panel). Colours are B3h's input colours. Writes ``fig_I11d_carry_summary.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
from b3h_pc_overview import COLORS  # noqa: E402
from i3_tdr_dynamics import INPUTS, N_TRANSITIONS, SEED  # noqa: E402
from i3b_stay_switch import LIM, N_PLOT  # noqa: E402
from i4_distilled_model import value_coordinate  # noqa: E402
from i11_value_across_lobes import WINDOW  # noqa: E402


def main() -> None:
    s12.apply_presentation_style()
    t = s12.load_trials(["ses_idx", "trial_idx", "action", "reward", "prev_action", "prev_outcome", "task"])
    ses, action = t["ses_idx"].to_numpy(), t["action"].to_numpy()
    reward = (t["reward"].to_numpy() > 0).astype(int)
    pa, po = t["prev_action"].to_numpy(), t["prev_outcome"].to_numpy()
    v = value_coordinate()
    ax_file = np.load(s12.CACHE / "i2_tdr_axes.npz")
    terms = [str(x) for x in ax_file["terms"]]
    states = s12.load_trials(s12.STATE_COLUMNS).to_numpy(np.float32)
    c = (states.astype(np.float64) - s12.load_pca()["mean"]) @ ax_file["basis"][:, terms.index("choice")]
    del states

    fig, axes = plt.subplots(1, 3, figsize=(24, 8.6), layout="constrained")

    # (1) I11's example session in the (c, v) plane.
    ax = axes[0]
    bg = np.random.default_rng(11).choice(len(t), 60_000, replace=False)
    ax.scatter(c[bg], v[bg], s=1, color="0.88", linewidths=0, rasterized=True)
    rows = np.flatnonzero(ses == s12.example_sessions(t)["Uncoupled Baiting"])[WINDOW[0]:WINDOW[1]]
    label = np.full(len(t), "", dtype=object)
    for name, (a, r) in INPUTS.items():
        label[(pa == a) & (po == r)] = name
    for k0, k1 in zip(rows[:-1], rows[1:]):
        switch = (not np.isnan(pa[k0])) and action[k0] != pa[k0]
        ax.plot([c[k0], c[k1]], [v[k0], v[k1]], color="k", lw=2.8 if switch else 0.8, alpha=0.9 if switch else 0.5, zorder=3)
    ax.scatter(c[rows], v[rows], s=90, c=[COLORS.get(x, "0.5") for x in label[rows]], edgecolors="k", linewidths=0.5, zorder=4)
    ax.set_xlabel("choice axis c", fontsize=23)
    ax.set_ylabel("v", fontsize=23)
    ax.set_title(f"one session, trials {WINDOW[0]}–{WINDOW[1]}\n(bold: switches)", fontsize=24)
    handles = [Line2D([], [], marker="o", ls="none", ms=13, color=col, markeredgecolor="k", label=n) for n, col in COLORS.items()]
    ax.legend(handles=handles, title="t−1 input", frameon=False, fontsize=16, title_fontsize=17, loc="upper left")

    # (2–3) I3b's stay and switch maps.
    ok = np.flatnonzero((ses[1:-1] == ses[2:]) & (ses[1:-1] == ses[:-2])) + 1
    tr = np.sort(np.random.default_rng(SEED).choice(ok, size=min(N_TRANSITIONS, len(ok)), replace=False))
    x_all, y_all = v[tr], v[tr + 1]
    stay = action[tr] == action[tr - 1]
    i3 = json.loads((s12.ANALYSIS / "i3_tdr_dynamics.json").read_text())
    v_idx = i3["axes"].index("value")
    attract = {k: [p[v_idx] for p in d["tdr"]] for k, d in i3["attractors_all_sessions_context"].items()}
    rng = np.random.default_rng(1)
    for ax, (kind, mask) in zip(axes[1:], (("stay", stay), ("switch", ~stay))):
        ax.plot(LIM, LIM, color="0.35", lw=3, zorder=1, label="unchanged")
        for name, (ch, rw) in INPUTS.items():
            m = mask & (action[tr] == ch) & (reward[tr] == rw)
            x, y = x_all[m], y_all[m]
            sub = rng.permutation(len(x))[:N_PLOT]
            slope, intercept = np.polyfit(x, y, 1)
            ax.scatter(x[sub], y[sub], s=2, alpha=0.25, color=COLORS[name], linewidths=0, rasterized=True, zorder=2)
            xs = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 2)
            ax.plot(xs, slope * xs + intercept, color=COLORS[name], lw=3, zorder=3, label=f"{name}: slope {slope:.2f}")
            if kind == "stay":
                for a in attract[name]:
                    ax.axhline(a, color=COLORS[name], lw=1.5, ls="--", zorder=1)
        ax.set_xlim(LIM)
        ax.set_ylim(LIM)
        ax.set_aspect("equal")
        ax.set_xlabel("v before trial t", fontsize=23)
        ax.set_ylabel("v after trial t", fontsize=23)
        ax.set_title(f"{kind}: choice t {'same side as' if kind == 'stay' else 'other side from'}\nchoice t−1", fontsize=24)
        if kind == "stay":  # lower right, above the L attractors' dashed lines
            ax.legend(frameon=False, fontsize=15, loc="lower right", bbox_to_anchor=(1.0, 0.14))
        else:
            ax.legend(frameon=False, fontsize=15, loc="upper left")
    for ax in axes:
        ax.tick_params(labelsize=18)
    fig.savefig(s12.figure_path("fig_I11d_carry_summary.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
