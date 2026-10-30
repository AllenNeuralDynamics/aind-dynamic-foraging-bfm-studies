"""I4: distil the GRU into a one-variable model and score it on the animals.

    python analysis/i4_distilled_model.py

The value coordinate v (I2's value axis) of every trial's state is the teacher. Per
transition group (input × stay/switch/first, see ``_transitions``) fit v(t+1) = v* + λ(v(t) − v*);
fit the readout logit ≈ a·v + b(last input) to the GRU's logit. Then *simulate* v from a fixed
start along each session, driven only by the animal's observed choices and rewards (no GRU),
read out P(right), and score the animals' choices. Two versions: pooled parameters (≈ 30
numbers for all mice) and per-mouse parameters (fallback to pooled for groups with < 50
transitions). Compared with the GRU, E2's 1-d carried state, and the per-mouse RL fits.

Also writes ``_cache/value_coord_<key>.npy`` and ``_cache/i4_mouse_params.npz`` for I5.
Writes ``i4_distilled_model.json`` and ``fig_I4_distilled_model.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
import _transitions as tr  # noqa: E402

RL = ("bari", "ctt", "hattori")


def value_coordinate() -> np.ndarray:
    path = s12.CACHE / f"value_coord_{s12.MODEL_KEY}.npy"
    if path.exists():
        return np.load(path)
    basis = np.load(s12.keyed_cache("i2_tdr_axes.npz"))["basis"]
    terms = [str(x) for x in np.load(s12.keyed_cache("i2_tdr_axes.npz"))["terms"]]
    axis, mean = basis[:, terms.index("value")], s12.load_pca()["mean"]
    states = s12.load_trials(s12.STATE_COLUMNS).to_numpy(dtype=np.float32)
    v = ((states.astype(np.float64) - mean) @ axis).astype(np.float64)
    np.save(path, v)
    return v


def main() -> None:
    s12.apply_presentation_style()
    t = s12.load_trials(["subject_id", "ses_idx", "trial_idx", "action", "reward", "logit_diff"])
    v, logit, action = value_coordinate(), t["logit_diff"].to_numpy(dtype=np.float64), t["action"].to_numpy()
    lay = tr.session_layout(t)
    v0 = float(v[lay["first"]].mean())

    # Pooled distillation.
    lam, target, n = tr.fit_groups(v, lay)
    a, b = tr.fit_readout(v, logit, lay)
    teacher_r2 = 1 - np.var(logit - (a * v + b[lay["cluster"]])) / np.var(logit)
    v_sim = tr.simulate(lay, v0, lam[lay["group"]], target[lay["group"]])
    logit_sim = a * v_sim + b[lay["cluster"]]

    # Per-mouse distillation (dynamics and readout per mouse; pooled fallback per group).
    subjects = t["subject_id"].to_numpy()
    by_mouse = pd.Series(np.arange(len(t))).groupby(subjects).indices
    lam_rows, target_rows, a_rows, b_rows = (np.empty(len(t)) for _ in range(4))
    mouse_params = {}
    for subject, rows in by_mouse.items():
        lm, tm, nm = tr.fit_groups(v, lay, rows=rows, fallback=(lam, target))
        am, bm = tr.fit_readout(v, logit, lay, rows=rows)
        mouse_params[subject] = {"lam": lm, "target": tm, "n": nm, "a": am, "b": bm}
        lam_rows[rows], target_rows[rows] = lm[lay["group"][rows]], tm[lay["group"][rows]]
        a_rows[rows], b_rows[rows] = am, bm[lay["cluster"][rows]]
    v_sim_m = tr.simulate(lay, v0, lam_rows, target_rows)
    logit_sim_m = a_rows * v_sim_m + b_rows
    np.savez(s12.keyed_cache("i4_mouse_params.npz"), subjects=np.array(list(mouse_params)),
             lam=np.array([p["lam"] for p in mouse_params.values()]),
             target=np.array([p["target"] for p in mouse_params.values()]),
             n=np.array([p["n"] for p in mouse_params.values()]),
             a=np.array([p["a"] for p in mouse_params.values()]),
             b=np.array([p["b"] for p in mouse_params.values()]))

    lik = {"GRU": tr.likelihood(action, logit),
           "distilled, per mouse": tr.likelihood(action, logit_sim_m),
           "distilled, pooled": tr.likelihood(action, logit_sim),
           "distilled readout on GRU's own v (teacher-forced)": tr.likelihood(action, a * v + b[lay["cluster"]])}
    e2 = s12.ANALYSIS / "e2_ablations.json"
    if e2.exists():
        import json  # noqa: PLC0415

        lik["GRU, 1-d carried state (E2)"] = json.loads(e2.read_text())["results"]["readout-null space (127-d) | full"]["likelihood"]
    for name in RL:
        p = pd.read_parquet(s12.CACHE / f"rl_trials_{name}_{s12.MODEL_KEY}.parquet", columns=["p_rl_right"])["p_rl_right"].to_numpy()
        lik[name] = tr.likelihood(action, np.log(np.clip(p, 1e-10, 1) / np.clip(1 - p, 1e-10, 1)))
    per_mouse_beats_ctt = None
    p_ctt = pd.read_parquet(s12.CACHE / f"rl_trials_ctt_{s12.MODEL_KEY}.parquet", columns=["p_rl_right"])["p_rl_right"].to_numpy()
    lp_m = np.log(np.clip(np.where(action == 1, 1 / (1 + np.exp(-logit_sim_m)), 1 / (1 + np.exp(logit_sim_m))), 1e-10, 1))
    lp_c = np.log(np.clip(np.where(action == 1, p_ctt, 1 - p_ctt), 1e-10, 1))
    gain = pd.Series(lp_m - lp_c).groupby(subjects).mean()
    per_mouse_beats_ctt = float((gain > 0).mean())
    sim_vs_gru = {"corr_v": float(np.corrcoef(v_sim_m, v)[0, 1]), "corr_logit": float(np.corrcoef(logit_sim_m, logit)[0, 1]),
                  "corr_v_pooled": float(np.corrcoef(v_sim, v)[0, 1])}

    fig, axes = plt.subplots(1, 3, figsize=(24, 6.6), gridspec_kw={"width_ratios": [1.1, 1.2, 1]})
    names = list(lik)
    colors = ["#222222", "#E69F00", "#F0C060", "#999999", "#777777", "#4C72B0", "#C44E52", "#55A868"][:len(names)]
    axes[0].barh(range(len(names)), [lik[k] for k in names], color=colors)
    axes[0].set_yticks(range(len(names)), names, fontsize=11)
    axes[0].invert_yaxis()
    axes[0].set_xlim(min(lik.values()) - 0.01, max(lik.values()) + 0.005)
    for i, k in enumerate(names):
        axes[0].text(lik[k] + 0.0005, i, f"{lik[k]:.4f}", va="center", fontsize=10)
    axes[0].set_xlabel("eval likelihood on the animals' choices")
    axes[0].set_title("A one-variable model distilled from the GRU", fontsize=13)
    xs = np.linspace(-3.5, 4.5, 50)
    for g, name in enumerate(tr.GROUP_NAMES):
        if name.endswith("(first)") or not np.isfinite(lam[g]):
            continue
        ls = "-" if "(stay)" in name else "--"
        color = {"R rewarded": "#0072B2", "R unrewarded": "#56B4E9", "L rewarded": "#D55E00", "L unrewarded": "#E69F00"}[name.split(" (")[0]]
        axes[1].plot(xs, target[g] + lam[g] * (xs - target[g]), color=color, ls=ls, lw=2,
                     label=f"{name}: λ {lam[g]:.2f}, v* {target[g]:+.2f}")
    axes[1].plot(xs, xs, color="0.6", ls=":", lw=1)
    axes[1].set_xlabel("v(t)")
    axes[1].set_ylabel("v(t+1)")
    axes[1].set_title("Fitted update rules (solid: same side, dashed: switch)", fontsize=13)
    axes[1].legend(frameon=False, fontsize=8.5)
    ses_example = t["ses_idx"].iloc[np.flatnonzero(lay["first"])[len(np.flatnonzero(lay["first"])) // 2]]
    rows = np.flatnonzero((t["ses_idx"] == ses_example).to_numpy())[:200]
    axes[2].plot(v[rows], color="#222222", lw=1.6, label="GRU value coordinate")
    axes[2].plot(v_sim_m[rows], color="#E69F00", lw=1.6, label="distilled model (per mouse), simulated")
    axes[2].plot(v_sim[rows], color="#F0C060", lw=1, ls="--", label="distilled model (pooled)")
    axes[2].set_xlabel("trial")
    axes[2].set_ylabel("value coordinate")
    axes[2].set_title(f"Example session (corr over all trials {sim_vs_gru['corr_v']:.2f})", fontsize=13)
    axes[2].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_I4_distilled_model.png"))
    plt.close(fig)

    s12.write_json("i4_distilled_model.json", {
        "groups": tr.GROUP_NAMES, "pooled": {"lam": lam, "target": target, "n": n, "readout_a": a, "readout_b": b,
                                              "readout_clusters": tr.INPUT_NAMES + ["first trial"], "v0": v0},
        "n_pooled_parameters": int(np.isfinite(lam).sum() * 2 + 1 + len(b) + 1),
        "teacher_forced_readout_r2": float(teacher_r2), "likelihood": lik,
        "per_mouse_frac_beats_ctt": per_mouse_beats_ctt, "simulated_vs_gru": sim_vs_gru,
        "n_mice": len(mouse_params), "min_per_group": tr.MIN_PER_GROUP,
    }, "analysis/i4_distilled_model.py", extra_groups=s12.rl_groups())
    print({k: round(x, 5) for k, x in lik.items()})
    print("teacher readout R2", round(teacher_r2, 3), "| sim vs GRU", {k: round(x, 3) for k, x in sim_vs_gru.items()},
          "| per-mouse beats CTT", per_mouse_beats_ctt)
    print({tr.GROUP_NAMES[g]: (round(lam[g], 2), round(target[g], 2), int(n[g])) for g in range(tr.N_GROUPS)})


if __name__ == "__main__":
    main()
