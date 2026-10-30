"""H4: canonical scenarios — the same scripted choice/reward histories into the GRU and the RL fits.

    PYTHONPATH=<wrapper>/code python analysis/h4_scenarios.py

Every mouse is run from the start of a fresh session through the same scripted history:
  GRU      — the wrapper's ``GruStep`` with the mouse's mean context (as F2), first input
             (−1, −1) as in a real session;
  RL       — the mouse's fitted Hattori, Bari and CTT agents (``perform_closed_loop`` on the
             scripted history, Q starting at 0).
Outputs P(right) before every trial. Scenarios (R = right, + rewarded, − unrewarded):
  A  8 R+, then 8 R−                    — staying through losses after a win streak
  B  8 R+, then L−, then 6 R+           — a failed switch
  C  8 R+, then 6 L+                    — a rewarded switch
  D  (R+, R−) × 8                       — alternating outcomes on one side
  E  8 R+, 4 R−, then 8 L+              — a reversal
Sweeps:
  F  n R+ then L− (n = 1…12): P(return to R) — return after a failed switch vs run length
  G  6 R+ then k L− (k = 1…10): P(stay on L) — staying through losses on a new side
Panels are side by side per model (median and interquartile range across mice). Writes
``h4_scenarios.json`` and ``fig_H4_scenarios.png``.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

RL_RUNS = {"hattori": "unhmbrk4", "bari": "bg3nzqz9", "ctt": "lmg1i9yd"}
MODELS = {"gru": "GRU", "hattori": "Hattori (L2F1)", "bari": "Bari (L1F1_CK1)", "ctt": "CTT"}
COLORS = {"gru": "#222222", "hattori": "#55A868", "bari": "#4C72B0", "ctt": "#C44E52"}


def seq(*parts) -> tuple[np.ndarray, np.ndarray]:
    """parts: (n, side, rewarded) → choices (1 = R), rewards."""
    c = np.concatenate([np.full(n, 1 if side == "R" else 0) for n, side, _ in parts])
    r = np.concatenate([np.full(n, int(rew)) for n, _, rew in parts])
    return c, r


SCENARIOS = {
    "A  8 R+, then 8 R−": seq((8, "R", 1), (8, "R", 0)),
    "B  8 R+, L−, then 6 R+": seq((8, "R", 1), (1, "L", 0), (6, "R", 1)),
    "C  8 R+, then 6 L+": seq((8, "R", 1), (6, "L", 1)),
    "D  (R+, R−) × 8": (np.ones(16, int), np.tile([1, 0], 8)),
    "E  8 R+, 4 R−, then 8 L+": seq((8, "R", 1), (4, "R", 0), (8, "L", 1)),
}
SWEEP_F = {n: seq((n, "R", 1), (1, "L", 0)) for n in range(1, 13)}
SWEEP_G = {k: seq((6, "R", 1), (k, "L", 0)) for k in range(1, 11)}


def rl_p_right(fit: dict, choices: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    """P(right) before trials 0…n (a dummy last trial exposes the probability after the history)."""
    from aind_dynamic_foraging_models import generative_model  # noqa: PLC0415

    agent = getattr(generative_model, fit["agent_class"])(**fit["agent_kwargs"], seed=0)
    agent.set_params(**{k: float(v) for k, v in fit["fitted_params"].items()})
    agent.perform_closed_loop(np.r_[choices, 0], np.r_[rewards, 0])
    return np.asarray(agent.choice_prob, dtype=float)[1, :len(choices) + 1]


def main() -> None:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from post_training_analysis.gru_counterfactual import load_gru_split  # noqa: PLC0415
    from post_training_analysis.gru_step import GruStep  # noqa: PLC0415

    s12.apply_presentation_style()
    split = load_gru_split(s12.CACHE / "models" / s12.MODEL_KEY / "run")
    step = GruStep.from_params(split.params)
    ctx = np.load(s12.CACHE / f"contexts_{s12.MODEL_KEY}.npz")
    t = s12.load_trials(["subject_id", "ses_idx"])
    ses_subject = t.drop_duplicates("ses_idx").set_index("ses_idx")["subject_id"]
    ctx_df = pd.DataFrame(ctx["contexts"], index=ctx["session_ids"])
    ctx_df["subject_id"] = ses_subject.reindex(ctx_df.index).to_numpy()
    mouse_ctx = ctx_df.groupby("subject_id").mean()
    fits = {m: json.loads((s12.CACHE / "models" / "rl" / run / "baseline_rl_output.json").read_text())["fitted_params_per_subject"]
            for m, run in RL_RUNS.items()}
    mice = [m for m in mouse_ctx.index if all(m in f for f in fits.values())]
    c_all = jnp.asarray(mouse_ctx.loc[mice].to_numpy(dtype=np.float32))
    run = jax.jit(lambda o, c: step.logit_diff(step.run(o, c)))

    def gru_p_right(choices, rewards):
        obs = np.vstack([[-1.0, -1.0], np.column_stack([choices, rewards])]).astype(np.float32)  # [n+1, 2]
        o = jnp.broadcast_to(jnp.asarray(obs)[:, None], (len(obs), len(mice), 2))
        cc = jnp.broadcast_to(c_all[None], (len(obs), len(mice), c_all.shape[1]))
        return 1 / (1 + np.exp(-np.asarray(run(o, cc))))  # [n+1, mice]: before trials 0…n

    def all_models(choices, rewards) -> dict:
        out = {"gru": gru_p_right(choices, rewards).T}
        for m in RL_RUNS:
            out[m] = np.stack([rl_p_right(fits[m][s], choices, rewards) for s in mice])
        return out  # each [mice, n+1]

    traces = {name: all_models(*sc) for name, sc in SCENARIOS.items()}
    print("scenarios done", flush=True)
    sweep_f = {n: all_models(*sc) for n, sc in SWEEP_F.items()}
    sweep_g = {k: all_models(*sc) for k, sc in SWEEP_G.items()}
    # Probe = P before the trial after the history: return to R (F) and stay on L (G).
    f_vals = {m: np.stack([sweep_f[n][m][:, -1] for n in SWEEP_F], axis=1) for m in MODELS}
    g_vals = {m: np.stack([1 - sweep_g[k][m][:, -1] for k in SWEEP_G], axis=1) for m in MODELS}

    rows = len(SCENARIOS) + 2
    fig, axes = plt.subplots(rows, 4, figsize=(28, 4.6 * rows), sharey=True)
    for r_, (name, (choices, rewards)) in enumerate(SCENARIOS.items()):
        x = np.arange(len(choices) + 1)
        for c_, m in enumerate(MODELS):
            ax = axes[r_, c_]
            v = traces[name][m]
            ax.fill_between(x, np.percentile(v, 25, 0), np.percentile(v, 75, 0), color=COLORS[m], alpha=0.2)
            ax.plot(x, np.median(v, 0), color=COLORS[m], lw=2.2, marker="o", ms=3)
            for i, (ch, rw) in enumerate(zip(choices, rewards)):
                ax.scatter(i, 1.04 if ch == 1 else -0.04, marker="|" if rw else "x", s=60 if rw else 25,
                           color="#0072B2" if ch == 1 else "#D55E00")
            ax.axhline(0.5, color="0.8", lw=0.8)
            ax.set_ylim(-0.08, 1.08)
            if r_ == 0:
                ax.set_title(MODELS[m], fontsize=15)
            if c_ == 0:
                ax.set_ylabel(f"{name}\nP(right) before trial")
            ax.set_xlabel("trial (ticks: scripted choice; | rewarded, × unrewarded)")
    for r_, (vals, xs, xlabel, ylabel) in enumerate((
            (f_vals, list(SWEEP_F), "n rewarded R choices, then one unrewarded L", "F  P(return to R)"),
            (g_vals, list(SWEEP_G), "k unrewarded L choices after 6 rewarded R", "G  P(stay on L)")), start=len(SCENARIOS)):
        for c_, m in enumerate(MODELS):
            ax = axes[r_, c_]
            v = vals[m]
            ax.fill_between(xs, np.percentile(v, 25, 0), np.percentile(v, 75, 0), color=COLORS[m], alpha=0.2)
            ax.plot(xs, np.median(v, 0), color=COLORS[m], lw=2.2, marker="o", ms=4)
            ax.axhline(0.5, color="0.8", lw=0.8)
            ax.set_xlabel(xlabel)
            if c_ == 0:
                ax.set_ylabel(ylabel)
    fig.suptitle("Identical scripted histories from a fresh session: median and IQR across mice", fontsize=18, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(s12.figure_path("fig_H4_scenarios.png"))
    plt.close(fig)

    def med(v):
        return np.median(v, 0).round(4).tolist()

    s12.write_json("h4_scenarios.json", {
        "n_mice": len(mice),
        "scenarios": {name: {"choices": SCENARIOS[name][0].tolist(), "rewards": SCENARIOS[name][1].tolist(),
                             "median_p_right": {m: med(traces[name][m]) for m in MODELS}} for name in SCENARIOS},
        "sweep_F_return_after_failed_switch": {"n": list(SWEEP_F), "median": {m: med(f_vals[m]) for m in MODELS}},
        "sweep_G_stay_through_losses_new_side": {"k": list(SWEEP_G), "median": {m: med(g_vals[m]) for m in MODELS}},
        "alternation_zigzag_D": {m: float(np.median(np.abs(np.diff(traces["D  (R+, R−) × 8"][m][:, 6:], axis=1)).mean(1)))
                                 for m in MODELS},
        "context": "per-mouse mean of its eval-session contexts", "start": "fresh session (GRU h0 = 0; Q-learning Q = 0; CTT at its initial state)",
    }, "analysis/h4_scenarios.py", extra_groups=s12.rl_groups())
    for m in MODELS:
        print(m, "F", med(f_vals[m]), "\n   G", med(g_vals[m]))


if __name__ == "__main__":
    main()
