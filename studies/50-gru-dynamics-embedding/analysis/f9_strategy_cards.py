"""F9: a strategy profile for every mouse from its embedding — table, prototypes, cards.

    PYTHONPATH=<wrapper>/code python analysis/f9_strategy_cards.py [--mouse SUBJECT_ID ...]

Uses F7's four named embedding axes (varimax-rotated, whitened embedding; unit variance across
mice) and names them from F7's loadings and their correlations with F5's per-mouse table:
  axis 1  determinism / stickiness — history-driven, sticky, few switches (GRU likelihood, depth)
  axis 2  side bias
  axis 3  slow learning / long choice memory — low RL learning rates, high v retention,
          more poorer-side choices and returns after failed switches
  axis 4  dynamics-only — v retention ↓ and attractor asymmetry ↑, little behavioural signature
(the names are asserted against F7's loadings so a re-run that reorders axes fails loudly).
Outputs:
  * ``f9_mouse_strategy_profiles.csv`` — per mouse: axis coordinates and percentiles, and the
    animal-source F7 features with population percentiles;
  * ``fig_F9_strategy_prototypes.png`` — for each axis, the mice nearest its 5th and 95th
    percentile (others near the median, ≥ 20 sessions), side by side: animal and GRU history
    kernels and the F7 features;
  * ``_cache/explore/cards/card_<subject>.png`` — one card per mouse (default: 12 examples
    spread over the axes; or ``--mouse``): axis percentiles, animal vs GRU kernels, features vs
    population, an example session, and the context trajectory across training on the named axes.
"""

from __future__ import annotations

import argparse
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

AXIS_NAMES = ["determinism / stickiness", "side bias", "slow learning / long choice memory", "dynamics-only (v retention ↓)"]
AXIS_CHECK = [("determinism", +1), ("bias", +1), ("choice memory span", +1), None]  # top-loading feature and sign
N_BACK = 10


def kernels(coef: list[float]) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(coef)
    return c[:N_BACK][::-1], c[N_BACK:2 * N_BACK][::-1]  # RewC, UnrC at lags 1..10


def main() -> None:
    from f7_feature_library import FEATURES  # noqa: PLC0415

    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse", nargs="*", default=None)
    args = parser.parse_args()
    s12.apply_presentation_style()

    f7 = json.loads((s12.ANALYSIS / "f7_feature_library.json").read_text())
    load = np.array([f7["named_axes"]["loadings"][f] for f in FEATURES])
    for k, chk in enumerate(AXIS_CHECK):
        if chk:
            f, sign = chk
            if FEATURES[int(np.argmax(np.abs(load[:, k])))] != f or np.sign(load[FEATURES.index(f), k]) != sign:
                raise AssertionError(f"F7 axis {k + 1} no longer matches its name ({AXIS_NAMES[k]}); rename")
    axes_df = pd.read_parquet(s12.CACHE / f"f7_named_axes_{s12.MODEL_KEY}.parquet")
    axes_df.columns = AXIS_NAMES
    feats = pd.read_parquet(s12.CACHE / f"f7_mouse_features_{s12.MODEL_KEY}.parquet")
    animal = feats.query("part == 'all' and source == 'animal'").set_index("subject_id")[FEATURES].loc[axes_df.index]
    prof = pd.concat([axes_df.add_prefix("z: "), axes_df.rank(pct=True).add_prefix("pct: "),
                      animal.add_prefix("animal: "), animal.rank(pct=True).add_prefix("animal pct: ")], axis=1)
    t = s12.load_trials(["subject_id", "ses_idx", "trial_idx", "action", "reward", "p_model1_right",
                         "reward_probabilityL", "reward_probabilityR", "task"])
    n_sess = t.groupby("subject_id")["ses_idx"].nunique()
    prof["n eval sessions"] = n_sess.reindex(prof.index)
    task = t.groupby("subject_id")["task"].agg(lambda s: s.value_counts().index[0])
    prof["main task"] = task.reindex(prof.index)
    prof.index.name = "subject_id"
    prof.round(4).to_csv(s12.ANALYSIS / "f9_mouse_strategy_profiles.csv")
    coefs = json.loads((s12.ANALYSIS / "c1_kernels.json").read_text())["per_mouse_coef"]

    # Prototypes: per axis, mice near the 5th / 95th percentile with others near the median.
    eligible = prof.index[prof["n eval sessions"] >= 20]
    z = axes_df.loc[eligible]
    proto = {}
    for k, name in enumerate(AXIS_NAMES):
        others = z.drop(columns=name).abs().sum(axis=1)
        for end, q in (("low", 0.05), ("high", 0.95)):
            target = z[name].quantile(q)
            score = (z[name] - target).abs() + 0.5 * others
            proto[(name, end)] = score.idxmin()
    fig, axs = plt.subplots(4, 3, figsize=(26, 24), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    lags = np.arange(1, N_BACK + 1)
    for r, name in enumerate(AXIS_NAMES):
        for c, end in enumerate(("low", "high")):
            m = proto[(name, end)]
            ax = axs[r, c]
            for src, ls in (("animal", "-"), ("gru", "--")):
                if m in coefs[src]:
                    rew, unr = kernels(coefs[src][m])
                    ax.plot(lags, rew, color="#0072B2", ls=ls, marker="o", ms=3, label=f"{src}: rewarded choice")
                    ax.plot(lags, unr, color="#D55E00", ls=ls, marker="o", ms=3, label=f"{src}: unrewarded choice")
            ax.axhline(0, color="0.7", lw=0.8)
            ax.set_xlabel("trials back")
            ax.set_ylabel("logistic weight (repeat that side)")
            ax.set_title(f"{name}: {end} ({100 * prof.at[m, 'pct: ' + name]:.0f}th pct) — mouse {m}, "
                         f"{prof.at[m, 'main task']}", fontsize=11)
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=9)
        ax = axs[r, 2]
        lo, hi = proto[(name, "low")], proto[(name, "high")]
        y = np.arange(len(FEATURES))
        ax.barh(y - 0.2, prof.loc[lo, [f"animal pct: {f}" for f in FEATURES]].to_numpy(dtype=float), 0.4, color="#4C72B0", label=f"low ({lo})")
        ax.barh(y + 0.2, prof.loc[hi, [f"animal pct: {f}" for f in FEATURES]].to_numpy(dtype=float), 0.4, color="#E69F00", label=f"high ({hi})")
        ax.axvline(0.5, color="0.6", lw=0.8)
        ax.set_yticks(y, FEATURES, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("animal feature, percentile across mice")
        ax.set_title(f"{name}: low vs high prototype", fontsize=11)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_F9_strategy_prototypes.png"))
    plt.close(fig)

    # Cards.
    if args.mouse:
        card_mice = args.mouse
    else:
        pick = []
        for name in AXIS_NAMES[:3]:
            for end in ("low", "high"):
                pick.append(proto[(name, end)])
        pick += list(prof.loc[eligible].sample(6, random_state=0).index)
        card_mice = list(dict.fromkeys(pick))
    traj = None
    try:
        from post_training_analysis.gru_counterfactual import load_gru_split  # noqa: PLC0415
        from post_training_analysis.gru_step import subject_context  # noqa: PLC0415

        cmap = json.loads((s12.CACHE / "models" / s12.MODEL_KEY / "run" / "outputs" / "session_context_map.json").read_text())
        split = load_gru_split(s12.CACHE / "models" / s12.MODEL_KEY / "run")
        npz = np.load(s12.CACHE / f"f7_named_axes_{s12.MODEL_KEY}.npz")
        rows = [(e["subject_id"], e["subject_index"], s + 1, len(e["ordered_session_ids"])) for e in cmap["per_subject"]
                if e["subject_id"] in card_mice for s in range(len(e["ordered_session_ids"]))]
        traj = pd.DataFrame(rows, columns=["subject_id", "i", "s", "n"])
        c = subject_context(split.make_network, split.params, traj["i"].to_numpy(), traj["s"].to_numpy())
        coords = ((c - npz["mean"]) @ npz["whiten"]) @ npz["rotation"]
        for k, name in enumerate(AXIS_NAMES):
            traj[name] = coords[:, k]
    except Exception as exc:  # noqa: BLE001 — the card is still useful without the trajectory
        print("context trajectories unavailable:", exc, flush=True)
    out_dir = s12.EXPLORE / "cards"
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in card_mice:
        fig, axs = plt.subplots(2, 3, figsize=(26, 12))
        ax = axs[0, 0]
        pct = [prof.at[m, f"pct: {n}"] for n in AXIS_NAMES]
        ax.barh(range(4), pct, color=["#222222", "#C44E52", "#E69F00", "#999999"])
        ax.axvline(0.5, color="0.6", lw=0.8)
        ax.set_yticks(range(4), AXIS_NAMES, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("percentile across 614 mice")
        ax.set_title(f"Mouse {m}: position on the embedding axes ({int(prof.at[m, 'n eval sessions'])} eval sessions, {prof.at[m, 'main task']})",
                     fontsize=12)
        ax = axs[0, 1]
        for src, ls in (("animal", "-"), ("gru", "--")):
            if m in coefs[src]:
                rew, unr = kernels(coefs[src][m])
                ax.plot(lags, rew, color="#0072B2", ls=ls, marker="o", ms=3, label=f"{src}: rewarded choice")
                ax.plot(lags, unr, color="#D55E00", ls=ls, marker="o", ms=3, label=f"{src}: unrewarded choice")
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_xlabel("trials back")
        ax.set_ylabel("logistic weight")
        ax.set_title("History kernels: animal (solid) vs GRU (dashed)", fontsize=12)
        ax.legend(frameon=False, fontsize=9)
        ax = axs[0, 2]
        y = np.arange(len(FEATURES))
        ax.barh(y, prof.loc[m, [f"animal pct: {f}" for f in FEATURES]].to_numpy(dtype=float), color="#4C72B0")
        ax.axvline(0.5, color="0.6", lw=0.8)
        ax.set_yticks(y, [f"{f} = {prof.at[m, 'animal: ' + f]:.2f}" for f in FEATURES], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("percentile across mice (animal's own choices)")
        ax.set_title("Behavioural features", fontsize=12)
        d = t[t["subject_id"] == m]
        ses = d["ses_idx"].unique()[len(d["ses_idx"].unique()) // 2]
        d = d[d["ses_idx"] == ses].head(150)
        ax = fig.add_subplot(2, 3, (4, 5))
        axs[1, 0].remove()
        axs[1, 1].remove()
        x = np.arange(len(d))
        ax.plot(x, d["reward_probabilityR"], color="#0072B2", alpha=0.5, label="P(reward | R)")
        ax.plot(x, d["reward_probabilityL"], color="#D55E00", alpha=0.5, label="P(reward | L)")
        ax.plot(x, d["p_model1_right"], color="k", lw=1.5, label="GRU P(right)")
        ch, rw = d["action"].to_numpy(), d["reward"].to_numpy() > 0
        ax.vlines(x[ch == 1], 1.02, np.where(rw[ch == 1], 1.12, 1.06), color="#0072B2")
        ax.vlines(x[ch == 0], -0.02, np.where(rw[ch == 0], -0.12, -0.06), color="#D55E00")
        ax.set_ylim(-0.15, 1.15)
        ax.set_xlabel("trial (ticks: choice; tall = rewarded)")
        ax.set_title(f"Example session {ses}", fontsize=12)
        ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower center")
        ax = axs[1, 2]
        if traj is not None and (traj["subject_id"] == m).any():
            tm = traj[traj["subject_id"] == m].sort_values("s")
            for name, color in zip(AXIS_NAMES, ("#222222", "#C44E52", "#E69F00", "#999999")):
                ax.plot(tm["s"] / tm["n"], tm[name], color=color, lw=2, label=name)
            ax.axhline(0, color="0.8", lw=0.8)
            ax.set_xlabel("training phase (session / number of sessions)")
            ax.set_ylabel("context on the named axis (z across mice)")
            ax.set_title("Context across training", fontsize=12)
            ax.legend(frameon=False, fontsize=9)
        else:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"card_{m}.png")
        plt.close(fig)
    s12.write_json("f9_strategy_cards.json", {
        "axis_names": AXIS_NAMES, "prototypes": {f"{n} | {e}": m for (n, e), m in proto.items()},
        "cards": card_mice, "profiles_csv": "analysis/f9_mouse_strategy_profiles.csv",
        "n_mice": int(len(prof)),
    }, "analysis/f9_strategy_cards.py")
    print("prototypes", {f"{n} | {e}": m for (n, e), m in proto.items()}, "cards", card_mice)


if __name__ == "__main__":
    main()
