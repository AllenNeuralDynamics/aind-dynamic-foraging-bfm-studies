"""F4 (behavioural half): does the GRU follow each mouse's change between early and late sessions?

    python analysis/f4_early_late.py

Each mouse's eval sessions are split at the median training-session index into early and
late halves; C1's Su2022 logistic fit is repeated per half for the animal (choices) and
the GRU (P(right) soft labels). For bias, reward gain and persistence the late − early
change of the GRU is compared with the animal's across mice. Mice move through tasks as they
train, so the split is also repeated within each mouse's most common task only.

Noise ceiling: the animal's Δ is estimated from sampled choices, the GRU's from soft labels,
so even a perfect model would not reach Spearman 1. Refitting the GRU with choices *sampled*
from its own P(right) gives the Spearman a perfect model would reach at this noise level
(Spearman of sampled-Δ vs soft-Δ). Six features x two splits are tested, so single p-values
around 0.01 should not be read as significant on their own.
Writes ``fig_F4_early_late.png`` and ``f4_early_late.json``; caches the per-mouse deltas and the
session halves in ``_cache/f4_mouse_deltas_<key>.parquet`` / ``_cache/f4_session_halves_<key>.parquet``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
from c1_kernels import N_BACK, design, fit_rows  # noqa: E402

MIN_ROWS = 1500  # per mouse x half x source, for a usable 31-parameter logistic fit
FEATURES = ("bias", "reward gain (1 back)", "persistence (1 back)")


def features(coef: np.ndarray) -> dict:
    rew, unr = (coef[i * N_BACK:(i + 1) * N_BACK][::-1] for i in range(2))  # Su2022 terms
    return {"bias": coef[-1], "reward gain (1 back)": rew[0] - unr[0], "persistence (1 back)": (rew[0] + unr[0]) / 2}


def main() -> None:
    s12.apply_presentation_style()
    x, t = design()
    order = s12.load_trials(["ses_idx", "subject_session_index"]).drop_duplicates("ses_idx").set_index("ses_idx")
    t["session_order"] = t["ses_idx"].map(order["subject_session_index"]).to_numpy()
    sessions = t.drop_duplicates("ses_idx")[["subject_id", "ses_idx", "task", "session_order"]]
    dominant = sessions.groupby("subject_id")["task"].agg(lambda s: s.value_counts().index[0])

    def halves(subset: pd.DataFrame) -> pd.Series:
        median = subset.groupby("subject_id")["session_order"].transform("median")
        return pd.Series(np.where(subset["session_order"] <= median, "early", "late"), index=subset["ses_idx"])

    splits = {"all sessions": halves(sessions),
              "dominant task only": halves(sessions[sessions["task"] == sessions["subject_id"].map(dominant)])}
    sampled = (np.random.default_rng(0).random(len(t)) < t["p_model1_right"].to_numpy()).astype(float)
    targets = {"animal": (t["action"].to_numpy().astype(float), False), "gru": (t["p_model1_right"].to_numpy(), True),
               "gru_sampled": (sampled, False)}
    results, figure_data, mouse_rows = {}, {}, []
    for split_name, half_of_session in splits.items():
        half = t["ses_idx"].map(half_of_session).to_numpy()
        jobs = []
        for subject, rows in t.groupby("subject_id", sort=False).indices.items():
            parts = {h: rows[half[rows] == h] for h in ("early", "late")}
            if all(len(p) >= MIN_ROWS for p in parts.values()):
                jobs += [(subject, h, source, parts[h]) for h in parts for source in targets]
        coefs = Parallel(n_jobs=24)(delayed(fit_rows)(x, targets[src][0], rows, targets[src][1]) for _, _, src, rows in jobs)
        table = pd.DataFrame([{"subject_id": s, "half": h, "source": src, **features(c)}
                              for (s, h, src, _), c in zip(jobs, coefs)])
        wide = table.pivot_table(index="subject_id", columns=["source", "half"], values=list(FEATURES))
        entry = {"n_mice": int(len(wide))}
        for f in FEATURES:
            delta = {src: (wide[(f, src, "late")] - wide[(f, src, "early")]).to_numpy() for src in targets}
            rho, p = spearmanr(delta["animal"], delta["gru"])
            ceiling, _ = spearmanr(delta["gru_sampled"], delta["gru"])
            entry[f] = {"spearman_delta_gru_vs_animal": float(rho), "p": float(p),
                        "noise_ceiling_spearman": float(ceiling),
                        **{f"median_delta_{src}": float(np.median(delta[src])) for src in targets},
                        **{f"wilcoxon_p_{src}": float(wilcoxon(delta[src]).pvalue) for src in targets}}
            figure_data[(split_name, f)] = delta
            mouse_rows += [{"split": split_name, "feature": f, "source": src, "subject_id": sid, "delta": float(dv)}
                           for src in targets for sid, dv in zip(wide.index, delta[src])]
        results[split_name] = entry

    # Per-mouse deltas and the session halves, for the context half of F4 (f4b_context_drift.py).
    pd.DataFrame(mouse_rows).to_parquet(s12.CACHE / f"f4_mouse_deltas_{s12.MODEL_KEY}.parquet")
    pd.concat({k: v.rename("half") for k, v in splits.items()}, names=["split", "ses_idx"]).reset_index().to_parquet(
        s12.CACHE / f"f4_session_halves_{s12.MODEL_KEY}.parquet")

    fig, axes = plt.subplots(2, len(FEATURES), figsize=(18, 11))
    for row, split_name in enumerate(splits):
        for ax, f in zip(axes[row], FEATURES):
            d = figure_data[(split_name, f)]
            lim = np.nanpercentile(np.abs(np.r_[d["animal"], d["gru"]]), 99)
            ax.scatter(d["animal"], d["gru"], s=9, alpha=0.5, color="#555555")
            ax.plot([-lim, lim], [-lim, lim], color="0.6", ls=":", lw=1)
            ax.axhline(0, color="0.85", lw=0.8)
            ax.axvline(0, color="0.85", lw=0.8)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            r = results[split_name][f]
            ax.set_title(f"{f} — {split_name}\nSpearman {r['spearman_delta_gru_vs_animal']:.2f} "
                         f"(noise ceiling {r['noise_ceiling_spearman']:.2f}), n={results[split_name]['n_mice']}", fontsize=12)
            ax.set_xlabel("animal: late − early")
            ax.set_ylabel("GRU: late − early")
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_F4_early_late.png"))
    plt.close(fig)
    s12.write_json("f4_early_late.json", {"split": "per mouse, at the median training-session index of its eval sessions",
                                          "min_rows": MIN_ROWS, "design": f"Su2022, {N_BACK} back",
                                          "results": results}, "analysis/f4_early_late.py")
    for k, v in results.items():
        print(k, v["n_mice"], {f: (round(v[f]["spearman_delta_gru_vs_animal"], 3), round(v[f]["noise_ceiling_spearman"], 3),
                                   round(v[f]["median_delta_animal"], 3), round(v[f]["median_delta_gru"], 3),
                                   round(v[f]["median_delta_gru_sampled"], 3)) for f in FEATURES})


if __name__ == "__main__":
    main()
