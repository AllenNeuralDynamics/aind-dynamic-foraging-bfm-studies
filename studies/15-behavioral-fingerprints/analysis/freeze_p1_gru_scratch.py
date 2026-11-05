#!/usr/bin/env python
"""P1 scratch: why the GRU leads on the subject-mean curves but not on the floor ratio.

Exploratory, for r2's scratch block. On the 146 test subjects every model holds out, against the
best literature models:

1. **Two errors** per model and rate statistic, both trial-weighted over the subject's kept bins
   (at least ``MIN_TRIALS``): the error of the subject-mean curve, sum_b N_b |P_b - O_b| / sum_b N_b
   (P_b, O_b the subject means of predicted and observed rate, N_b the bin's trials over subjects),
   and the mean over subjects of each subject's own gap (``replay_gap``), beside the binomial floor.
2. **Paired per subject**, the GRU against Lopez-DoubleTrace as 2-level EM: D = the mean over the
   four rate statistics of log(GRU gap / Lopez gap), its sign test, Wilcoxon test and the median
   ratio with a bootstrap interval.
3. **Session spread**: does D grow with the subject's session-to-session spread? Two measures:
   Lopez-DoubleTrace's 2-level mean log sigma (its fit), and the mean over statistics of
   log(split-half floor / binomial floor) (the data; ``noise_floor`` includes the differences
   between sessions, ``binomial_floor`` does not). Control: the same measures against Lopez's gain
   from per-session parameters, the mean log(per-subject gap / 2-level gap).

Needs /allen (the MEvolve freeze, study 12's 2-level file) and the library; on a compute node:

    srun --ntasks=1 --cpus-per-task=4 --mem=16G -p aind --time=00:30:00 \\
      env PYTHONPATH=<foraging-models>/src:<mevolve>/src <mevolve env>/bin/python \\
      analysis/freeze_p1_gru_scratch.py --models-repo <foraging-models>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
sys.path.insert(0, str(HERE))
from _meta import build_meta, write_artifact  # noqa: E402
from freeze_p0_golden_mevolve import FREEZE, FREEZE_MANIFEST, _git_sha, _sha256  # noqa: E402

GAPS, GRU, GROUPS = HERE / "p1_replay_gaps.json", HERE / "p1_gru.json", HERE / "p1_session_groups.json"
TALLIES = HERE / "p1_replay"
TWO_LEVEL_INDEX = (HERE.parents[1] / "12-cognitive-literature-leaderboard" / "analysis"
                   / "two_level_em_fits.json")
OUT = HERE / "p1_gru_scratch.json"
STATS = ("losses", "pattern", "hazard", "post_switch")
MODELS = (("GRU-D614", "per_subject"), ("Lopez-DoubleTrace", "two_level_em"),
          ("Lebedeva-PR", "two_level_em"), ("CompareToThreshold", "per_subject"),
          ("GLM-Su2022", "per_subject"))
N_BOOT, SEED = 5000, 0


def _errors(common: list[str], heads: set[str], min_trials: int) -> dict:
    """Per model and statistic: the subject-mean curve's error and the mean per-subject gap."""
    import pandas as pd

    animal = pd.read_parquet(TALLIES / "animal.parquet")
    animal = animal[animal.group.isin(heads) & animal.subject_id.isin(common) & animal.statistic.isin(STATS)]
    an = animal.groupby(["subject_id", "statistic", "bin"])[["n", "obs"]].sum()
    out = {}
    for m, reg in MODELS:
        d = pd.read_parquet(TALLIES / f"{m}.parquet")
        d = d[d.group.isin(heads) & d.subject_id.isin(common) & d.statistic.isin(STATS)]
        # the GRU's per-subject column is the mean over its seeds, as in the gaps freeze: its gap
        # is each seed's gap averaged; its mean curve averages the seeds' predictions (linear)
        d = d[d.regime.str.startswith("seed")] if m == "GRU-D614" else d[d.regime == reg]
        pred = d.groupby(["regime", "subject_id", "statistic", "bin"]).pred.sum()
        out[m] = {}
        for st in STATS:
            gaps, curves = [], []
            for regime, one in pred[pred.index.get_level_values("statistic") == st].groupby(level="regime"):
                t = an.join(one.droplevel("regime")).reset_index()
                t = t[(t.statistic == st) & (t.n >= min_trials)]
                t["o"], t["p"] = t.obs / t.n, t.pred / t.n
                gaps.append(t.groupby("subject_id").apply(
                    lambda x: float(np.sum(x.n * np.abs(x.p - x.o)) / x.n.sum())))
                curves.append(t.groupby("bin").agg(N=("n", "sum"), P=("p", "mean"), O=("o", "mean")))
            gap = sum(gaps) / len(gaps)
            b = curves[0].assign(P=sum(c.P for c in curves) / len(curves))
            out[m][st] = {"subject_mean_curve_error": float(np.sum(b.N * np.abs(b.P - b.O)) / b.N.sum()),
                          "mean_subject_gap": float(gap.mean()), "n_subjects": int(gap.size),
                          "n_regimes": len(gaps)}
    return out


def _rho(x: np.ndarray, y: np.ndarray, rng) -> dict:
    from scipy.stats import spearmanr

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    boot = [spearmanr(x[i], y[i]).statistic for i in (rng.integers(0, len(x), len(x)) for _ in range(N_BOOT))]
    return {"spearman_rho": float(spearmanr(x, y).statistic), "ci95": [float(np.percentile(boot, q)) for q in (2.5, 97.5)],
            "n": int(ok.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-repo", type=Path, required=True)
    a = ap.parse_args()
    from mevolve.data import Freeze
    from scipy.stats import binomtest, wilcoxon

    from aind_dynamic_foraging_models.fingerprints import MIN_TRIALS, noise_floor, session_tally

    gaps, gru = json.loads(GAPS.read_text()), json.loads(GRU.read_text())
    groups = json.loads(GROUPS.read_text())["groups"]
    head = gaps["headline_set"]
    heads, subs = set(gaps["sets"][head]), gaps["subjects"]
    trained = {s for g in gru["seeds"].values() for s in g["subjects_in_training"]}
    common = [s for s in subs if s not in trained]

    def cell(m, reg, st, key, s):
        v = gaps["models"][m][reg][head][st]["per_subject"][key][subs.index(s)]
        return np.nan if v is None else v

    run = next(r for r in json.loads(TWO_LEVEL_INDEX.read_text())["runs"] if r["literature_model"] == "Lopez-DoubleTrace")
    lopez_path = Path(run["path"])
    assert _sha256(lopez_path) == run["sha256"], "Lopez-DoubleTrace's 2-level file differs from study 12's"
    lopez = {str(r["subject_id"]): r for r in json.loads(lopez_path.read_text())["subjects"]}
    freeze = Freeze(FREEZE)
    assert _sha256(FREEZE / "manifest.json") == FREEZE_MANIFEST, "another data freeze"

    rows = []
    for s in common:
        sessions = [x for x in freeze.subject_sessions(s, "score") if groups[x.session_id] in heads]
        tallies = [session_tally(x.choices, x.rewards) for x in sessions]
        d, e, sh = [], [], []
        for st in STATS:
            floor = cell("GRU-D614", "per_subject", st, "floor", s)
            d.append(np.log(cell("GRU-D614", "per_subject", st, "gap", s)
                            / cell("Lopez-DoubleTrace", "two_level_em", st, "gap", s)))
            e.append(np.log(cell("Lopez-DoubleTrace", "per_subject", st, "gap", s)
                            / cell("Lopez-DoubleTrace", "two_level_em", st, "gap", s)))
            sh.append(np.log(noise_floor(tallies, st) / floor))
        rows.append({"subject_id": s, "gru_vs_lopez_2l": float(np.nanmean(d)),
                     "lopez_per_subject_vs_2l": float(np.nanmean(e)),
                     "lopez_mean_log_sigma": float(np.mean(lopez[s]["log_sigma"])),
                     "log_split_over_binomial_floor": float(np.nanmean(sh)),
                     "n_trials": int(sum(len(x.choices) for x in sessions)), "n_sessions": len(sessions)})

    rng = np.random.default_rng(SEED)
    D = np.array([r["gru_vs_lopez_2l"] for r in rows])
    E = np.array([r["lopez_per_subject_vs_2l"] for r in rows])
    col = {k: np.array([r[k] for r in rows], float) for k in
           ("lopez_mean_log_sigma", "log_split_over_binomial_floor", "n_trials", "n_sessions")}
    ok = np.isfinite(D)
    boot = [np.median(rng.choice(D[ok], ok.sum())) for _ in range(N_BOOT)]
    paired = {"n_subjects": int(ok.sum()), "n_gru_worse": int((D[ok] > 0).sum()),
              "sign_test_p": float(binomtest(int((D[ok] > 0).sum()), int(ok.sum())).pvalue),
              "wilcoxon_p": float(wilcoxon(D[ok]).pvalue),
              "median_ratio": float(np.exp(np.median(D[ok]))),
              "median_ratio_ci95": [float(np.exp(np.percentile(boot, q))) for q in (2.5, 97.5)]}
    spread = {"test": {k: _rho(v, D, rng) for k, v in col.items()},
              "control": {k: _rho(col[k], E, rng) for k in ("lopez_mean_log_sigma", "log_split_over_binomial_floor")},
              "measures_agree": _rho(col["lopez_mean_log_sigma"], col["log_split_over_binomial_floor"], rng),
              "control_median_ratio": float(np.exp(np.nanmedian(E)))}
    floors = {st: float(np.nanmean([cell("GRU-D614", "per_subject", st, "floor", s) for s in common])) for st in STATS}

    meta = build_meta("analysis/freeze_p1_gru_scratch.py", [], study_root=HERE.parent, kind="freeze",
                      code=[__file__, HERE / "freeze_p0_golden_mevolve.py"],
                      inputs=[GAPS, GRU, GROUPS, TWO_LEVEL_INDEX, TALLIES / "animal.parquet",
                              *(TALLIES / f"{m}.parquet" for m, _ in MODELS)],
                      sources=[{"type": "file", "id": str(FREEZE / "manifest.json"), "sha256": FREEZE_MANIFEST},
                               {"type": "file", "id": str(lopez_path), "sha256": run["sha256"]}])
    write_artifact(OUT, {
        "_meta": meta,
        "fingerprints_library": {"repo": "AllenNeuralDynamics/aind-dynamic-foraging-models",
                                 "git_sha": _git_sha(a.models_repo)},
        "headline_set": head, "n_subjects": len(common), "min_trials": MIN_TRIALS,
        "models": [m for m, _ in MODELS], "regimes": dict(MODELS), "binomial_floor": floors,
        "errors": _errors(common, heads, MIN_TRIALS), "paired": paired, "spread": spread, "subjects": rows,
    })
    print(json.dumps({"paired": paired, "spread_test": spread["test"]}, indent=1))


if __name__ == "__main__":
    main()
