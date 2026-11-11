#!/usr/bin/env python
"""P2: the GRU against the best literature models per mouse and on the average mouse, in rollout.

The rollout counterpart of ``freeze_p1_gru_scratch.py``, with its formulas, on the subjects every
model holds out (the GRU's checkpoint trained on 3 of the 149) and the rollout headline set:

1. **Two errors** per model and rate statistic, both trial-weighted over the subject's kept bins:
   the error of the subject-mean curve, sum_b N_b |P_b - O_b| / sum_b N_b (P_b, O_b the subject
   means of the rollouts' and the subject's rate, N_b the bin's trials over subjects), and the mean
   over subjects of each subject's rollout gap; beside the mean split-half floor.
2. **Paired per subject**, the GRU against Lopez-DoubleTrace as 2-level EM: D = the mean over the
   four rate statistics of log(GRU gap / Lopez gap), its sign test, Wilcoxon test and the median
   ratio with a bootstrap interval.
3. **Session spread**: does D grow with the subject's session-to-session spread? The two measures
   are properties of the mouse and of Lopez-DoubleTrace's fit, not of the mode, so they are read
   from ``p1_gru_scratch.json``: the fit's mean log sigma and log(split-half / binomial floor).
   Control: the same measures against Lopez's gain from per-session parameters in rollout.

The GRU's numbers are the mean over its three seeds: its gap is each seed's gap averaged, its curve
the seeds' rates averaged. Reads committed files only (the rollout tallies need pyarrow, which CI
lacks, so it runs as a ``freeze-*`` target, not in ``all``); on a compute node:

    env PYTHONPATH=<foraging-models>/src <mevolve env>/bin/python analysis/freeze_p2_gru_scratch.py \\
      --models-repo <foraging-models>
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
from freeze_p0_golden_mevolve import _git_sha  # noqa: E402
from freeze_p1_gru_scratch import MODELS, N_BOOT, SEED, STATS, _rho  # noqa: E402

GAPS, P1, SPECS = HERE / "p2_rollout_gaps.json", HERE / "p1_gru_scratch.json", HERE / "p2_task_specs.json"
TALLIES = HERE / "p2_rollouts"
OUT = HERE / "p2_gru_scratch.json"
GRU = "GRU-D614"


def _rates(df, keys):
    """Per (keys..., bin): summed n and obs."""
    return df.groupby([*keys, "bin"])[["n", "obs"]].sum()


def _errors(common: list[str], heads: set[str], min_trials: int, per_subject_gap) -> dict:
    import pandas as pd

    animal = pd.read_parquet(TALLIES / "animal_sessions.parquet")
    animal = animal[animal.group.isin(heads) & animal.subject_id.isin(common) & animal.statistic.isin(STATS)]
    an = _rates(animal, ["subject_id", "statistic"]).reset_index()
    out = {}
    for m, reg in MODELS:
        d = pd.read_parquet(TALLIES / f"{m}.parquet")
        d = d[d.group.isin(heads) & d.subject_id.isin(common) & d.statistic.isin(STATS)]
        d = d[d.regime.str.startswith("seed")] if m == GRU else d[d.regime == reg]
        roll = _rates(d, ["regime", "subject_id", "statistic"]).reset_index()
        out[m] = {}
        for st in STATS:
            a = an[(an.statistic == st) & (an.n >= min_trials)].assign(o=lambda x: x.obs / x.n)
            curves = []
            for _, one in roll[roll.statistic == st].groupby("regime"):
                t = a.merge(one[["subject_id", "bin", "n", "obs"]].rename(columns={"n": "mn", "obs": "mobs"}),
                            on=["subject_id", "bin"], how="left")
                t["p"] = np.where(t.mn > 0, t.mobs / t.mn, np.nan)   # a bin the rollout never visits drops
                curves.append(t.groupby("bin").agg(N=("n", "sum"), P=("p", "mean"), O=("o", "mean")))
            b = curves[0].assign(P=sum(c.P for c in curves) / len(curves))
            ok = np.isfinite(b.P)
            gap = per_subject_gap(m, reg, st)
            out[m][st] = {"subject_mean_curve_error": float(np.sum(b.N[ok] * np.abs(b.P[ok] - b.O[ok])) / b.N[ok].sum()),
                          "mean_subject_gap": float(np.nanmean(gap)), "n_subjects": int(np.isfinite(gap).sum()),
                          "n_regimes": len(curves)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-repo", type=Path, required=True)
    a = ap.parse_args()
    from scipy.stats import binomtest, wilcoxon

    gaps, p1 = json.loads(GAPS.read_text()), json.loads(P1.read_text())
    head = gaps["headline_set"]
    heads, subs, common = set(gaps["sets"][head]), gaps["subjects"], gaps["common_subjects"]
    idx = [subs.index(s) for s in common]

    def per_subject(m, reg, st, key):
        """Over the common subjects; the GRU's is the mean over its seeds."""
        regs = [r for r in gaps["models"][m] if r.startswith("seed")] if m == GRU else [reg]
        v = np.array([[np.nan if x is None else x for x in
                       np.array(gaps["models"][m][r][head]["statistics"][st]["per_subject"][key], dtype=object)[idx]]
                      for r in regs], float)
        with np.errstate(all="ignore"):
            return np.nanmean(v, axis=0) if len(regs) > 1 else v[0]   # all-NaN subjects stay NaN

    sub_p1 = {r["subject_id"]: r for r in p1["subjects"]}
    specs = [s for s in json.loads(SPECS.read_text())["sessions"] if s["group"] in heads]
    with np.errstate(all="ignore"):
        D = np.nanmean([np.log(per_subject(GRU, "per_subject", st, "gap")
                               / per_subject("Lopez-DoubleTrace", "two_level_em", st, "gap")) for st in STATS], axis=0)
        E = np.nanmean([np.log(per_subject("Lopez-DoubleTrace", "per_subject", st, "gap")
                               / per_subject("Lopez-DoubleTrace", "two_level_em", st, "gap")) for st in STATS], axis=0)
    rows = [{"subject_id": s, "gru_vs_lopez_2l": float(d), "lopez_per_subject_vs_2l": float(e),
             "lopez_mean_log_sigma": sub_p1[s]["lopez_mean_log_sigma"],
             "log_split_over_binomial_floor": sub_p1[s]["log_split_over_binomial_floor"],
             "n_trials": int(sum(x["n_trials"] for x in specs if x["subject_id"] == s)),
             "n_sessions": sum(x["subject_id"] == s for x in specs)} for s, d, e in zip(common, D, E)]

    rng = np.random.default_rng(SEED)
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
    floors = {st: float(np.nanmean(per_subject(GRU, "per_subject", st, "floor"))) for st in STATS}
    errors = _errors(common, heads, gaps["min_trials"], lambda m, reg, st: per_subject(m, reg, st, "gap"))

    meta = build_meta("analysis/freeze_p2_gru_scratch.py", [], study_root=HERE.parent,
                      code=[__file__, HERE / "freeze_p1_gru_scratch.py", HERE / "freeze_p0_golden_mevolve.py"],
                      inputs=[GAPS, P1, SPECS, TALLIES / "animal_sessions.parquet",
                              *(TALLIES / f"{m}.parquet" for m, _ in MODELS)])
    write_artifact(OUT, {
        "_meta": meta,
        "fingerprints_library": {"repo": "AllenNeuralDynamics/aind-dynamic-foraging-models",
                                 "git_sha": _git_sha(a.models_repo)},
        "headline_set": head, "n_subjects": len(common), "min_trials": gaps["min_trials"],
        "models": [m for m, _ in MODELS], "regimes": dict(MODELS), "floor": floors, "floor_kind": "split-half",
        "errors": errors, "paired": paired, "spread": spread, "subjects": rows,
    })
    print(json.dumps({"paired": paired, "spread_test": spread["test"], "errors_gru": errors[GRU]}, indent=1))


if __name__ == "__main__":
    main()
