"""G1: do the headline C3-vs-C results hold for C3 seeds 1 and 2?

    python analysis/g_seeds.py

Table-only; for each C3 seed with an A1 table, recomputes compact versions of the claims
r1–r6 rest on, on the shared engaged trials and against C seed 0 (Study 50 found C's seeds
within 0.0001 held-out likelihood of each other; C seeds 1–2 are not extracted):
  * C2: L/R likelihood, C3 − C per-trial advantage by gap;
  * C4: predicted P(repeat previous engaged choice) by gap vs the animals; C's for reference;
  * C1b: lag-1 kernel weight after k ignores as a fraction of no gap (soft labels);
  * I4: the one-variable ignore rule v(t+1) = λ v(t) + c (teacher-forced fit);
  * B8: calibration and bout-exit hazard gaps;
  * F: CV R² of each mouse's ignore rate from the seed's subject embedding.
Writes ``g_seeds.json``. The figure and r7 combine it with ``g_dynamics.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict

from aind_dynamic_foraging_models.logistic_regression import prepare_logistic_design_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk02 as s13  # noqa: E402

SEEDS = ["C3-s0", "C3-s1", "C3-s2"]
GAP_BINS = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 10_000)]
N_BACK = 10


def gap_label(lo: int, hi: int) -> str:
    return str(lo) if lo == hi else (f"{lo}+" if hi >= 10_000 else f"{lo}–{hi}")


def lr_ll(action: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(np.where(action == 1, p, 1 - p))


def kernel_ratios(t: pd.DataFrame) -> dict:
    blocks, keep = [], []
    for rows in t.groupby("ses_idx", sort=False).indices.values():
        if len(rows) < N_BACK + 2:
            continue
        d = prepare_logistic_design_matrix(t["action"].to_numpy()[rows], (t["reward"].to_numpy()[rows] > 0).astype(int),
                                           logistic_model="Su2022", n_trial_back=N_BACK)
        blocks.append(d["X"].to_numpy(dtype=np.float32))
        keep.append(rows[d.index.to_numpy()])
    x, rows = np.vstack(blocks), np.concatenate(keep)
    gap, p = t["gap"].to_numpy()[rows], t["p_C3"].to_numpy()[rows]
    out = {}
    for lo, hi in GAP_BINS:
        m = (gap >= lo) & (gap <= hi)
        model = LogisticRegression(C=1.0, max_iter=2000)
        model.fit(np.vstack([x[m], x[m]]), np.r_[np.ones(m.sum()), np.zeros(m.sum())], sample_weight=np.r_[p[m], 1 - p[m]])
        coef = model.coef_[0]
        out[gap_label(lo, hi)] = {"RewC": float(coef[N_BACK - 1]), "UnrC": float(coef[2 * N_BACK - 1])}
    base = out["0"]
    return {g: {k: v[k] / base[k] for k in v} for g, v in out.items()}


def embedding_decode(key: str, rate: pd.Series) -> float:
    outputs = s13.CACHE / "models" / key / "run" / "outputs"
    index_map = json.loads((outputs / "subject_index_map.json").read_text())["subject_id_to_index"]
    table = np.asarray(s13.best_params(key)["multisubject_gru"]["subject_embeddings"])
    mice = [m for m in rate.index if m in index_map]
    x = table[[index_map[m] for m in mice]]
    y = rate.loc[mice].to_numpy()
    pred = cross_val_predict(LinearRegression(), x, y, cv=KFold(5, shuffle=True, random_state=0))
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def seed_metrics(key: str) -> dict:
    t = s13.engaged_comparison_table(key)
    action = t["action"].to_numpy()
    ll3, llc = lr_ll(action, t["p_C3"].to_numpy()), lr_ll(action, t["p_C"].to_numpy())
    by_gap = {}
    for lo, hi in GAP_BINS:
        m = (t["gap"].to_numpy() >= lo) & (t["gap"].to_numpy() <= hi)
        by_gap[gap_label(lo, hi)] = {"n": int(m.sum()), "C3": float(np.exp(ll3[m].mean())), "C": float(np.exp(llc[m].mean())),
                                     "C3_minus_C_per_trial": float((ll3[m] - llc[m]).mean())}
    g = t.groupby("ses_idx", sort=False)
    prev = g["action"].shift(1)
    ok = prev.notna().to_numpy()
    prev_right = prev.to_numpy() == 1
    stay = {}
    for lo, hi in GAP_BINS:
        m = ok & (t["gap"].to_numpy() >= lo) & (t["gap"].to_numpy() <= hi)
        p3 = np.where(prev_right, t["p_C3"], 1 - t["p_C3"])[m]
        pc = np.where(prev_right, t["p_C"], 1 - t["p_C"])[m]
        stay[gap_label(lo, hi)] = {"animal": float((action[m] == prev.to_numpy()[m]).mean()), "C3": float(p3.mean()),
                                   "C": float(pc.mean())}

    full = s13.load_trials(["subject_id", "ses_idx", "action", "logit_diff", "p_model1_ignore"], key=key)
    ses, act = full["ses_idx"].to_numpy(), full["action"].to_numpy()
    v = full["logit_diff"].to_numpy(np.float64)
    nxt_same = np.r_[ses[1:] == ses[:-1], False]
    ign = (act == s13.IGNORE) & nxt_same
    lam, c = np.polyfit(v[ign], v[np.flatnonzero(ign) + 1], 1)
    pred = full["p_model1_ignore"].to_numpy(np.float64)
    obs = (act == s13.IGNORE).astype(float)
    edges = np.r_[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cal = pd.DataFrame({"b": np.digitize(pred, edges[1:-1]), "o": obs, "p": pred}).groupby("b").mean()
    prev_ign = np.r_[False, act[:-1] == s13.IGNORE] & np.r_[False, ses[1:] == ses[:-1]]
    rate = pd.Series(obs).groupby(full["subject_id"].astype(str).to_numpy()).mean()
    return {
        "lr_likelihood": {"C3": float(np.exp(ll3.mean())), "C": float(np.exp(llc.mean()))},
        "by_gap": by_gap, "stay_after_gap": stay,
        "stay_max_abs_gap_C3_vs_animal": max(abs(s["C3"] - s["animal"]) for s in stay.values()),
        "kernel_lag1_ratio_to_gap0": kernel_ratios(t),
        "ignore_rule_lambda": float(lam), "ignore_rule_c": float(c),
        "calibration_max_abs_gap": float((cal["o"] - cal["p"]).abs().max()),
        "p_ignore_after_ignore": {"observed": float(obs[prev_ign].mean()), "model": float(pred[prev_ign].mean())},
        "embedding_decodes_ignore_rate_cv_r2": embedding_decode(key, rate),
    }


def main() -> None:
    present = [k for k in SEEDS if s13.trial_table_path(k).exists()]
    out = {}
    for key in present:
        out[key] = seed_metrics(key)
        m = out[key]
        print(key, "LR", {k: round(v, 4) for k, v in m["lr_likelihood"].items()},
              "adv by gap", {g: round(v["C3_minus_C_per_trial"], 3) for g, v in m["by_gap"].items()},
              "λ", round(m["ignore_rule_lambda"], 3), "stay gap", round(m["stay_max_abs_gap_C3_vs_animal"], 3),
              "kernel", {g: round(v["RewC"], 2) for g, v in m["kernel_lag1_ratio_to_gap0"].items()},
              "emb R2", round(m["embedding_decodes_ignore_rate_cv_r2"], 2))
    s13.write_json("g_seeds.json", {"seeds": present, "C_reference": "C seed 0 (Study 50)", "metrics": out},
                   "analysis/g_seeds.py", extra_groups=s13.twin_wandb_groups())


if __name__ == "__main__":
    main()
