"""I4: distil C3 into two carried variables — side value v and engagement e — and score them.

    python analysis/i4_distilled_model.py

The teachers are C3's own readouts on every step: v = logit_diff (= log P(R | engaged) /
P(L | engaged)) and e = engage_logit (= log P(engaged) / P(ignore)). Per transition group
fit the affine update x(t+1) = A_g x(t) + c_g (x = (v, e), A_g 2×2, so the two may interact).
Groups follow Study 50's ``_transitions``: input at t = choice × reward × {stay, switch,
first}, stay/switch relative to the previous *engaged* choice, plus one group for an ignore.
Then simulate x from a fixed start along each session driven only by the animals' inputs
(no GRU), and score the animals: L/R on engaged trials via σ(v), engage vs ignore on every
step via σ(e). Variants, all pooled parameters, on the shared sessions:
  * C3 (the GRU itself);
  * distilled 2-var (v, e);
  * distilled v only, with the ignore rule (scored on L/R);
  * distilled v only, ignore steps skipped — the rule C uses implicitly;
  * C (Study 50's GRU) and the same 1-var model distilled from C's logit_diff.
L/R scores are over the same engaged trials for every variant. Also by gap (ignores just
before). Writes ``i4_distilled_model.json`` and ``fig_I4_distilled_model.png``.
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
import _pk02 as s13  # noqa: E402

INPUT_NAMES = ["L−", "L+", "R−", "R+"]  # index 2·choice + reward
KINDS = ["stay", "switch", "first"]
GROUPS = [f"{i} ({k})" for i in INPUT_NAMES for k in KINDS] + ["ignore"]
IGNORE_GROUP = len(GROUPS) - 1
MIN_PER_GROUP = 50
GAP_BINS = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 10_000)]


def layout(action: np.ndarray, reward: np.ndarray, ses: np.ndarray) -> dict:
    """Groups of the transition leaving each row, with stay/switch relative to the last engaged choice."""
    first = np.r_[True, ses[1:] != ses[:-1]]
    last = np.r_[ses[1:] != ses[:-1], True]
    engaged = action != s13.IGNORE
    prev_engaged = pd.Series(np.where(engaged, action, np.nan)).groupby(ses, sort=False).transform(
        lambda s: s.shift(1).ffill()).to_numpy()
    kind = np.where(np.isnan(prev_engaged), 2, np.where(action == prev_engaged, 0, 1))
    group = np.where(engaged, 3 * (2 * np.where(engaged, action, 0) + reward) + kind, IGNORE_GROUP)
    return {"first": first, "has_next": ~last, "group": group.astype(int)}


def fit(x: np.ndarray, lay: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-group least squares x(t+1) = A x(t) + c; first-kind groups fitted as constants."""
    d = x.shape[1]
    idx = np.flatnonzero(lay["has_next"])
    A, c, n = np.zeros((len(GROUPS), d, d)), np.zeros((len(GROUPS), d)), np.zeros(len(GROUPS), int)
    for g in range(len(GROUPS)):
        sel = idx[lay["group"][idx] == g]
        n[g] = len(sel)
        if n[g] < MIN_PER_GROUP:
            A[g], c[g] = np.eye(d), 0.0
        elif g != IGNORE_GROUP and g % 3 == 2:
            c[g] = x[sel + 1].mean(axis=0)
        else:
            design = np.column_stack([x[sel], np.ones(len(sel))])
            coef, *_ = np.linalg.lstsq(design, x[sel + 1], rcond=None)
            A[g], c[g] = coef[:d].T, coef[d]
    return A, c, n


def simulate(lay: dict, x0: np.ndarray, A: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = len(lay["group"])
    x = np.empty((n, len(x0)))
    starts = np.flatnonzero(lay["first"])
    lengths = np.diff(np.r_[starts, n])
    x[starts] = x0
    for k in range(1, lengths.max()):
        rows = starts[lengths > k] + k
        prev = rows - 1
        g = lay["group"][prev]
        x[rows] = np.einsum("nij,nj->ni", A[g], x[prev]) + c[g]
    return x


def lr_loglik(action: np.ndarray, v: np.ndarray) -> np.ndarray:
    p = 1 / (1 + np.exp(-v))
    return np.log(np.clip(np.where(action == 1, p, 1 - p), 1e-10, 1))


def main() -> None:
    s13.apply_presentation_style()
    drop = s13.unshared_sessions()
    t = s13.load_trials(["ses_idx", "session_id", "trial", "action", "reward", "logit_diff",
                         "choice_logit_0", "choice_logit_1", "choice_logit_2"])
    t = t[~t["session_id"].isin(drop["C3"])].reset_index(drop=True)
    action, reward, ses = t["action"].to_numpy(), (t["reward"].to_numpy() > 0).astype(int), t["ses_idx"].to_numpy()
    v, e = t["logit_diff"].to_numpy(np.float64), t["engage_logit"].to_numpy()
    lay = layout(action, reward, ses)
    engaged = action != s13.IGNORE
    cum = pd.Series((~engaged).astype(int)).groupby(ses, sort=False).cumsum().to_numpy()
    cum_e = pd.Series(cum[engaged])
    gap = (cum_e - cum_e.groupby(ses[engaged], sort=False).shift(1)).fillna(cum_e).to_numpy().astype(int)

    x = np.column_stack([v, e])
    A2, c2, n2 = fit(x, lay)
    sim2 = simulate(lay, x[lay["first"]].mean(axis=0), A2, c2)
    A1, c1, _ = fit(v[:, None], lay)
    sim1 = simulate(lay, np.array([v[lay["first"]].mean()]), A1, c1)[:, 0]
    # v only, ignore steps skipped (C's view): engaged rows only, groups re-derived on that sequence.
    ve, ae, re_, se = v[engaged], action[engaged], reward[engaged], ses[engaged]
    lay_e = layout(ae, re_, se)
    A0, c0, _ = fit(ve[:, None], lay_e)
    sim0 = simulate(lay_e, np.array([ve[lay_e["first"]].mean()]), A0, c0)[:, 0]

    c = s13.load_twin(["ses_idx", "session_id", "trial", "action", "reward", "logit_diff"])
    c = c[~c["session_id"].isin(drop["C"])].reset_index(drop=True)
    vc = c["logit_diff"].to_numpy(np.float64)
    lay_c = layout(c["action"].to_numpy(), (c["reward"].to_numpy() > 0).astype(int), c["ses_idx"].to_numpy())
    Ac, cc, _ = fit(vc[:, None], lay_c)
    simc = simulate(lay_c, np.array([vc[lay_c["first"]].mean()]), Ac, cc)[:, 0]
    key3 = pd.MultiIndex.from_arrays([ses[engaged], t["trial"].to_numpy()[engaged]])
    keyc = pd.MultiIndex.from_arrays([c["ses_idx"].to_numpy(), c["trial"].to_numpy()])
    if not keyc.equals(key3):
        raise AssertionError("C and C3 engaged trials are not in the same order")

    ae_action = action[engaged]
    ll = {
        "C3 (GRU)": lr_loglik(ae_action, v[engaged]),
        "distilled 2-var (v, e)": lr_loglik(ae_action, sim2[engaged, 0]),
        "distilled v, with ignore rule": lr_loglik(ae_action, sim1[engaged]),
        "distilled v, ignores skipped": lr_loglik(ae_action, sim0),
        "C (GRU)": lr_loglik(ae_action, vc),
        "C distilled v": lr_loglik(ae_action, simc),
    }
    lr = {k: float(np.exp(x_.mean())) for k, x_ in ll.items()}
    by_gap = {}
    for lo, hi in GAP_BINS:
        m = (gap >= lo) & (gap <= hi)
        by_gap[str(lo) if lo == hi else (f"{lo}+" if hi >= 10_000 else f"{lo}–{hi}")] = {
            "n": int(m.sum()), **{k: float(np.exp(x_[m].mean())) for k, x_ in ll.items()}}
    p_eng_gru = 1 / (1 + np.exp(-e))
    p_eng_sim = 1 / (1 + np.exp(-sim2[:, 1]))
    engage_lik = {k: float(np.exp(np.mean(np.log(np.clip(np.where(engaged, p, 1 - p), 1e-10, 1)))))
                  for k, p in (("C3 (GRU)", p_eng_gru), ("distilled 2-var (v, e)", p_eng_sim))}
    base_rate = float((~engaged).mean())
    engage_lik["constant base rate"] = float(np.exp(np.mean(np.log(np.where(engaged, 1 - base_rate, base_rate)))))
    fidelity = {"corr_v": float(np.corrcoef(sim2[:, 0], v)[0, 1]), "corr_e": float(np.corrcoef(sim2[:, 1], e)[0, 1])}

    params = {g: {"A": A2[i].round(4), "c": c2[i].round(4), "n": int(n2[i]),
                  "fixed_point": (np.linalg.solve(np.eye(2) - A2[i], c2[i]).round(3)
                                  if abs(np.linalg.det(np.eye(2) - A2[i])) > 1e-6 else None)}
              for i, g in enumerate(GROUPS)}

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), gridspec_kw={"width_ratios": [1.15, 1.2, 1]})
    names = list(lr)
    colors = [s13.ENGAGE_COLOR, "#B07CC6", "#D7B8E3", "#999999", s13.MODEL_COLOR, "#777777"]
    axes[0].barh(range(len(names)), [lr[k] for k in names], color=colors)
    axes[0].set_yticks(range(len(names)), names, fontsize=12)
    axes[0].invert_yaxis()
    axes[0].set_xlim(min(lr.values()) - 0.01, max(lr.values()) + 0.006)
    for i, k in enumerate(names):
        axes[0].text(lr[k] + 0.0004, i, f"{lr[k]:.4f}", va="center", fontsize=11)
    axes[0].set(xlabel="L/R likelihood, shared engaged trials", title="a. GRUs and their distilled models")
    labels = list(by_gap)
    for k, color in zip(names, colors):
        axes[1].plot(labels, [by_gap[g][k] for g in labels], marker="o", lw=2, color=color, label=k)
    axes[1].set(xlabel="ignored trials just before", ylabel="L/R likelihood", title="b. by gap")
    axes[1].legend(frameon=False, fontsize=10)
    xs = np.linspace(-5, 5, 50)
    for g, name in enumerate(GROUPS):
        if name.endswith("(first)") or n2[g] < MIN_PER_GROUP:
            continue
        lam = A2[g, 0, 0]
        color = s13.IGNORE_COLOR if g == IGNORE_GROUP else {"R+": "#0072B2", "R−": "#56B4E9", "L+": "#D55E00", "L−": "#E69F00"}[name.split(" (")[0]]
        ls = "-" if ("(stay)" in name or g == IGNORE_GROUP) else "--"
        e_mean = e[lay["group"] == g].mean()
        axes[2].plot(xs, A2[g, 0, 0] * xs + A2[g, 0, 1] * e_mean + c2[g, 0], color=color, ls=ls,
                     lw=3 if g == IGNORE_GROUP else 1.6,
                     label=f"{name}: λ {lam:.2f}")
    axes[2].plot(xs, xs, color="0.6", ls=":", lw=1)
    axes[2].set(xlabel="v(t) (e at its mean for that input)", ylabel="v(t+1)", title="c. side update per input (2-var model, v row)")
    axes[2].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(s13.figure_path("fig_I4_distilled_model.png"))
    plt.close(fig)

    s13.write_json("i4_distilled_model.json", {
        "groups": GROUPS, "trials": "shared sessions; L/R scored on the same engaged trials for every variant",
        "lr_likelihood": lr, "lr_likelihood_by_gap": by_gap, "engage_likelihood_all_steps": engage_lik,
        "simulated_vs_gru_C3": fidelity, "params_2var": params,
        "ignore_rule_1var": {"lambda": float(A1[IGNORE_GROUP, 0, 0]), "c": float(c1[IGNORE_GROUP, 0])},
        "min_per_group": MIN_PER_GROUP,
    }, "analysis/i4_distilled_model.py", extra_groups=s13.twin_wandb_groups())
    print({k: round(x_, 4) for k, x_ in lr.items()})
    print({g: {k: round(x_, 3) for k, x_ in v_.items() if k != "n"} for g, v_ in by_gap.items()})
    print("engage", {k: round(x_, 4) for k, x_ in engage_lik.items()}, "fidelity", fidelity)
    print("ignore group A, c:", A2[IGNORE_GROUP].round(3).tolist(), c2[IGNORE_GROUP].round(3).tolist(),
          "| 1-var ignore rule:", round(float(A1[IGNORE_GROUP, 0, 0]), 3), round(float(c1[IGNORE_GROUP, 0]), 3))


if __name__ == "__main__":
    main()
