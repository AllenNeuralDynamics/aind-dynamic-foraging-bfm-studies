"""I7: which second carried variable closes the gap between I4's distilled model and the GRU?

    python analysis/i7_second_variable.py

I4's one-variable model tracks the GRU's value coordinate v but scores 0.718 against 0.734;
its readout a·v + b[last input] misses ~6% of the logit's variance. Here a second coordinate
u (unit axis orthogonal to I2's value axis, state projected after removing the PCA mean) is
added and the I4 procedure repeated in 2-d: per transition group, [v, u](t+1) = A_g [v, u](t)
+ c_g (2×2 A_g, so v and u may interact); readout logit ≈ a_v·v + a_u·u + b[last input];
simulated from the animals' inputs without the GRU, pooled and per mouse (pooled fallback for
groups with < 50 transitions). Candidates:
  readout residual — the part of the readout r = w_R − w_L orthogonal to v. Because the logit
                     is exactly r·h + const, v and this u read out the logit exactly: an upper
                     bound for any single extra coordinate (teacher-forced it equals the GRU);
  choice / reward / value×choice / choice×reward axes — I2's orthonormal TDR axes;
  PC5 (time)       — the within-session drift, orthogonalised against v;
  choice + reward  — a 3-d model.
Also scored teacher-forced (the readout on the GRU's own coordinates), which bounds what the
simulated model can reach. Writes ``i7_second_variable.json`` and ``fig_I7_second_variable.png``.
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
from i4_distilled_model import value_coordinate  # noqa: E402

CHUNK = 500_000
N_BINS = 20
HISTORY = 10
COLORS = {"R rewarded": "#0072B2", "R unrewarded": "#56B4E9", "L rewarded": "#D55E00", "L unrewarded": "#E69F00"}
CANDIDATES = {"v only": [], "+ readout residual": ["readout residual"], "+ choice axis": ["choice"],
              "+ reward axis": ["reward"], "+ value×choice axis": ["value×choice"],
              "+ choice×reward axis": ["choice×reward"], "+ PC5 (time)": ["PC5"],
              "+ choice + reward axes": ["choice", "reward"]}


def unit_orthogonal(vec: np.ndarray, to: np.ndarray) -> np.ndarray:
    vec = vec - (vec @ to) * to
    return vec / np.linalg.norm(vec)


def coordinates() -> tuple[np.ndarray, list[str]]:
    """Projections of (h − PCA mean) onto v and every candidate axis, [N, 1 + n_axes]."""
    tdr = np.load(s12.keyed_cache("i2_tdr_axes.npz"))
    terms = [str(x) for x in tdr["terms"]]
    q = tdr["basis"]
    e_v = q[:, terms.index("value")]
    pca = s12.load_pca()
    w = np.asarray(s12.best_params()["multisubject_gru/~/readout"]["w"], dtype=np.float64)
    axes = {"v": e_v, "readout residual": unit_orthogonal(w[:, 1] - w[:, 0], e_v),
            "choice": q[:, terms.index("choice")], "reward": q[:, terms.index("reward")],
            "value×choice": q[:, terms.index("value×choice")], "choice×reward": q[:, terms.index("choice×reward")], "PC5": unit_orthogonal(pca["components"][4], e_v)}
    basis = np.column_stack(list(axes.values()))
    states = s12.load_trials(s12.STATE_COLUMNS).to_numpy(dtype=np.float32)
    out = np.empty((len(states), basis.shape[1]))
    for start in range(0, len(states), CHUNK):
        out[start:start + CHUNK] = (states[start:start + CHUNK].astype(np.float64) - pca["mean"]) @ basis
    return out, list(axes)


def distil(x: np.ndarray, logit: np.ndarray, lay: dict, by_mouse: dict, action: np.ndarray) -> dict:
    """I4's procedure for a d-dim carried state; returns likelihoods, fits and simulations."""
    A, c, n = tr.fit_groups_nd(x, lay)
    a, b = tr.fit_readout_nd(x, logit, lay)
    teacher = x @ a + b[lay["cluster"]]
    x0 = x[lay["first"]].mean(axis=0)
    sim = tr.simulate_nd(lay, x0, A[lay["group"]], c[lay["group"]])
    A_rows, c_rows = np.empty((len(x),) + A.shape[1:]), np.empty((len(x), x.shape[1]))
    a_rows, b_rows = np.empty((len(x), x.shape[1])), np.empty(len(x))
    for rows in by_mouse.values():
        Am, cm, _ = tr.fit_groups_nd(x, lay, rows=rows, fallback=(A, c))
        am, bm = tr.fit_readout_nd(x, logit, lay, rows=rows)
        A_rows[rows], c_rows[rows] = Am[lay["group"][rows]], cm[lay["group"][rows]]
        a_rows[rows], b_rows[rows] = am, bm[lay["cluster"][rows]]
    sim_m = tr.simulate_nd(lay, x0, A_rows, c_rows)
    idx = np.flatnonzero(lay["has_next"] & ~lay["first"])
    resid = x[idx + 1] - (np.einsum("nij,nj->ni", A[lay["group"][idx]], x[idx]) + c[lay["group"][idx]])
    return {
        "teacher_forced": tr.likelihood(action, teacher),
        "teacher_forced_r2": float(1 - np.var(logit - teacher) / np.var(logit)),
        "teacher_forced_per_mouse": tr.likelihood(action, np.einsum("nd,nd->n", a_rows, x) + b_rows),
        "simulated_per_mouse": tr.likelihood(action, np.einsum("nd,nd->n", a_rows, sim_m) + b_rows),
        "simulated_pooled": tr.likelihood(action, sim @ a + b[lay["cluster"]]),
        "update_r2": (1 - resid.var(axis=0) / x[idx + 1].var(axis=0)).tolist(),
        "corr_sim_vs_gru": [float(np.corrcoef(sim_m[:, k], x[:, k])[0, 1]) for k in range(x.shape[1])],
        "max_abs_sim": float(np.abs(sim_m).max()), "A": A, "c": c, "n": n, "readout_a": a, "readout_b": b,
        "_sim_m": sim_m,
    }


def describe_residual(coords: np.ndarray, names: list[str], t: pd.DataFrame, lay: dict) -> dict:
    """What the readout-residual coordinate is: variance share, alignment with PCs, and how
    much of its within-cluster variation choice / choice×reward history (lags 2–10) explains."""
    pca = s12.load_pca()
    tdr = np.load(s12.keyed_cache("i2_tdr_axes.npz"))
    e_v = tdr["basis"][:, 0]
    w = np.asarray(s12.best_params()["multisubject_gru/~/readout"]["w"], dtype=np.float64)
    axis = unit_orthogonal(w[:, 1] - w[:, 0], e_v)
    eig = np.clip(pca["eigvals"], 0, None)
    share = float(((pca["components"] @ axis) ** 2 * eig).sum() / eig.sum())
    g = t.groupby("ses_idx", sort=False)
    hist = {}
    for lag in range(2, HISTORY + 1):  # lag 1 is fixed within a cluster (collinear with b)
        c = g["action"].shift(lag).to_numpy() * 2 - 1
        r = (g["reward"].shift(lag) > 0).to_numpy().astype(float)
        hist[f"choice-{lag}"], hist[f"choice×reward-{lag}"] = c, c * r
    h = pd.DataFrame(hist)
    ok = h.notna().all(axis=1).to_numpy()
    design = np.column_stack([h.to_numpy()[ok], np.eye(tr.N_CLUSTERS)[lay["cluster"][ok]]])
    out = {"variance_share": share,
           "cos_with_pcs_1_6": (pca["components"][:6] @ axis).round(3).tolist(),
           "cos_with_tdr_axes": dict(zip([str(x) for x in tdr["terms"]], (tdr["basis"].T @ axis).round(3).tolist())),
           "history_lags": [2, HISTORY]}
    for name in ("v", "readout residual"):
        y = coords[ok, names.index(name)]
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        within = y - np.eye(tr.N_CLUSTERS)[lay["cluster"][ok]] @ np.linalg.lstsq(np.eye(tr.N_CLUSTERS)[lay["cluster"][ok]], y, rcond=None)[0]
        resid = y - design @ coef
        out[f"{name}: within-cluster R² from lags 2–{HISTORY}"] = float(1 - resid.var() / within.var())
        out[f"{name}: history weights"] = dict(zip(h.columns, coef[:h.shape[1]].round(3).tolist()))
    return out


def binned(ax, u: np.ndarray, lay: dict, title: str) -> None:
    idx = np.flatnonzero(lay["has_next"])
    lo, hi = np.percentile(u, [1, 99])
    edges = np.linspace(lo, hi, N_BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    for g, name in enumerate(tr.GROUP_NAMES):
        if name.endswith("(first)"):
            continue
        sel = idx[lay["group"][idx] == g]
        b = np.digitize(u[sel], edges) - 1
        ok = (b >= 0) & (b < N_BINS)
        m = pd.Series(u[sel + 1][ok]).groupby(b[ok]).agg(["mean", "size"])
        m = m[m["size"] >= 50]
        ax.plot(centres[m.index], m["mean"], color=COLORS[name.split(" (")[0]],
                ls="-" if "(stay)" in name else "--", marker="o", ms=3, lw=1.6, label=name)
    ax.plot([lo, hi], [lo, hi], color="0.6", ls=":", lw=1)
    ax.set_xlabel("u(t)")
    ax.set_ylabel("u(t+1)")
    ax.set_title(title, fontsize=13)
    ax.legend(frameon=False, fontsize=8)


def main() -> None:
    s12.apply_presentation_style()
    t = s12.load_trials(["subject_id", "ses_idx", "trial_idx", "action", "reward", "logit_diff"])
    lay = tr.session_layout(t)
    logit, action = t["logit_diff"].to_numpy(dtype=np.float64), t["action"].to_numpy()
    by_mouse = pd.Series(np.arange(len(t))).groupby(t["subject_id"].to_numpy()).indices
    coords, names = coordinates()
    v_check = float(np.abs(coords[:, 0] - value_coordinate()).max())
    if v_check > 1e-3:
        raise AssertionError(f"value coordinate does not match I4's cache (max diff {v_check})")

    character = describe_residual(coords, names, t, lay)
    print("residual axis", character, flush=True)
    results = {}
    for label, extra in CANDIDATES.items():
        x = coords[:, [0] + [names.index(e) for e in extra]]
        results[label] = distil(x, logit, lay, by_mouse, action)
        r = results[label]
        print(label, {k: round(r[k], 5) for k in ("teacher_forced", "teacher_forced_per_mouse", "simulated_per_mouse", "simulated_pooled")},
              "update R2", np.round(r["update_r2"], 3), "corr", np.round(r["corr_sim_vs_gru"], 3),
              "max|x|", round(r["max_abs_sim"], 1), flush=True)
    gru = tr.likelihood(action, logit)
    p_ctt = pd.read_parquet(s12.CACHE / f"rl_trials_ctt_{s12.MODEL_KEY}.parquet", columns=["p_rl_right"])["p_rl_right"].to_numpy()
    ctt = tr.likelihood(action, np.log(np.clip(p_ctt, 1e-10, 1) / np.clip(1 - p_ctt, 1e-10, 1)))
    base = results["v only"]["simulated_per_mouse"]
    for r in results.values():
        r["gap_closed_per_mouse"] = (r["simulated_per_mouse"] - base) / (gru - base)

    labels = list(results)
    best = max((k for k in labels if k not in ("v only", "+ readout residual")), key=lambda k: results[k]["simulated_per_mouse"])
    fig, axes = plt.subplots(1, 4, figsize=(32, 7), gridspec_kw={"width_ratios": [1.3, 1, 1, 1.1]})
    y = np.arange(len(labels))
    for k, (key, color, name) in enumerate((("teacher_forced_per_mouse", "#999999", "readout on the GRU's own coordinates (per mouse)"),
                                            ("simulated_per_mouse", "#E69F00", "simulated, per mouse"),
                                            ("simulated_pooled", "#F0C060", "simulated, pooled"))):
        axes[0].barh(y + (k - 1) * 0.27, [results[n][key] for n in labels], height=0.27, color=color, label=name)
    axes[0].axvline(gru, color="k", lw=1.5, label=f"GRU {gru:.4f}")
    axes[0].axvline(ctt, color="#55A868", lw=1.5, ls="--", label=f"CTT {ctt:.4f}")
    axes[0].set_yticks(y, labels, fontsize=11)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.695, gru + 0.004)
    axes[0].set_xlabel("eval likelihood on the animals' choices")
    axes[0].set_title("Adding one carried coordinate to the distilled model", fontsize=13)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    for ax, label in zip(axes[1:3], ("+ readout residual", best)):
        r = results[label]
        binned(ax, coords[:, names.index(CANDIDATES[label][0])], lay,
               f"{label[2:]}: update R² {r['update_r2'][1]:.2f}, sim corr {r['corr_sim_vs_gru'][1]:.2f}")
    ses = t["ses_idx"].iloc[np.flatnonzero(lay["first"])[len(np.flatnonzero(lay["first"])) // 2]]
    rows = np.flatnonzero((t["ses_idx"] == ses).to_numpy())[:100]
    k = names.index(CANDIDATES[best][0])
    axes[3].plot(coords[rows, k], color="#222222", lw=1.8, label=f"GRU: {CANDIDATES[best][0]} coordinate")
    axes[3].plot(results[best]["_sim_m"][rows, 1], color="#4C72B0", lw=1.4, ls="--", label="distilled 2-d model (per mouse), simulated")
    axes[3].set_xlabel("trial")
    axes[3].set_ylabel("u")
    axes[3].set_title(f"Example session: {CANDIDATES[best][0]} coordinate", fontsize=13)
    axes[3].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_I7_second_variable.png"))
    plt.close(fig)

    s12.write_json("i7_second_variable.json", {
        "gru_likelihood": gru, "ctt_likelihood": ctt, "axes": names, "candidates": CANDIDATES, "groups": tr.GROUP_NAMES,
        "results": {k: {kk: vv for kk, vv in r.items() if not kk.startswith("_")} for k, r in results.items()},
        "best_interpretable": best, "min_per_group": tr.MIN_PER_GROUP, "readout_residual_axis": character,
    }, "analysis/i7_second_variable.py", extra_groups=s12.rl_groups())
    print("gru", round(gru, 5), "ctt", round(ctt, 5), "gap closed", {k: round(r["gap_closed_per_mouse"], 3) for k, r in results.items()})


if __name__ == "__main__":
    main()
