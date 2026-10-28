"""H3: the GRU's dynamics drawn in Q-learning's own coordinates (Q_L, Q_R).

    python analysis/h3_q_phase_portraits.py

Linear decoders read Hattori's (per-mouse fitted) Q_L and Q_R out of the GRU state (fit on a
subsample, held-out R² on separate sessions), so every trial gets decoded values Q̂_L, Q̂_R.
For each input on trial t (choice × reward), the transition t → t+1 is drawn as the binned
mean displacement in the (Q_L, Q_R) plane:
  GRU      — decoded Q̂ from its own states;
  Hattori  — its true Q (delta rule with reward-dependent learning rate + forgetting);
  Bari     — its true Q (one learning rate + forgetting; its choice kernel is not in Q).
Quantified per input by regressing the change of the chosen and the unchosen value on both
values: ΔQ_x = a + b·Q_chosen + c·Q_unchosen. A delta rule with forgetting has b = −α, c = 0
for the chosen side and b = 0, c = −f for the unchosen side; cross-terms (chosen ↔ unchosen)
are what the GRU adds. Caveat: the GRU is read through Hattori's coordinates, which explain
part, not all, of its state. Writes ``h3_q_phase_portraits.json`` and ``fig_H3_q_phase_portraits.png``.
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

SEED = 10
N_FIT = 1_000_000
N_BINS = 14
MIN_PER_BIN = 100
INPUTS = {"R rewarded": (1, 1), "R unrewarded": (1, 0), "L rewarded": (0, 1), "L unrewarded": (0, 0)}
MODELS = {"gru": "GRU (Q̂ decoded from its state)", "hattori": "Hattori (L2F1): true Q", "bari": "Bari (L1F1_CK1): true Q"}
COLORS = {"gru": "#222222", "hattori": "#55A868", "bari": "#4C72B0"}


def decode(states: np.ndarray, target: np.ndarray, ses: np.ndarray, rng) -> tuple[np.ndarray, float]:
    uniq = np.unique(ses)
    test = np.isin(ses, rng.choice(uniq, size=len(uniq) // 5, replace=False))
    fit_rows = rng.choice(np.flatnonzero(~test), size=min(N_FIT, int((~test).sum())), replace=False)
    x = np.column_stack([states[fit_rows], np.ones(len(fit_rows), np.float32)]).astype(np.float64)
    coef, *_ = np.linalg.lstsq(x, target[fit_rows], rcond=None)
    pred = states.astype(np.float64) @ coef[:-1] + coef[-1]
    r2 = 1 - np.var(target[test] - pred[test]) / np.var(target[test])
    return pred, float(r2)


def main() -> None:
    s12.apply_presentation_style()
    t = s12.load_trials(["subject_id", "ses_idx", "trial_idx", "action", "reward"])
    ses = t["ses_idx"].to_numpy()
    lay = tr.session_layout(t)
    q = {}
    for m in ("hattori", "bari"):
        rl = pd.read_parquet(s12.CACHE / f"rl_trials_{m}_{s12.MODEL_KEY}.parquet")
        if not (rl["ses_idx"].to_numpy() == ses).all():
            raise AssertionError(f"{m} rollout rows are not aligned with the trial table")
        q[m] = np.column_stack([rl["q_rl_left"].to_numpy(), rl["q_rl_right"].to_numpy()])
    states = s12.load_trials(s12.STATE_COLUMNS).to_numpy(dtype=np.float32)
    rng = np.random.default_rng(SEED)
    q_gru, r2 = np.empty_like(q["hattori"]), {}
    for k, side in enumerate(("Q_L", "Q_R")):
        q_gru[:, k], r2[side] = decode(states, q["hattori"][:, k], ses, rng)
    del states
    q["gru"] = q_gru
    print("decoding R²", r2, flush=True)

    idx = np.flatnonzero(lay["has_next"])
    action, reward = t["action"].to_numpy(), (t["reward"].to_numpy() > 0).astype(int)
    lo, hi = np.percentile(q["hattori"], [0.5, 99.5])
    edges = np.linspace(lo, hi, N_BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(4, 4, figsize=(28, 26), gridspec_kw={"width_ratios": [1, 1, 1, 1.1]})
    coefs = {}
    for r_, (name, (ch, rw)) in enumerate(INPUTS.items()):
        rows = idx[(action[idx] == ch) & (reward[idx] == rw)]
        coefs[name] = {}
        for c_, m in enumerate(MODELS):
            x0, x1 = q[m][rows], q[m][rows + 1]
            d = x1 - x0
            ax = axes[r_, c_]
            ix = np.clip(np.digitize(x0[:, 0], edges) - 1, -1, N_BINS)
            iy = np.clip(np.digitize(x0[:, 1], edges) - 1, -1, N_BINS)
            ok = (ix >= 0) & (ix < N_BINS) & (iy >= 0) & (iy < N_BINS)
            flat = ix[ok] * N_BINS + iy[ok]
            n = np.bincount(flat, minlength=N_BINS ** 2)
            mean_d = np.stack([np.bincount(flat, weights=d[ok, k], minlength=N_BINS ** 2) for k in range(2)], axis=1)
            mean_d = mean_d / np.maximum(n, 1)[:, None]
            keep = n >= MIN_PER_BIN
            gx, gy = np.meshgrid(centres, centres, indexing="ij")
            ax.imshow(np.log10(np.maximum(n, 1)).reshape(N_BINS, N_BINS).T, origin="lower", cmap="Greys", alpha=0.5,
                      extent=(lo, hi, lo, hi), aspect="equal")
            ax.quiver(gx.ravel()[keep], gy.ravel()[keep], mean_d[keep, 0], mean_d[keep, 1], color=COLORS[m],
                      angles="xy", scale_units="xy", scale=1, width=0.006)
            ax.plot([lo, hi], [lo, hi], color="0.7", lw=0.8, ls=":")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            if r_ == 0:
                ax.set_title(MODELS[m], fontsize=14)
            if c_ == 0:
                ax.set_ylabel(f"trial t: {name}\nQ_R")
            ax.set_xlabel("Q_L")
            # ΔQ_x = a + b·Q_chosen + c·Q_unchosen for the chosen (x = ch) and unchosen side.
            qc, qu = x0[:, ch], x0[:, 1 - ch]
            design = np.column_stack([np.ones(len(rows)), qc, qu])
            out = {}
            for side, col in (("chosen", ch), ("unchosen", 1 - ch)):
                beta, *_ = np.linalg.lstsq(design, d[:, col], rcond=None)
                res = d[:, col] - design @ beta
                out[side] = {"a": float(beta[0]), "b_Qchosen": float(beta[1]), "c_Qunchosen": float(beta[2]),
                             "r2": float(1 - res.var() / d[:, col].var())}
            coefs[name][m] = out
        ax = axes[r_, 3]
        labels = ["chosen ← Q_chosen", "chosen ← Q_unchosen", "unchosen ← Q_chosen", "unchosen ← Q_unchosen"]
        keys = [("chosen", "b_Qchosen"), ("chosen", "c_Qunchosen"), ("unchosen", "b_Qchosen"), ("unchosen", "c_Qunchosen")]
        y = np.arange(len(labels))
        for j, m in enumerate(MODELS):
            ax.barh(y + (j - 1) * 0.27, [coefs[name][m][s][k] for s, k in keys], height=0.27, color=COLORS[m],
                    label=m.upper() if m == "gru" else m.capitalize())
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_yticks(y, labels, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel("slope of the one-trial change")
        if r_ == 0:
            ax.set_title("Update slopes: ΔQ = a + b·Q_chosen + c·Q_unchosen", fontsize=14)
            ax.legend(frameon=False, fontsize=11)
    fig.suptitle(f"Mean one-trial change of (Q_L, Q_R) after each input; GRU decoded at held-out R² "
                 f"Q_L {r2['Q_L']:.2f}, Q_R {r2['Q_R']:.2f}; grey = occupancy (log)", fontsize=17)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_H3_q_phase_portraits.png"))
    plt.close(fig)
    s12.write_json("h3_q_phase_portraits.json", {
        "decoding_heldout_r2": r2, "decoder_target": "Hattori (L2F1) per-mouse Q", "update_regressions": coefs,
        "n_bins": N_BINS, "min_per_bin": MIN_PER_BIN, "n_fit": N_FIT, "seed": SEED,
    }, "analysis/h3_q_phase_portraits.py", extra_groups=s12.rl_groups())
    for name, d in coefs.items():
        for m, out in d.items():
            print(name, m, {s: {k: round(v, 3) for k, v in o.items()} for s, o in out.items()})


if __name__ == "__main__":
    main()
