"""H1: a model ladder — relax Q-learning one component at a time, fit to the GRU, score on the animals.

    PYTHONPATH=<wrapper>/code python analysis/h1_model_ladder.py     # GPU (JAX)

Every rung is fitted per mouse to the GRU's own predicted P(right) on the eval trials
(cross-entropy to the GRU's soft labels, Adam, all mice in parallel), so the ladder asks which
components reproduce the GRU's policy; each rung is then scored on the animals' choices
(pooled likelihood, share of mice beating CTT, C4e's two behavioural signatures). Rungs are
cumulative:
  R0  Q-learning: α_rew, α_unrew, forgetting of the unchosen side f, β, bias  (Hattori's form)
  R1  + one-trial choice kernel w_ck · c(t−1)                                  (Bari's form)
  R2  + choice trace T ← λ_T T + (1 − λ_T) c(t), read out as w_T · T          (its own timescale)
  R3  + outcome-scaled trace: unrewarded choices add κ_unrew · c instead of c  (r6 I8)
  R4  + on a switch the trace is first multiplied by ρ                          (r8 H4)
  R5  + the one-trial kernel split by outcome: w_rew / w_unrew · c(t−1)        (r6 I4/I7's b[last input])
c = +1 right / −1 left. Q and T start at 0 in every session (as the RL fits). Writes
``h1_model_ladder.json``, ``_cache/h1_ladder_p_right_<key>.npz`` and ``fig_H1_model_ladder.png``.
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
from c4e_signatures import signature_curves  # noqa: E402

RUNGS = ["R0 Q-learning", "R1 + one-trial choice kernel", "R2 + choice trace", "R3 + outcome-scaled trace",
         "R4 + reset on switch", "R5 + outcome-split last-choice kernel"]
N_STEPS = 1500
LR = 0.03
SEED = 11
# Unconstrained parameter vector per mouse; transforms in ``unpack``.
NAMES = ["a_rew", "a_unrew", "forget", "beta", "bias", "w_ck", "lam_T", "w_T", "kappa_unrew", "rho", "w_ck_unrew"]
# New components start neutral: w = 0, κ_unrew = 1 (outcome-independent trace), ρ ≈ 1 (no reset);
# R5's w_ck_unrew starts at the fitted w_ck. λ_T starts at 0.5 (logit 0) so its fitted value is not the init.
INIT = {"a_rew": -1.0, "a_unrew": -2.0, "forget": -0.5, "beta": 1.5, "bias": 0.0, "w_ck": 0.0, "lam_T": 0.0,
        "w_T": 0.0, "kappa_unrew": 1.0, "rho": 6.0, "w_ck_unrew": 0.0}


def session_arrays(t: pd.DataFrame, p_gru: np.ndarray) -> dict:
    """Pad every session to [S, T] (trial order within session)."""
    ses = t["ses_idx"].to_numpy()
    codes, first_idx = np.unique(ses, return_index=True)
    order = np.argsort(first_idx)
    codes = codes[order]
    col = pd.Series(np.arange(len(codes)), index=codes).reindex(ses).to_numpy()
    pos = pd.Series(np.arange(len(t))).groupby(ses).cumcount().to_numpy()
    n_s, n_t = len(codes), int(pos.max()) + 1
    out = {"col": col, "pos": pos, "shape": (n_s, n_t)}
    for name, v, fill in (("choice", t["action"].to_numpy().astype(np.float32), 0.0),
                          ("reward", (t["reward"].to_numpy() > 0).astype(np.float32), 0.0),
                          ("target", p_gru.astype(np.float32), 0.5)):
        arr = np.full((n_s, n_t), fill, np.float32)
        arr[col, pos] = v
        out[name] = arr
    mask = np.zeros((n_s, n_t), np.float32)
    mask[col, pos] = 1
    out["mask"] = mask
    subjects = t["subject_id"].to_numpy()
    mice = list(dict.fromkeys(subjects))
    mouse_of_session = pd.Series(subjects).groupby(col).first().map({m: i for i, m in enumerate(mice)}).to_numpy()
    out["mouse"], out["mice"] = mouse_of_session, mice
    return out


def main() -> None:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import optax  # noqa: PLC0415

    s12.apply_presentation_style()
    print("devices", jax.devices(), flush=True)
    t = s12.load_trials(["subject_id", "ses_idx", "trial_idx", "action", "reward", "p_model1_right"])
    p_rl = {m: pd.read_parquet(s12.CACHE / f"rl_trials_{m}_{s12.MODEL_KEY}.parquet", columns=["p_rl_right"])["p_rl_right"].to_numpy()
            for m in ("hattori", "bari", "ctt")}
    d = session_arrays(t, t["p_model1_right"].to_numpy())
    n_mice = len(d["mice"])
    choice, reward, target, mask = (jnp.asarray(d[k]) for k in ("choice", "reward", "target", "mask"))
    mouse = jnp.asarray(d["mouse"])

    def unpack(theta):
        sig, sp = jax.nn.sigmoid, jax.nn.softplus
        th = dict(zip(NAMES, theta.T))
        return {"a_rew": sig(th["a_rew"]), "a_unrew": sig(th["a_unrew"]), "forget": sig(th["forget"]),
                "beta": sp(th["beta"]), "bias": th["bias"], "w_ck": th["w_ck"], "lam_T": sig(th["lam_T"]),
                "w_T": th["w_T"], "kappa_unrew": th["kappa_unrew"], "rho": sig(th["rho"]), "w_ck_unrew": th["w_ck_unrew"]}

    def policy(theta, rung: int):
        """logit(right) before every trial, [S, T]."""
        prm = {k: v[mouse] for k, v in unpack(theta).items()}  # per session
        n_s = choice.shape[0]

        def step(carry, inputs):
            ql, qr, tr_, prev_c, prev_r, started = carry
            c, r = inputs
            s_prev = jnp.where(started, 2 * prev_c - 1, 0.0)
            logit = prm["beta"] * (qr - ql) + prm["bias"]
            if rung >= 1:
                w_last = prm["w_ck"] if rung < 5 else jnp.where(prev_r > 0, prm["w_ck"], prm["w_ck_unrew"])
                logit = logit + w_last * s_prev
            if rung >= 2:
                logit = logit + prm["w_T"] * tr_
            a = jnp.where(r > 0, prm["a_rew"], prm["a_unrew"])
            ql_new = jnp.where(c == 0, ql + a * (r - ql), (1 - prm["forget"]) * ql)
            qr_new = jnp.where(c == 1, qr + a * (r - qr), (1 - prm["forget"]) * qr)
            s = 2 * c - 1
            if rung >= 2:
                gain = jnp.where(r > 0, 1.0, prm["kappa_unrew"]) if rung >= 3 else 1.0
                t_in = tr_ * jnp.where((rung >= 4) & started & (c != prev_c), prm["rho"], 1.0)
                tr_new = prm["lam_T"] * t_in + (1 - prm["lam_T"]) * gain * s
            else:
                tr_new = tr_
            return (ql_new, qr_new, tr_new, c, r, jnp.ones_like(started)), logit

        zeros = jnp.zeros(n_s)
        carry = (zeros, zeros, zeros, zeros, zeros, jnp.zeros(n_s, bool))
        _, logits = jax.lax.scan(step, carry, (choice.T, reward.T))
        return logits.T

    def loss(theta, rung):
        z = policy(theta, rung)
        ll = target * jax.nn.log_sigmoid(z) + (1 - target) * jax.nn.log_sigmoid(-z)
        return -jnp.sum(ll * mask) / jnp.sum(mask)

    results, p_rungs = {}, {}
    theta = jnp.tile(jnp.asarray([INIT[k] for k in NAMES]), (n_mice, 1))
    per_mouse = {}
    for rung, name in enumerate(RUNGS):
        if rung == 5:  # split kernel starts from the shared one
            theta = theta.at[:, NAMES.index("w_ck_unrew")].set(theta[:, NAMES.index("w_ck")])
        opt = optax.adam(LR)
        state = opt.init(theta)

        @jax.jit
        def update(th, st, rung=rung):
            val, g = jax.value_and_grad(loss)(th, rung)
            upd, st = opt.update(g, st)
            return optax.apply_updates(th, upd), st, val

        trace = []
        for i in range(N_STEPS):
            theta, state, val = update(theta, state)
            if i % 100 == 0 or i == N_STEPS - 1:
                trace.append(float(val))
        z = np.asarray(policy(theta, rung))
        p = 1 / (1 + np.exp(-z[d["col"], d["pos"]]))
        p_rungs[name] = p
        a_ = t["action"].to_numpy()
        lp = np.log(np.clip(np.where(a_ == 1, p, 1 - p), 1e-10, 1))
        per_mouse[name] = {k: np.asarray(v) for k, v in unpack(theta).items()}
        results[name] = {"loss_trace": trace, "fit_cross_entropy_to_gru": trace[-1],
                         "animal_likelihood": float(np.exp(lp.mean())),
                         "params_median": {k: float(np.median(np.asarray(v))) for k, v in unpack(theta).items()},
                         "params_iqr": {k: np.percentile(np.asarray(v), [25, 75]).tolist() for k, v in unpack(theta).items()},
                         "_lp": lp}
        print(name, "fit CE", round(trace[-1], 5), "trace", [round(x, 5) for x in trace[::3]],
              "animal lik", round(results[name]["animal_likelihood"], 5), flush=True)

    # Reference models.
    a_ = t["action"].to_numpy()
    ref = {"GRU": t["p_model1_right"].to_numpy(), **{m.capitalize() if m != "ctt" else "CTT": v for m, v in p_rl.items()}}
    ref_lik = {k: float(np.exp(np.log(np.clip(np.where(a_ == 1, v, 1 - v), 1e-10, 1)).mean())) for k, v in ref.items()}
    lp_ctt = np.log(np.clip(np.where(a_ == 1, p_rl["ctt"], 1 - p_rl["ctt"]), 1e-10, 1))
    subj = t["subject_id"].to_numpy()
    base, top = ref_lik["CTT"], ref_lik["GRU"]
    for name, r_ in results.items():
        r_["frac_mice_beat_ctt"] = float((pd.Series(r_.pop("_lp") - lp_ctt).groupby(subj).mean() > 0).mean())
        r_["gap_closed_vs_ctt"] = (r_["animal_likelihood"] - base) / (top - base)
    # C4e signatures per rung.
    sig_p = {"gru": ref["GRU"], **{name: p_rungs[name] for name in RUNGS}}
    row1, row2 = signature_curves(t["ses_idx"].to_numpy(), a_.astype(int), t["reward"].to_numpy() > 0, sig_p)
    sig_err = {}
    for label, dd in {**{f"return, {k}": v for k, v in row1.items()}, **{f"stay, {k}": v for k, v in row2.items()}}.items():
        sig_err[label] = {m: float(np.average(np.abs(dd[m] - dd["animal"]), weights=dd["n"])) for m in sig_p}
    np.savez_compressed(s12.CACHE / f"h1_ladder_p_right_{s12.MODEL_KEY}.npz", **{f"r{i}": p_rungs[n] for i, n in enumerate(RUNGS)})
    pd.DataFrame([{"rung": n, "subject_id": d["mice"][i], **{k: float(v[i]) for k, v in per_mouse[n].items()}}
                  for n in RUNGS for i in range(n_mice)]).to_parquet(s12.CACHE / f"h1_ladder_params_{s12.MODEL_KEY}.parquet")

    fig, axes = plt.subplots(1, 4, figsize=(32, 7), gridspec_kw={"width_ratios": [1.2, 1, 1, 1]})
    ax = axes[0]
    lik = [results[n]["animal_likelihood"] for n in RUNGS]
    ax.plot(range(len(RUNGS)), lik, color="#E69F00", marker="o", lw=2.5, label="ladder (fit to the GRU)")
    for k, color, ls in (("GRU", "#222222", "-"), ("CTT", "#C44E52", "--"), ("Hattori", "#55A868", ":"), ("Bari", "#4C72B0", ":")):
        ax.axhline(ref_lik[k], color=color, ls=ls, lw=1.5, label=f"{k} {ref_lik[k]:.4f}")
    ax.set_xticks(range(len(RUNGS)), [n.split(" ", 1)[0] for n in RUNGS])
    ax.set_ylabel("eval likelihood on the animals' choices")
    ax.set_title("Which component closes the gap to the GRU?", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    ax.text(0.98, 0.04, "\n".join(RUNGS), transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "0.8"})
    ax = axes[1]
    ax.plot(range(len(RUNGS)), [results[n]["fit_cross_entropy_to_gru"] for n in RUNGS], color="#4C72B0", marker="o", lw=2.5)
    ax.set_xticks(range(len(RUNGS)), [n.split(" ", 1)[0] for n in RUNGS])
    ax.set_ylabel("cross-entropy to the GRU's P(right) (nats/trial)")
    ax.set_title("Fit to the GRU", fontsize=13)
    ax = axes[2]
    ax.plot(range(len(RUNGS)), [results[n]["frac_mice_beat_ctt"] for n in RUNGS], color="#55A868", marker="o", lw=2.5)
    ax.axhline(0.5, color="0.6", lw=0.8)
    ax.set_xticks(range(len(RUNGS)), [n.split(" ", 1)[0] for n in RUNGS])
    ax.set_ylabel("fraction of mice where the rung beats CTT")
    ax.set_title("Per mouse", fontsize=13)
    ax = axes[3]
    for label, color in zip(sig_err, ("#C44E52", "#E69F00", "#4C72B0", "#55A868")):
        ax.plot(range(len(RUNGS)), [sig_err[label][n] for n in RUNGS], marker="o", lw=2, color=color, label=label)
        ax.axhline(sig_err[label]["gru"], color=color, ls="--", lw=1.8)
    ax.set_xticks(range(len(RUNGS)), [n.split(" ", 1)[0] for n in RUNGS])
    ax.set_ylabel("mean |model − animals| on the C4e curve")
    ax.set_yscale("log")
    ax.set_title("C4e signatures (dashed: GRU; log scale)", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_H1_model_ladder.png"))
    plt.close(fig)

    s12.write_json("h1_model_ladder.json", {
        "rungs": RUNGS, "results": results, "reference_likelihood": ref_lik, "signature_errors": sig_err,
        "fit": {"target": "GRU P(right), soft labels", "optimizer": "adam", "lr": LR, "steps_per_rung": N_STEPS,
                "warm_start": "each rung starts from the previous rung's fit; new components start neutral (w = 0, κ = 1, ρ ≈ 1; R5 w_ck_unrew = w_ck)",
                "init": INIT, "per_mouse": True, "per_mouse_params": f"_cache/h1_ladder_params_{s12.MODEL_KEY}.parquet"},
        "n_mice": n_mice,
    }, "analysis/h1_model_ladder.py", extra_groups=s12.rl_groups())
    print(ref_lik)
    print({n: (round(r_["animal_likelihood"], 5), round(r_["frac_mice_beat_ctt"], 3)) for n, r_ in results.items()})


if __name__ == "__main__":
    main()
