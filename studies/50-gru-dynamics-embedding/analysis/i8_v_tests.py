"""I8: is v a self-contained leaky integrator, and an integrator of what? (stage A, CPU)

    JAX_PLATFORMS=cpu python analysis/i8_v_tests.py

v = I2's value coordinate (I4 ``value_coordinate``). Three tests:

1. Closure. Per transition group (input × stay/switch), predict v(t+1) from v(t) alone, from v
   plus the top 20 PCs, and from the full 128-d state (held-out R², 80/20 split by session).
   If v is self-contained, the rest of the state adds little.
2. Perturbation. Add ±δ along v's axis to the GRU state before choice t, run the GRU forward on
   the observed inputs (wrapper ``GruStep`` with the cached session contexts) and measure
   Δv(t+k)/δ, k = 0…K. A one-variable leaky integrator predicts Π_j λ_g(t+j), with λ_g I4's
   pooled per-group retention along the actual inputs. Also the share of the perturbation that
   leaks out of v's axis, and the logit effect.
3. Causal kernels, GRU vs RL. Flip trial t's reward (choice kept), or its choice (outcome kept),
   and measure the effect on later trials, signed by trial t's side:
     reward effect   R(k) = c · [x | rewarded − x | unrewarded]
     choice effect   C(k) = x | chose R − x | chose L, separately for unrewarded and rewarded t.
   For the GRU, x = a·v (v in logit units, a = I7's v-only readout slope) and the full logit.
   For Hattori and Bari (per-mouse fitted parameters, numpy replay from the cached Q values,
   validated against the cached rollouts), x = β·ΔQ and Bari's choice-kernel term
   β·w_ck·(CK_R − CK_L). A choice kernel has C(unrewarded) ≈ C(rewarded) > 0 and R ≈ 0; a
   reward-driven value has R > 0 and C(unrewarded) ≤ 0.
Writes ``i8_v_tests.json`` and ``fig_I8_v_tests.png``.
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
import _transitions as tr  # noqa: E402
from i4_distilled_model import value_coordinate  # noqa: E402

SEED = 8
N_PROBES = 20_000
K = 10
DELTA = 0.5
N_PCS = 20
BATCH = 512
RL_RUNS = {"hattori": "unhmbrk4", "bari": "bg3nzqz9"}
FIT_GROUPS = [g for g, n in enumerate(tr.GROUP_NAMES) if not n.endswith("(first)")]


def closure(v: np.ndarray, lay: dict, ses: np.ndarray) -> dict:
    """Held-out R² of v(t+1) from v(t) / v + top PCs / the full state, per group."""
    pca = s12.load_pca()
    states = s12.load_trials(s12.STATE_COLUMNS).to_numpy(dtype=np.float32)
    rng = np.random.default_rng(SEED)
    uniq = np.unique(ses)
    test_ses = set(rng.choice(uniq, size=len(uniq) // 5, replace=False))
    is_test = np.isin(ses, list(test_ses))
    idx = np.flatnonzero(lay["has_next"])
    out = {}
    for g in FIT_GROUPS:
        sel = idx[lay["group"][idx] == g]
        y = v[sel + 1]
        h = states[sel].astype(np.float64) - pca["mean"]
        designs = {"v(t) only": v[sel, None], f"v + top {N_PCS} PCs": np.column_stack([v[sel], h @ pca["components"][:N_PCS].T]),
                   "full 128-d state": h}
        tr_, te = ~is_test[sel], is_test[sel]
        out[tr.GROUP_NAMES[g]] = {}
        for name, x in designs.items():
            x1 = np.column_stack([x, np.ones(len(x))])
            coef, *_ = np.linalg.lstsq(x1[tr_], y[tr_], rcond=None)
            res = y[te] - x1[te] @ coef
            out[tr.GROUP_NAMES[g]][name] = float(1 - res.var() / y[te].var())
    return out


def rl_params(name: str, subjects: np.ndarray) -> dict:
    fits = json.loads((s12.CACHE / "models" / "rl" / RL_RUNS[name] / "baseline_rl_output.json").read_text())["fitted_params_per_subject"]
    keys = {"hattori": ("learn_rate_rew", "learn_rate_unrew"), "bari": ("learn_rate", "learn_rate")}[name]
    out = {k: np.empty(len(subjects)) for k in ("a_rew", "a_unrew", "forget", "beta", "biasL", "w_ck")}
    for i, s in enumerate(subjects):
        p = fits[str(s)]["fitted_params"]
        out["a_rew"][i], out["a_unrew"][i] = p[keys[0]], p[keys[1]]
        out["forget"][i], out["beta"][i], out["biasL"][i] = p["forget_rate_unchosen"], p["softmax_inverse_temperature"], p["biasL"]
        out["w_ck"][i] = p.get("choice_kernel_relative_weight", 0.0)
    return out


def rl_step(qL, qR, choice, reward, prm):
    a = np.where(reward > 0, prm["a_rew"], prm["a_unrew"])
    new_l = np.where(choice == 0, qL + a * (reward - qL), (1 - prm["forget"]) * qL)
    new_r = np.where(choice == 1, qR + a * (reward - qR), (1 - prm["forget"]) * qR)
    return new_l, new_r


def rl_kernels(name: str, t: pd.DataFrame, probes: np.ndarray, action: np.ndarray, reward: np.ndarray) -> tuple[dict, dict]:
    """Reward / choice flips for an RL model from the cached Q values; returns kernels and checks."""
    rl = pd.read_parquet(s12.CACHE / f"rl_trials_{name}_{s12.MODEL_KEY}.parquet")
    qL_all, qR_all = rl["q_rl_left"].to_numpy(), rl["q_rl_right"].to_numpy()
    prm = rl_params(name, t["subject_id"].to_numpy()[probes])
    # Validation: one update from Q(t) with the observed input reproduces Q(t+1).
    l1, r1 = rl_step(qL_all[probes], qR_all[probes], action[probes], reward[probes], prm)
    checks = {"max_abs_q_update_error": float(max(np.abs(l1 - qL_all[probes + 1]).max(), np.abs(r1 - qR_all[probes + 1]).max()))}
    p = np.clip(rl["p_rl_right"].to_numpy()[probes + 1], 1e-12, 1 - 1e-12)
    ck = 2.0 * action[probes] - 1.0 if name == "bari" else 0.0
    logit = prm["beta"] * (qR_all[probes + 1] - qL_all[probes + 1]) - prm["biasL"] + prm["beta"] * prm["w_ck"] * ck
    checks["max_abs_logit_error"] = float(np.abs(logit - np.log(p / (1 - p))).max())

    def run(edit: str):
        qL, qR = qL_all[probes].copy(), qR_all[probes].copy()
        q_term, ck_term = np.empty((len(probes), K)), np.empty((len(probes), K))
        for j in range(K):
            c, r = action[probes + j].copy(), reward[probes + j].copy()
            if j == 0 and edit == "reward":
                r = 1.0 - r
            if j == 0 and edit == "choice":
                c = 1 - c
            qL, qR = rl_step(qL, qR, c, r, prm)
            q_term[:, j] = prm["beta"] * (qR - qL)
            ck_term[:, j] = prm["beta"] * prm["w_ck"] * (2.0 * c - 1.0)
        return q_term, ck_term

    base, rew, cho = run("none"), run("reward"), run("choice")
    kern = {"β·ΔQ": flip_kernels(base[0], rew[0], cho[0], action[probes], reward[probes])}
    if name == "bari":
        kern["choice-kernel term"] = flip_kernels(base[1], rew[1], cho[1], action[probes], reward[probes])
        kern["full logit"] = flip_kernels(base[0] + base[1], rew[0] + rew[1], cho[0] + cho[1], action[probes], reward[probes])
    return kern, checks


def flip_kernels(base, rew_flip, cho_flip, choice, reward) -> dict:
    """Signed reward and choice effects (means over probes) from baseline and flipped traces [N, K]."""
    c = 2.0 * choice - 1.0
    r_eff = c[:, None] * np.where(reward[:, None] > 0, base - rew_flip, rew_flip - base)
    c_eff = c[:, None] * (base - cho_flip)  # x | chose R − x | chose L at the observed outcome
    out = {"reward effect": r_eff.mean(axis=0)}
    for lab, m in (("choice effect, unrewarded", reward == 0), ("choice effect, rewarded", reward > 0)):
        out[lab] = c_eff[m].mean(axis=0)
        out[lab + " (sem)"] = c_eff[m].std(axis=0, ddof=1) / np.sqrt(m.sum())
    out["reward effect (sem)"] = r_eff.std(axis=0, ddof=1) / np.sqrt(len(r_eff))
    return out


def main() -> None:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from post_training_analysis.gru_counterfactual import load_gru_split  # noqa: PLC0415
    from post_training_analysis.gru_step import GruStep, split_inputs  # noqa: PLC0415

    s12.apply_presentation_style()
    t = s12.load_trials(["subject_id", "ses_idx", "trial_idx", "action", "reward", "logit_diff"])
    action = t["action"].to_numpy().astype(int)
    reward = (t["reward"].to_numpy() > 0).astype(float)
    ses = t["ses_idx"].to_numpy()
    lay = tr.session_layout(t)
    v = value_coordinate()

    # 1. Closure.
    clo = closure(v, lay, ses)
    print("closure", {g: {k: round(x, 4) for k, x in d.items()} for g, d in clo.items()}, flush=True)

    # Probes: trials t with K later trials in the session and a previous trial (no first rows).
    pos = pd.Series(np.arange(len(t))).groupby(ses).cumcount().to_numpy()
    n_in = pd.Series(ses).map(pd.Series(ses).value_counts()).to_numpy()
    eligible = np.flatnonzero((pos >= 1) & (pos + K < n_in))
    probes = np.sort(np.random.default_rng(SEED).choice(eligible, size=N_PROBES, replace=False))

    # GRU set-up (as E2b): free-context step, cached contexts, split session order.
    split = load_gru_split(s12.CACHE / "models" / s12.MODEL_KEY / "run")
    step = GruStep.from_params(split.params)
    _, _, obs = split_inputs(split.xs, session_conditioning=split.run.session_conditioning_enabled)
    ctx = np.load(s12.CACHE / f"contexts_{s12.MODEL_KEY}.npz")
    if list(ctx["session_ids"]) != split.session_ids:
        raise AssertionError("Cached contexts are not in the split's session order")
    contexts = ctx["contexts"].astype(np.float32)
    col = pd.Series(np.arange(len(split.session_ids)), index=split.session_ids).reindex(ses[probes]).to_numpy()
    tau = t["trial_idx"].to_numpy()[probes] - 1  # state index before choice t; trial t's input is obs[tau + 1]

    run_full = jax.jit(step.run)
    h_t = np.empty((N_PROBES, step.hidden_size), dtype=np.float32)
    for start in range(0, obs.shape[1], BATCH):
        cols = np.arange(start, min(start + BATCH, obs.shape[1]))
        mine = np.flatnonzero(np.isin(col, cols))
        if len(mine) == 0:
            continue
        pad = np.r_[cols, np.full(BATCH - len(cols), cols[-1])]
        c = np.broadcast_to(contexts[pad][None], obs[:, pad].shape[:2] + (contexts.shape[1],))
        states = np.asarray(run_full(jnp.asarray(obs[:, pad]), jnp.asarray(c)))
        h_t[mine] = states[tau[mine], col[mine] - start]
    pca = s12.load_pca()
    tdr = np.load(s12.keyed_cache("i2_tdr_axes.npz"))
    terms = [str(x) for x in tdr["terms"]]
    e_v = tdr["basis"][:, terms.index("value")].astype(np.float32)
    v_check = float(np.median(np.abs((h_t.astype(np.float64) - pca["mean"]) @ e_v - v[probes])))
    print("CPU replay vs cached v (median |diff|)", round(v_check, 4), flush=True)

    seq = obs[tau[:, None] + 1 + np.arange(K)[None], col[:, None]]  # [N, K, 2], inputs of trials t … t+K−1
    ctx_p = np.broadcast_to(contexts[col][:, None], (N_PROBES, K, contexts.shape[1]))
    run_short = jax.jit(lambda h0, o, cc: step.run(jnp.swapaxes(o, 0, 1), jnp.swapaxes(cc, 0, 1), h0=h0))

    def traces(h0, o):
        out_v, out_l, out_h = [], [], []
        for s in range(0, N_PROBES, 4096):
            sl = slice(s, s + 4096)
            st = np.asarray(run_short(jnp.asarray(h0[sl]), jnp.asarray(o[sl]), jnp.asarray(ctx_p[sl])))  # [K, n, H]
            st = np.swapaxes(st, 0, 1)
            out_h.append(st)
            out_v.append(st @ e_v)
            out_l.append(np.asarray(step.logit_diff(jnp.asarray(st))))
        return np.concatenate(out_v), np.concatenate(out_l), np.concatenate(out_h)

    base_v, base_l, base_h = traces(h_t, seq)

    # 2. Perturbation along v's axis.
    up_v, up_l, up_h = traces(h_t + DELTA * e_v, seq)
    dn_v, dn_l, dn_h = traces(h_t - DELTA * e_v, seq)
    dv = (up_v - dn_v) / (2 * DELTA)  # [N, K], k = 1..K
    dl = (up_l - dn_l) / (2 * DELTA)
    dh = (up_h - dn_h) / (2 * DELTA)
    spill = np.linalg.norm(dh - dv[..., None] * e_v, axis=-1) / np.linalg.norm(dh, axis=-1)
    i4 = json.loads((s12.ANALYSIS / "i4_distilled_model.json").read_text())
    lam = np.array(i4["pooled"]["lam"], dtype=float)
    g = lay["group"][probes[:, None] + np.arange(K)[None]]
    pred = np.cumprod(lam[g], axis=1)
    r_hat = np.asarray(step.readout_w[:, 1] - step.readout_w[:, 0])
    pert = {"k": list(range(1, K + 1)), "measured_mean": dv.mean(0), "measured_q25": np.percentile(dv, 25, 0),
            "measured_q75": np.percentile(dv, 75, 0), "predicted_mean": pred.mean(0),
            "per_probe_corr": [float(np.corrcoef(dv[:, k], pred[:, k])[0, 1]) for k in range(K)],
            "spill_median": np.median(spill, 0), "logit_per_unit_v": dl.mean(0),
            "immediate_logit_per_unit_v": float(r_hat @ e_v)}
    print("perturbation", {k: np.round(v_, 3).tolist() if isinstance(v_, np.ndarray) else v_ for k, v_ in pert.items()}, flush=True)

    # 3. Causal kernels by input flips.
    rew_seq, cho_seq = seq.copy(), seq.copy()
    rew_seq[:, 0, 1] = 1.0 - rew_seq[:, 0, 1]
    cho_seq[:, 0, 0] = 1.0 - cho_seq[:, 0, 0]
    rv, rl_, _ = traces(h_t, rew_seq)
    cv, cl, _ = traces(h_t, cho_seq)
    a_v = float(json.loads((s12.ANALYSIS / "i7_second_variable.json").read_text())["results"]["v only"]["readout_a"][0])
    obs_c, obs_r = action[probes], reward[probes]
    if not (np.array_equal(seq[:, 0, 0], obs_c) and np.array_equal(seq[:, 0, 1], obs_r)):
        raise AssertionError("GRU input row does not match the trial table's choice/reward")
    kernels = {"GRU": {"a·v": flip_kernels(a_v * base_v, a_v * rv, a_v * cv, obs_c, obs_r),
                       "full logit": flip_kernels(base_l, rl_, cl, obs_c, obs_r)}}
    checks = {"cpu_vs_cached_v_median_abs": v_check}
    for name in RL_RUNS:
        kernels[name], checks[name] = rl_kernels(name, t, probes, action, reward)
    print("checks", checks, flush=True)

    # Figure.
    ks = np.arange(1, K + 1)
    fig, axes = plt.subplots(2, 5, figsize=(37, 13))
    ax = axes[0, 0]
    labels = list(clo)
    y = np.arange(len(labels))
    for j, (name, color) in enumerate(zip(clo[labels[0]], ("#E69F00", "#4C72B0", "#222222"))):
        ax.barh(y + (j - 1) * 0.27, [clo[lab][name] for lab in labels], height=0.27, color=color, label=name)
    ax.set_yticks(y, labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(min(min(d.values()) for d in clo.values()) - 0.01, 1.0)
    ax.set_xlabel("held-out R² of v(t+1)")
    ax.set_title("1. Closure: does the rest of the state predict v's next value?", fontsize=13)
    ax.legend(frameon=False, fontsize=10, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=3)
    ax = axes[0, 1]
    ax.fill_between(ks, pert["measured_q25"], pert["measured_q75"], color="#E69F00", alpha=0.25)
    ax.plot(ks, pert["measured_mean"], color="#E69F00", marker="o", lw=2, label="GRU: Δv(t+k) / δ (mean, IQR)")
    ax.plot(ks, pert["predicted_mean"], color="k", ls="--", marker="s", lw=1.5, label="one-variable model: Π λ (I4)")
    ax.set_xlabel("trials after the perturbation (k)")
    ax.set_ylabel("remaining perturbation of v")
    ax.set_title(f"2. Perturb v by ±{DELTA} and run the GRU forward", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    ax = axes[0, 2]
    ax.plot(ks, pert["spill_median"], color="#C44E52", marker="o", lw=2, label="share of Δh off v's axis (median)")
    ax.plot(ks, pert["per_probe_corr"], color="#4C72B0", marker="s", lw=2, label="per-probe corr(measured, Π λ)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("k")
    ax.set_title("2. Does the perturbation stay on v's axis?", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    styles = {"reward effect": ("#222222", "-"), "choice effect, unrewarded": ("#C44E52", "-"),
              "choice effect, rewarded": ("#0072B2", "--")}
    panels = [(axes[0, 3], "GRU full logit", kernels["GRU"]["full logit"]),
              (axes[1, 0], "GRU: v (logit units, a·v)", kernels["GRU"]["a·v"]),
              (axes[1, 1], "Hattori (L2F1): β·ΔQ", kernels["hattori"]["β·ΔQ"]),
              (axes[1, 2], "Bari (L1F1_CK1): β·ΔQ", kernels["bari"]["β·ΔQ"]),
              (axes[1, 3], "Bari: choice-kernel term", kernels["bari"]["choice-kernel term"])]
    for ax, title, kern in panels:
        for lab, (color, ls) in styles.items():
            ax.errorbar(ks, kern[lab], yerr=kern[lab + " (sem)"], color=color, ls=ls, marker="o", ms=4, lw=2, label=lab)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlabel("trials after the flipped trial (k)")
        ax.set_ylabel("effect on log-odds of right (signed by side)")
        ax.set_title(f"3. Flip one input: {title}", fontsize=13)
        ax.legend(frameon=False, fontsize=10)
    lim = max(np.nanmax(np.abs(np.r_[kern[lab]])) for _, _, kern in panels for lab in styles)
    for ax, _, _ in panels:
        ax.set_ylim(-0.25 * lim, 1.05 * lim)
    ratios = {"GRU full logit": kernels["GRU"]["full logit"], "GRU v": kernels["GRU"]["a·v"],
              "Hattori β·ΔQ (= its logit − bias)": kernels["hattori"]["β·ΔQ"], "Bari full logit": kernels["bari"]["full logit"],
              "Bari β·ΔQ": kernels["bari"]["β·ΔQ"]}
    ax = axes[0, 4]
    for (lab, kern), color, ls in zip(ratios.items(), ("#222222", "#E69F00", "#55A868", "#4C72B0", "#4C72B0"),
                                      ("-", "-", "-", "-", ":")):
        ax.plot(ks, kern["choice effect, unrewarded"] / kern["choice effect, rewarded"], color=color, ls=ls, marker="o",
                lw=2, label=lab)
    ax.set_ylim(0, 1.45)
    ax.set_xlabel("trials after the flipped trial (k)")
    ax.set_ylabel("choice effect: unrewarded / rewarded")
    ax.set_title("Is the choice trace separable (outcome × time)?", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    ax = axes[1, 4]
    for (lab, kern), color, ls in zip(ratios.items(), ("#222222", "#E69F00", "#55A868", "#4C72B0", "#4C72B0"),
                                      ("-", "-", "-", "-", ":")):
        ax.plot(ks, kern["reward effect"] / kern["choice effect, rewarded"], color=color, ls=ls, marker="o", lw=2, label=lab)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("trials after the flipped trial (k)")
    ax.set_ylabel("reward effect / rewarded-choice effect")
    ax.set_title("How much of a rewarded choice's effect is the reward?", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_I8_v_tests.png"))
    plt.close(fig)

    s12.write_json("i8_v_tests.json", {
        "closure_heldout_r2": clo, "perturbation": pert, "flip_kernels": kernels, "checks": checks,
        "n_probes": N_PROBES, "K": K, "delta": DELTA, "n_pcs": N_PCS, "seed": SEED, "readout_a_v": a_v,
        "conventions": {"reward effect": "c·(x|rewarded − x|unrewarded) at trial t+k",
                        "choice effect": "x|chose R − x|chose L, outcome kept as observed",
                        "units": "log-odds of right; GRU v scaled by I7's v-only readout slope"},
    }, "analysis/i8_v_tests.py", extra_groups=s12.rl_groups())
    for model, parts in kernels.items():
        for part, kern in parts.items():
            print(model, part, {lab: np.round(kern[lab], 3).tolist() for lab in styles})


if __name__ == "__main__":
    main()
