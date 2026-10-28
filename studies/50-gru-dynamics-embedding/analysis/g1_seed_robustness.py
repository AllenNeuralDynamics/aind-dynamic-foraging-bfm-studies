"""G1: do Study 50's headline findings hold for seeds 1 and 2 of model C?

    python analysis/g1_seed_robustness.py

Reads each seed's producer JSONs (seed 0 from ``analysis/``, others from
``analysis/seeds/<key>/``, written by ``g_run_seed.sh <key>``), pulls the number behind every
headline claim in SUMMARY.md, and checks each claim with the same criterion on every
seed. Seeds differ in their hidden-unit basis and embedding axes, so only basis-free
quantities are compared here (likelihoods, variance spectra, attractor counts, v's dynamics,
behavioural decoding from the embedding); G3 aligns the representations themselves.

Writes ``g1_seed_robustness.json`` and ``fig_G1_seed_robustness.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

KEYS = ["C-s0", "C-s1", "C-s2"]
SEED_COLORS = {"C-s0": "#222222", "C-s1": "#0072B2", "C-s2": "#D55E00"}
INPUTS = ["R rewarded", "R unrewarded", "L rewarded", "L unrewarded"]


def seed_dir(key: str) -> Path:
    return s12.STUDY / "analysis" if key == s12.DEFAULT_KEY else s12.STUDY / "analysis" / "seeds" / key


def load(key: str, name: str) -> dict | None:
    path = seed_dir(key) / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def metrics(key: str) -> dict[str, float]:
    """Every headline number for one seed; a missing producer leaves its metrics out."""
    m: dict[str, float] = {}
    a1 = json.loads((s12.STUDY / "analysis" / f"trial_states_{key}.json").read_text())
    m["eval likelihood (GRU)"] = a1["checks"]["recomputed_eval_likelihood"]

    if (c2 := load(key, "c2_compare")):
        for rl in ("ctt", "bari", "hattori"):
            m[f"likelihood {rl}"] = c2["likelihood"]["all"][rl]
            m[f"frac mice GRU > {rl}"] = c2["per_mouse_frac_gru_better"][rl]
        m["median per-mouse gain vs CTT"] = c2["per_mouse_median_gain"]["ctt"]
    if (c1 := load(key, "c1_kernels")):
        pooled = c1["pooled"]
        for model in ("gru", "ctt", "bari", "hattori"):
            gap = max(np.max(np.abs(np.subtract(pooled[model][t], pooled["animal"][t]))) for t in c1["terms"])
            m[f"kernel max |{model} − animal|"] = float(gap)
    if (c4 := load(key, "c4_behaviour")):
        for model, err in c4["history_calibration"]["weighted_mean_abs_error_pp"].items():
            m[f"3-trial history calibration error pp ({model})"] = err
    if (c4e := load(key, "c4e_signatures")):
        for model, err in c4e["mean_abs_error"]["return_after_switch"]["failed"].items():
            m[f"return after failed switch error ({model})"] = err

    if (pca := load(key, "b3_pca")):
        for i in range(5):
            m[f"PC{i + 1} variance share"] = pca["variance_explained_ratio"][i]
        m["participation ratio"] = pca["participation_ratio"]
        m["PCs for 90% variance"] = pca["n_pcs_for_variance"]["90%"]
    if (e1 := load(key, "e1_subspaces")):
        m["readout variance share"] = e1["readout_variance_share"]

    if (d := load(key, "d_dynamics")):
        ctx = d["contexts"]["all sessions"]["inputs"]
        for inp in INPUTS:
            fps = ctx[inp]["fixed_points"]
            side = 1 if inp.startswith("R") else -1
            m[f"fixed points ({inp})"] = ctx[inp]["n_fixed_points"]
            # "On the chosen side": every fixed point's logit has the sign of the input's choice.
            m[f"attractor on chosen side ({inp})"] = float(all(np.sign(fp["logit_right"]) == side for fp in fps))
            m[f"main attractor logit ({inp})"] = max((fp["logit_right"] for fp in fps), key=lambda x: side * x)
        for side in "RL":
            m[f"reward depth {side} (|rew| − |unrew| logit)"] = (abs(m[f"main attractor logit ({side} rewarded)"])
                                                              - abs(m[f"main attractor logit ({side} unrewarded)"]))
        # Same comparison in every context D searched (pooled, low/high gain, low/high bias).
        depths = [abs(max((f["logit_right"] for f in e["inputs"][f"{sd} rewarded"]["fixed_points"]), key=lambda x: sg * x))
                  - abs(max((f["logit_right"] for f in e["inputs"][f"{sd} unrewarded"]["fixed_points"]), key=lambda x: sg * x))
                  for e in d["contexts"].values() for sd, sg in (("R", 1), ("L", -1))]
        m["reward depth > 0, share of context × side"] = float(np.mean(np.array(depths) > 0))

    if (i3 := load(key, "i3_tdr_dynamics")):
        slopes = [t["slope"] for t in i3["transitions"].values()]
        r2s = [t["r2"] for t in i3["transitions"].values()]
        m["v carry slope min"], m["v carry slope max"] = min(slopes), max(slopes)
        m["v carry R² min"], m["v carry R² max"] = min(r2s), max(r2s)
    if (i4 := load(key, "i4_distilled_model")):
        m["1-variable model corr(v)"] = i4["simulated_vs_gru"]["corr_v"]
        m["1-variable model likelihood (per mouse)"] = i4["likelihood"]["distilled, per mouse"]
    if (i8 := load(key, "i8_v_tests")):
        m["v closure R² (v only), min over groups"] = min(g["v(t) only"] for g in i8["closure_heldout_r2"].values())
        m["v perturbation retained at k=1"] = i8["perturbation"]["measured_mean"][0]
    if (i9 := load(key, "i9_v_kernel")):
        m["v kernel held-out R² (free)"] = i9["fits"]["GRU v"]["free"]["heldout_r2"]
    if (i9d := load(key, "i9d_v_kernel_long")):
        pm = i9d["per_mouse"]["GRU v"]["K20"]
        m["three traces best BIC share (mice)"] = pm["bic_best_share"]["three traces"]
        m["share mice λ_unr > λ_rew (two outcomes)"] = pm["share_lambda_unr_gt_rew"]
        for p, v in pm["three_traces_param_medians"].items():
            m[f"three traces median {p}"] = v
    if (i10 := load(key, "i10_gates")):
        m["v axis participation ratio (units)"] = i10["participation_ratio_of_v_axis"]
        m["v retention eᵀJe (median over groups)"] = float(np.median([g["eJe_mean"] for g in i10["groups"].values()]))
    if (e2b := load(key, "e2b_conditional_ablations")):
        r = e2b["results"]
        for ax in ("value axis", "reward axis", "choice axis", "readout direction"):
            m[f"ablation Δ {ax}"] = r[ax]["delta_vs_none"]
        m["ablation Δ random 1-d (mean)"] = e2b["random_summary"]["random 1-d"]["mean_delta"]
        m["ablation Δ random 1-d (worst)"] = e2b["random_summary"]["random 1-d"]["min_delta"]

    if (h2 := load(key, "h2_learning_rate")):
        m["reward effect on logit (GRU)"] = h2["mean_effect"]["gru"]
        stay_switch = h2["by_state"]["stay vs switch"]["gru"]
        m["reward effect switch / stay (GRU)"] = stay_switch["switch"] / stay_switch["stay"]
    if (h3 := load(key, "h3_q_phase_portraits")):
        m["Q decoding R² (min of Q_L, Q_R)"] = min(h3["decoding_heldout_r2"].values())
        for inp in ("R rewarded", "R unrewarded"):
            reg = h3["update_regressions"][inp]["gru"]["chosen"]
            m[f"Q_chosen update a ({inp})"] = reg["a"]
            m[f"Q_chosen update b ({inp})"] = reg["b_Qchosen"]

    if (f2 := load(key, "f2_mouse_dynamics")):
        m["mice with all four attractors"] = f2["n_mice_with_all_four_attractors"] / f2["n_mice"]
        m["mice bistable for any input"] = float(sum(f2["n_bistable_mice_by_input"].values()))
        m["per-mouse reward depth (median)"] = f2["feature_summary"]["reward depth (rew − unrew)"][1]
        m["per-mouse slowest τ (median, trials)"] = f2["feature_summary"]["slowest τ (trials)"][1]
        for feat in ("attractor depth", "v retention (e_vᵀJe_v)"):
            m[f"embedding decodes {feat} (rank R²)"] = f2["embedding_decoding"][feat]["rank_r2"]
    if (f5 := load(key, "f5_embedding_atlas")):
        for i, share in enumerate(f5["embedding_pc_variance_share"]):
            m[f"embedding PC{i + 1} variance share"] = share
        for feat in ("gru bias", "gru reward gain (1 back)", "ctt learn_rate", "hattori softmax_inverse_temperature"):
            m[f"embedding decodes {feat} (rank R²)"] = f5["numeric_decoding"][feat]["rank_r2"]
    if (f6 := load(key, "f6_embedding_types")):
        mix = f6["mixtures"]["embedding"]
        m["mixture best k (BIC)"] = mix["best_k"]
        m["mixture BIC gain − heavy-tail null 95%"] = mix["gain_best_vs_1"] - mix["t_null_gain_95"]
        m["mixture k=2 separation (Mahalanobis)"] = mix["k2_mean_separation_mahalanobis"]
        m["Mantel ρ embedding vs animal behaviour"] = f6["mantel"]["animal behaviour"]["rho"]
    if (f10 := load(key, "f10_embedding_behaviour_maps")):
        for feat in ("determinism", "switch rate", "bias"):
            if feat in f10["in_sample_r2"]["animal"]:
                m[f"embedding explains animal {feat} (R²)"] = f10["in_sample_r2"]["animal"][feat]
        # Angles between behaviour directions in the whitened embedding: basis-free, unlike PC loadings.
        cos = f10["direction_cosines"]["embedding_covariance"]
        for a, b in (("animal determinism", "ctt learn_rate"), ("animal determinism", "animal bias"),
                     ("ctt learn_rate", "animal bias"), ("animal determinism", "animal reward gain (1 back)")):
            m[f"cos({a.removeprefix('animal ')}, {b.removeprefix('animal ')}) in embedding"] = cos[a][b]
    return m


# Each headline claim (SUMMARY.md) and the per-seed test that decides whether it holds.
CLAIMS = [
    ("GRU beats every RL model, in ≥ 99% of mice",
     lambda m: all(m[f"likelihood {rl}"] < m["eval likelihood (GRU)"] and m[f"frac mice GRU > {rl}"] >= 0.99
                   for rl in ("ctt", "bari", "hattori"))),
    ("GRU's choice kernels match the animals' (< 0.03) and ≥ 3× better than any RL model",
     lambda m: m["kernel max |gru − animal|"] < 0.03 and m["kernel max |gru − animal|"] * 3
     < min(m[f"kernel max |{r} − animal|"] for r in ("ctt", "bari", "hattori"))),
    ("GRU calibrated over 3-trial histories (< 0.5 pp; every RL model ≥ 3× worse)",
     lambda m: m["3-trial history calibration error pp (gru)"] < 0.5
     and min(m[f"3-trial history calibration error pp ({r})"] for r in ("ctt", "bari", "hattori"))
     > 3 * m["3-trial history calibration error pp (gru)"]),
    ("Low-dimensional state (PC1 > 35%, participation ratio < 6)",
     lambda m: m["PC1 variance share"] > 0.35 and m["participation ratio"] < 6),
    ("Attractors on the chosen side; one per input for ≥ 3 of 4 inputs",
     lambda m: all(m[f"attractor on chosen side ({i})"] == 1 for i in INPUTS)
     and sum(m[f"fixed points ({i})"] == 1 for i in INPUTS) >= 3),
    ("Rewarded attractors deeper than unrewarded (every context × side)",
     lambda m: m["reward depth > 0, share of context × side"] == 1),
    ("Readout carries < 15% of the state variance (memory sits off the readout)",
     lambda m: m["readout variance share"] < 0.15),
    ("v carried through every jump (slopes 0.6–0.85, R² > 0.9)",
     lambda m: m["v carry slope min"] > 0.6 and m["v carry slope max"] < 0.85 and m["v carry R² min"] > 0.9),
    ("A one-variable model of v tracks the GRU (corr > 0.97)",
     lambda m: m["1-variable model corr(v)"] > 0.97),
    ("v closes on itself (v-only R² > 0.9 in every transition group)",
     lambda m: m["v closure R² (v only), min over groups"] > 0.9),
    ("Three-trace kernel wins by BIC in > 95% of mice",
     lambda m: m["three traces best BIC share (mice)"] > 0.95),
    ("Ablating v costs more than any random direction; the reward axis costs most",
     lambda m: m["ablation Δ value axis"] < m["ablation Δ random 1-d (worst)"]
     and m["ablation Δ reward axis"] < m["ablation Δ value axis"]),
    ("Non-saturating learning rate (reward effect 0.8–1.2 logit; switch/stay 0.9–1.1)",
     lambda m: 0.8 < m["reward effect on logit (GRU)"] < 1.2 and 0.9 < m["reward effect switch / stay (GRU)"] < 1.1),
    ("Delta rule after rewards in Q coordinates (|a + b| < 0.1), small update after losses (a < 0.15)",
     lambda m: abs(m["Q_chosen update a (R rewarded)"] + m["Q_chosen update b (R rewarded)"]) < 0.1
     and m["Q_chosen update a (R unrewarded)"] < 0.15),
    ("Embedding decodes bias, reward gain and learning rate (rank R² > 0.5)",
     lambda m: min(m[f"embedding decodes {f} (rank R²)"] for f in ("gru bias", "gru reward gain (1 back)", "ctt learn_rate")) > 0.5),
    ("Embedding is a continuum (k = 2 components < 1.5 SD apart)",
     lambda m: m["mixture best k (BIC)"] == 1 or m["mixture k=2 separation (Mahalanobis)"] < 1.5),
    ("Bias is orthogonal to stickiness and learning rate (|cos| < 0.3); those two are not (cos > 0.4)",
     lambda m: abs(m["cos(determinism, bias) in embedding"]) < 0.3 and abs(m["cos(ctt learn_rate, bias) in embedding"]) < 0.3
     and abs(m["cos(determinism, ctt learn_rate) in embedding"]) > 0.4),
    ("Every mouse's context has all four attractors (> 99% of mice)",
     lambda m: m["mice with all four attractors"] > 0.99),
]


def evaluate(per_seed: dict[str, dict]) -> list[dict]:
    out = []
    for claim, test in CLAIMS:
        row = {"claim": claim}
        for key, m in per_seed.items():
            try:
                row[key] = bool(test(m))
            except KeyError as missing:
                row[key] = f"missing {missing}"
        out.append(row)
    return out


def rl_only(name: str) -> bool:
    """RL baselines' own numbers: the same fits on the same trials for every seed, so equal by construction."""
    return (name.startswith("likelihood ") or any(f"|{r} " in name or name.endswith(f"({r})") for r in ("ctt", "bari", "hattori")))


def plot(per_seed: dict[str, dict], claims: list[dict]) -> None:
    names = [n for n in per_seed[KEYS[0]] if all(n in per_seed[k] for k in per_seed) and not rl_only(n)]
    fig = plt.figure(figsize=(15, 0.2 * len(names) + 1.7))
    gs = fig.add_gridspec(1, len(per_seed) + 1, width_ratios=[0.45] * len(per_seed) + [1.2], wspace=0.04)
    # One column per seed with the values printed, then the spread relative to seed 0.
    for j, key in enumerate(per_seed):
        ax = fig.add_subplot(gs[0, j])
        ax.set_xlim(0, 1)
        ax.set_ylim(len(names) - 0.5, -0.5)
        for i, n in enumerate(names):
            ax.text(0.5, i, f"{per_seed[key][n]:.4g}", ha="center", va="center", fontsize=7, color=SEED_COLORS[key])
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color="0.94", zorder=-1)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names if j == 0 else [], fontsize=7)
        ax.set_xticks([])
        ax.set_title(key, color=SEED_COLORS[key], fontsize=10)
    ax = fig.add_subplot(gs[0, -1])
    lim = 0
    for i, n in enumerate(names):
        ref = per_seed[KEYS[0]][n]
        for key in per_seed:
            pct = (per_seed[key][n] - ref) / max(abs(ref), 1e-9) * 100
            lim = max(lim, abs(pct))
            ax.plot(pct, i, "o", ms=3.5, color=SEED_COLORS[key])
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="0.94", zorder=-1)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-lim * 1.2 - 1, lim * 1.2 + 1)
    ax.set_xlabel("% difference from seed 0 (symlog; full range)")
    ax.set_title("spread across seeds", fontsize=10)
    held = sum(all(r[k] is True for k in per_seed) for r in claims)
    fig.suptitle(f"G1 · model C seeds 0–2: headline numbers; {held} of {len(claims)} claims hold in all three seeds\n"
                 "(RL baselines' own numbers omitted: same fits on the same trials, identical by construction)",
                 fontsize=11)
    fig.subplots_adjust(top=1 - 1.1 / fig.get_figheight(), bottom=0.6 / fig.get_figheight())
    fig.savefig(s12.figure_path("fig_G1_seed_robustness.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    per_seed = {k: metrics(k) for k in KEYS}
    claims = evaluate(per_seed)
    for r in claims:
        print(" | ".join([f"{r[k]!s:>5}" for k in KEYS] + [r["claim"]]))
    spread = {}
    for n in per_seed[KEYS[0]]:
        vals = [per_seed[k].get(n) for k in KEYS]
        if None not in vals:
            spread[n] = {"values": vals, "range": float(np.ptp(vals)),
                         "rel_range": float(np.ptp(vals) / max(abs(np.mean(vals)), 1e-12))}
    s12.write_json("g1_seed_robustness.json", {
        "seeds": KEYS, "claims": claims, "metrics": per_seed, "spread": spread,
        "note": "basis-free quantities only; G3 aligns the representations across seeds",
    }, produced_by="analysis/g1_seed_robustness.py")
    plot(per_seed, claims)


if __name__ == "__main__":
    main()
