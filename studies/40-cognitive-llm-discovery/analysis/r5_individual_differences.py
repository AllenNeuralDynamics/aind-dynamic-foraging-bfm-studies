#!/usr/bin/env python
"""r5: individual differences in mechanism fit on the 100 monitor subjects (issue #146). Offline.

Reads ``individual/sessions.json`` (programs, score sessions and covariates) and
``individual/scores/<tag>.json`` (17 programs; per subject and score session the 2-level and per-subject
held-out log-likelihood), written by ``freeze_individual.py``. Every statistic was fixed on #146 before
scoring:

- each subject's score sessions are split by order into two halves (A: 1st, 3rd, ...; B: 2nd, 4th, ...);
  a cell is a subject's log-likelihood per trial on one half; a subject that falls back (MEvolve ADR 0004)
  uses its per-subject values;
- primary: the correlation between halves of the double-centred program x subject matrix (program and
  subject means removed), with a bootstrap over subjects;
- secondary: per-program split-half correlations of the advantage over the all-program mean; a
  cross-validated per-subject choice (best program on one half, scored on the other) against the single
  best program; mechanism effects (search programs with an add-on minus those without) and their
  reliability and covariates; the union test (arch08 against arch14, arch04, arch03); correlations of
  per-subject advantage between programs.

Writes ``r5_individual_differences.json``, three figures and the report's result blocks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
import figure_style as fs  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402
from freeze_seeded_islands_2 import names  # noqa: E402
from r2_search_curve import ADD_ONS, tags  # noqa: E402

PLAN = HERE / "individual" / "sessions.json"
SCORES = HERE / "individual" / "scores"
REPORT = HERE / "reports" / "r5-individual-differences.md"
FIGS = {k: HERE / f"fig_r5_{k}.svg" for k in ("reliability", "mechanisms", "pairs")}
LITERATURE = ("miller_rhg", "lebedeva_pr", "lopez_double_trace")
UNION, PARTS = "arch08", ("arch14", "arch04", "arch03")
N_BOOT, SEED = 2000, 0
OI = fs.OKABE_ITO


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def write_blocks(blocks: dict[int, str]) -> None:
    text = REPORT.read_text()
    for k, body in blocks.items():
        start, end = f"<!-- BEGIN result-{k} -->", f"<!-- END result-{k} -->"
        assert text.count(start) == 1 and text.count(end) == 1, f"result-{k} markers"
        i, j = text.index(start) + len(start), text.index(end)
        text = text[:i] + "\n" + body.strip() + "\n" + text[j:]
    REPORT.write_text(text)


def load():
    plan = json.loads(PLAN.read_text())
    tags_ = list(plan["programs"])
    subjects = list(plan["subjects"])
    ll = {h: np.zeros((len(tags_), len(subjects))) for h in ("A", "B", "all")}
    n = {h: np.zeros((len(tags_), len(subjects))) for h in ("A", "B", "all")}
    fell = np.zeros((len(tags_), len(subjects)), bool)
    inputs = [PLAN]
    for i, t in enumerate(tags_):
        path = SCORES / f"{t}.json"
        inputs.append(path)
        rows = {r["subject_id"]: r for r in json.loads(path.read_text())["subjects"]}
        assert set(rows) == set(subjects), t
        for j, s in enumerate(subjects):
            r = rows[s]
            assert r["session_ids"] == [x["session_id"] for x in plan["subjects"][s]], (t, s)
            fell[i, j] = r["fell_back"]
            x = np.array(r["session_per_subject_log_lik" if r["fell_back"] else "session_log_lik"])
            k = np.array(r["session_n_trials"])
            assert np.isfinite(x).all(), (t, s)
            for h, sl in (("A", slice(0, None, 2)), ("B", slice(1, None, 2)), ("all", slice(None))):
                ll[h][i, j], n[h][i, j] = x[sl].sum(), k[sl].sum()
    X = {h: ll[h] / n[h] for h in ll}
    return plan, tags_, subjects, X, fell, inputs


def dc(x):
    return x - x.mean(0, keepdims=True) - x.mean(1, keepdims=True) + x.mean()


def corr(a, b):
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])


def boot(fn, n_subj, rng):
    """Bootstrap over subjects: fn(index array) -> float; returns the 2.5 and 97.5 percentiles."""
    vals = np.array([fn(rng.integers(0, n_subj, n_subj)) for _ in range(N_BOOT)])
    return np.percentile(vals, [2.5, 97.5], axis=0)


def cv_choice(XA, XB, idx):
    """Per-subject best on one half scored on the other, minus the single best (chosen on the first half)."""
    a, b = XA[:, idx], XB[:, idx]
    per = b[a.argmax(0), np.arange(a.shape[1])].mean()
    single = b[a.mean(1).argmax()].mean()
    return per - single


def covariates(plan, subjects):
    ig, ns, rise, unc = [], [], [], []
    for s in subjects:
        rows = plan["subjects"][s]
        nt = sum(r["n_trials"] for r in rows)
        ig.append(1 - sum(r["n_choice"] for r in rows) / nt)
        ns.append(len(rows))
        rise.append(np.mean([r["ignore_halves"][1] - r["ignore_halves"][0] for r in rows]))
        unc.append(np.mean(["Uncoupled" in r["task"] for r in rows]))
    return {"ignore_rate": np.array(ig), "n_sessions": np.array(ns), "ignore_rise_within_session": np.array(rise),
            "share_uncoupled": np.array(unc)}


def analyse(plan, tags_, subjects, X, fell):
    rng = np.random.default_rng(SEED)
    A, B, F = X["A"], X["B"], X["all"]
    P, M = F.shape
    allix = np.arange(M)
    out: dict = {"programs": tags_, "n_subjects": M,
                 "n_fallback": {t: int(fell[i].sum()) for i, t in enumerate(tags_)}}

    # primary: reliability of the program x subject interaction
    lo, hi = boot(lambda ix: corr(dc(A[:, ix]), dc(B[:, ix])), M, rng)
    cov_int = float((dc(A) * dc(B)).mean())
    prog_cov = float(np.cov(A.mean(1), B.mean(1))[0, 1])
    out["primary"] = {"r": corr(dc(A), dc(B)), "ci": [float(lo), float(hi)],
                      "interaction_sd_reliable": float(np.sqrt(max(cov_int, 0.0))),
                      "interaction_sd_raw": float(dc(F).std()),
                      "program_sd": float(F.mean(1).std()),
                      "program_sd_reliable": float(np.sqrt(max(prog_cov, 0.0)))}

    # per program: split-half reliability of the advantage over the all-program mean
    dA, dB, dF = A - A.mean(0), B - B.mean(0), F - F.mean(0)
    per = {}
    for i, t in enumerate(tags_):
        lo, hi = boot(lambda ix: corr(dA[i, ix], dB[i, ix]), M, rng)
        per[t] = {"r": corr(dA[i], dB[i]), "ci": [float(lo), float(hi)], "mean_advantage": float(dF[i].mean()),
                  "mean_ll_per_trial": float(F[i].mean())}
    out["per_program"] = per

    # cross-validated per-subject choice, both directions averaged
    def cv(ix):
        return 0.5 * (cv_choice(A, B, ix) + cv_choice(B, A, ix))
    lo, hi = boot(cv, M, rng)
    out["choice"] = {"cv_gain": float(cv(allix)), "ci": [float(lo), float(hi)],
                     "in_sample_gain": float(F.max(0).mean() - F.mean(1).max()),
                     "single_best": tags_[int(F.mean(1).argmax())],
                     "n_distinct_winners": int(len(set(A.argmax(0)) | set(B.argmax(0))))}

    # mechanism effects among the search programs
    search = [i for i, t in enumerate(tags_) if t not in LITERATURE]
    code = {t: (HERE / "individual" / "programs" / plan["programs"][t]["file"]).read_text() for t in tags_}
    mech = {t: sorted(tags(sum(names(code[t]), []))) for t in tags_}
    out["mechanisms_by_program"] = mech
    cov = covariates(plan, subjects)
    out["covariates"] = {k: v.tolist() for k, v in cov.items()}
    effects = {}
    for k in ADD_ONS:
        has = [i for i in search if k in mech[tags_[i]]]
        without = [i for i in search if k not in mech[tags_[i]]]
        if not has or not without:
            continue

        def eff(d, ix=allix):
            return d[has][:, ix].mean(0) - d[without][:, ix].mean(0)
        eA, eB, eF = eff(dA), eff(dB), eff(dF)
        lo, hi = boot(lambda ix: corr(eff(dA, ix), eff(dB, ix)), M, rng)
        rho = {c: stats.spearmanr(eF, v) for c, v in cov.items()}
        effects[k] = {"programs_with": [tags_[i] for i in has], "n_without": len(without),
                      "mean_effect": float(eF.mean()), "sd_effect": float(eF.std()),
                      "share_subjects_positive": float((eF > 0).mean()),
                      "split_half_r": corr(eA, eB), "ci": [float(lo), float(hi)],
                      "spearman": {c: {"rho": float(v.statistic), "p": float(v.pvalue)} for c, v in rho.items()}}
    out["mechanisms"] = effects

    # union test: arch08 against its parts
    u, parts = tags_.index(UNION), [tags_.index(p) for p in PARTS]

    def union_cv(ix):
        g = 0.0
        for H1, H2 in ((A, B), (B, A)):
            best = np.array(parts)[H1[parts][:, ix].argmax(0)]
            g += 0.5 * (H2[u, ix].mean() - H2[best, ix].mean())
        return g
    lo, hi = boot(union_cv, M, rng)
    out["union"] = {"union": UNION, "parts": list(PARTS),
                    "union_minus_part": {tags_[p]: float(F[u].mean() - F[p].mean()) for p in parts},
                    "share_subjects_union_beats_best_part": float((F[u] > F[parts].max(0)).mean()),
                    "cv_union_minus_best_part": float(union_cv(allix)), "ci": [float(lo), float(hi)]}

    # pairs: correlation of per-subject advantage, all sessions and across halves
    X2 = np.array([[corr(dA[i], dB[j]) for j in range(P)] for i in range(P)])
    out["pairs"] = {"corr_all": np.corrcoef(dF).round(4).tolist(), "corr_cross_half": ((X2 + X2.T) / 2).round(4).tolist()}
    return out, A, B


LABEL = {"arch03": "satiety_engagement (best)", "s75": "75% floor", "s50": "50% floor", "miller_rhg": "Miller-RHG",
         "lebedeva_pr": "Lebedeva-PR", "lopez_double_trace": "Lopez-DoubleTrace"}


def label(plan, t):
    return LABEL.get(t, plan["programs"][t]["name"])


def figures(plan, out, A, B):
    fs.apply_paper_style()
    tags_ = out["programs"]
    pr, ch = out["primary"], out["choice"]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=fs.mm(183, 70), layout="constrained", width_ratios=[4, 5, 3])
    ra, rb = dc(A).ravel() * 1e3, dc(B).ravel() * 1e3
    lim = float(np.percentile(np.abs(np.r_[ra, rb]), 99.5))
    a.scatter(ra, rb, s=2, color=OI["blue"], alpha=0.4, linewidths=0)
    a.plot([-lim, lim], [-lim, lim], color="0.6", lw=0.5)
    a.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="interaction, half A (mnats/trial)",
          ylabel="interaction, half B (mnats/trial)")
    a.set_title(f"r = {pr['r']:.2f} [{pr['ci'][0]:.2f}, {pr['ci'][1]:.2f}]")
    order = sorted(tags_, key=lambda t: out["per_program"][t]["r"])
    y = np.arange(len(order))
    r = np.array([out["per_program"][t]["r"] for t in order])
    ci = np.array([out["per_program"][t]["ci"] for t in order])
    b.errorbar(r, y, xerr=[r - ci[:, 0], ci[:, 1] - r], fmt="o", ms=2.5, color=OI["black"], lw=0.7)
    b.axvline(0, color="0.6", lw=0.5)
    b.set_yticks(y, [label(plan, t) for t in order])
    b.set_xlabel("split-half r of per-mouse advantage")
    g, lo, hi = ch["cv_gain"] * 1e3, ch["ci"][0] * 1e3, ch["ci"][1] * 1e3
    c.bar([0, 1], [ch["in_sample_gain"] * 1e3, g], color=[OI["sky"], OI["blue"]], width=0.6)
    c.errorbar([1], [g], yerr=[[g - lo], [hi - g]], fmt="none", color=OI["black"], lw=0.7)
    c.axhline(0, color="0.6", lw=0.5)
    c.set_xticks([0, 1], ["in-sample\n(biased)", "cross-\nvalidated"])
    c.set_ylabel("per-mouse choice − single best\n(mnats/trial)")
    for ax, k in zip((a, b, c), "abc"):
        fs.panel_label(ax, k)
    fs.save_figure(fig, FIGS["reliability"])

    mech = out["mechanisms"]
    ks = list(mech)
    covs = list(mech[ks[0]]["spearman"])
    fig, (a, b) = plt.subplots(1, 2, figsize=fs.mm(150, 60), layout="constrained", width_ratios=[2, 3])
    y = np.arange(len(ks))
    r = np.array([mech[k]["split_half_r"] for k in ks])
    ci = np.array([mech[k]["ci"] for k in ks])
    a.errorbar(r, y, xerr=[r - ci[:, 0], ci[:, 1] - r], fmt="o", ms=2.5, color=OI["black"], lw=0.7)
    a.axvline(0, color="0.6", lw=0.5)
    a.set_yticks(y, ks)
    a.set_xlabel("split-half r of per-mouse\nmechanism effect")
    R = np.array([[mech[k]["spearman"][cv]["rho"] for cv in covs] for k in ks])
    im = b.imshow(R, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    b.set_xticks(range(len(covs)), [cv.replace("_", " ") for cv in covs], rotation=30, ha="right")
    b.set_yticks(y, ks)
    for i in range(len(ks)):
        for j in range(len(covs)):
            b.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=5)
    fig.colorbar(im, ax=b, shrink=0.8, label="Spearman ρ")
    fs.panel_label(a, "a"), fs.panel_label(b, "b")
    fs.save_figure(fig, FIGS["mechanisms"])

    fig, axes = plt.subplots(1, 2, figsize=fs.mm(183, 85), layout="constrained")
    labs = [label(plan, t) for t in tags_]
    for ax, key, title in ((axes[0], "corr_all", "all sessions"), (axes[1], "corr_cross_half", "across halves")):
        im = ax.imshow(np.array(out["pairs"][key]), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labs)), labs, rotation=90)
        ax.set_yticks(range(len(labs)), labs if ax is axes[0] else [])
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.7, label="correlation of per-mouse advantage")
    fs.panel_label(axes[0], "a"), fs.panel_label(axes[1], "b")
    fs.save_figure(fig, FIGS["pairs"])


def blocks(plan, out) -> dict[int, str]:
    def m(x):          # mnats per trial
        return fs.num(x * 1e3, "+.2f")
    pr, ch, un = out["primary"], out["choice"], out["union"]
    b1 = (f"Program × mouse interaction (program and mouse means removed), correlated between the two halves "
          f"of each mouse's score sessions: **r = {pr['r']:.2f}** (95% bootstrap CI over mice "
          f"{pr['ci'][0]:.2f} to {pr['ci'][1]:.2f}; {out['n_subjects']} monitor mice, {len(out['programs'])} "
          f"programs). Reliable interaction SD {pr['interaction_sd_reliable'] * 1e3:.2f} mnats per trial "
          f"(raw {pr['interaction_sd_raw'] * 1e3:.2f}); SD of the program means {pr['program_sd'] * 1e3:.2f} "
          f"(reliable {pr['program_sd_reliable'] * 1e3:.2f}). Reliable SD = square root of the covariance "
          f"between halves.\n\n")
    rows = [[label(plan, t), f"{fs.num(p['r'], '.2f')} [{fs.num(p['ci'][0], '.2f')}, {fs.num(p['ci'][1], '.2f')}]",
             m(p["mean_advantage"]),
             out["n_fallback"][t]] for t, p in sorted(out["per_program"].items(), key=lambda kv: -kv[1]["r"])]
    b1 += table(["program", "split-half r of per-mouse advantage [95% CI]", "mean advantage (mnats/trial)",
                 "mice on fallback"], rows)
    b1 += "\n\n![reliability](../fig_r5_reliability.svg)"
    b2 = (f"Choosing each mouse's best program on one half and scoring it on the other gains "
          f"**{m(ch['cv_gain'])} mnats per trial** over the single best program (95% CI {m(ch['ci'][0])} to "
          f"{m(ch['ci'][1])}; both directions averaged; mice weighted equally). Chosen and scored on the same "
          f"sessions the gain would read {m(ch['in_sample_gain'])}, which noise inflates. Single best on all "
          f"sessions: {label(plan, ch['single_best'])}; {ch['n_distinct_winners']} programs win at least one "
          f"mouse on a half.")
    covs = list(next(iter(out["mechanisms"].values()))["spearman"])
    rows = []
    for k, e in out["mechanisms"].items():
        sp = e["spearman"]
        rows.append([k, len(e["programs_with"]), e["n_without"], m(e["mean_effect"]), f"{e['sd_effect'] * 1e3:.2f}",
                     f"{e['share_subjects_positive']:.2f}",
                     f"{fs.num(e['split_half_r'], '.2f')} [{fs.num(e['ci'][0], '.2f')}, {fs.num(e['ci'][1], '.2f')}]"]
                    + [f"{fs.num(sp[c]['rho'], '+.2f')} (p {sp[c]['p']:.2g})" for c in covs])
    b3 = ("Mechanism effect per mouse = mean advantage of the search programs with the add-on minus those "
          "without (14 search programs; literature models left out). Spearman ρ against each mouse's "
          "covariates on all sessions; with 4 covariates per mechanism, read single p values with care.\n\n"
          + table(["add-on", "with", "without", "mean (mnats/trial)", "SD across mice", "share of mice > 0",
                   "split-half r [95% CI]"] + [f"ρ {c.replace('_', ' ')}" for c in covs], rows)
          + "\n\n![mechanisms](../fig_r5_mechanisms.svg)")
    b4 = (f"Union test: {label(plan, UNION)} (engagement, experience and satiety) minus each part on all "
          f"sessions: " + ", ".join(f"{label(plan, p)} {m(d)}" for p, d in un["union_minus_part"].items())
          + f" mnats per trial. It beats every part on {un['share_subjects_union_beats_best_part']:.0%} of mice. "
          f"Against each mouse's best part chosen on the other half: {m(un['cv_union_minus_best_part'])} "
          f"(95% CI {m(un['ci'][0])} to {m(un['ci'][1])}).\n\n"
          "Correlation of per-mouse advantage between programs: positive means two programs help the same "
          "mice (interchangeable); zero or negative, different mice. Panel b correlates one program's half A "
          "with the other's half B, so noise shared within a half does not enter.\n\n"
          "![pairs](../fig_r5_pairs.svg)")
    return {1: b1, 2: b2, 3: b3, 4: b4}


def main() -> None:
    plan, tags_, subjects, X, fell, inputs = load()
    out, A, B = analyse(plan, tags_, subjects, X, fell)
    figures(plan, out, A, B)
    write_blocks(blocks(plan, out))
    inputs += [HERE / "individual" / "programs" / plan["programs"][t]["file"] for t in tags_]
    meta = build_meta("analysis/r5_individual_differences.py", [], study_root=STUDY,
                      code=[__file__, HERE / "r2_search_curve.py", HERE / "freeze_seeded_islands_2.py"],
                      inputs=inputs)
    write_artifact(HERE / "r5_individual_differences.json", {"_meta": meta, **out},
                   outputs=list(FIGS.values()) + [f"{REPORT}#result-{k}" for k in (1, 2, 3, 4)])


if __name__ == "__main__":
    main()
