#!/usr/bin/env python
"""r3: the search's best, chosen on the selection subjects and scored once on the test subjects. Offline.

Reads the selection (``selection_selection.json``, ADR 0008), the test scores of ``freeze_test.py``
(``test/test/*.json``, ``test/gru_test/*.json``), the curves of ``freeze_test_curves.py``
(``test/curves.json``), and other studies' committed per-subject values: HB-CTT (study 13 r5,
``ctt_final_runs.json``), GRU v2 H128 (study 01, ``per_subject_rows.json``, full training set, three
seeds) and study 15's mouse-average curves (``p1_replay_gaps.json``, ``p2_rollout_gaps.json``).

Headline (ADR 0003, ADR 0008): pooled held-out likelihood per trial on the 146 test subjects every model
holds out, each model re-pooled from its per-subject values (the GRU's as the mean over seeds of each
seed's pooled value, as study 01 reports it); gap closed = (program − HB-CTT) / (GRU − HB-CTT), with a
bootstrap over subjects. Writes ``r3_search_best.json``, two figures and the report's result blocks.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
STUDIES = STUDY.parent
sys.path.insert(0, str(STUDIES / "util"))
import figure_style as fs  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402

SEL = HERE / "selection_selection.json"
CHOSEN = "rank3_e3819885"
TEST = {"test": HERE / "test" / "test" / f"{CHOSEN}.json", "gru_test": HERE / "test" / "gru_test" / f"{CHOSEN}.json",
        "reference": HERE / "test" / "test" / "reference_miller_rhg.json"}
CURVES = HERE / "test" / "curves.json"
HB = STUDIES / "13-hb-subject-vs-gru-heldout" / "analysis" / "ctt_final_runs.json"
HB_POOLED = STUDIES / "13-hb-subject-vs-gru-heldout" / "analysis" / "r5_ctt_pooled.json"
GRU = STUDIES / "01-gru-scaling-law" / "analysis" / "per_subject_rows.json"
S15 = {"replay": STUDIES / "15-behavioral-fingerprints" / "analysis" / "p1_replay_gaps.json",
       "rollout": STUDIES / "15-behavioral-fingerprints" / "analysis" / "p2_rollout_gaps.json"}
GRU_MODEL = "GRU-D614"
PROG = "search's best"
REPORT = HERE / "reports" / "r3-search-best.md"
FIGS = {"test": HERE / "fig_r3_test.svg", "curves": HERE / "fig_r3_curves.svg"}
N_BOOT = 2000
OI = fs.OKABE_ITO
PROG_COLOR, GRU_COLOR, HB_COLOR, REF_COLOR = OI["blue"], OI["purple"], OI["orange"], OI["black"]


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows])


def used_ll(run: dict) -> dict[str, tuple[float, int]]:
    """Per subject, the log-likelihood the score uses (MEvolve's per-subject fallback) and the trials."""
    failed = run["score"]["two_level_failed"]
    return {s["subject_id"]: (s["per_subject_log_lik"] if failed or s["fell_back"] else s["log_lik"], s["n_trials"])
            for s in run["subjects"]}


def pooled(ll: np.ndarray, n: np.ndarray, idx=None) -> float:
    idx = slice(None) if idx is None else idx
    return float(np.exp(ll[idx].sum() / n[idx].sum()))


# ---------------------------------------------------------------- 1. selection
def selection() -> tuple[str, dict]:
    sel = json.loads(SEL.read_text())
    ref = sel["reference"]
    rows = [["Miller-RHG (reference)", "—", f"{ref['score_x100']:.3f}", str(ref["n_fallback"]),
             f"{ref['replay_score']:.3f}", f"{ref['rollout_score']:.3f}", f"{ref['replay_curve_rmse']:.4f}",
             f"{ref['rollout_curve_rmse']:.4f}", "—"]]
    for c in sel["candidates"]:
        name = f"**`{c['patch_name']}`**" if c["name"] == sel["chosen"] else f"`{c['patch_name']}`"
        rows.append([name, f"{c['rank']} ({c['validation_x100']:.3f})", f"{c['score_x100']:.3f}", str(c["n_fallback"]),
                     f"{c['replay_score']:.3f}", f"{c['rollout_score']:.3f}", f"{c['replay_curve_rmse']:.4f}",
                     f"{c['rollout_curve_rmse']:.4f}", "pass" if c["passes_gate"] else "fail"])
    chosen = next(c for c in sel["candidates"] if c["name"] == sel["chosen"])
    order = sorted(sel["candidates"], key=lambda c: -c["score_x100"])
    ordinal = ["first", "second", "third", "fourth", "fifth"][[c["rank"] for c in order].index(1)]
    text = ("![selection](../fig_selection_selection.svg)\n\n"
            f"All {len(sel['candidates'])} candidates pass the gate: each beats Miller-RHG on all four behaviour "
            f"scores, and on held-out likelihood by {min(c['score_x100'] for c in sel['candidates']) - ref['score_x100']:+.3f} "
            f"to {max(c['score_x100'] for c in sel['candidates']) - ref['score_x100']:+.3f} (×100). The highest "
            f"selection score, **{chosen['score_x100']:.3f}**, is `{chosen['patch_name']}` (search rank "
            f"{chosen['rank']}). The validation order did not hold: the search's rank 1 is {ordinal} of five here, and the five span only {order[0]['score_x100'] - order[-1]['score_x100']:.3f}.\n\n"
            + table(["program", "search rank (validation)", "selection score", "fallback subjects", "replay score",
                     "rollout score", "replay curve RMSE", "rollout curve RMSE", "gate"], rows)
            + "\n\n**Figure 1.** The five candidates and Miller-RHG on the 100 selection subjects (`selection_gate.py`): "
              "(a) held-out likelihood, 2-level EM, `max_em` 100; (b, c) replay and rollout fingerprint scores "
              "(study 15's floor ratios, geometric mean); (d, e) the mouse-average curves' per-bin RMSE, geometric "
              "mean over statistics. Lower is closer to the subjects in b–e; dotted: Miller-RHG.")
    return text, {"chosen": sel["chosen"], "chosen_patch_name": chosen["patch_name"], "candidates": sel["candidates"],
                  "reference": ref}


# ---------------------------------------------------------------- 2. test likelihood
def test() -> tuple[str, dict]:
    prog = used_ll(json.loads(TEST["test"].read_text()))
    prog_gru = used_ll(json.loads(TEST["gru_test"].read_text()))
    ref = used_ll(json.loads(TEST["reference"].read_text()))
    hb = {r["subject_id"]: (r["heldout/eval_log_lik_sum"], r["heldout/num_eval_trials"])
          for r in json.loads(HB.read_text())["rows"]}
    rows = [r for r in json.loads(GRU.read_text())["rows"] if r["variant"] == "v2" and r["ratio"] == 1.0]
    seeds = sorted({r["seed"] for r in rows})
    gru = {s: {r["subject"]: (r["n_trials"] * math.log(r["ll"]), r["n_trials"]) for r in rows if r["seed"] == s}
           for s in seeds}
    shared = sorted(set(prog) & set(gru[seeds[0]]))
    assert len(shared) == 146 and set(shared) <= set(hb), "the 146 shared subjects"
    mism = [s for s in shared if not (prog[s][1] == hb[s][1] == gru[seeds[0]][s][1] == ref[s][1])]
    assert not mism, f"trial counts differ on {len(mism)} subjects, e.g. {mism[:3]}"
    n = np.array([prog[s][1] for s in shared], float)
    P = np.array([prog[s][0] for s in shared])
    R = np.array([ref[s][0] for s in shared])
    H = np.array([hb[s][0] for s in shared])
    G = np.array([[gru[k][s][0] for s in shared] for k in seeds])

    def stats(idx=None):
        p, h, g, r = pooled(P, n, idx), pooled(H, n, idx), float(np.mean([pooled(x, n, idx) for x in G])), pooled(R, n, idx)
        return p, h, g, r, (p - h) / (g - h)

    p, h, g, r, closed = stats()
    rng = np.random.default_rng(0)
    boot = np.array([stats(rng.integers(0, len(shared), len(shared))) for _ in range(N_BOOT)])
    ci = np.percentile(boot[:, 4], [2.5, 97.5])
    se = boot[:, :4].std(axis=0)
    paired_se = {"program": float((boot[:, 0] - boot[:, 1]).std()), "gru_v2_h128": float((boot[:, 2] - boot[:, 1]).std()),
                 "miller_rhg_2level": float((boot[:, 3] - boot[:, 1]).std())}
    # the published lists: 149 test subjects (program, HB-CTT) and the GRU's own 149
    run149 = json.loads(TEST["test"].read_text())["score"]["score"]
    run_gru149 = json.loads(TEST["gru_test"].read_text())["score"]["score"]
    hb149 = json.loads(HB_POOLED.read_text())["hb_ctt"]
    n_g = {s: np.array([gru[s][x][1] for x in sorted(gru[s])], float) for s in seeds}
    gru149 = float(np.mean([np.exp(sum(v[0] for v in gru[s].values()) / n_g[s].sum()) for s in seeds]))
    res = {"n_subjects": len(shared), "n_trials": int(n.sum()),
           "pooled_x100": {"program": 100 * p, "hb_ctt": 100 * h, "gru_v2_h128": 100 * g, "miller_rhg_2level": 100 * r},
           "bootstrap_se_x100": {"program": 100 * se[0], "hb_ctt": 100 * se[1], "gru_v2_h128": 100 * se[2],
                                 "miller_rhg_2level": 100 * se[3]},
           "above_hb_ctt_x100": {"program": 100 * (p - h), "gru_v2_h128": 100 * (g - h),
                                 "miller_rhg_2level": 100 * (r - h)},
           "above_hb_ctt_paired_se_x100": {k: 100 * v for k, v in paired_se.items()},
           "gap_closed": closed, "gap_closed_ci95": ci.tolist(),
           "published_lists_x100": {"program_test149": 100 * run149, "hb_ctt_test149": 100 * hb149,
                                    "program_gru149": 100 * run_gru149, "gru_v2_h128_gru149": 100 * gru149},
           "fallback_subjects": {"test": len(json.loads(TEST["test"].read_text())["score"]["fallback_subjects"]),
                                 "gru_test": len(json.loads(TEST["gru_test"].read_text())["score"]["fallback_subjects"])}}

    fs.apply_paper_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=fs.mm(2 * fs.AXIS_MM + 3 * fs.GROUP_MM + 50, 60), layout="constrained",
                               width_ratios=[0.8, 1.2])
    names = ["Miller-RHG\n(2-level EM)", "search's best", "GRU v2 H128"]
    vals = [100 * (r - h), 100 * (p - h), 100 * (g - h)]
    errs = [100 * paired_se[k] for k in ("miller_rhg_2level", "program", "gru_v2_h128")]
    cols = [REF_COLOR, PROG_COLOR, GRU_COLOR]
    for i, (v, e, c) in enumerate(zip(vals, errs, cols)):
        a.bar(i, v, yerr=e, width=0.6, color=c, error_kw={"elinewidth": 0.6})
    a.axhline(0, color="0.5", lw=0.6)
    a.set_xticks(range(3), names)
    a.set_ylabel("Pooled likelihood above HB-CTT (×100)")
    a.annotate(f"gap closed {closed:.2f}\n[{ci[0]:.2f}, {ci[1]:.2f}]", (1, vals[1] + errs[1]), xytext=(0, 3),
               textcoords="offset points", fontsize=5, ha="center", va="bottom")
    dp, dg = 100 * (P / n - H / n), 100 * (G.mean(0) / n - H / n)
    b.scatter(dg, dp, s=6, color=PROG_COLOR, lw=0, alpha=0.7)
    lim = [min(dg.min(), dp.min()) - 0.2, max(dg.max(), dp.max()) + 0.2]
    b.plot(lim, lim, color="0.6", ls=":", lw=0.6)
    b.axhline(0, color="0.8", lw=0.5), b.axvline(0, color="0.8", lw=0.5)
    b.set(xlim=lim, ylim=lim, xlabel="GRU − HB-CTT (log-lik per trial ×100)",
          ylabel="Search's best − HB-CTT (log-lik per trial ×100)")
    fs.panel_label(a, "a"), fs.panel_label(b, "b")
    fs.save_figure(fig, FIGS["test"])
    above = int((dp > 0).sum())
    text = ("![test](../fig_r3_test.svg)\n\n"
            f"On the {len(shared)} shared test subjects ({res['n_trials']:,} trials), the search's best scores "
            f"**{100 * p:.3f}** against HB-CTT's {100 * h:.3f} and the GRU's {100 * g:.3f}: it closes **{closed:.2f}** of "
            f"the gap from HB-CTT to the GRU (95% CI {ci[0]:.2f} to {ci[1]:.2f}, bootstrap over subjects). Miller-RHG, "
            f"fitted the same way, scores {100 * r:.3f}. The program beats HB-CTT on {above} of {len(shared)} subjects.\n\n"
            + table(["model", "146 shared (×100)", "above HB-CTT ± paired SE", "published list (×100)"],
                    [["Miller-RHG, 2-level EM (this study)", f"{100 * r:.3f}",
                      f"{fs.num(100 * (r - h), '+.3f')} ± {100 * paired_se['miller_rhg_2level']:.3f}", "—"],
                     ["HB-CTT (study 13 r5)", f"{100 * h:.3f}", "—", f"{100 * hb149:.3f} (149 test)"],
                     [f"search's best (`{CHOSEN}`)", f"**{100 * p:.3f}**",
                      f"{fs.num(100 * (p - h), '+.3f')} ± {100 * paired_se['program']:.3f}",
                      f"{100 * run149:.3f} (149 test); {100 * run_gru149:.3f} (GRU's 149)"],
                     ["GRU v2 H128 (study 01)", f"{100 * g:.3f}",
                      f"{fs.num(100 * (g - h), '+.3f')} ± {100 * paired_se['gru_v2_h128']:.3f}", f"{100 * gru149:.3f} (GRU's 149)"]])
            + f"\n\n**Figure 2.** (a) Pooled held-out likelihood per trial above HB-CTT's on the {len(shared)} shared "
              f"test subjects, each model re-pooled from its committed per-subject values (the GRU: mean over its "
              f"{len(seeds)} seeds); bars: paired bootstrap SE over subjects ({N_BOOT:,} resamples). (b) Per subject, the program's and the GRU's "
              f"held-out log-likelihood per trial above HB-CTT's; dotted: the program matches the GRU.")
    return text, res


# ---------------------------------------------------------------- 3. mouse-average curves
def _rmse(model, subject) -> float:
    x = np.array([np.nan if v is None else v for v in subject], float)
    y = np.array([np.nan if v is None else v for v in model], float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.sqrt(np.mean((y[ok] - x[ok]) ** 2)))


def curves() -> tuple[str, dict]:
    ours = json.loads(CURVES.read_text())
    out = {}
    for mode, path in S15.items():
        r = json.loads(path.read_text())
        set_name = r["headline_set"]
        rows = {}
        for m, regimes in r["models"].items():
            if "two_level_em" in regimes or "per_subject" in regimes:
                reg = "two_level_em" if "two_level_em" in regimes else "per_subject"
                cells = regimes[reg][set_name]
                cells = cells.get("statistics", cells)
                rows[m] = {st: _rmse(c["per_bin_common"]["model"], c["per_bin_common"]["subject"])
                           for st, c in cells.items()}
            else:   # the GRU's seeds, averaged per bin as study 15's figures do (fingerprint_figures._rollout_cells)
                seeds = [regimes[k][set_name].get("statistics", regimes[k][set_name]) for k in sorted(regimes)]
                rows[m] = {}
                for st in seeds[0]:
                    pb = [c[st]["per_bin_common"] for c in seeds]
                    model = [None if all(v is None for v in vs) else float(np.mean([v for v in vs if v is not None]))
                             for vs in zip(*(q["model"] for q in pb))]
                    rows[m][st] = _rmse(model, pb[0]["subject"])
        for name, label in ((CHOSEN, "search's best"), ("reference_miller_rhg", "Miller-RHG (this study)")):
            rows[label] = {st: _rmse(c["model"], c["subject"]) for st, c in ours["programs"][name][mode].items()
                           if st in next(iter(rows.values()))}
        out[mode] = {"set": set_name, "n_subjects": ours["programs"][CHOSEN][mode + "_n_common"], "rmse": rows}

    fs.apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=fs.mm(183, 70), layout="constrained", width_ratios=[4, 5])
    for ax, mode in zip(axes, ("replay", "rollout")):
        rows = out[mode]["rmse"]
        stats = list(next(iter(rows.values())))
        lit = [m for m in rows if m not in (GRU_MODEL, "search's best", "Miller-RHG (this study)", "GLM-Su2022")]
        for i, st in enumerate(stats):
            ax.scatter([rows[m][st] for m in lit], [i] * len(lit), s=6, color="0.7", lw=0, zorder=2,
                       label="literature models (study 15)" if i == 0 else None)
            ax.scatter(rows["Miller-RHG"][st], i, s=14, marker="s", color=REF_COLOR, lw=0, zorder=3,
                       label="Miller-RHG (study 15)" if i == 0 else None)
            ax.scatter(rows[GRU_MODEL][st], i, s=22, marker="D", color=GRU_COLOR, lw=0, zorder=4,
                       label="GRU (study 15)" if i == 0 else None)
            ax.scatter(rows["search's best"][st], i, s=26, marker="o", color=PROG_COLOR, lw=0, zorder=5,
                       label="search's best" if i == 0 else None)
        ax.set_yticks(range(len(stats)), stats)
        ax.invert_yaxis()
        ax.set_xlim(0, min(0.1, 1.1 * max(rows[m][st] for m in rows for st in stats)))
        ax.set_xlabel("Per-bin RMSE of the mouse-average curve")
        ax.set_title(f"{mode} ({out[mode]['set']}, {out[mode]['n_subjects']} subjects)", fontsize=6)
    axes[1].legend(loc="lower right", fontsize=5, handletextpad=0.2, borderaxespad=0.2)
    fs.panel_label(axes[0], "a"), fs.panel_label(axes[1], "b")
    fs.save_figure(fig, FIGS["curves"])

    lines, summary = [], {}
    for mode in ("replay", "rollout"):
        rows = out[mode]["rmse"]
        stats = list(next(iter(rows.values())))
        models = [m for m in rows if m not in ("Miller-RHG (this study)",)]
        for st in stats:
            rank = sorted(models, key=lambda m: rows[m][st])
            summary[f"{mode}|{st}"] = {"best": rank[0], "program_rank": rank.index(PROG) + 1,
                                       "gru_rank": rank.index(GRU_MODEL) + 1, "n_models": len(rank)}
            best_lit = min(rows[m][st] for m in models if m not in (GRU_MODEL, PROG, "GLM-Su2022"))
            lines.append([mode, st, f"{rows[PROG][st]:.4f}", f"{rows[GRU_MODEL][st]:.4f}",
                          f"{rows['Miller-RHG'][st]:.4f}", f"{best_lit:.4f}",
                          f"{rank.index(PROG) + 1} / {len(rank)}"])
    text = ("![curves](../fig_r3_curves.svg)\n\n"
            + table(["mode", "statistic", "search's best", "GRU", "Miller-RHG", "best literature model",
                     "search's best, rank"], lines)
            + f"\n\n**Figure 3.** Study 15's Figure 6 view on the test subjects: per statistic, the per-bin RMSE "
              f"between the subjects' mean curve and the model's, on study 15's headline sets and the subjects every "
              f"model holds out (replay: {out['replay']['n_subjects']}; rollout: {out['rollout']['n_subjects']}). "
              f"Literature models and the GRU from study 15 (2-level EM where it has it); the search's best from its "
              f"test run. Its subjects' curves equal study 15's exactly (`test/curves.json`, `subject_check`).")
    return text, {"rmse": out, "ranks": summary}


def write_blocks(blocks: dict[int, str]) -> None:
    text = REPORT.read_text()
    for k, body in blocks.items():
        start, end = f"<!-- BEGIN result-{k} -->", f"<!-- END result-{k} -->"
        assert text.count(start) == 1 and text.count(end) == 1, f"result-{k} markers"
        i, j = text.index(start) + len(start), text.index(end)
        text = text[:i] + "\n" + body.strip() + "\n" + text[j:]
    REPORT.write_text(text)


def main() -> None:
    b1, r1 = selection()
    b2, r2 = test()
    b3, r3 = curves()
    write_blocks({1: b1, 2: b2, 3: b3})
    meta = build_meta("analysis/r3_search_best.py", [], study_root=STUDY, code=[__file__],
                      inputs=[SEL, *TEST.values(), CURVES, HB, HB_POOLED, GRU, *S15.values()])
    write_artifact(HERE / "r3_search_best.json", {"_meta": meta, "selection": r1, "test": r2, "curves": r3},
                   outputs=[*FIGS.values(), *(Path(f"{REPORT}#result-{k}") for k in range(1, 4))])
    print(json.dumps({"test": {k: r2[k] for k in ("pooled_x100", "gap_closed", "gap_closed_ci95")}, "ranks": r3["ranks"]},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
