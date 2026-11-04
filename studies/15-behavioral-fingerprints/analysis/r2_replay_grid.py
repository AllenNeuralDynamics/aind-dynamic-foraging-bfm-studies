"""Producer for r2: replay fingerprints of the literature models, the GLM and Win-Stay-Lose-Shift.

Reads the committed P1 freezes: ``p1_replay.json`` and ``p1_glm.json`` (the gates) and
``p1_replay_gaps.json`` (per model, regime, session set and rate statistic: the subject-balanced
replay gap, binomial floor and floor ratio, and the per-bin subject means). Models are ordered by
study 12 r3's headline ranking. Writes the grid figure, ``r2_replay_grid.json`` and the report's
result block. Offline: reads committed files only.
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
import cross_study_literature as lit  # noqa: E402
import figure_style as fs  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402

WANDB_GROUPS: list[str] = []
REPLAY = HERE / "p1_replay.json"
GLM = HERE / "p1_glm.json"
GAPS = HERE / "p1_replay_gaps.json"
REPORT = HERE / "reports" / "r2-replay-grid.md"
FIG = HERE / "fig_r2_replay_grid.svg"
FIG_BINS = HERE / "fig_r2_bins.svg"
LOCAL = ("losses", "pattern", "hazard")   # the local switching statistics
LABEL_OFFSETS = {"Lebedeva-PR": (-4, -5), "CompareToThreshold": (-4, 5)}   # panel c, points: pt
C_MIN_LIKELIHOOD = 0.5   # panel c's x axis starts here; models below are named in the caption
STATS = {"losses": "losses", "pattern": "last 1–3 trials", "hazard": "hazard",
         "post_switch": "after a switch"}
REGIMES = {"per_subject": "per subject", "two_level_em": "2-level EM"}
EXTRA = ["GLM-Su2022", "Win-Stay-Lose-Shift"]
CURVES = {"Lebedeva-PR": "vermillion", "CompareToThreshold": "blue", "GLM-Su2022": "green",
          "Win-Stay-Lose-Shift": "orange"}


def ordered_models(gaps: dict) -> tuple[list[str], set[str]]:
    """Study 12 r3's headline order, then the GLM and Win-Stay-Lose-Shift; and the models whose
    2-level fit failed (r3 places them by their per-subject score)."""
    headline = lit.literature_headline_ranking()
    names = [r["literature_model"] for r in headline] + EXTRA
    assert set(names) == set(gaps["models"]), set(names) ^ set(gaps["models"])
    failed = {r["literature_model"] for r in headline if r["source"] != "2-level EM"}
    return names, failed


def best_regime(cells: dict) -> str:
    return "two_level_em" if "two_level_em" in cells else "per_subject"


def _floats(values) -> list[float]:
    return [float("nan") if v is None else v for v in values]


def draw(gaps: dict, names: list[str], failed: set[str]) -> None:
    fs.apply_paper_style()
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    head = gaps["headline_set"]
    excluded = {(e["model"], e["regime"]) for e in gaps["excluded"]}
    cols = [(reg, s) for reg in REGIMES for s in STATS]
    grid = np.full((len(names), len(cols)), np.nan)
    for i, m in enumerate(names):
        for j, (reg, s) in enumerate(cols):
            cell = gaps["models"][m].get(reg, {}).get(head, {}).get(s)
            if cell and cell["floor_ratio"] is not None:
                grid[i, j] = cell["floor_ratio"]

    fig = plt.figure(figsize=fs.mm(183, 95))
    ax = fig.add_axes([0.215, 0.14, 0.30, 0.72])
    x = np.arange(len(cols)) + np.repeat([0.0, 0.4], len(STATS))
    norm = LogNorm(vmin=1, vmax=float(np.nanmax(grid)))
    cmap = plt.get_cmap("magma_r")
    for i in range(len(names)):
        for j in range(len(cols)):
            v = grid[i, j]
            if not np.isfinite(v):
                if (names[i], cols[j][0]) in excluded:
                    ax.text(x[j], i, "n.r.", ha="center", va="center", fontsize=5, color="0.5")
                continue
            shade = norm(max(v, 1.0))
            ax.add_patch(plt.Rectangle((x[j] - 0.5, i - 0.5), 1, 1, color=cmap(shade), lw=0))
            ax.text(x[j], i, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="center",
                    fontsize=5, color="white" if shade > 0.55 else "black")
    ax.set_xlim(-0.5, x[-1] + 0.5)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([m + (" *" if m in failed else "") for m in names])
    ax.set_xticks(x)
    ax.set_xticklabels([STATS[s] for _, s in cols], rotation=60, ha="right")
    for k, reg in enumerate(REGIMES):
        ax.text(x[k * len(STATS):(k + 1) * len(STATS)].mean(), -1.0, REGIMES[reg], ha="center",
                va="bottom")
    ax.tick_params(length=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    fs.panel_label(ax, "a", dy_pt=18)
    cax = fig.add_axes([0.525, 0.14, 0.008, 0.3])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("floor ratio")
    fs.plain_log_axis(cax, which="y")

    half = len(gaps["bins"]["hazard"]) // 2
    runs = np.arange(1, half + 1)
    for k, (title, sl) in enumerate((("last trial unrewarded", slice(0, half)),
                                     ("last trial rewarded", slice(half, None)))):
        axk = fig.add_axes([0.645 + k * 0.19, 0.2, 0.145, 0.56])
        first = next(iter(CURVES))
        pb = gaps["models"][first][best_regime(gaps["models"][first])][head]["hazard"]["per_bin"]
        axk.plot(runs, _floats(pb["subject"][sl]), "o", color="black", ms=2.5, label="subjects",
                 zorder=3)
        for m, c in CURVES.items():
            pb = gaps["models"][m][best_regime(gaps["models"][m])][head]["hazard"]["per_bin"]
            axk.plot(runs, _floats(pb["model"][sl]), "-", color=fs.OKABE_ITO[c], lw=1, label=m)
        axk.set_xticks(runs)
        axk.set_xticklabels([str(r) for r in runs[:-1]] + [f"{runs[-1]}+"])
        axk.set_xlabel("run length")
        axk.set_title(title)
        axk.set_ylim(0, None)
        axk.spines[["top", "right"]].set_visible(False)
        if k == 0:
            axk.set_ylabel("p(switch)")
            fs.panel_label(axk, "b", dy_pt=18)
            axk.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.9))
    fs.save_figure(fig, FIG)
    plt.close(fig)


def likelihoods(glm: dict) -> dict:
    """Pooled held-out likelihood on the 149 test subjects per model and regime, re-pooled from
    study 12's per-subject scores (and the GLM's own)."""
    out = {}
    for regime, table in (("per_subject", lit.literature_per_subject("heldout")),
                          ("two_level_em", lit.two_level_em_per_subject("heldout"))):
        for m, subjects in table.items():
            ll = sum(v[0] for v in subjects.values())
            n = sum(v[1] for v in subjects.values())
            out.setdefault(m, {})[regime] = math.exp(ll / n)
    out[glm["model"]] = {"per_subject": glm["pooled_heldout_likelihood"]}
    return out


def mean_local_ratio(cells: dict, head: str) -> float:
    return sum(cells[head][s]["floor_ratio"] for s in LOCAL) / len(LOCAL)


def draw_bins(gaps: dict, names: list[str], lik: dict) -> list[dict]:
    """a, p(switch) by consecutive losses; b, by the last two trials; c, held-out likelihood
    against the mean floor ratio of the local statistics, per model and regime."""
    fs.apply_paper_style()
    import matplotlib.pyplot as plt
    import numpy as np

    head = gaps["headline_set"]
    fig = plt.figure(figsize=fs.mm(183, 62))
    panels = [("losses", slice(0, len(gaps["bins"]["losses"])), "consecutive losses"),
              ("pattern", slice(2, 10), "last two trials (most recent first)")]
    lefts = (0.07, 0.33)
    for k, ((stat, sl, xlabel), left) in enumerate(zip(panels, lefts)):
        ax = fig.add_axes([left, 0.24, 0.2 if k == 0 else 0.25, 0.62])
        labels = gaps["bins"][stat][sl]
        xs = np.arange(len(labels))
        first = next(iter(CURVES))
        pb = gaps["models"][first][best_regime(gaps["models"][first])][head][stat]["per_bin"]
        ax.plot(xs, _floats(pb["subject"][sl]), "o", color="black", ms=2.5, label="subjects", zorder=3)
        for m, c in CURVES.items():
            pb = gaps["models"][m][best_regime(gaps["models"][m])][head][stat]["per_bin"]
            ax.plot(xs, _floats(pb["model"][sl]), "-" if stat == "losses" else "_", color=fs.OKABE_ITO[c],
                    lw=1, ms=7, mew=1.2, label=m)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=0 if stat == "losses" else 60)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("p(switch)")
        ax.set_ylim(0, None)
        ax.spines[["top", "right"]].set_visible(False)
        fs.panel_label(ax, "ab"[k])
        if k == 0:
            ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.6))

    ax = fig.add_axes([0.70, 0.24, 0.27, 0.62])
    points = []
    for m in names:
        for reg, cells in gaps["models"][m].items():
            if m in lik and reg in lik[m]:
                points.append({"model": m, "regime": reg, "likelihood": lik[m][reg],
                               "mean_local_floor_ratio": mean_local_ratio(cells, head)})
    shown = [p for p in points if p["likelihood"] >= C_MIN_LIKELIHOOD]
    for reg, marker in (("per_subject", "o"), ("two_level_em", "s")):
        pts = [p for p in shown if p["regime"] == reg]
        ax.scatter([p["likelihood"] for p in pts], [p["mean_local_floor_ratio"] for p in pts], s=10,
                   marker=marker, facecolors="none" if reg == "per_subject" else "0.3",
                   edgecolors="0.3", lw=0.7, label=REGIMES[reg])
    for p in shown:
        if p["model"] in CURVES and p["regime"] == best_regime(gaps["models"][p["model"]]):
            ax.annotate(p["model"], (p["likelihood"], p["mean_local_floor_ratio"]), fontsize=5,
                        xytext=LABEL_OFFSETS.get(p["model"], (-4, 0)), textcoords="offset points",
                        ha="right", va="center",
                        color=fs.OKABE_ITO[CURVES[p["model"]]])
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_ylim(1, None)
    ax.set_xlabel("held-out likelihood (149 test subjects)")
    ax.set_ylabel("floor ratio, mean of local statistics")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fs.panel_label(ax, "c")
    fs.save_figure(fig, FIG_BINS)
    plt.close(fig)
    return points


def summary(gaps: dict, names: list[str]) -> list[dict]:
    """Per model and regime: the headline floor ratio of each statistic, and the worst one."""
    head = gaps["headline_set"]
    rows = []
    for m in names:
        row = {"model": m}
        for reg, cells in gaps["models"][m].items():
            ratios = {s: cells[head][s]["floor_ratio"] for s in STATS}
            row[reg] = {"floor_ratio": ratios, "worst": max(ratios, key=ratios.get),
                        "n_subjects": {s: cells[head][s]["n_subjects"] for s in STATS}}
        rows.append(row)
    return rows


def gate_summary(replay: dict, glm: dict, gaps: dict) -> dict:
    """The replay gates, over the cells the grid shows; a cell the gaps freeze left out (a failed
    2-level fit that does not reproduce) is listed, not counted."""
    skip = {(e["model"], e["regime"]) for e in gaps["excluded"]}
    out = {"library_git_sha": replay["fingerprints_library"]["git_sha"],
           "mevolve_git_sha": replay["mevolve_git_sha"], "tolerance": replay["tolerance"],
           "n_models": len(replay["models"]), "n_subjects": replay["n_subjects"],
           "max_abs_log_lik_diff": {}, "max_fingerprint_diff": 0.0, "excluded": gaps["excluded"],
           "glm_max_abs_diff": glm["max_abs_diff_sklearn"],
           "glm_pooled_heldout_likelihood": glm["pooled_heldout_likelihood"]}
    shown = [(m, reg, g) for m, regimes in replay["models"].items() for reg, g in regimes.items()
             if (m, reg) not in skip]
    out["passed"] = glm["passed"] and all(g["passed"] for _, _, g in shown)
    for _, reg, g in shown:
        out["max_abs_log_lik_diff"][reg] = max(out["max_abs_log_lik_diff"].get(reg, 0.0),
                                               g["max_abs_log_lik_diff"])
        out["max_fingerprint_diff"] = max(out["max_fingerprint_diff"], g.get("max_fingerprint_diff", 0.0))
    return out


def render(gates: dict, rows: list[dict], gaps: dict, failed: set[str], off_axis: list[dict]) -> str:
    head = gaps["headline_set"]
    lines = [
        f"![replay grid](../{FIG.name})",
        "",
        f"**a**, floor ratio (replay gap ÷ binomial floor, each averaged over the "
        f"{len(gaps['subjects'])} test subjects) per model, regime and statistic, on the subjects' "
        f"held-out sessions except warm-ups (set `{head}`). 1 means as close as the subjects' own "
        "trials allow. \\* the 2-level fit failed, so study 12 r3 places the model by its "
        "per-subject score; n.r., not reproduced (see the gates). **b**, p(switch) by the last trial's reward and the current run length "
        "(the hazard): the subjects' mean and four models' replayed predictions, over subjects that "
        "keep the bin (2-level EM where the model has it).",
        "",
        "| model | per subject: worst statistic | floor ratio | 2-level EM: worst statistic | floor ratio |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        cells = [r["model"] + (" \\*" if r["model"] in failed else "")]
        for reg in REGIMES:
            if reg in r:
                cells += [STATS[r[reg]["worst"]], f"{r[reg]['floor_ratio'][r[reg]['worst']]:.1f}"]
            else:
                cells += ["—", "—"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        f"![replay bins](../{FIG_BINS.name})",
        "",
        "**a**, p(switch) by the count of consecutive unrewarded trials on the current side, and "
        "**b**, by the last two trials (S same side, O other side, + rewarded, − not; most recent "
        "first): subjects and the same four models as above, over subjects that keep the bin. "
        "**c**, each model's pooled held-out likelihood on the 149 test subjects (study 12's "
        "per-subject scores, re-pooled; the GLM's own) against its mean floor ratio over the "
        "losses, last 1–3 trials and hazard statistics, per regime; labels mark the four models of "
        "the curves in the regime drawn there. Off the axis: "
        + "; ".join(f"{p['model']} ({REGIMES[p['regime']]}, likelihood {p['likelihood']:.3f}, "
                    f"floor ratio {p['mean_local_floor_ratio']:.0f})" for p in off_axis) + ".",
        "",
        f"**Gates** (library `{gates['library_git_sha'][:7]}`, MEvolve `{gates['mevolve_git_sha'][:7]}`). "
        f"Every replay shown ({gates['n_models']} of study 12's models × {gates['n_subjects']} subjects) "
        "reproduces the held-out log likelihood study 12 stored for the subject: largest difference "
        + ", ".join(f"{REGIMES[k]} {v:.1e}" for k, v in gates["max_abs_log_lik_diff"].items())
        + f" (tolerance {gates['tolerance']['log_lik']:.0e}). "
        + "".join(f"Left out: {e['model']} ({REGIMES[e['regime']]}; {e['n_subjects_replay_differs']} "
                  f"subjects differ): {e['reason']}. " for e in gates["excluded"])
        + "MEvolve's `replay_gap`, rebuilt from the 2-level tallies, equals the "
        f"stored one to {gates['max_fingerprint_diff']:.1e}. The GLM's replay equals scikit-learn's "
        f"prediction to {gates['glm_max_abs_diff']:.1e}; its pooled held-out likelihood is "
        f"{gates['glm_pooled_heldout_likelihood']:.4f}.",
    ]
    return "\n".join(lines)


def _rewrite(text: str, block: str, body: str) -> str:
    begin, end = f"<!-- BEGIN {block} -->", f"<!-- END {block} -->"
    head, rest = text.split(begin)
    _, tail = rest.split(end)
    return f"{head}{begin}\n{body}\n{end}{tail}"


def main() -> None:
    replay, glm, gaps = (json.loads(p.read_text()) for p in (REPLAY, GLM, GAPS))
    gates = gate_summary(replay, glm, gaps)
    if not gates["passed"]:
        raise SystemExit("a P1 replay gate failed; fix it before rendering r2")
    names, failed = ordered_models(gaps)
    draw(gaps, names, failed)
    points = draw_bins(gaps, names, likelihoods(glm))
    rows = summary(gaps, names)
    REPORT.write_text(_rewrite(REPORT.read_text(), "result-1", render(gates, rows, gaps, failed,
                                       [p for p in points if p['likelihood'] < C_MIN_LIKELIHOOD])))
    by_set = {m: {reg: {name: {s: c["floor_ratio"] for s, c in cells.items()}
                        for name, cells in sets.items()}
                  for reg, sets in gaps["models"][m].items()} for m in names}
    meta = build_meta("analysis/r2_replay_grid.py", WANDB_GROUPS, study_root=HERE.parent,
                      code=[__file__], inputs=[REPLAY, GLM, GAPS, lit.HEADLINE_SOURCE, lit.LITERATURE_SOURCE,
                              lit.TWO_LEVEL_EM_SOURCE])
    write_artifact(HERE / "r2_replay_grid.json",
                   {"_meta": meta, "gates": gates, "headline_set": gaps["headline_set"],
                    "order": names, "two_level_failed": sorted(failed), "rows": rows,
                    "floor_ratio_by_set": by_set, "likelihood_vs_floor_ratio": points},
                   outputs=[FIG, FIG_BINS, f"{REPORT}#result-1"])


if __name__ == "__main__":
    main()
