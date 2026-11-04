"""Producer for r2: replay fingerprints of the literature models, the GLM and Win-Stay-Lose-Shift.

Reads the committed P1 freezes: ``p1_replay.json`` and ``p1_glm.json`` (the gates) and
``p1_replay_gaps.json`` (per model, regime, session set and rate statistic: the subject-balanced
replay gap, binomial floor and floor ratio, and the per-bin subject means). Models are ordered by
study 12 r3's headline ranking. Writes the grid figure, ``r2_replay_grid.json`` and the report's
result block. Offline: reads committed files only.
"""
import json
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


def render(gates: dict, rows: list[dict], gaps: dict, failed: set[str]) -> str:
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
    rows = summary(gaps, names)
    REPORT.write_text(_rewrite(REPORT.read_text(), "result-1", render(gates, rows, gaps, failed)))
    by_set = {m: {reg: {name: {s: c["floor_ratio"] for s, c in cells.items()}
                        for name, cells in sets.items()}
                  for reg, sets in gaps["models"][m].items()} for m in names}
    meta = build_meta("analysis/r2_replay_grid.py", WANDB_GROUPS, study_root=HERE.parent,
                      code=[__file__], inputs=[REPLAY, GLM, GAPS, lit.HEADLINE_SOURCE])
    write_artifact(HERE / "r2_replay_grid.json",
                   {"_meta": meta, "gates": gates, "headline_set": gaps["headline_set"],
                    "order": names, "two_level_failed": sorted(failed), "rows": rows,
                    "floor_ratio_by_set": by_set},
                   outputs=[FIG, f"{REPORT}#result-1"])


if __name__ == "__main__":
    main()
