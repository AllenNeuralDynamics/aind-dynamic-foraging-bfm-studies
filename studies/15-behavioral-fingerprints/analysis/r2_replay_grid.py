"""Producer for r2: replay fingerprints of the literature models, the GLM and the GRU.

Reads the committed P1 freezes: ``p1_replay.json``, ``p1_glm.json`` and ``p1_gru.json`` (the gates) and
``p1_replay_gaps.json`` (per model, regime, session set and rate statistic: the subject-balanced
replay gap, binomial floor and floor ratio, and the per-bin subject means). Models are ordered by
study 12 r3's headline ranking, then the GLM and the GRU; the GRU's per-subject column is the
mean over its three seeds, which the gaps freeze also keeps one by one. Writes the grid figure, ``r2_replay_grid.json`` and the report's
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
GRU = HERE / "p1_gru.json"
GAPS = HERE / "p1_replay_gaps.json"
REPORT = HERE / "reports" / "r2-replay-grid.md"
FIG = HERE / "fig_r2_replay_grid.svg"
FIG_BINS = HERE / "fig_r2_bins.svg"
FIG_CORR = HERE / "fig_r2_scatter.svg"
# study 01 r9 rolled out the per-subject CompareToThreshold, Bari2019 and Hattori2019; then the best
# 2-level model and the GLM
SCATTER = [("CompareToThreshold", "per_subject"), ("Bari2019", "per_subject"), ("Hattori2019", "per_subject"),
           ("Lebedeva-PR", "two_level_em"), ("GLM-Su2022", "per_subject"), ("GRU-D614", "per_subject")]
DEPTH_COLORS = {1: "black", 2: "blue", 3: "orange"}
C_MIN_LIKELIHOOD = 0.5   # panel c's x axis starts here; models below are named in the caption
STATS = {"losses": "losses", "pattern": "last 1–3 trials", "hazard": "hazard",
         "post_switch": "after a switch"}
REGIMES = {"per_subject": "per subject", "two_level_em": "2-level EM"}
EXTRA = ["GLM-Su2022", "GRU-D614"]
CURVES = {"Lebedeva-PR": "vermillion", "CompareToThreshold": "blue", "GLM-Su2022": "green",
          "GRU-D614": "purple"}


def ordered_models(gaps: dict) -> tuple[list[str], set[str]]:
    """Study 12 r3's headline order, then the GLM and the GRU; and the models whose
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
    cols = [(reg, s) for reg in REGIMES for s in list(STATS) + ["score"]]
    grid = np.full((len(names), len(cols)), np.nan)
    for i, m in enumerate(names):
        for j, (reg, s) in enumerate(cols):
            cells = gaps["models"][m].get(reg)
            if cells is None:
                continue
            grid[i, j] = fingerprint_score(gaps, m, reg) if s == "score" else cells[head][s]["floor_ratio"]

    fig = plt.figure(figsize=fs.mm(183, 95))
    ax = fig.add_axes([0.215, 0.14, 0.33, 0.72])
    per = len(STATS) + 1
    x = np.arange(len(cols)) + np.repeat([0.0, 0.6], per) + np.tile([0.0] * len(STATS) + [0.2], 2)
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
                    fontsize=5, color="white" if shade > 0.55 else "black",
                    fontweight="bold" if cols[j][1] == "score" else "normal")
    ax.set_xlim(-0.5, x[-1] + 0.5)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([m + (" *" if m in failed else "") for m in names])
    ax.set_xticks(x)
    ax.set_xticklabels([STATS.get(s, "score") for _, s in cols], rotation=60, ha="right")
    for k, reg in enumerate(REGIMES):
        ax.text(x[k * per:(k + 1) * per].mean(), -1.0, REGIMES[reg], ha="center", va="bottom")
    ax.tick_params(length=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    fs.panel_label(ax, "a", dy_pt=18)
    cax = fig.add_axes([0.555, 0.14, 0.008, 0.3])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("floor ratio")
    fs.plain_log_axis(cax, which="y")

    half = len(gaps["bins"]["hazard"]) // 2
    runs = np.arange(1, half + 1)
    for k, (title, sl) in enumerate((("last trial unrewarded", slice(0, half)),
                                     ("last trial rewarded", slice(half, None)))):
        axk = fig.add_axes([0.665 + k * 0.175, 0.2, 0.135, 0.56])
        first = next(iter(CURVES))
        pb = gaps["models"][first][best_regime(gaps["models"][first])][head]["hazard"]["per_bin_common"]
        axk.plot(runs, _floats(pb["subject"][sl]), "o", color="black", ms=2.5, label="subjects",
                 zorder=3)
        for m, c in CURVES.items():
            pb = gaps["models"][m][best_regime(gaps["models"][m])][head]["hazard"]["per_bin_common"]
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
    handles, labels = axk.get_legend_handles_labels()   # one legend, under both hazard panels
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.82, 0.0))
    fs.save_figure(fig, FIG)
    plt.close(fig)


def likelihoods(glm: dict, gru: dict) -> dict:
    """Pooled held-out likelihood on the 149 test subjects per model and regime, re-pooled from
    study 12's per-subject scores (the GLM's own; the GRU's per seed over the subjects it holds out,
    then the mean over seeds)."""
    out = {}
    for regime, table in (("per_subject", lit.literature_per_subject("heldout")),
                          ("two_level_em", lit.two_level_em_per_subject("heldout"))):
        for m, subjects in table.items():
            ll = sum(v[0] for v in subjects.values())
            n = sum(v[1] for v in subjects.values())
            out.setdefault(m, {})[regime] = math.exp(ll / n)
    out[glm["model"]] = {"per_subject": glm["pooled_heldout_likelihood"]}
    seeds = {}
    for r in gru["subjects"]:
        if r["in_training"]:
            continue
        ll, n = seeds.get(r["seed"], (0.0, 0))
        seeds[r["seed"]] = (ll + r["n_trials"] * math.log(r["likelihood"]), n + r["n_trials"])
    out[gru["model"]] = {"per_subject": sum(math.exp(ll / n) for ll, n in seeds.values()) / len(seeds)}
    return out


def fingerprint_score(gaps: dict, model: str, regime: str) -> float:
    """ADR 0009's replay fingerprint score of the headline set, as the gaps freeze computed it
    with the library (``fingerprints.aggregate_subjects``)."""
    return gaps["scores"][model][regime][gaps["headline_set"]]


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
        pb = gaps["models"][first][best_regime(gaps["models"][first])][head][stat]["per_bin_common"]
        ax.plot(xs, _floats(pb["subject"][sl]), "o", color="black", ms=2.5, label="subjects", zorder=3)
        for m, c in CURVES.items():
            pb = gaps["models"][m][best_regime(gaps["models"][m])][head][stat]["per_bin_common"]
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
        for reg in REGIMES:
            if reg in gaps["models"][m] and m in lik and reg in lik[m]:
                points.append({"model": m, "regime": reg, "likelihood": lik[m][reg],
                               "replay_score": fingerprint_score(gaps, m, reg)})
    shown = [p for p in points if p["likelihood"] >= C_MIN_LIKELIHOOD]
    for reg, marker in (("per_subject", "o"), ("two_level_em", "s")):
        pts = [p for p in shown if p["regime"] == reg]
        ax.scatter([p["likelihood"] for p in pts], [p["replay_score"] for p in pts], s=10,
                   marker=marker, facecolors="none" if reg == "per_subject" else "0.3",
                   edgecolors="0.3", lw=0.7, label=REGIMES[reg])
    for p in shown:   # the curves' models, in their colours (panel a's legend)
        if p["model"] in CURVES and p["regime"] == best_regime(gaps["models"][p["model"]]):
            ax.scatter([p["likelihood"]], [p["replay_score"]], s=14, zorder=3,
                       marker="o" if p["regime"] == "per_subject" else "s",
                       color=fs.OKABE_ITO[CURVES[p["model"]]], edgecolors="white", lw=0.4)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_ylim(1, None)
    ax.set_xlabel("held-out likelihood (test subjects)")
    ax.set_ylabel("replay fingerprint score")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fs.panel_label(ax, "c")
    fs.save_figure(fig, FIG_BINS)
    plt.close(fig)
    return points


def _bin_groups(stat: str, labels: list[str]) -> list[tuple[str, str, list[int]]]:
    """Colour groups of a statistic's bins: patterns by depth, reward x run statistics by the
    last trial's reward; losses in one colour."""
    if stat == "pattern":
        return [(f"{d} trial{'s' if d > 1 else ''} back", c, [i for i, lab in enumerate(labels) if len(lab) == 2 * d])
                for d, c in ((1, "black"), (2, "blue"), (3, "orange"))]
    if stat in ("hazard", "post_switch"):
        return [("last trial unrewarded", "vermillion", [i for i, lab in enumerate(labels) if lab[0] == "-"]),
                ("last trial rewarded", "blue", [i for i, lab in enumerate(labels) if lab[0] == "+"])]
    return [("all bins", "black", list(range(len(labels))))]


def draw_scatter(gaps: dict) -> list[dict]:
    """Study 01 r9's view, in replay, for every rate statistic: per bin, the subjects' mean
    p(switch) against the model's, over subjects that keep the bin; each panel prints r and RMSE
    over the bins and the model's floor ratio for that statistic (the grid's cell)."""
    fs.apply_paper_style()
    import matplotlib.pyplot as plt
    import numpy as np

    head = gaps["headline_set"]
    fig, axes = plt.subplots(len(STATS), len(SCATTER), figsize=fs.mm(183, 158))
    out = []
    for row, stat in enumerate(STATS):
        labels = gaps["bins"][stat]
        cells = [gaps["models"][m][reg][head][stat] for m, reg in SCATTER]
        top = max(np.nanmax(_floats(c["per_bin"][k])) for c in cells for k in ("model", "subject"))
        lim = float(np.ceil(top * 10.5) / 10)
        for col, ((m, reg), cell) in enumerate(zip(SCATTER, cells)):
            ax = axes[row, col]
            x, y = np.array(_floats(cell["per_bin"]["subject"])), np.array(_floats(cell["per_bin"]["model"]))
            ok = np.isfinite(x) & np.isfinite(y)
            r = float(np.corrcoef(x[ok], y[ok])[0, 1])
            rmse = float(np.sqrt(np.mean((y[ok] - x[ok]) ** 2)))
            out.append({"statistic": stat, "model": m, "regime": reg, "n_bins": int(ok.sum()), "r": r,
                        "rmse": rmse, "floor_ratio": cell["floor_ratio"]})
            ax.plot([0, lim], [0, lim], color="0.7", lw=0.5, zorder=0)
            for name, c, idx in _bin_groups(stat, labels):
                sel = np.zeros(len(x), bool)
                sel[idx] = True
                sel &= ok
                ax.scatter(x[sel], y[sel], s=5, color=fs.OKABE_ITO[c], lw=0, label=name)
            ax.text(0.98, 0.03, f"r = {r:.3f}\nRMSE = {rmse:.3f}\nfloor ratio = {cell['floor_ratio']:.1f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=5,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5}, zorder=4)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect("equal")
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(f"{m}\n{REGIMES[reg]}", fontsize=5)
            if row == len(STATS) - 1:
                ax.set_xlabel("subjects p(switch)")
            if col == 0:
                ax.set_ylabel(f"{STATS[stat]}\nmodel p(switch)")
                fs.panel_label(ax, "abcd"[row])
            if col == 0 and stat != "losses":   # above the diagonal's left end: empty for this model
                ax.legend(frameon=False, loc="upper left", fontsize=5, handletextpad=0.2,
                          borderaxespad=0.1)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.05, top=0.95, wspace=0.3, hspace=0.3)
    fs.save_figure(fig, FIG_CORR)
    plt.close(fig)
    return out


def summary(gaps: dict, names: list[str]) -> list[dict]:
    """Per model and regime: the headline floor ratio of each statistic, and the worst one."""
    head = gaps["headline_set"]
    rows = []
    for m in names:
        row = {"model": m}
        for reg, cells in gaps["models"][m].items():
            if reg not in REGIMES:   # a GRU seed; its mean is the per-subject column
                continue
            ratios = {s: cells[head][s]["floor_ratio"] for s in STATS}
            row[reg] = {"floor_ratio": ratios, "worst": max(ratios, key=ratios.get),
                        "score": fingerprint_score(gaps, m, reg),
                        "n_subjects": {s: cells[head][s]["n_subjects"] for s in STATS}}
        rows.append(row)
    return rows


def gate_summary(replay: dict, glm: dict, gru: dict, gaps: dict) -> dict:
    """The replay gates, over the cells the grid shows; a cell the gaps freeze left out (a failed
    2-level fit that does not reproduce) is listed, not counted."""
    skip = {(e["model"], e["regime"]) for e in gaps["excluded"]}
    out = {"library_git_sha": replay["fingerprints_library"]["git_sha"],
           "mevolve_git_sha": replay["mevolve_git_sha"], "tolerance": replay["tolerance"],
           "n_models": len(replay["models"]), "n_subjects": replay["n_subjects"],
           "max_abs_log_lik_diff": {}, "max_fingerprint_diff": 0.0, "excluded": gaps["excluded"],
           "glm_max_abs_diff": glm["max_abs_diff_sklearn"],
           "glm_pooled_heldout_likelihood": glm["pooled_heldout_likelihood"],
           "gru": {"wandb_group": gru["launch"]["wandb_group"], "tolerance": gru["tolerance"]["likelihood"],
                   "n_seeds": len(gru["seeds"]),
                   "n_subjects_referenced": min(g["n_subjects_referenced"] for g in gru["seeds"].values()),
                   "subjects_unreferenced": gru["seeds"]["seed0"]["subjects_unreferenced"],
                   "subjects_in_training": gru["seeds"]["seed0"]["subjects_in_training"],
                   "max_abs_likelihood_diff": max(g["max_abs_likelihood_diff"] for g in gru["seeds"].values()),
                   "seed_scores": {reg: fingerprint_score(gaps, gru["model"], reg) for reg in gru["seeds"]}}}
    shown = [(m, reg, g) for m, regimes in replay["models"].items() for reg, g in regimes.items()
             if (m, reg) not in skip]
    out["passed"] = glm["passed"] and gru["passed"] and all(g["passed"] for _, _, g in shown)
    for _, reg, g in shown:
        out["max_abs_log_lik_diff"][reg] = max(out["max_abs_log_lik_diff"].get(reg, 0.0),
                                               g["max_abs_log_lik_diff"])
        out["max_fingerprint_diff"] = max(out["max_fingerprint_diff"], g.get("max_fingerprint_diff", 0.0))
    return out


def render(gates: dict, rows: list[dict], gaps: dict, failed: set[str], off_axis: list[dict],
           scatter: list[dict]) -> str:
    head = gaps["headline_set"]
    lines = [
        f"![replay grid](../{FIG.name})",
        "",
        f"**a**, floor ratio (replay gap ÷ binomial floor, each averaged over the "
        f"{len(gaps['subjects'])} test subjects) per model, regime and statistic, on the subjects' "
        f"held-out sessions except warm-ups (set `{head}`). 1 means as close as the subjects' own "
        "trials allow. **score** is the replay fingerprint score, the geometric mean of the four "
        "ratios (ADR 0009). The GRU's row is the mean over its three seeds. \\* the 2-level fit failed, so study 12 r3 places the model by its "
        "per-subject score; n.r., not reproduced (see the gates). **b**, p(switch) by the last trial's reward and the current run length "
        "(the hazard): the subjects' mean and four models' replayed predictions, over the "
        f"{len(gaps['subjects']) - len(gates['gru']['subjects_in_training'])} subjects every model "
        "holds out that keep the bin (2-level EM where the model has it).",
        "",
        "| model | per subject: score | worst statistic | its floor ratio | 2-level EM: score | worst statistic | its floor ratio |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cells = [r["model"] + (" \\*" if r["model"] in failed else "")]
        for reg in REGIMES:
            if reg in r:
                cells += [f"**{r[reg]['score']:.1f}**", STATS[r[reg]["worst"]],
                          f"{r[reg]['floor_ratio'][r[reg]['worst']]:.1f}"]
            else:
                cells += ["—", "—", "—"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        f"![replay bins](../{FIG_BINS.name})",
        "",
        "**a**, p(switch) by the count of consecutive unrewarded trials on the current side, and "
        "**b**, by the last two trials (S same side, O other side, + rewarded, − not; most recent "
        "first): subjects and the same four models as above, over the same subjects. "
        "**c**, each model's pooled held-out likelihood on the 149 test subjects (study 12's "
        "per-subject scores, re-pooled; the GLM's own; the GRU's mean over seeds, on its 146) against its replay fingerprint score (the "
        "geometric mean of its four floor ratios, ADR 0009), per regime; the models of the curves are "
        "coloured as in **a**, in the regime drawn there."
        + ("" if not off_axis else " Off the axis: " + "; ".join(
            f"{p['model']} ({REGIMES[p['regime']]}, likelihood {p['likelihood']:.3f}, "
            f"replay score {p['replay_score']:.0f})" for p in off_axis) + "."),
        "",
        f"![scatter](../{FIG_CORR.name})",
        "",
        "Study 01 r9's view, here in replay, for each statistic (rows **a** to **d**): each point is "
        "one bin, the subjects' mean p(switch) against the model's replayed prediction, over "
        "subjects that keep the bin. Each panel prints r and RMSE over the bins and the model's "
        "floor ratio for that statistic (the first figure's cell): the floor ratio weights each bin "
        "by its trials and compares with what the subjects' data allow; r and RMSE count every bin "
        "once. Patterns are coloured by depth, hazard and after-a-switch bins by the last trial's "
        "reward.",
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
        f"{gates['glm_pooled_heldout_likelihood']:.4f}. "
        f"The GRU (study 01's v1 GRU at D = 614, re-fine-tuned on the held-out subjects, launch "
        f"`{gates['gru']['wandb_group']}`, which logs no W&B runs) scores exactly the freeze's sessions and choices; on the "
        f"{gates['gru']['n_subjects_referenced']} subjects study 01 r2 also scored, each seed's "
        f"likelihood equals study 01's to {gates['gru']['max_abs_likelihood_diff']:.1e} (tolerance "
        f"{gates['gru']['tolerance']:.0e}; study 01 scored on GPUs, this re-fine-tune on CPUs). Its "
        f"{len(gates['gru']['subjects_in_training'])} other subjects "
        f"({', '.join(gates['gru']['subjects_in_training'])}) were among the 614 its checkpoint was "
        f"trained on, so the GRU's numbers leave them out: its cells and score are over "
        f"{len(gaps['subjects']) - len(gates['gru']['subjects_in_training'])} subjects, every other "
        f"model's over {len(gaps['subjects'])}. Replay scores per seed: "
        + ", ".join(f"{v:.2f}" for v in gates["gru"]["seed_scores"].values()) + ".",
    ]
    return "\n".join(lines)


def _rewrite(text: str, block: str, body: str) -> str:
    begin, end = f"<!-- BEGIN {block} -->", f"<!-- END {block} -->"
    head, rest = text.split(begin)
    _, tail = rest.split(end)
    return f"{head}{begin}\n{body}\n{end}{tail}"


def main() -> None:
    replay, glm, gru, gaps = (json.loads(p.read_text()) for p in (REPLAY, GLM, GRU, GAPS))
    gates = gate_summary(replay, glm, gru, gaps)
    if not gates["passed"]:
        raise SystemExit("a P1 replay gate failed; fix it before rendering r2")
    names, failed = ordered_models(gaps)
    draw(gaps, names, failed)
    points = draw_bins(gaps, names, likelihoods(glm, gru))
    scatter = draw_scatter(gaps)
    rows = summary(gaps, names)
    REPORT.write_text(_rewrite(REPORT.read_text(), "result-1", render(gates, rows, gaps, failed,
                                       [p for p in points if p['likelihood'] < C_MIN_LIKELIHOOD], scatter)))
    by_set = {m: {reg: {name: {s: c["floor_ratio"] for s, c in cells.items()}
                        for name, cells in sets.items()}
                  for reg, sets in gaps["models"][m].items()} for m in names}
    meta = build_meta("analysis/r2_replay_grid.py", WANDB_GROUPS, study_root=HERE.parent,
                      code=[__file__], inputs=[REPLAY, GLM, GRU, GAPS, lit.HEADLINE_SOURCE, lit.LITERATURE_SOURCE,
                              lit.TWO_LEVEL_EM_SOURCE])
    write_artifact(HERE / "r2_replay_grid.json",
                   {"_meta": meta, "gates": gates, "headline_set": gaps["headline_set"],
                    "order": names, "two_level_failed": sorted(failed), "rows": rows,
                    "floor_ratio_by_set": by_set, "likelihood_vs_floor_ratio": points,
                    "scatter": scatter},
                   outputs=[FIG, FIG_BINS, FIG_CORR, f"{REPORT}#result-1"])


if __name__ == "__main__":
    main()
