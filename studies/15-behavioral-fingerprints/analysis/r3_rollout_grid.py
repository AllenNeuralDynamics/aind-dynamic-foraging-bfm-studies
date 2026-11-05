"""Producer for r3: rollout fingerprints of the literature models, the GLM and the GRU.

Reads the committed P2 freezes: ``p2_rollouts.json`` (the self-replay and GRU gates),
``p2_rollout_gaps.json`` (per model, regime, session set and statistic: the subject-balanced
rollout gap, split-half floor, floor ratio and coverage, and the per-bin subject means),
``p2_holds.json`` (the hold on/off check) and ``p2_task_specs.json`` (the tasks); and r2's committed
output for the model order, the replay scores and the scatter's models. The GRU's column is the
mean over its three seeds' floor ratios (the seeds share the subjects, so their floors are the
same). Writes the figures, ``r3_rollout_grid.json`` and the report's result blocks. Offline: reads
committed files only.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
sys.path.insert(0, str(HERE))
import cross_study_literature as lit  # noqa: E402
import figure_style as fs  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402
from r2_replay_grid import C_MIN_LIKELIHOOD, CURVES, REGIMES, _bin_groups, _floats, _rewrite, likelihoods  # noqa: E402

WANDB_GROUPS: list[str] = []
ROLLOUTS = HERE / "p2_rollouts.json"
GAPS = HERE / "p2_rollout_gaps.json"
HOLDS = HERE / "p2_holds.json"
SPECS = HERE / "p2_task_specs.json"
R2 = HERE / "r2_replay_grid.json"
GLM = HERE / "p1_glm.json"
GRU = HERE / "p1_gru.json"
REPORT = HERE / "reports" / "r3-rollout-grid.md"
FIG = HERE / "fig_r3_rollout_grid.svg"
FIG_VS = HERE / "fig_r3_scores.svg"
FIG_CORR = HERE / "fig_r3_scatter.svg"
FIG_SETS = HERE / "fig_r3_sets.svg"
FIG_HOLDS = HERE / "fig_r3_holds.svg"
FIG_REG = HERE / "fig_r3_regimes.svg"
FIG_CURVES = HERE / "fig_r3_curves.svg"
FIG_REPLAY_CURVES = HERE / "fig_r3_replay_curves.svg"
REPLAY_GAPS = HERE / "p1_replay_gaps.json"
FIG_RMSE = HERE / "fig_r3_rmse.svg"
RMSE_MAX = 0.10   # Figure 7's axis; longer bars are cut and labelled
CURVE_PANELS = [("losses", slice(0, 6), "consecutive losses", "losses"),
                ("pattern", slice(2, 10), "last two trials (most recent first)", "last 1–3 trials (depth 2)"),
                ("hazard", slice(0, 6), "run length", "hazard: last trial unrewarded"),
                ("hazard", slice(6, 12), "run length", "hazard: last trial rewarded"),
                ("post_switch", slice(0, 6), "run length before the switch", "after a switch: switch unrewarded"),
                ("post_switch", slice(6, 12), "run length before the switch", "after a switch: switch rewarded")]
REGIME_MODELS = ["CompareToThreshold", "Lopez-DoubleTrace", "Bari2019"]   # Figure 3: both regimes' loss curves
GRU_MODEL = "GRU-D614"
STATS = {"losses": "losses", "pattern": "last 1–3 trials", "hazard": "hazard",
         "post_switch": "after a switch", "run_length": "run lengths"}
LOW_COVERAGE = 0.95   # a cell whose rollouts visit less of the subjects' bin weight is marked
MODEL_COLORS = {"CompareToThreshold": "blue", "Miller-RHG": "orange", "Bari2019": "sky",
                "Lopez-DoubleTrace": "green", "Lebedeva-PR": "vermillion", GRU_MODEL: "purple"}
MIN_GROUP_SUBJECTS = 8   # Figure 8 shows a session group only if this many subjects have a floor in it


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _score(ratios):
    """ADR 0009's score: the geometric mean of the floor ratios; None if one is missing."""
    import numpy as np

    r = np.array([np.nan if x is None else x for x in ratios], float)
    return float(np.exp(np.mean(np.log(r)))) if np.all(np.isfinite(r) & (r > 0)) else None


def regimes(gaps: dict, model: str) -> dict:
    """``{regime: {set: {"score", "statistics": {stat: cell}}}}``; the GRU's three seeds become one
    ``per_subject`` regime, each cell's gap, floor, floor ratio and coverage averaged over seeds."""
    cells = gaps["models"][model]
    if model != GRU_MODEL:
        return cells
    seeds = sorted(cells)
    out = {}
    for name in cells[seeds[0]]:
        stats = {}
        for st in STATS:
            per = [cells[s][name]["statistics"][st] for s in seeds]
            stats[st] = {k: _mean([c.get(k) for c in per]) for k in ("gap", "floor", "floor_ratio", "coverage")}
            stats[st]["n_subjects"] = per[0]["n_subjects"]
            if "per_bin_common" in per[0]:
                pb = [c["per_bin_common"] for c in per]
                stats[st]["per_bin_common"] = {"subject": pb[0]["subject"],
                                               "model": [_mean(v) for v in zip(*(p["model"] for p in pb))]}
        out[name] = {"statistics": stats, "score": _score(stats[st]["floor_ratio"] for st in STATS)}
    return {"per_subject": out}


def _best(cells: dict, m: str) -> str:
    return "two_level_em" if "two_level_em" in cells[m] else "per_subject"


def _curve_cell(cells: dict, m: str, head: str, stat: str) -> dict:
    return cells[m][_best(cells, m)][head]["statistics"][stat]


def draw_grid(gaps: dict, names: list[str], cells: dict, failed: set[str], skipped: set) -> None:
    """a, floor ratio per model, regime and statistic on the headline set, with the score; cells
    whose rollouts miss part of the subjects' bins are marked. b, the hazard curves."""
    fs.apply_paper_style()
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    head = gaps["headline_set"]
    cols = [(reg, s) for reg in REGIMES for s in list(STATS) + ["score"]]
    grid = np.full((len(names), len(cols)), np.nan)
    cover = np.full_like(grid, np.nan)
    for i, m in enumerate(names):
        for j, (reg, s) in enumerate(cols):
            c = cells[m].get(reg, {}).get(head)
            if c is None:
                continue
            v = c["score"] if s == "score" else c["statistics"][s]["floor_ratio"]
            grid[i, j] = np.nan if v is None else v
            if s != "score" and c["statistics"][s]["coverage"] is not None:
                cover[i, j] = c["statistics"][s]["coverage"]

    fig = plt.figure(figsize=fs.mm(183, 100))
    ax = fig.add_axes([0.2, 0.13, 0.37, 0.74])
    per = len(STATS) + 1
    x = np.arange(len(cols)) + np.repeat([0.0, 0.6], per) + np.tile([0.0] * len(STATS) + [0.2], 2)
    norm = LogNorm(vmin=1, vmax=float(np.nanmax(grid)))
    cmap = plt.get_cmap("magma_r")
    for i, m in enumerate(names):
        for j, (reg, s) in enumerate(cols):
            v = grid[i, j]
            if not np.isfinite(v):
                if (m, reg) in skipped:
                    ax.text(x[j], i, "n.r.", ha="center", va="center", fontsize=5, color="0.5")
                continue
            shade = norm(max(v, 1.0))
            ax.add_patch(plt.Rectangle((x[j] - 0.5, i - 0.5), 1, 1, color=cmap(shade), lw=0))
            ax.text(x[j], i, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="center",
                    fontsize=5, color="white" if shade > 0.55 else "black",
                    fontweight="bold" if s == "score" else "normal")
            if cover[i, j] < LOW_COVERAGE:
                ax.plot(x[j] + 0.36, i - 0.3, marker="v", ms=2, color=fs.OKABE_ITO["sky"], lw=0)
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
    cax = fig.add_axes([0.585, 0.13, 0.008, 0.3])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("floor ratio")
    fs.plain_log_axis(cax, which="y")

    half = len(gaps["bins"]["hazard"]) // 2
    runs = np.arange(1, half + 1)
    for k, (title, sl) in enumerate((("last trial unrewarded", slice(0, half)),
                                     ("last trial rewarded", slice(half, None)))):
        axk = fig.add_axes([0.69 + k * 0.16, 0.2, 0.125, 0.56])
        first = _curve_cell(cells, next(iter(CURVES)), head, "hazard")
        axk.plot(runs, _floats(first["per_bin_common"]["subject"][sl]), "o", color="black", ms=2.5,
                 label="subjects", zorder=3)
        for m, c in CURVES.items():
            pb = _curve_cell(cells, m, head, "hazard")["per_bin_common"]
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
    handles, labels = axk.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.82, 0.0))
    fs.save_figure(fig, FIG)
    plt.close(fig)


def _on_axis(p: dict, key: str) -> bool:
    """Panel a starts at r2's likelihood cutoff; the models below it are named in the caption."""
    return p[key] is not None and (key != "likelihood" or p[key] >= C_MIN_LIKELIHOOD)


def draw_scores(gaps: dict, names: list[str], cells: dict, lik: dict, replay: dict) -> list[dict]:
    """a, held-out likelihood against the rollout score; b, the replay score against it."""
    fs.apply_paper_style()
    import matplotlib.pyplot as plt

    head = gaps["headline_set"]
    points = []
    for m in names:
        for reg in REGIMES:
            c = cells[m].get(reg, {}).get(head)
            if c is None or c["score"] is None:
                continue
            points.append({"model": m, "regime": reg, "rollout_score": c["score"],
                           "likelihood": lik.get(m, {}).get(reg), "replay_score": replay.get(m, {}).get(reg)})
    fig, axes = plt.subplots(1, 2, figsize=fs.mm(fs.fit_width(square_panels=2, axes=2), 60))
    for k, (ax, key, label) in enumerate(zip(axes, ("likelihood", "replay_score"),
                                             ("held-out likelihood (test subjects)",
                                              "replay fingerprint score (r2)"))):
        for reg, marker in (("per_subject", "o"), ("two_level_em", "s")):
            pts = [p for p in points if p["regime"] == reg and _on_axis(p, key)]
            ax.scatter([p[key] for p in pts], [p["rollout_score"] for p in pts], s=10, marker=marker,
                       facecolors="none" if reg == "per_subject" else "0.3", edgecolors="0.3", lw=0.7,
                       label=REGIMES[reg])
        for p in points:
            if p["model"] in CURVES and p["regime"] == _best(cells, p["model"]) and _on_axis(p, key):
                ax.scatter([p[key]], [p["rollout_score"]], s=14, zorder=3,
                           marker="o" if p["regime"] == "per_subject" else "s",
                           color=fs.OKABE_ITO[CURVES[p["model"]]], edgecolors="white", lw=0.4, label=p["model"])
        ax.set_yscale("log")
        if key == "replay_score":
            ax.set_xscale("log")
            vals = [p[key] for p in points if p[key]] + [p["rollout_score"] for p in points]
            ax.plot([min(vals), max(vals)], [min(vals), max(vals)], color="0.7", lw=0.5, zorder=0)
            fs.plain_log_axis(ax)
        else:
            fs.plain_log_axis(ax, which="y")
        ax.set_xlabel(label)
        ax.set_ylabel("rollout fingerprint score")
        ax.spines[["top", "right"]].set_visible(False)
        fs.panel_label(ax, "ab"[k])
    handles, labels = axes[0].get_legend_handles_labels()
    keep = dict(zip(labels, handles))
    axes[1].legend(keep.values(), keep.keys(), frameon=False, loc="lower right", fontsize=5)
    fig.tight_layout()
    fs.save_figure(fig, FIG_VS)
    plt.close(fig)
    return points


def draw_scatter(gaps: dict, cells: dict, shown: list[tuple[str, str]]) -> list[dict]:
    """Study 01 r9's view, in rollout: per bin, the subjects' mean against the rollouts' mean,
    over the subjects every model holds out that keep the bin; one row per statistic."""
    fs.apply_paper_style()
    import matplotlib.pyplot as plt
    import numpy as np

    head = gaps["headline_set"]
    fig, axes = plt.subplots(len(STATS), len(shown), figsize=fs.mm(183, 192))
    out = []
    for row, stat in enumerate(STATS):
        labels = gaps["bins"][stat]
        cs = [cells[m][reg][head]["statistics"][stat] for m, reg in shown]
        top = max(np.nanmax(_floats(c["per_bin_common"][k])) for c in cs for k in ("model", "subject"))
        bottom = min(np.nanmin(_floats(c["per_bin_common"][k])) for c in cs for k in ("model", "subject"))
        lim, low = float(np.ceil(top * 10.5) / 10), 0.0
        if stat == "losses" or bottom > 0.5 * top:   # a narrow band: zoom to it, axes still equal
            low, lim = float(np.floor(bottom * 100 - 1) / 100), float(np.ceil(top * 100 + 1) / 100)
        for col, ((m, reg), cell) in enumerate(zip(shown, cs)):
            ax = axes[row, col]
            x = np.array(_floats(cell["per_bin_common"]["subject"]))
            y = np.array(_floats(cell["per_bin_common"]["model"]))
            ok = np.isfinite(x) & np.isfinite(y)
            r = float(np.corrcoef(x[ok], y[ok])[0, 1])
            rmse = float(np.sqrt(np.mean((y[ok] - x[ok]) ** 2)))
            out.append({"statistic": stat, "model": m, "regime": reg, "n_bins": int(ok.sum()), "r": r,
                        "rmse": rmse, "floor_ratio": cell["floor_ratio"]})
            ax.plot([low, lim], [low, lim], color="0.7", lw=0.5, zorder=0)
            for name, c, idx in _bin_groups(stat, labels):
                sel = np.zeros(len(x), bool)
                sel[idx] = True
                sel &= ok
                ax.scatter(x[sel], y[sel], s=5, color=fs.OKABE_ITO[c], lw=0, label=name)
            ax.text(0.98, 0.03, f"r = {r:.3f}\nRMSE = {rmse:.3f}\nfloor ratio = {cell['floor_ratio']:.1f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=5,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5}, zorder=4)
            ax.set_xlim(low, lim)
            ax.set_ylim(low, lim)
            if low > 0:
                ticks = np.arange(np.ceil(low * 25) / 25, lim + 1e-9, 0.04)   # every 0.04: 0.02 overlaps
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
            ax.set_aspect("equal")
            ax.spines[["top", "right"]].set_visible(False)
            unit = "share of runs" if stat == "run_length" else "p(switch)"
            if row == 0:
                ax.set_title(f"{m}\n{REGIMES[reg]}", fontsize=5)
            if row == len(STATS) - 1:
                ax.set_xlabel(f"subjects {unit}")
            if col == 0:
                ax.set_ylabel(f"{STATS[stat]}\nmodel {unit}")
                fs.panel_label(ax, "abcde"[row])
            if col == 0 and stat not in ("losses", "run_length"):
                ax.legend(frameon=False, loc="upper left", fontsize=5, handletextpad=0.2, borderaxespad=0.1)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.04, top=0.95, wspace=0.3, hspace=0.35)
    fs.save_figure(fig, FIG_CORR)
    plt.close(fig)
    return out


def draw_sets(gaps: dict, cells: dict, shown: list[tuple[str, str]]) -> list[dict]:
    """Rollout score per session group for the scatter's models: the headline and its groups, then
    the early coupled stages and the warm-ups, which the gym plays without the coupled holds."""
    fs.apply_paper_style()
    import matplotlib.pyplot as plt
    import numpy as np

    head = set(gaps["sets"][gaps["headline_set"]])
    m0, r0 = shown[-1]

    def n_subjects(m, reg, g):
        c = cells[m][reg].get(g)
        return 0 if c is None else min(v["n_subjects"] for v in c["statistics"].values())

    groups = sorted((g for g in gaps["sets"] if g not in (gaps["headline_set"], "non-warm-up")
                     and n_subjects(m0, r0, g) >= MIN_GROUP_SUBJECTS), key=lambda g: (g not in head, g))
    order = [gaps["headline_set"]] + groups
    rows = []
    fig, ax = plt.subplots(figsize=fs.mm(min(183, 9 * len(order) + 55), 62))   # dot columns, ~9 mm each
    xs = np.arange(len(order))
    for k, (m, reg) in enumerate(shown):
        ys = []
        for g in order:
            c = cells[m][reg].get(g)
            sc = None if c is None else c["score"]
            ys.append(np.nan if sc is None else sc)
            rows.append({"model": m, "regime": reg, "set": g, "score": sc, "n_subjects": n_subjects(m, reg, g)})
        ax.plot(xs + (k - (len(shown) - 1) / 2) * 0.1, ys, "o", ms=2.5,
                color=fs.OKABE_ITO[MODEL_COLORS[m]], label=f"{m} ({REGIMES[reg]})")
    ax.axvline(sum(g in head for g in groups) + 0.5, color="0.7", lw=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{'headline' if g == gaps['headline_set'] else g.replace('|', ', ')} "
                        f"({n_subjects(m0, r0, g)})" for g in order], rotation=60, ha="right")
    ax.set_yscale("log")
    ax.set_yticks([1, 1.5, 2, 3])
    fs.plain_log_axis(ax, which="y")
    ax.set_ylim(1, 3.2)
    ax.set_ylabel("rollout fingerprint score")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fs.save_figure(fig, FIG_SETS)
    plt.close(fig)
    return rows


def draw_regimes(gaps: dict, cells: dict) -> list[dict]:
    """p(switch) by consecutive losses, per subject and as 2-level EM, for ``REGIME_MODELS``, with
    the subjects' mean and the GRU; over the subjects every model holds out that keep the bin."""
    fs.apply_paper_style()
    import matplotlib.pyplot as plt
    import numpy as np

    head = gaps["headline_set"]
    labels = gaps["bins"]["losses"]
    xs = np.arange(len(labels))
    fig, axes = plt.subplots(1, len(REGIME_MODELS), figsize=fs.mm(fs.fit_width(square_panels=3, axes=1), 52),
                             sharey=True)
    out = []
    for k, (ax, m) in enumerate(zip(axes, REGIME_MODELS)):
        subj = _floats(cells[m]["per_subject"][head]["statistics"]["losses"]["per_bin_common"]["subject"])
        ax.plot(xs, subj, "o", color="black", ms=2.5, label="subjects", zorder=3)
        gru = cells[GRU_MODEL]["per_subject"][head]["statistics"]["losses"]["per_bin_common"]["model"]
        ax.plot(xs, _floats(gru), "-", color=fs.OKABE_ITO["purple"], lw=0.8, label="GRU-D614")
        for reg, ls in (("per_subject", "--"), ("two_level_em", "-")):
            c = cells[m][reg][head]["statistics"]["losses"]
            ax.plot(xs, _floats(c["per_bin_common"]["model"]), ls, color=fs.OKABE_ITO[MODEL_COLORS[m]], lw=1,
                    label=f"{REGIMES[reg]}")
            out.append({"model": m, "regime": reg, "floor_ratio": c["floor_ratio"],
                        "model_curve": c["per_bin_common"]["model"], "subject_curve": c["per_bin_common"]["subject"]})
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlabel("consecutive losses")
        ax.set_title(m)
        ax.spines[["top", "right"]].set_visible(False)
        if k == 0:
            ax.set_ylabel("p(switch)")
        ax.legend(frameon=False, loc="lower right", fontsize=5)
        fs.panel_label(ax, "abc"[k])
    fig.tight_layout()
    fs.save_figure(fig, FIG_REG)
    plt.close(fig)
    return out


def _rmse(pb: dict) -> float:
    """Figure 6's RMSE: over the bins where both subject means are finite, each bin once."""
    import numpy as np

    x, y = np.array(_floats(pb["subject"])), np.array(_floats(pb["model"]))
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.sqrt(np.mean((y[ok] - x[ok]) ** 2)))


def draw_rmse(gaps: dict, names: list[str], cells: dict) -> list[dict]:
    """Figure 6's per-bin RMSE for every model and regime, one panel per statistic."""
    fs.apply_paper_style()
    import matplotlib.pyplot as plt
    import numpy as np

    head = gaps["headline_set"]
    rows = []
    fig, axes = plt.subplots(1, len(STATS), figsize=fs.mm(183, 118), sharey=True)
    ys = np.arange(len(names))
    for k, (ax, st) in enumerate(zip(axes, STATS)):
        for off, reg in ((-0.2, "per_subject"), (0.2, "two_level_em")):
            for i, m in enumerate(names):
                c = cells[m].get(reg, {}).get(head)
                if c is None:
                    continue
                v = _rmse(c["statistics"][st]["per_bin_common"])
                rows.append({"model": m, "regime": reg, "statistic": st, "rmse": v})
                color = (fs.OKABE_ITO["purple"] if m == GRU_MODEL else fs.OKABE_ITO["green"] if m == "GLM-Su2022"
                         else "0.65" if reg == "per_subject" else "0.25")
                ax.barh(i + off, min(v, RMSE_MAX), height=0.38, color=color, lw=0,
                        label=REGIMES[reg] if i == 0 and k == 0 else None)
                if v > RMSE_MAX:
                    ax.text(RMSE_MAX * 0.98, i + off, f"{v:.2f}", ha="right", va="center", fontsize=5, color="white")
        gru = [r["rmse"] for r in rows if r["model"] == GRU_MODEL and r["statistic"] == st][0]
        ax.axvline(gru, color=fs.OKABE_ITO["purple"], lw=0.5, ls="--", zorder=0)
        ax.set_xlim(0, RMSE_MAX)
        ax.set_xticks([0, 0.05, 0.1])
        ax.set_xlabel("RMSE")
        ax.set_title(STATS[st])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        fs.panel_label(ax, "abcde"[k], dx_pt=-6 if k else -24)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(names)
    axes[0].set_ylim(len(names) - 0.5, -0.5)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fs.save_figure(fig, FIG_RMSE)
    plt.close(fig)
    return rows


def draw_curves(bins: dict, per_bin, shown: list[tuple[str, str]], path: Path) -> None:
    """The four rate statistics as curves: the subjects, the five scatter models as 2-level EM
    (thin) and the GRU (thick, mean of its seeds), over the subjects every model holds out.
    ``per_bin(model, regime, stat)`` returns that cell's ``per_bin_common``."""
    fs.apply_paper_style()
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=fs.mm(183, 110))
    for k, (ax, (st, sl, xlabel, title)) in enumerate(zip(axes.flat, CURVE_PANELS)):
        labels = bins[st][sl]
        xs = np.arange(len(labels))
        for m, reg in shown:
            pb = per_bin(m, reg, st)
            gru = m == GRU_MODEL
            ax.plot(xs, _floats(pb["model"])[sl], "-", color=fs.OKABE_ITO[MODEL_COLORS[m]], lw=1.6 if gru else 0.8,
                    label=m if gru else f"{m} (2-level EM)", zorder=3 if gru else 2)
        ax.plot(xs, _floats(pb["subject"])[sl], "o", color="black", ms=3, label="subjects", zorder=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels if st == "pattern" else [lab.lstrip("+-") for lab in labels],
                           rotation=60 if st == "pattern" else 0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("p(switch)")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        fs.panel_label(ax, "abcdef"[k])
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=4)
    fs.save_figure(fig, path)
    plt.close(fig)


def draw_holds(holds: dict) -> None:
    """Per model and statistic, the gap between hold-on and hold-off rollouts as a multiple of the
    subjects' split-half floor on the same sessions."""
    fs.apply_paper_style()
    import matplotlib.pyplot as plt
    import numpy as np

    models = list(holds["models"])
    fig, ax = plt.subplots(figsize=fs.mm(fs.fit_width(groups=len(STATS)) + 30, 50))
    for k, m in enumerate(models):
        ys = [holds["models"][m]["statistics"][s]["ratio"] for s in STATS]
        ax.plot(np.arange(len(STATS)) + (k - (len(models) - 1) / 2) * 0.1, ys, "o", ms=2.5,
                color=fs.OKABE_ITO[MODEL_COLORS[m]], label=m)
    ax.axhline(1, color="0.5", lw=0.5, ls="--")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(range(len(STATS)))
    ax.set_xticklabels(list(STATS.values()), rotation=30, ha="right")
    ax.set_ylabel("hold on vs off: gap ÷ floor")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fs.save_figure(fig, FIG_HOLDS)
    plt.close(fig)


def render(result: dict, gaps: dict, cells: dict, failed: set[str]) -> dict[str, str]:
    """The report's generated blocks: one per figure, then the gates."""
    head = gaps["headline_set"]
    n, n_common = len(gaps["subjects"]), len(gaps["common_subjects"])
    lines = [
        f"![rollout grid](../{FIG.name})", "",
        f"**Figure 1.** **a**, floor ratio (rollout gap ÷ split-half floor, each averaged over the {n} test "
        f"subjects with a gym task) per model, regime and statistic, on the headline sessions (set `{head}`: "
        "uncoupled and final-stage coupled, ADR 0006). 1 means the model's sessions differ from the "
        "subject's no more than the subject's own sessions differ from each other. **score** is the rollout "
        "fingerprint score, the geometric mean of the five ratios (ADR 0009). The GRU's row is the mean over "
        f"its three seeds, on the {n_common} subjects it holds out. ▾ the rollouts visit less than "
        f"{LOW_COVERAGE:.0%} of the subjects' bin weight. \\* the 2-level fit failed (study 12 r3); n.r., not "
        "rolled out (its replay did not reproduce, r2). **b**, p(switch) by the last trial's reward and the "
        f"run length: the subjects' mean and four models' rollouts, over the {n_common} subjects every model "
        "holds out that keep the bin (2-level EM where the model has it).",
        "", "| model | per subject: score | worst statistic | its floor ratio | 2-level EM: score | worst statistic | its floor ratio |",
        "|---|---|---|---|---|---|---|"]
    for m in result["order"]:
        row = [m + (" \\*" if m in failed else "")]
        for reg in REGIMES:
            c = cells[m].get(reg, {}).get(head)
            if c is None:
                row += ["—", "—", "—"]
                continue
            ratios = {s: c["statistics"][s]["floor_ratio"] for s in STATS}
            worst = max(ratios, key=lambda s: -1 if ratios[s] is None else ratios[s])
            row += ["—" if c["score"] is None else f"**{c['score']:.1f}**", STATS[worst],
                    "—" if ratios[worst] is None else f"{ratios[worst]:.1f}"]
        lines.append("| " + " | ".join(row) + " |")
    blocks = {"result-1": "\n".join(lines)}
    blocks["result-2"] = "\n".join([
        f"![scores](../{FIG_VS.name})", "",
        "**Figure 2.** Each model's rollout fingerprint score (Figure 1a) against **a**, its pooled held-out "
        "likelihood on the test subjects (r2 Figure 2c), and **b**, its replay fingerprint score (r2), per "
        "regime; the grey line marks equal scores. The models of Figure 1b are coloured."
        + ("" if not result["off_axis"] else " Off panel a's axis: " + "; ".join(
            f"{p['model']} ({REGIMES[p['regime']]}, likelihood {p['likelihood']:.3f}, rollout score "
            f"{p['rollout_score']:.0f})" for p in result["off_axis"]) + ".")])
    blocks["result-3"] = "\n".join([
        f"![scatter](../{FIG_CORR.name})", "",
        "**Figure 6.** Study 01 r9's view, in rollout, for each statistic (rows **a** to **e**): each point is "
        "one bin, the subjects' mean against the mean of the model's rollouts, over the subjects every model "
        "holds out that keep the bin. Columns as in r2 Figure 3: the five literature models with the best "
        "2-level EM likelihood, as 2-level EM, from the lowest likelihood to the highest, then the GRU. Each "
        "panel prints r and RMSE over the bins (each bin once) and the model's floor ratio for the statistic "
        "(its cell in Figure 1a, trial-weighted). Run lengths are shares of runs, not p(switch)."])
    blocks["result-4"] = "\n".join([
        f"![sets](../{FIG_SETS.name})", "",
        "**Figure 8.** Rollout fingerprint score per session group, for the models of Figure 6, on groups "
        f"where at least {MIN_GROUP_SUBJECTS} subjects have a floor (in brackets: the GRU's count; a floor "
        "needs two sessions of the group). Left of the line: the headline and its groups. Right: the early "
        "coupled stages and the warm-ups, where the gym does not hold a block as the real task's "
        "`AdvancedBlockAuto` does (ADR 0006)."])
    blocks["result-7"] = "\n".join([
        f"![regimes](../{FIG_REG.name})", "",
        "**Figure 3.** p(switch) by the count of consecutive unrewarded trials on the current side, in rollout: "
        "the subjects' mean (dots), three literature models with their per-subject parameters (dashed) and as "
        "2-level EM, each session drawing its own parameters (solid), and the GRU, over the subjects every "
        "model holds out that keep the bin. Floor ratios of this statistic: "
        + "; ".join(f"{c['model']} {REGIMES[c['regime']]} {c['floor_ratio']:.1f}" for c in result["regime_curves"])
        + "."])
    blocks["result-8"] = "\n".join([
        f"![curves](../{FIG_CURVES.name})", "",
        "**Figure 4.** The four switch statistics as curves, in rollout: the subjects' mean (dots), the five "
        "models of Figure 6 as 2-level EM (thin) and the GRU (thick, the mean of its three seeds), over the "
        "subjects every model holds out that keep the bin. **a**, by consecutive losses; **b**, by the last "
        "two trials (S same side, O other side, + rewarded, − not; most recent first); **c**, **d**, the "
        "hazard, by the current run's length after an unrewarded or a rewarded trial; **e**, **f**, right "
        "after a switch, by the length of the run before it and the switch trial's reward."])
    blocks["result-10"] = "\n".join([
        f"![rmse](../{FIG_RMSE.name})", "",
        "**Figure 7.** Figure 6's per-bin RMSE (subjects' mean against the rollouts' mean, each bin once) for "
        "every model and statistic: per subject (light) and as 2-level EM (dark); the GLM in green and the GRU "
        f"in purple (dashed line: the GRU's value). The axis stops at {RMSE_MAX:g}; a longer bar is cut and "
        "labelled. Models in r2's order."])
    blocks["result-9"] = "\n".join([
        f"![replay curves](../{FIG_REPLAY_CURVES.name})", "",
        "**Figure 5.** The same curves as Figure 4, in replay (r2): each model predicts every trial from the "
        "subject's real history. r2's sessions (all but the warm-ups) and its subjects (the 146 every model "
        "holds out), so the subjects' dots differ from Figure 4's."])
    h = result["holds"]
    blocks["result-5"] = "\n".join([
        f"![holds](../{FIG_HOLDS.name})", "",
        f"**Figure 9.** The `BlockMinReward` check, on the {h['n_sessions']} score sessions of {h['n_subjects']} "
        "subjects that record a positive minimum. The other coupled and warm-up sessions record 0 ("
        + ", ".join(f"{k.split('|')[0]} with minimum {k.split('|')[1]}: {v}" for k, v in h["n_sessions_by_minimum"].items())
        + "). Per model and statistic: the gap between rollouts with the hold on and off (same seeds), over "
        "the subjects' bins, as a multiple of the subjects' split-half floor on those sessions. "
        f"Largest: {h['max_ratio']:.2f}."])
    g, t = result["gates"], result["tasks"]
    blocks["result-6"] = "\n".join([
        "**Gates.** Each agent, replayed on the first rollout it played, returns the probabilities it played "
        f"with (largest difference {max(v.get('max_self_diff', 0) for v in g.values()):.1e}). The numpy GRU "
        f"replays every held-out session to the wrapper's stored p(right) within {g[GRU_MODEL]['max_gate_diff']:.1e}. "
        f"Tasks: {t['n_sessions']} score sessions; {t['n_excluded']} left out (no recorded block parameters, or one "
        f"uncoupled reward probability), which removes every score session of {t['n_subjects_lost']} of the 149 "
        "subjects. Coupled blocks in the gym average "
        f"{t['block_length_check']['coupled']['gym_mean']:.1f} trials against "
        f"{t['block_length_check']['coupled']['real_mean']:.1f} in the real sessions, which count ignored trials "
        "and held blocks. GRU rollout scores per seed: "
        + ", ".join(f"{v:.2f}" for v in result["gru_seed_scores"].values()) + "."])
    return blocks


def _spearman(x, y) -> float:
    """Spearman's rank correlation (average ranks for ties)."""
    import numpy as np
    from scipy.stats import rankdata

    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def claims(points: list[dict], likelihood_cut: float = 0.70) -> dict:
    """The numbers the report's prose quotes, computed here so they cannot go stale."""
    rep = [p for p in points if p["replay_score"] is not None]
    lik = [p for p in points if p["likelihood"] is not None]
    good = [p for p in lik if p["likelihood"] > likelihood_cut]
    both: dict = {}
    for p in points:
        both.setdefault(p["model"], {})[p["regime"]] = p
    pairs = [v for v in both.values() if len(v) == 2 and all(q["replay_score"] for q in v.values())]
    return {
        "spearman_rollout_replay": _spearman([p["rollout_score"] for p in rep], [p["replay_score"] for p in rep]),
        "spearman_rollout_likelihood": _spearman([p["rollout_score"] for p in lik], [p["likelihood"] for p in lik]),
        "likelihood_cut": likelihood_cut, "n_cells_above_cut": len(good),
        "spearman_rollout_likelihood_above_cut": _spearman([p["rollout_score"] for p in good],
                                                           [p["likelihood"] for p in good]),
        "n_models_both_regimes": len(pairs),
        "two_level_better_in_replay": sum(v["two_level_em"]["replay_score"] < v["per_subject"]["replay_score"]
                                          for v in pairs),
        "two_level_better_in_rollout": sum(v["two_level_em"]["rollout_score"] < v["per_subject"]["rollout_score"]
                                           for v in pairs),
        "below_equal_line": {reg: [sum(p["rollout_score"] < p["replay_score"] for p in rep if p["regime"] == reg),
                                   sum(p["regime"] == reg for p in rep)] for reg in REGIMES},
    }


def main() -> None:
    rolls, gaps, holds, specs, r2 = (json.loads(p.read_text()) for p in (ROLLOUTS, GAPS, HOLDS, SPECS, R2))
    glm, gru = json.loads(GLM.read_text()), json.loads(GRU.read_text())
    if not rolls["passed"]:
        raise SystemExit("a P2 rollout gate failed; fix it before rendering r3")
    names = r2["order"]
    assert set(names) == set(gaps["models"]), set(names) ^ set(gaps["models"])
    failed = set(r2["two_level_failed"])
    skipped = {(e["model"], e["regime"]) for e in r2["gates"]["excluded"]}
    cells = {m: regimes(gaps, m) for m in names}
    lik = likelihoods(glm, gru)
    replay = {r["model"]: {reg: r[reg]["score"] for reg in REGIMES if reg in r} for r in r2["rows"]}
    shown = list(dict.fromkeys((c["model"], c["regime"]) for c in r2["scatter"]))
    draw_grid(gaps, names, cells, failed, skipped)
    points = draw_scores(gaps, names, cells, lik, replay)
    scatter = draw_scatter(gaps, cells, shown)
    sets = draw_sets(gaps, cells, shown)
    draw_holds(holds)
    regime_curves = draw_regimes(gaps, cells)
    rmse = draw_rmse(gaps, names, cells)
    head = gaps["headline_set"]
    draw_curves(gaps["bins"], lambda m, reg, st: cells[m][reg][head]["statistics"][st]["per_bin_common"],
                shown, FIG_CURVES)
    rg = json.loads(REPLAY_GAPS.read_text())   # r2's replay cells; the GRU's per_subject is its seed mean
    draw_curves(rg["bins"], lambda m, reg, st: rg["models"][m][reg][rg["headline_set"]][st]["per_bin_common"],
                shown, FIG_REPLAY_CURVES)
    kept = {s["subject_id"] for s in specs["sessions"]}
    result = {
        "order": names, "headline_set": gaps["headline_set"], "points": points,
        "claims": claims(points),
        "off_axis": [p for p in points if p["likelihood"] is not None and p["likelihood"] < C_MIN_LIKELIHOOD], "scatter": scatter, "sets": sets, "regime_curves": regime_curves, "rmse": rmse,
        "gru_seed_scores": {s: gaps["models"][GRU_MODEL][s][gaps["headline_set"]]["score"]
                            for s in sorted(gaps["models"][GRU_MODEL])},
        "holds": {k: holds[k] for k in ("n_sessions", "n_subjects", "max_ratio", "holds_move_less_than_floor",
                                        "n_sessions_by_minimum")},
        "tasks": {**{k: specs[k] for k in ("n_sessions", "n_excluded", "by_kind_and_group", "block_length_check")},
                  "n_subjects_lost": len({e["subject_id"] for e in specs["excluded"]} - kept)},
        "gates": {m: {k: v for k, v in r.items() if k.startswith("max_") or k == "n_subjects"}
                  for m, r in rolls["models"].items()},
    }
    text = REPORT.read_text()
    for name, body in render(result, gaps, cells, failed).items():
        text = _rewrite(text, name, body)
    REPORT.write_text(text)
    meta = build_meta("analysis/r3_rollout_grid.py", WANDB_GROUPS, study_root=HERE.parent,
                      code=[__file__, HERE / "r2_replay_grid.py"],
                      inputs=[ROLLOUTS, GAPS, HOLDS, SPECS, R2, REPLAY_GAPS, GLM, GRU, lit.LITERATURE_SOURCE,
                              lit.TWO_LEVEL_EM_SOURCE])
    write_artifact(HERE / "r3_rollout_grid.json", {"_meta": meta, **result},
                   outputs=[FIG, FIG_VS, FIG_CORR, FIG_SETS, FIG_HOLDS, FIG_REG, FIG_CURVES,
                            FIG_REPLAY_CURVES, FIG_RMSE, *(f"{REPORT}#result-{k}" for k in range(1, 11))])


if __name__ == "__main__":
    main()
