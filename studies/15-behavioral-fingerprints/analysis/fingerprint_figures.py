"""The figures r2 (replay) and r3 (rollout) share: one format, one order, one caption per figure.

Both producers build a ``Mode`` from their committed gaps freeze (``replay_mode`` from
``p1_replay_gaps.json``, ``rollout_mode`` from ``p2_rollout_gaps.json``) and draw ``FIGURES`` in
order; figure k is the report's ``result-k`` block. A mode holds its cells in one shape,
``cells[model][regime][set] = {"score", "statistics": {stat: {gap, floor, floor_ratio, ...}}}``,
with the GRU's seeds averaged into one ``per_subject`` regime. Replay has four rate statistics;
rollout adds the run-length distribution. The rollout report adds one figure of its own, the
holds check, after these.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
import cross_study_literature as lit  # noqa: E402
import figure_style as fs  # noqa: E402

GRU_MODEL, GLM_MODEL = "GRU-D614", "GLM-Su2022"
EXTRA = [GLM_MODEL, GRU_MODEL]
REGIMES = {"per_subject": "per subject", "two_level_em": "2-level EM"}
STAT_LABELS = {"losses": "losses", "pattern": "last 1–3 trials", "hazard": "hazard",
               "post_switch": "after a switch", "run_length": "run lengths"}
CURVES = {"Lebedeva-PR": "vermillion", "CompareToThreshold": "blue", GLM_MODEL: "green", GRU_MODEL: "purple"}
MODEL_COLORS = {"CompareToThreshold": "blue", "Miller-RHG": "orange", "Bari2019": "sky",
                "Lopez-DoubleTrace": "green", "Lebedeva-PR": "vermillion", GRU_MODEL: "purple"}
SCRATCH_LABELS = {GRU_MODEL: ("GRU", "purple"), "Lopez-DoubleTrace": ("Lopez-DT 2L", "vermillion"),
                  "Lebedeva-PR": ("Lebedeva 2L", "orange"), "CompareToThreshold": ("CTT per subj.", "blue"),
                  GLM_MODEL: ("GLM", "green")}
REGIME_MODELS = ["CompareToThreshold", "Lopez-DoubleTrace", "Bari2019"]   # the regimes figure
N_SCATTER = 5            # the scatter: the best literature models by 2-level EM likelihood, then the GRU
C_MIN_LIKELIHOOD = 0.5   # the scores figure's x axis starts here; models below are named in the caption
RMSE_MAX = 0.10          # the RMSE figure's axis; longer bars are cut and labelled
LOW_COVERAGE = 0.95      # the grid marks a rollout cell that visits less of the subjects' bin weight
MIN_GROUP_SUBJECTS = 8   # the sets figure shows a group only if this many subjects have a floor in it
CURVE_PANELS = [("losses", slice(0, 6), "consecutive losses", "losses"),
                ("pattern", slice(2, 10), "last two trials (most recent first)", "last 1–3 trials (depth 2)"),
                ("hazard", slice(0, 6), "run length", "hazard: last trial unrewarded"),
                ("hazard", slice(6, 12), "run length", "hazard: last trial rewarded"),
                ("post_switch", slice(0, 6), "run length before the switch", "after a switch: switch unrewarded"),
                ("post_switch", slice(6, 12), "run length before the switch", "after a switch: switch rewarded")]
FIGURES = ["grid", "scores", "regimes", "curves", "scatter", "rmse", "gru_errors", "gru_spread", "sets"]


@dataclass
class Mode:
    """One report's data, in the shared shape."""
    key: str                  # "replay" or "rollout"
    report: str               # "r2" or "r3"
    head: str                 # the headline set
    head_text: str            # what the headline set is, for captions
    sets: dict
    stats: tuple
    bins: dict
    subjects: list
    common: list              # the subjects every model holds out
    names: list               # the models, in study 12 r3's order, then the GLM and the GRU
    failed: set               # 2-level fits study 12 r3 marks failed
    skipped: set              # (model, regime) cells not reproduced in replay (r2 gates)
    cells: dict
    scratch: dict             # the GRU per-mouse comparison (p1/p2_gru_scratch.json)
    floor_text: str           # "binomial floor" or "split-half floor"
    gap_text: str             # "replay gap" or "rollout gap"
    model_text: str           # how the model's value is made, for captions
    other: dict = field(default_factory=dict)   # the other mode's headline scores {model: {regime: score}}
    lik: dict = field(default_factory=dict)

    def fig(self, name: str) -> Path:
        return HERE / f"fig_{self.report}_{name}.svg"

    @property
    def score_text(self) -> str:
        return f"{self.key} fingerprint score"

    @property
    def other_key(self) -> str:
        return "rollout" if self.key == "replay" else "replay"

    def cell(self, m: str, reg: str, st: str, name: str | None = None) -> dict | None:
        c = self.cells.get(m, {}).get(reg, {}).get(name or self.head)
        return None if c is None else c["statistics"].get(st)

    def score(self, m: str, reg: str, name: str | None = None):
        c = self.cells.get(m, {}).get(reg, {}).get(name or self.head)
        return None if c is None else c["score"]


def _f(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v


def _floats(values) -> list[float]:
    return [float("nan") if v is None else v for v in values]


def _mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else None


def fingerprint_score(ratios):
    """ADR 0009's score: the geometric mean of the floor ratios; None if one is missing."""
    r = [None if x is None else x for x in ratios]
    if not r or any(x is None or not math.isfinite(x) or x <= 0 for x in r):
        return None
    return math.exp(sum(math.log(x) for x in r) / len(r))


def model_order(models) -> tuple[list[str], set[str]]:
    """Study 12 r3's headline order, then the GLM and the GRU; and the failed 2-level fits."""
    headline = lit.literature_headline_ranking()
    names = [r["literature_model"] for r in headline] + EXTRA
    assert set(names) == set(models), set(names) ^ set(models)
    return names, {r["literature_model"] for r in headline if r["source"] != "2-level EM"}


def scatter_models(replay_gaps: dict) -> list[tuple[str, str]]:
    """The ``N_SCATTER`` literature models with the best pooled 2-level EM likelihood whose 2-level fit
    did not fail (study 12 r3) and reproduced in replay, as 2-level EM, worst to best; then the GRU."""
    _, failed = model_order(replay_gaps["models"])
    skip = {e["model"] for e in replay_gaps["excluded"] if e["regime"] == "two_level_em"}
    pooled = {m: sum(v[0] for v in t.values()) / sum(v[1] for v in t.values())
              for m, t in lit.two_level_em_per_subject("heldout").items() if m not in failed | skip}
    return [(m, "two_level_em") for m in sorted(pooled, key=pooled.get)[-N_SCATTER:]] + [(GRU_MODEL, "per_subject")]


def likelihoods(glm: dict, gru: dict) -> dict:
    """Pooled held-out likelihood on the 149 test subjects per model and regime, re-pooled from
    study 12's per-subject scores (the GLM's own; the GRU's per seed over the subjects it holds out,
    then the mean over seeds)."""
    out = {}
    for regime, table in (("per_subject", lit.literature_per_subject("heldout")),
                          ("two_level_em", lit.two_level_em_per_subject("heldout"))):
        for m, subjects in table.items():
            out.setdefault(m, {})[regime] = math.exp(sum(v[0] for v in subjects.values())
                                                     / sum(v[1] for v in subjects.values()))
    out[glm["model"]] = {"per_subject": glm["pooled_heldout_likelihood"]}
    seeds = {}
    for r in gru["subjects"]:
        if r["in_training"]:
            continue
        ll, n = seeds.get(r["seed"], (0.0, 0))
        seeds[r["seed"]] = (ll + r["n_trials"] * math.log(r["likelihood"]), n + r["n_trials"])
    out[gru["model"]] = {"per_subject": sum(math.exp(ll / n) for ll, n in seeds.values()) / len(seeds)}
    return out


def _replay_cells(g: dict) -> dict:
    stats = ("losses", "pattern", "hazard", "post_switch")
    return {m: {reg: {name: {"score": g["scores"][m][reg][name], "statistics": {st: cell[st] for st in stats}}
                      for name, cell in regs[reg].items()}
                for reg in REGIMES if reg in regs}
            for m, regs in g["models"].items()}


def _rollout_cells(g: dict) -> dict:
    """The rollout cells; the GRU's seeds become one ``per_subject`` regime, each value averaged."""
    out = {}
    for m, regs in g["models"].items():
        if m != GRU_MODEL:
            out[m] = regs
            continue
        seeds = sorted(regs)
        one = {}
        for name in regs[seeds[0]]:
            stats = {}
            for st in STAT_LABELS:
                per = [regs[s][name]["statistics"][st] for s in seeds]
                c = {k: _mean([p.get(k) for p in per]) for k in ("gap", "floor", "floor_ratio", "coverage")}
                c["n_subjects"] = per[0]["n_subjects"]
                if "per_bin_common" in per[0]:
                    pb = [p["per_bin_common"] for p in per]
                    c["per_bin_common"] = {"subject": pb[0]["subject"],
                                           "model": [_mean(v) for v in zip(*(p["model"] for p in pb))]}
                if "per_subject" in per[0]:
                    c["per_subject"] = {k: [_mean(v) for v in zip(*(p["per_subject"][k] for p in per))]
                                        for k in ("gap", "floor")}
                stats[st] = c
            one[name] = {"statistics": stats, "score": fingerprint_score(stats[st]["floor_ratio"] for st in stats)}
        out[m] = {"per_subject": one}
    return out


def _headline_scores(cells: dict, head: str) -> dict:
    return {m: {reg: c[head]["score"] for reg, c in regs.items() if head in c} for m, regs in cells.items()}


def replay_mode(g: dict, other: dict, scratch: dict, lik: dict, gru: dict) -> Mode:
    names, failed = model_order(g["models"])
    trained = set(gru["seeds"]["seed0"]["subjects_in_training"])
    return Mode(key="replay", report="r2", head=g["headline_set"],
                head_text=f"the subjects' held-out sessions except warm-ups (set `{g['headline_set']}`)",
                sets=g["sets"], stats=("losses", "pattern", "hazard", "post_switch"), bins=g["bins"],
                subjects=g["subjects"], common=[s for s in g["subjects"] if s not in trained], names=names,
                failed=failed, skipped={(e["model"], e["regime"]) for e in g["excluded"]},
                cells=_replay_cells(g), scratch=scratch, floor_text="binomial floor", gap_text="replay gap",
                model_text="the model's replayed prediction", other=_headline_scores(_rollout_cells(other),
                                                                                      other["headline_set"]),
                lik=lik)


def rollout_mode(g: dict, other: dict, scratch: dict, lik: dict) -> Mode:
    names, failed = model_order(g["models"])
    return Mode(key="rollout", report="r3", head=g["headline_set"],
                head_text=(f"the headline sessions (set `{g['headline_set']}`: uncoupled and final-stage "
                           "coupled, ADR 0006)"),
                sets=g["sets"], stats=tuple(STAT_LABELS), bins=g["bins"], subjects=g["subjects"],
                common=g["common_subjects"], names=names, failed=failed,
                skipped={(e["model"], e["regime"]) for e in other["excluded"]}, cells=_rollout_cells(g),
                scratch=scratch, floor_text="split-half floor", gap_text="rollout gap",
                model_text="the mean of the model's rollouts",
                other=_headline_scores(_replay_cells(other), other["headline_set"]), lik=lik)


def _best(mode: Mode, m: str) -> str:
    return "two_level_em" if "two_level_em" in mode.cells[m] else "per_subject"


def _unit(st: str) -> str:
    return "share of runs" if st == "run_length" else "p(switch)"


def _rmse(pb: dict) -> float:
    """Over the bins where both subject means are finite, each bin once."""
    import numpy as np

    x, y = np.array(_floats(pb["subject"])), np.array(_floats(pb["model"]))
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.sqrt(np.mean((y[ok] - x[ok]) ** 2)))


# --- the figures, in FIGURES order --------------------------------------------------------------

def draw_grid(mode: Mode) -> list[dict]:
    """a, floor ratio per model, regime and statistic on the headline set, with the score; b, the
    hazard curves of ``CURVES``. Returns the table rows."""
    fs.apply_paper_style()
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    names, stats = mode.names, list(mode.stats)
    cols = [(reg, s) for reg in REGIMES for s in stats + ["score"]]
    grid = np.full((len(names), len(cols)), np.nan)
    cover = np.full_like(grid, np.nan)
    for i, m in enumerate(names):
        for j, (reg, s) in enumerate(cols):
            v = mode.score(m, reg) if s == "score" else (mode.cell(m, reg, s) or {}).get("floor_ratio")
            grid[i, j] = np.nan if v is None else v
            c = mode.cell(m, reg, s) if s != "score" else None
            if c and c.get("coverage") is not None:
                cover[i, j] = c["coverage"]
    fig = plt.figure(figsize=fs.mm(183, 100))
    ax = fig.add_axes([0.2, 0.13, 0.37, 0.74])
    per = len(stats) + 1
    x = np.arange(len(cols)) + np.repeat([0.0, 0.6], per) + np.tile([0.0] * len(stats) + [0.2], 2)
    norm = LogNorm(vmin=1, vmax=float(np.nanmax(grid)))
    cmap = plt.get_cmap("magma_r")
    for i, m in enumerate(names):
        for j, (reg, s) in enumerate(cols):
            v = grid[i, j]
            if not np.isfinite(v):
                if (m, reg) in mode.skipped:
                    ax.text(x[j], i, "n.r.", ha="center", va="center", fontsize=5, color="0.5")
                continue
            shade = norm(max(v, 1.0))
            ax.add_patch(plt.Rectangle((x[j] - 0.5, i - 0.5), 1, 1, color=cmap(shade), lw=0))
            ax.text(x[j], i, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="center", fontsize=5,
                    color="white" if shade > 0.55 else "black", fontweight="bold" if s == "score" else "normal")
            if cover[i, j] < LOW_COVERAGE:
                ax.plot(x[j] + 0.36, i - 0.3, marker="v", ms=2, color=fs.OKABE_ITO["sky"], lw=0)
    ax.set_xlim(-0.5, x[-1] + 0.5)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([m + (" *" if m in mode.failed else "") for m in names])
    ax.set_xticks(x)
    ax.set_xticklabels([STAT_LABELS.get(s, "score") for _, s in cols], rotation=60, ha="right")
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

    half = len(mode.bins["hazard"]) // 2
    runs = np.arange(1, half + 1)
    for k, (title, sl) in enumerate((("last trial unrewarded", slice(0, half)),
                                     ("last trial rewarded", slice(half, None)))):
        axk = fig.add_axes([0.69 + k * 0.16, 0.2, 0.125, 0.56])
        first = next(iter(CURVES))
        axk.plot(runs, _floats(mode.cell(first, _best(mode, first), "hazard")["per_bin_common"]["subject"][sl]),
                 "o", color="black", ms=2.5, label="subjects", zorder=3)
        for m, c in CURVES.items():
            pb = mode.cell(m, _best(mode, m), "hazard")["per_bin_common"]
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
    fs.save_figure(fig, mode.fig("grid"))
    plt.close(fig)

    rows = []
    for m in names:
        row = {"model": m}
        for reg in REGIMES:
            if mode.score(m, reg) is None and mode.cell(m, reg, stats[0]) is None:
                continue
            ratios = {s: mode.cell(m, reg, s)["floor_ratio"] for s in stats}
            row[reg] = {"floor_ratio": ratios, "score": mode.score(m, reg),
                        "worst": max(ratios, key=lambda s: -1 if ratios[s] is None else ratios[s]),
                        "n_subjects": {s: mode.cell(m, reg, s)["n_subjects"] for s in stats}}
        rows.append(row)
    return rows


def _on_axis(p: dict, key: str) -> bool:
    return p[key] is not None and (key != "likelihood" or p[key] >= C_MIN_LIKELIHOOD)


def draw_scores(mode: Mode) -> list[dict]:
    """a, held-out likelihood against this mode's score; b, the other mode's score against it."""
    fs.apply_paper_style()
    import matplotlib.pyplot as plt

    points = [{"model": m, "regime": reg, "score": mode.score(m, reg), "likelihood": mode.lik.get(m, {}).get(reg),
               "other_score": mode.other.get(m, {}).get(reg)}
              for m in mode.names for reg in REGIMES if mode.score(m, reg) is not None]
    fig, axes = plt.subplots(1, 2, figsize=fs.mm(fs.fit_width(square_panels=2, axes=2), 60))
    for k, (ax, key, label) in enumerate(zip(axes, ("likelihood", "other_score"),
                                             ("held-out likelihood (test subjects)",
                                              f"{mode.other_key} fingerprint score"))):
        for reg, marker in (("per_subject", "o"), ("two_level_em", "s")):
            pts = [p for p in points if p["regime"] == reg and _on_axis(p, key)]
            ax.scatter([p[key] for p in pts], [p["score"] for p in pts], s=10, marker=marker,
                       facecolors="none" if reg == "per_subject" else "0.3", edgecolors="0.3", lw=0.7,
                       label=REGIMES[reg])
        for p in points:
            if p["model"] in CURVES and p["regime"] == _best(mode, p["model"]) and _on_axis(p, key):
                ax.scatter([p[key]], [p["score"]], s=14, zorder=3, marker="o" if p["regime"] == "per_subject" else "s",
                           color=fs.OKABE_ITO[CURVES[p["model"]]], edgecolors="white", lw=0.4, label=p["model"])
        ax.set_yscale("log")
        if key == "other_score":
            ax.set_xscale("log")
            vals = [p[key] for p in points if p[key]] + [p["score"] for p in points]
            ax.plot([min(vals), max(vals)], [min(vals), max(vals)], color="0.7", lw=0.5, zorder=0)
            fs.plain_log_axis(ax)
        else:
            fs.plain_log_axis(ax, which="y")
        ax.set_xlabel(label)
        ax.set_ylabel(mode.score_text)
        ax.spines[["top", "right"]].set_visible(False)
        fs.panel_label(ax, "ab"[k])
    handles, labels = axes[0].get_legend_handles_labels()
    keep = dict(zip(labels, handles))
    axes[1].legend(keep.values(), keep.keys(), frameon=False, loc="lower right", fontsize=5)
    fig.tight_layout()
    fs.save_figure(fig, mode.fig("scores"))
    plt.close(fig)
    return points


def draw_regimes(mode: Mode) -> list[dict]:
    """p(switch) by consecutive losses, per subject (dashed) and as 2-level EM (solid), for
    ``REGIME_MODELS``, with the subjects' mean and the GRU."""
    fs.apply_paper_style()
    import matplotlib.pyplot as plt
    import numpy as np

    labels = mode.bins["losses"]
    xs = np.arange(len(labels))
    fig, axes = plt.subplots(1, len(REGIME_MODELS), figsize=fs.mm(fs.fit_width(square_panels=3, axes=1), 52),
                             sharey=True)
    out = []
    for k, (ax, m) in enumerate(zip(axes, REGIME_MODELS)):
        ax.plot(xs, _floats(mode.cell(m, "per_subject", "losses")["per_bin_common"]["subject"]), "o",
                color="black", ms=2.5, label="subjects", zorder=3)
        ax.plot(xs, _floats(mode.cell(GRU_MODEL, "per_subject", "losses")["per_bin_common"]["model"]), "-",
                color=fs.OKABE_ITO["purple"], lw=0.8, label=GRU_MODEL)
        for reg, ls in (("per_subject", "--"), ("two_level_em", "-")):
            c = mode.cell(m, reg, "losses")
            ax.plot(xs, _floats(c["per_bin_common"]["model"]), ls, color=fs.OKABE_ITO[MODEL_COLORS[m]], lw=1,
                    label=REGIMES[reg])
            out.append({"model": m, "regime": reg, "floor_ratio": c["floor_ratio"],
                        "model_curve": c["per_bin_common"]["model"], "subject_curve": c["per_bin_common"]["subject"]})
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlabel("consecutive losses")
        ax.set_title(m)
        ax.spines[["top", "right"]].set_visible(False)
        if k == 0:
            ax.set_ylabel("p(switch)")
        fs.panel_label(ax, "abc"[k])
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    handles, labels = axes[0].get_legend_handles_labels()   # one legend, below; the colour is the panel's model
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=4)
    fs.save_figure(fig, mode.fig("regimes"))
    plt.close(fig)
    return out


def draw_curves(mode: Mode, shown: list[tuple[str, str]]) -> None:
    """The four rate statistics as curves: the subjects, the five scatter models as 2-level EM
    (thin) and the GRU (thick)."""
    fs.apply_paper_style()
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=fs.mm(183, 110))
    for k, (ax, (st, sl, xlabel, title)) in enumerate(zip(axes.flat, CURVE_PANELS)):
        labels = mode.bins[st][sl]
        xs = np.arange(len(labels))
        for m, reg in shown:
            pb = mode.cell(m, reg, st)["per_bin_common"]
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
    fs.save_figure(fig, mode.fig("curves"))
    plt.close(fig)


def _bin_groups(stat: str, labels: list[str]) -> list[tuple[str, str, list[int]]]:
    if stat == "pattern":
        return [(f"{d} trial{'s' if d > 1 else ''} back", c, [i for i, lab in enumerate(labels) if len(lab) == 2 * d])
                for d, c in ((1, "black"), (2, "blue"), (3, "orange"))]
    if stat in ("hazard", "post_switch"):
        return [("last trial unrewarded", "vermillion", [i for i, lab in enumerate(labels) if lab[0] == "-"]),
                ("last trial rewarded", "blue", [i for i, lab in enumerate(labels) if lab[0] == "+"])]
    return [("all bins", "black", list(range(len(labels))))]


def draw_scatter(mode: Mode, shown: list[tuple[str, str]]) -> list[dict]:
    """Study 01 r9's view: per bin, the subjects' mean against the model's, one row per statistic."""
    fs.apply_paper_style()
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(len(mode.stats), len(shown), figsize=fs.mm(183, 38 * len(mode.stats) + 2))
    out = []
    for row, stat in enumerate(mode.stats):
        labels = mode.bins[stat]
        cs = [mode.cell(m, reg, stat) for m, reg in shown]
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
                step = 0.02 if lim - low <= 0.08 else 0.04   # a wide zoom at 0.02 overlaps
                ticks = np.arange(np.ceil(low / step) * step, lim + 1e-9, step)
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
            ax.set_aspect("equal")
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(f"{m}\n{REGIMES[reg]}", fontsize=5)
            if row == len(mode.stats) - 1:
                ax.set_xlabel(f"subjects {_unit(stat)}")
            if col == 0:
                ax.set_ylabel(f"{STAT_LABELS[stat]}\nmodel {_unit(stat)}")
                fs.panel_label(ax, "abcde"[row])
            if col == 0 and stat not in ("losses", "run_length"):
                ax.legend(frameon=False, loc="upper left", fontsize=5, handletextpad=0.2, borderaxespad=0.1)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.2 / len(mode.stats), top=1 - 0.25 / len(mode.stats),
                        wspace=0.3, hspace=0.35)
    fs.save_figure(fig, mode.fig("scatter"))
    plt.close(fig)
    return out


def draw_rmse(mode: Mode) -> list[dict]:
    """The scatter's per-bin RMSE for every model, regime and statistic."""
    fs.apply_paper_style()
    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    fig, axes = plt.subplots(1, len(mode.stats), figsize=fs.mm(min(183, 50 + 26 * len(mode.stats)), 118),
                             sharey=True)
    for k, (ax, st) in enumerate(zip(axes, mode.stats)):
        for off, reg in ((-0.2, "per_subject"), (0.2, "two_level_em")):
            for i, m in enumerate(mode.names):
                c = mode.cell(m, reg, st)
                if c is None:
                    continue
                v = _rmse(c["per_bin_common"])
                rows.append({"model": m, "regime": reg, "statistic": st, "rmse": v})
                color = (fs.OKABE_ITO["purple"] if m == GRU_MODEL else fs.OKABE_ITO["green"] if m == GLM_MODEL
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
        ax.set_title(STAT_LABELS[st])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        fs.panel_label(ax, "abcde"[k], dx_pt=-6 if k else -24)
    axes[0].set_yticks(np.arange(len(mode.names)))
    axes[0].set_yticklabels(mode.names)
    axes[0].set_ylim(len(mode.names) - 0.5, -0.5)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fs.save_figure(fig, mode.fig("rmse"))
    plt.close(fig)
    return rows


def draw_gru_errors(mode: Mode) -> None:
    """a, b: each subject's floor ratio, the GRU against Lopez-DoubleTrace as 2-level EM; c, d: two
    errors per model (``mode.scratch``)."""
    fs.apply_paper_style()
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    sc = mode.scratch
    keep = [mode.subjects.index(r["subject_id"]) for r in sc["subjects"]]

    def per_subject(m, reg, st, key):
        v = mode.cell(m, reg, st)["per_subject"][key]
        return np.array([np.nan if v[i] is None else v[i] for i in keep])

    shown = {"pattern": "last 1–3 trials", "hazard": "hazard"}
    floor = sc.get("binomial_floor") or sc["floor"]
    fig = plt.figure(figsize=fs.mm(178, 66))
    for k, st in enumerate(shown):
        ax = fig.add_axes([0.07 + k * 0.25, 0.22, 0.17, 0.6])
        f = per_subject(GRU_MODEL, "per_subject", st, "floor")
        x = per_subject("Lopez-DoubleTrace", "two_level_em", st, "gap") / f
        y = per_subject(GRU_MODEL, "per_subject", st, "gap") / f
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=4, color=fs.OKABE_ITO["purple"], lw=0)
        lim = [0.4, 5]
        ax.plot(lim, lim, color="0.6", lw=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ticks = [0.5, 1, 2, 4]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.minorticks_off()
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.set_xlabel("Lopez-DoubleTrace 2L")
        ax.set_ylabel("GRU")
        ax.set_title(f"{shown[st]}: per-subject floor ratio\nGRU lower in {int((y[ok] < x[ok]).sum())} of {int(ok.sum())}")
        ax.spines[["top", "right"]].set_visible(False)
        fs.panel_label(ax, "ab"[k])
    for k, st in enumerate(shown):
        ax = fig.add_axes([0.58 + k * 0.215, 0.22, 0.17, 0.6])
        for j, m in enumerate(sc["models"]):
            lab, c = SCRATCH_LABELS[m]
            e = sc["errors"][m][st]
            ax.bar(j - 0.2, e["subject_mean_curve_error"], 0.38, color=fs.OKABE_ITO[c], alpha=0.4)
            ax.bar(j + 0.2, e["mean_subject_gap"], 0.38, color=fs.OKABE_ITO[c])
        ax.axhline(floor[st], color="0.3", lw=0.6, ls="--")
        ax.set_xticks(range(len(sc["models"])))
        ax.set_xticklabels([SCRATCH_LABELS[m][0] for m in sc["models"]], rotation=45, ha="right")
        ax.set_title(shown[st])
        ax.spines[["top", "right"]].set_visible(False)
        if k == 0:
            ax.set_ylabel("p(switch) error (trial-weighted)")
        fs.panel_label(ax, "cd"[k])
    fig.legend(handles=[Patch(color="0.6", alpha=0.4, label="subject-mean curve"),
                        Patch(color="0.3", label="each subject, then averaged (gap)"),
                        plt.Line2D([], [], color="0.3", lw=0.6, ls="--", label=mode.floor_text)],
               frameon=False, loc="upper center", bbox_to_anchor=(0.78, 1.0), ncol=3, fontsize=5)
    fs.save_figure(fig, mode.fig("gru_errors"))
    plt.close(fig)


def draw_gru_spread(mode: Mode) -> None:
    """Per subject, the GRU's log gap ratio to Lopez-DoubleTrace (2-level) against two measures of
    session spread; and the control."""
    fs.apply_paper_style()
    import matplotlib.pyplot as plt
    import numpy as np

    sc = mode.scratch
    rows = sc["subjects"]
    col = {k: np.array([r[k] for r in rows], float) for k in rows[0] if k != "subject_id"}
    panels = [("lopez_mean_log_sigma", "gru_vs_lopez_2l", "test", "purple",
               "Lopez-DT 2-level: mean log σ\n(session spread, from the fit)"),
              ("log_split_over_binomial_floor", "gru_vs_lopez_2l", "test", "purple",
               "log(split-half floor / binomial floor)\n(session spread, from the data)"),
              ("lopez_mean_log_sigma", "lopez_per_subject_vs_2l", "control", "vermillion",
               "Lopez-DT 2-level: mean log σ")]
    ylabels = {"gru_vs_lopez_2l": "GRU vs Lopez-DT 2L\nlog(gap ratio)",
               "lopez_per_subject_vs_2l": "Lopez-DT: per subject vs 2L\nlog(gap ratio)"}
    fig, axes = plt.subplots(1, 3, figsize=fs.mm(165, 58))
    for k, (ax, (xk, yk, kind, c, xl)) in enumerate(zip(axes, panels)):
        x, y = col[xk], col[yk]
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=4, color=fs.OKABE_ITO[c], lw=0)
        ax.axhline(0, color="0.6", lw=0.5)
        rho = sc["spread"][kind][xk]["spearman_rho"]
        ax.set_title(f"{kind}: Spearman ρ = {fs.num(rho, '+.2f')}", fontsize=6)
        ax.set_xlabel(xl)
        ax.set_ylabel(ylabels[yk])
        if xk == "lopez_mean_log_sigma":   # one diverged fit sits far right; ranks are unaffected
            off = int((x[ok] > -1.5).sum())
            ax.set_xlim(-3.4, -1.8)
            if off:
                ax.text(0.98, 0.03, f"{off} subject off axis (diverged fit)", transform=ax.transAxes,
                        ha="right", fontsize=5, color="0.4")
        ax.text(0.02, 0.97, "above 0: GRU worse" if kind == "test" else "above 0: 2-level better",
                transform=ax.transAxes, va="top", fontsize=5, color="0.4")
        ax.spines[["top", "right"]].set_visible(False)
        fs.panel_label(ax, "abc"[k])
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.25, top=0.88, wspace=0.45)
    fs.save_figure(fig, mode.fig("gru_spread"))
    plt.close(fig)


def draw_sets(mode: Mode, shown: list[tuple[str, str]]) -> list[dict]:
    """The score per session group for the scatter's models: the headline and its groups, then the
    groups outside it."""
    fs.apply_paper_style()
    import matplotlib.pyplot as plt
    import numpy as np

    head = set(mode.sets[mode.head])
    m0, r0 = shown[-1]

    def n_subjects(m, reg, g):
        c = mode.cells[m][reg].get(g)
        return 0 if c is None else min(v["n_subjects"] for v in c["statistics"].values())

    groups = sorted((g for g in mode.sets if g not in (mode.head, "headline", "non-warm-up")
                     and n_subjects(m0, r0, g) >= MIN_GROUP_SUBJECTS), key=lambda g: (g not in head, g))
    order = [mode.head] + groups
    rows = []
    fig, ax = plt.subplots(figsize=fs.mm(min(183, 9 * len(order) + 55), 62))
    xs = np.arange(len(order))
    for k, (m, reg) in enumerate(shown):
        ys = []
        for g in order:
            sc = mode.score(m, reg, g)
            ys.append(np.nan if sc is None else sc)
            rows.append({"model": m, "regime": reg, "set": g, "score": sc, "n_subjects": n_subjects(m, reg, g)})
        ax.plot(xs + (k - (len(shown) - 1) / 2) * 0.1, ys, "o", ms=2.5, color=fs.OKABE_ITO[MODEL_COLORS[m]],
                label=f"{m} ({REGIMES[reg]})")
    ax.axvline(sum(g in head for g in groups) + 0.5, color="0.7", lw=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{'headline' if g == mode.head else g.replace('|', ', ')} ({n_subjects(m0, r0, g)})"
                        for g in order], rotation=60, ha="right")
    ax.set_yscale("log")
    ax.set_yticks([1, 1.5, 2, 3])
    fs.plain_log_axis(ax, which="y")
    ax.set_ylim(1, 3.2)
    ax.set_ylabel(mode.score_text)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fs.save_figure(fig, mode.fig("sets"))
    plt.close(fig)
    return rows


def claims(points: list[dict], likelihood_cut: float = 0.70) -> dict:
    """The rank correlations and regime counts the reports quote."""
    import numpy as np
    from scipy.stats import rankdata

    def rho(x, y):
        return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])

    oth = [p for p in points if p["other_score"] is not None]
    lik = [p for p in points if p["likelihood"] is not None]
    good = [p for p in lik if p["likelihood"] > likelihood_cut]
    both: dict = {}
    for p in points:
        both.setdefault(p["model"], {})[p["regime"]] = p
    pairs = [v for v in both.values() if len(v) == 2]
    return {"spearman_score_other_mode": rho([p["score"] for p in oth], [p["other_score"] for p in oth]),
            "spearman_score_likelihood": rho([p["score"] for p in lik], [p["likelihood"] for p in lik]),
            "likelihood_cut": likelihood_cut, "n_cells_above_cut": len(good),
            "spearman_score_likelihood_above_cut": rho([p["score"] for p in good], [p["likelihood"] for p in good]),
            "n_models_both_regimes": len(pairs),
            "two_level_better": sum(v["two_level_em"]["score"] < v["per_subject"]["score"] for v in pairs),
            "below_equal_line": {reg: [sum(p["score"] < p["other_score"] for p in oth if p["regime"] == reg),
                                       sum(p["regime"] == reg for p in oth)] for reg in REGIMES}}


def draw_all(mode: Mode, shown: list[tuple[str, str]]) -> dict:
    """Every shared figure, in ``FIGURES`` order; returns what each one measured."""
    out = {"rows": draw_grid(mode), "points": draw_scores(mode), "regime_curves": draw_regimes(mode)}
    draw_curves(mode, shown)
    out["scatter"] = draw_scatter(mode, shown)
    out["rmse"] = draw_rmse(mode)
    draw_gru_errors(mode)
    draw_gru_spread(mode)
    out["sets"] = draw_sets(mode, shown)
    out["claims"] = claims(out["points"])
    out["off_axis"] = [p for p in out["points"] if p["likelihood"] is not None and p["likelihood"] < C_MIN_LIKELIHOOD]
    return out


# --- captions, one per figure ---------------------------------------------------------------------

def captions(mode: Mode, res: dict, shown: list[tuple[str, str]]) -> dict[str, str]:
    """The shared figures' blocks, ``result-1`` to ``result-9``, with mode-specific words."""
    n_all, n_common, n_stats = len(mode.subjects), len(mode.common), len(mode.stats)
    word = {4: "four", 5: "five"}[n_stats]
    sc = mode.scratch
    p, sp = sc["paired"], sc["spread"]
    floor = sc.get("binomial_floor") or sc["floor"]

    def img(name, alt):
        return f"![{alt}](../{mode.fig(name).name})"

    def rho(d):
        return f"ρ = {fs.num(d['spearman_rho'], '+.2f')} [{fs.num(d['ci95'][0], '+.2f')}, {fs.num(d['ci95'][1], '+.2f')}]"

    def err(m, st, key):
        return f"{sc['errors'][m][st][key]:.4f}"

    table = ["", "| model | per subject: score | worst statistic | its floor ratio | 2-level EM: score | worst statistic | its floor ratio |",
             "|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        row = [r["model"] + (" \\*" if r["model"] in mode.failed else "")]
        for reg in REGIMES:
            c = r.get(reg)
            if c is None:
                row += ["—", "—", "—"]
                continue
            w = c["worst"]
            row += ["—" if c["score"] is None else f"**{c['score']:.1f}**", STAT_LABELS[w],
                    "—" if c["floor_ratio"][w] is None else f"{c['floor_ratio'][w]:.1f}"]
        table.append("| " + " | ".join(row) + " |")
    cover = (f" ▾ the rollouts visit less than {LOW_COVERAGE:.0%} of the subjects' bin weight."
             if mode.key == "rollout" else "")
    blocks = {"result-1": "\n".join([
        img("grid", f"{mode.key} grid"), "",
        f"**Figure 1.** **a**, floor ratio ({mode.gap_text} ÷ {mode.floor_text}, each averaged over the {n_all} "
        f"test subjects) per model, regime and statistic, on {mode.head_text}. 1 means as close as the "
        f"subjects' own data allow. **score** is the {mode.score_text}, the geometric mean of the {word} ratios "
        f"(ADR 0009). The GRU's row is the mean over its three seeds, on the {n_common} subjects it holds out."
        f"{cover} \\* the 2-level fit failed, so study 12 r3 places the model by its per-subject score; n.r., "
        "not reproduced in replay (r2 gates). **b**, p(switch) by the last trial's reward and the current run "
        f"length (the hazard): the subjects' mean and {mode.model_text} for four models, over the {n_common} "
        "subjects every model holds out that keep the bin (2-level EM where the model has it).",
        *table])}
    off = res["off_axis"]
    blocks["result-2"] = "\n".join([
        img("scores", "scores"), "",
        f"**Figure 2.** Each model's {mode.score_text} (Figure 1a) against **a**, its pooled held-out likelihood "
        "on the 149 test subjects (study 12's per-subject scores, re-pooled; the GLM's own; the GRU's mean over "
        f"seeds, on its 146), and **b**, its {mode.other_key} fingerprint score "
        f"({'r3' if mode.key == 'replay' else 'r2'}), per regime; the grey line marks equal scores. The models "
        "of Figure 1b are coloured."
        + ("" if not off else " Off panel a's axis: " + "; ".join(
            f"{q['model']} ({REGIMES[q['regime']]}, likelihood {q['likelihood']:.3f}, score {q['score']:.0f})"
            for q in off) + ".")])
    blocks["result-3"] = "\n".join([
        img("regimes", "regimes"), "",
        f"**Figure 3.** p(switch) by the count of consecutive unrewarded trials on the current side, in {mode.key}: "
        "the subjects' mean (dots), three literature models with their per-subject parameters (dashed) and as "
        "2-level EM, each session with its own parameters (solid), and the GRU, over the subjects every model "
        "holds out that keep the bin. Floor ratios of this statistic: "
        + "; ".join(f"{c['model']} {REGIMES[c['regime']]} {c['floor_ratio']:.1f}" for c in res["regime_curves"]) + "."])
    blocks["result-4"] = "\n".join([
        img("curves", "curves"), "",
        f"**Figure 4.** The four switch statistics as curves, in {mode.key}: the subjects' mean (dots), the five "
        "literature models of Figure 5 as 2-level EM (thin) and the GRU (thick, the mean of its three seeds), over "
        "the subjects every model holds out that keep the bin. **a**, by consecutive losses; **b**, by the last "
        "two trials (S same side, O other side, + rewarded, − not; most recent first); **c**, **d**, the hazard, "
        "by the current run's length after an unrewarded or a rewarded trial; **e**, **f**, right after a "
        "switch, by the length of the run before it and the switch trial's reward."])
    rows_text = "rows **a** to **e**" if n_stats == 5 else "rows **a** to **d**"
    blocks["result-5"] = "\n".join([
        img("scatter", "scatter"), "",
        f"**Figure 5.** Study 01 r9's view, in {mode.key}, for each statistic ({rows_text}): each point is one bin, "
        f"the subjects' mean against {mode.model_text}, over the subjects every model holds out that keep the bin. "
        "Columns: the five literature models with the best 2-level EM likelihood, as 2-level EM, from the lowest "
        "likelihood to the highest, then the GRU. Each panel prints r and RMSE over the bins (each bin once) and "
        "the model's floor ratio for the statistic (its cell in Figure 1a, trial-weighted)."
        + (" Run lengths are shares of runs, not p(switch)." if "run_length" in mode.stats else "")])
    blocks["result-6"] = "\n".join([
        img("rmse", "rmse"), "",
        "**Figure 6.** Figure 5's per-bin RMSE for every model and statistic: per subject (light) and as 2-level "
        "EM (dark); the GLM in green and the GRU in purple (dashed line: the GRU's value). The axis stops at "
        f"{RMSE_MAX:g}; a longer bar is cut and labelled. Models in Figure 1's order."])
    blocks["result-7"] = "\n".join([
        img("gru_errors", "two errors"), "",
        f"**Figure 7.** On the {sc['n_subjects']} test subjects every model holds out. **a**, **b**, each subject's "
        f"floor ratio in {mode.key}, the GRU against Lopez-DoubleTrace as 2-level EM (the diagonal is a tie). **c**, "
        "**d**, two errors per model, both trial-weighted over the same kept bins: of the subject-mean curve "
        "(light; what Figure 5 compares) and of each subject, then averaged (dark; the gap the floor ratio "
        f"divides). The dashed line is the mean {mode.floor_text}.", "",
        "| | GRU | Lopez-DoubleTrace 2L |", "|---|---|---|",
        f"| subject-mean curve error: last 1–3 trials, hazard | {err(GRU_MODEL, 'pattern', 'subject_mean_curve_error')}, "
        f"{err(GRU_MODEL, 'hazard', 'subject_mean_curve_error')} | {err('Lopez-DoubleTrace', 'pattern', 'subject_mean_curve_error')}, "
        f"{err('Lopez-DoubleTrace', 'hazard', 'subject_mean_curve_error')} |",
        f"| mean per-subject gap: last 1–3 trials, hazard (floors {floor['pattern']:.4f}, {floor['hazard']:.4f}) | "
        f"{err(GRU_MODEL, 'pattern', 'mean_subject_gap')}, {err(GRU_MODEL, 'hazard', 'mean_subject_gap')} | "
        f"{err('Lopez-DoubleTrace', 'pattern', 'mean_subject_gap')}, {err('Lopez-DoubleTrace', 'hazard', 'mean_subject_gap')} |",
        f"| subjects where the GRU's gap is larger (four statistics) | {p['n_gru_worse']} of {p['n_subjects']} "
        f"(sign test p = {fs.num(p['sign_test_p'], '.2g')}; Wilcoxon p = {fs.num(p['wilcoxon_p'], '.2g')}) | |",
        f"| median per-subject gap ratio, GRU ÷ Lopez 2L | {p['median_ratio']:.2f} "
        f"[{p['median_ratio_ci95'][0]:.2f}, {p['median_ratio_ci95'][1]:.2f}] | |"])
    blocks["result-8"] = "\n".join([
        img("gru_spread", "session spread"), "",
        f"**Figure 8.** Per subject, the GRU's log gap ratio to Lopez-DoubleTrace (2-level) in {mode.key}, mean over "
        "the four rate statistics, against **a**, the subject's session spread from Lopez-DoubleTrace's 2-level fit "
        "(mean log σ), and **b**, from the data (log of the split-half floor, which includes the differences "
        "between sessions, over the binomial floor, which does not; both from the replay sessions, as they "
        "describe the mouse and the fit, not the mode). **c**, control: the same spread against "
        f"Lopez-DoubleTrace's own gain from per-session parameters in {mode.key} (log per-subject gap over 2-level "
        "gap). Spearman ρ with 95% bootstrap intervals:", "",
        "| | ρ |", "|---|---|",
        f"| GRU's ratio to Lopez 2L vs spread from the fit (a) | {rho(sp['test']['lopez_mean_log_sigma'])} |",
        f"| GRU's ratio to Lopez 2L vs spread from the data (b) | {rho(sp['test']['log_split_over_binomial_floor'])} |",
        f"| GRU's ratio to Lopez 2L vs trials; vs sessions | {rho(sp['test']['n_trials'])}; {rho(sp['test']['n_sessions'])} |",
        f"| control: Lopez's 2-level gain vs spread from the fit (c); from the data | "
        f"{rho(sp['control']['lopez_mean_log_sigma'])}; {rho(sp['control']['log_split_over_binomial_floor'])} |",
        f"| the two spread measures, against each other | {rho(sp['measures_agree'])} |", "",
        f"Lopez-DoubleTrace's median gain from per-session parameters: its per-subject gap is "
        f"{sp['control_median_ratio']:.2f} times its 2-level gap."])
    outside = "the warm-ups" if mode.key == "replay" else ("the early coupled stages and the warm-ups, where the gym "
                                                          "does not hold a block as the real task's "
                                                          "`AdvancedBlockAuto` does (ADR 0006)")
    blocks["result-9"] = "\n".join([
        img("sets", "sets"), "",
        f"**Figure 9.** {mode.score_text[0].upper() + mode.score_text[1:]} per session group, for the models of "
        f"Figure 5, on groups where at least {MIN_GROUP_SUBJECTS} subjects have a floor (in brackets: the GRU's "
        f"count). Left of the line: the headline and its groups. Right: {outside}."])
    return blocks


def rewrite(text: str, block: str, body: str) -> str:
    begin, end = f"<!-- BEGIN {block} -->", f"<!-- END {block} -->"
    head, rest = text.split(begin)
    _, tail = rest.split(end)
    return f"{head}{begin}\n{body}\n{end}{tail}"


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())
