#!/usr/bin/env python
"""ADR 0008's selection: the candidates and the reference on one cohort, the fingerprint gate, the choice.

Reads the frozen ``selection/<cohort>/<name>.json`` of ``freeze_selection.py`` (offline). A candidate passes
the gate if its replay and rollout fingerprint scores are both no larger than the reference's (lower is
closer to the subjects); the search's best is the passing candidate with the highest held-out score. Writes
``selection_<cohort>.json`` and ``fig_selection_<cohort>.svg``.

    python analysis/selection_gate.py --cohort selection
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
import figure_style as fs  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402

CANDIDATES = STUDY / "reference" / "selection_candidates_seeded-islands-2.json"
REFERENCE = "reference_miller_rhg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["generation", "selection"])
    a = ap.parse_args()
    cand = json.loads(CANDIDATES.read_text())
    files = {p.stem: p for p in sorted((HERE / "selection" / a.cohort).glob("*.json"))}
    assert REFERENCE in files, f"no {REFERENCE} on {a.cohort}"
    runs = {k: json.loads(p.read_text()) for k, p in files.items()}
    ref = runs[REFERENCE]
    rows = []
    for x in cand["ranked"]:
        name = f"rank{x['rank']}_{x['id'][:8]}"
        if name not in runs:
            continue
        r = runs[name]
        rp, ro = r["replay"]["score"], r["rollout"]["score"]
        ok = rp is not None and ro is not None and rp <= ref["replay"]["score"] and ro <= ref["rollout"]["score"]
        rows.append({"name": name, "rank": x["rank"], "patch_name": x["patch_name"], "island": x["island"],
                     "validation_x100": x["validation_score_x100"], "score_x100": 100 * r["score"]["score"],
                     "n_fallback": r["score"]["n_fallback"], "replay_score": rp, "rollout_score": ro,
                     "passes_gate": bool(ok)})
    passing = [r for r in rows if r["passes_gate"]]
    best = max(passing, key=lambda r: r["score_x100"]) if passing else None
    reference = {"name": REFERENCE, "score_x100": 100 * ref["score"]["score"], "n_fallback": ref["score"]["n_fallback"],
                 "replay_score": ref["replay"]["score"], "rollout_score": ref["rollout"]["score"]}

    fs.apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=fs.mm(fs.fit_width(groups=len(rows) + 1) * 3 / 2 + 36, 62),
                             layout="constrained")
    labels = ["Miller-RHG"] + [f"#{r['rank']}" for r in rows]
    cols = [fs.OKABE_ITO["black"]] + [fs.OKABE_ITO["blue"] if r["passes_gate"] else fs.OKABE_ITO["vermillion"]
                                      for r in rows]
    for ax, key, title in zip(axes, ("score_x100", "replay_score", "rollout_score"),
                              ("held-out score (×100)", "replay fingerprint score", "rollout fingerprint score")):
        vals = [reference[key]] + [r[key] for r in rows]
        ax.bar(range(len(vals)), vals, color=cols, width=0.7)
        ax.set_xticks(range(len(vals)), labels, rotation=45, ha="right")
        ax.set_ylabel(title)
        if key != "score_x100":
            ax.axhline(reference[key], color="0.4", ls=":", lw=0.6)
        else:
            lo = min(vals) - 0.1
            ax.set_ylim(lo, max(vals) + 0.05)
        if best is not None:
            i = 1 + rows.index(best)
            ax.annotate("chosen", (i, vals[i]), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=5)
    for ax, letter in zip(axes, "abc"):
        fs.panel_label(ax, letter)
    fig.suptitle(f"Selection on the {a.cohort} subjects: blue passes the fingerprint gate (both scores ≤ Miller-RHG's)")
    fig_path = HERE / f"fig_selection_{a.cohort}.svg"
    fs.save_figure(fig, fig_path)
    out = HERE / f"selection_{a.cohort}.json"
    meta = build_meta(f"analysis/{Path(__file__).name}", [], study_root=STUDY, code=[__file__],
                      inputs=[CANDIDATES] + list(files.values()))
    write_artifact(out, {"_meta": meta, "cohort": a.cohort, "reference": reference, "candidates": rows,
                         "chosen": best and best["name"],
                         "rule": "passes if replay and rollout fingerprint scores are both <= the reference's; "
                                 "the chosen program is the passing candidate with the highest held-out score "
                                 "(study 40 ADR 0008)"},
                   outputs=[fig_path])
    print(json.dumps({"reference": reference, "candidates": rows, "chosen": best and best["name"]}, indent=1))


if __name__ == "__main__":
    main()
