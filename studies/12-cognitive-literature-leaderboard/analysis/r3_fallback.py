"""Producer for r3's result-3: failed 2-level fits and the per-mouse fallback (transparency only).

MEvolve's per-mouse fallback rule (MEvolve ADR 0004, #81): a subject whose 2-level held-out score is
non-finite or more than FALLBACK_TOL nats per trial below its per-subject score counts as a failed
2-level fit and is scored by its per-subject fit. Per literature model, this writes how many held-out
subjects fail and the pooled score with the rule applied. The headline (result-1) stays the raw 2-level
score: Han judged raw more transparent (2026-11-12). The atlas leaderboard's hover card reads
``n_fallback_heldout`` and ``pooled_heldout_fallback`` per ``EM-`` label.

Reads the two frozen fit tables and ``r3_2level_em.json`` (labels, raw scores); never rewrites the latter.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
from _meta import build_meta, write_artifact  # noqa: E402
from r3_2level_em import EM, PS, REPORT, _pooled, _read, _rewrite  # noqa: E402

R3 = HERE / "r3_2level_em.json"
OUT = HERE / "r3_fallback.json"
FALLBACK_TOL = 0.005  # nats per trial: MEvolve evaluate.FALLBACK_TOL


def compute() -> dict:
    em, ps = _read(EM), _read(PS)
    out = {}
    for label, row in json.loads(R3.read_text())["by_model"].items():
        m = row["literature_model"]
        held = [s for s in em[m] if em[m][s]["pool"] == "heldout"]
        e_ll = {s: (float(em[m][s]["log_lik"]), int(em[m][s]["n_trials"])) for s in held}
        p_ll = {s: (float(ps[m][s]["eval_total_log_likelihood"]), int(ps[m][s]["eval_total_trials"])) for s in held}
        fell = sorted(s for s in held
                      if not math.isfinite(e_ll[s][0]) or e_ll[s][0] < p_ll[s][0] - FALLBACK_TOL * e_ll[s][1])
        raw = _pooled([e_ll[s] for s in held])
        assert abs(raw - row["pooled_heldout"]) < 1e-12, f"{m}: raw score differs from r3"
        out[label] = {"literature_model": m, "n_heldout": len(held), "n_fallback_heldout": len(fell),
                      "fallback_subjects": fell, "pooled_heldout": raw,
                      "pooled_heldout_fallback": _pooled([p_ll[s] if s in fell else e_ll[s] for s in held])}
    return out


def render(f: dict) -> str:
    lines = [f"Failed 2-level fits: held-out subjects whose 2-level score is non-finite or more than {FALLBACK_TOL} "
             "nats per trial below their per-subject score (r1), by MEvolve's per-mouse fallback rule (ADR 0004). The "
             "last column scores those subjects by their per-subject fit instead. Shown for transparency: the headline "
             "(result-1) stays the raw 2-level score.",
             "",
             "| literature model | 2-level EM (raw) | failed 2-level fits | with per-mouse fallback |",
             "|---|---:|---:|---:|"]
    for e in sorted(f.values(), key=lambda e: -e["pooled_heldout"]):
        lines.append(f"| {e['literature_model']} | {e['pooled_heldout']:.4f} | {e['n_fallback_heldout']} of "
                     f"{e['n_heldout']} | {e['pooled_heldout_fallback']:.4f} |")
    return "\n".join(lines)


def main() -> None:
    f = compute()
    REPORT.write_text(_rewrite(REPORT.read_text(), "result-3", render(f)))
    meta = build_meta("analysis/r3_fallback.py", [], study_root=HERE.parent,
                      code=[__file__, HERE / "r3_2level_em.py"], inputs=[EM, PS, R3])
    write_artifact(OUT, {"_meta": meta, "tolerance_nats_per_trial": FALLBACK_TOL, "by_model": f},
                   outputs=[f"{REPORT}#result-3"])


if __name__ == "__main__":
    main()
