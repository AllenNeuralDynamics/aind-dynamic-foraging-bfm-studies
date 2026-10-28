"""r6: regenerate the numbers block of ``reports/r6-integration-axis.md`` from I1 and I4.

    python analysis/report_r6.py

Offline formatter: reads only committed JSON, rewrites what sits between the r6 markers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent
REPORT = ANALYSIS / "reports" / "r6-integration-axis.md"
BEGIN, END = "<!-- BEGIN result-1 -->", "<!-- END result-1 -->"


def matrix(res: dict) -> list[str]:
    names = res["clusters"]
    return (["| | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
            + [f"| {a} | " + " | ".join(f"{res['cosine'][i][j]:.2f}" for j in range(len(names))) + " |"
               for i, a in enumerate(names)])


def block() -> str:
    i1 = json.loads((ANALYSIS / "i1_cluster_axes.json").read_text())
    i4 = json.loads((ANALYSIS / "i4_distilled_model.json").read_text())
    i2 = json.loads((ANALYSIS / "i2_tdr.json").read_text())
    i3 = json.loads((ANALYSIS / "i3_tdr_dynamics.json").read_text())
    tdr = ["### I2 TDR with ignored trials", "",
           f"Terms (orthogonalised in this order): {', '.join(i2['terms'])}; {i2['n_steps']:,} steps; "
           f"task + nuisance R² {i2['task_plus_nuisance_r2']:.3f}.", "",
           f"Value axis vs a refit of Study 50's design (5 terms, steps after an engaged step only): cosine "
           f"{i2['study12_design']['value_axis_cosine_with_full_fit']:.3f}. Lobe axes (β_v ± β_vc) cosine "
           f"{i2['lobe_axes_cosine']:.2f}.", "",
           "| raw axis | cosine with side readout | cosine with engage direction |", "|---|---:|---:|",
           *[f"| {n} | {a:+.2f} | {b:+.2f} |" for n, a, b in zip(i2["terms"], i2["raw_axis_cosine_with_side_readout"],
                                                                  i2["raw_axis_cosine_with_engage_direction"])],
           "", "![I2](../fig_I2_tdr.png)", "",
           "### I3 Value coordinate carried through each input (C3); C from Study 50", "",
           "| input | source | n | slope | R² | intercept |", "|---|---|---:|---:|---:|---:|"]
    for key, r in i3["transitions"].items():
        inp, src = key.split(" | from ")
        tdr.append(f"| {inp} | {src} | {r['n']:,} | {r['slope']:.3f} | {r['r2']:.3f} | {r['intercept']:+.2f} |")
    tdr += ["", "C (Study 50, value axis of its own TDR): " + "; ".join(
        f"{k.replace(' | from ', ' from ')} {v['slope']:.2f} / {v['r2']:.2f}" for k, v in i3["C_study12_transitions"].items()) + ".",
        "", "Side readout ≈ a·v + b within each source cluster (state at t):", "",
        "| cluster | a | b | v at neutral readout | mean v | R² |", "|---|---:|---:|---:|---:|---:|",
        *[f"| {c} | {r['a']:.2f} | {r['b']:+.2f} | {r['value_at_neutral_readout']:+.2f} | {r['value_mean']:+.2f} | {r['r2']:.2f} |"
          for c, r in i3["readout_vs_value_by_cluster"].items()],
        "", "Value coordinate of each input's stable attractor (mean context): " + "; ".join(
            f"{k} {', '.join(f'{p[0]:+.2f}' for p in f['tdr'])}" for k, f in i3["attractors_all_sessions_context"].items()) + ".",
        "", "![I3](../fig_I3_tdr_dynamics.png)", ""]
    c = json.loads((ANALYSIS / "i3c_value_causal.json").read_text())
    tdr += ["### I3c Causal test: nudge ±δ along the value axis, re-run on the observed inputs", "",
            f"{c['n_starts']:,} starts ({c['n_per_input']:,} per input), δ = {c['delta']}; unperturbed rollout matches the "
            f"stored states to {c['parity_median_abs_state_err']:.1e}. At the nudged state itself: side readout "
            f"{c['nudge_to_readouts_at_k0']['side']:+.2f} per unit, engage {c['nudge_to_readouts_at_k0']['engage']:+.3f}.", "",
            "| input | source | n | causal Δv(t+1)/δ (mean ± SD) | I3 slope |", "|---|---|---:|---|---:|",
            *[f"| {k.split(' | from ')[0]} | {k.split(' | from ')[1]} | {v['n']:,} | {v['causal_k1_mean']:.3f} ± "
              f"{v['causal_k1_sd']:.3f} | {v['i3_slope']:.3f} |" for k, v in c["by_cell_k1"].items()],
            "", "| steps after the nudge | " + " | ".join(str(k) for k in c["k"]) + " |", "|---|" + "---:|" * len(c["k"]),
            "| value nudge → Δv/δ | " + " | ".join(f"{x:.2f}" for x in c["linear_chain"]["observed_mean"]) + " |",
            "| product of I3 slopes | " + " | ".join(f"{x:.2f}" for x in c["linear_chain"]["predicted_mean"]) + " |",
            "| value nudge → Δside | " + " | ".join(f"{x:.2f}" for x in c["summary"]["value"]["side_per_delta_mean"]) + " |",
            "| value nudge → Δengage | " + " | ".join(f"{x:+.2f}" for x in c["summary"]["value"]["engage_per_delta_mean"]) + " |",
            "| engage nudge → Δv/δ | " + " | ".join(f"{x:+.2f}" for x in c["summary"]["engage"]["value_per_delta_mean"]) + " |",
            "| engage nudge → Δengage | " + " | ".join(f"{x:.2f}" for x in c["summary"]["engage"]["engage_per_delta_mean"]) + " |",
            *[f"| {n} → Δv/δ | " + " | ".join(f"{x:+.2f}" for x in v["value_per_delta_mean"]) + " |"
              for n, v in c["summary"].items() if n.startswith("random")],
            "", "Correlation across starts of observed vs chain-predicted Δv/δ, per step: "
            + ", ".join(f"{x:.2f}" for x in c["linear_chain"]["corr_per_k"]) + ".", "",
            "![I3c](../fig_I3c_value_causal.png)", ""]
    lines = ["[regenerated by `analysis/report_r6.py` — do not edit by hand]", "", *tdr,
             "### I1 Encoding axes per last-input cluster (cosines; shared sessions)", "",
             "C3, side readout:", "", *matrix(i1["C3"]["side"]), "",
             "C (Study 50), side readout:", "", *matrix(i1["C"]["side"]), "",
             "C3, engage readout:", "", *matrix(i1["C3"]["engage"]), "",
             "| cluster | C3 side mean / std | C side mean / std | C3 engage mean / std |", "|---|---|---|---|"]
    for c in i1["C3"]["side"]["clusters"]:
        s3, e3 = i1["C3"]["side"]["stats"][c], i1["C3"]["engage"]["stats"][c]
        s2 = i1["C"]["side"]["stats"].get(c)
        lines.append(f"| {c} | {s3['readout_mean']:+.2f} / {s3['readout_std']:.2f} | "
                     + (f"{s2['readout_mean']:+.2f} / {s2['readout_std']:.2f}" if s2 else "—")
                     + f" | {e3['readout_mean']:+.2f} / {e3['readout_std']:.2f} |")
    lines += ["", "![I1](../fig_I1_cluster_axes.png)", "",
              "### I4 Distilled models, simulated from the animals' inputs (shared sessions)", "",
              "| model | L/R likelihood | " + " | ".join(f"gap {g}" for g in i4["lr_likelihood_by_gap"]) + " |",
              "|---|---:|" + "---:|" * len(i4["lr_likelihood_by_gap"])]
    for name, value in i4["lr_likelihood"].items():
        lines.append(f"| {name} | {value:.4f} | " + " | ".join(
            f"{g[name]:.3f}" for g in i4["lr_likelihood_by_gap"].values()) + " |")
    lines += ["", "Trials per gap bin: " + "; ".join(f"{g} {v['n']:,}" for g, v in i4["lr_likelihood_by_gap"].items()) + ".",
              "", "Engage-vs-ignore likelihood on every step: " + "; ".join(
                  f"{k} {v:.4f}" for k, v in i4["engage_likelihood_all_steps"].items()) + ".",
              f"Simulated vs GRU: corr(v) {i4['simulated_vs_gru_C3']['corr_v']:.3f}, corr(e) {i4['simulated_vs_gru_C3']['corr_e']:.3f}.",
              "", "2-var update per group, x(t+1) = A x(t) + c, x = (v, e):", "",
              "| group | n | A (v←v, v←e; e←v, e←e) | c (v, e) |", "|---|---:|---|---|"]
    for g, p in i4["params_2var"].items():
        a = p["A"]
        lines.append(f"| {g} | {p['n']:,} | {a[0][0]:.2f}, {a[0][1]:+.3f}; {a[1][0]:+.3f}, {a[1][1]:.2f} | "
                     f"{p['c'][0]:+.2f}, {p['c'][1]:+.2f} |")
    r = i4["ignore_rule_1var"]
    lines += ["", f"1-var ignore rule: v ← {r['lambda']:.3f}·v {r['c']:+.3f}.", "", "![I4](../fig_I4_distilled_model.png)"]
    return "\n".join(lines)


def main() -> None:
    text = REPORT.read_text()
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        raise AssertionError(f"{REPORT} lacks the {BEGIN} / {END} markers")
    REPORT.write_text(pattern.sub(lambda _: f"{BEGIN}\n{block()}\n{END}", text))


if __name__ == "__main__":
    main()
