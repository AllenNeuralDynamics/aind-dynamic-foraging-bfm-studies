"""K scan report: table, figure (SVG) and out/kscan/kscan_results.json with provenance."""
import json
import subprocess
from importlib import metadata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PILOT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe")
OUT = PILOT / "out"
KOUT = OUT / "kscan"
SETS = ["latents", "logits", "history", "untrained"]
LABELS = {"latents": "GRU latents", "logits": "GRU logits", "history": "History", "untrained": "Untrained GRU"}
COLORS = {"latents": "#1b9e77", "logits": "#d95f02", "history": "#7570b3", "untrained": "#999999"}
plt.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
                     "lines.linewidth": 0.9, "lines.markersize": 2.5})
MM = 1 / 25.4


def sha(path):
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def main():
    parts = {}
    for f in sorted((KOUT / "parts").glob("*.json")):
        d = json.loads(f.read_text())
        parts[(d["emission"], d["K"], d["feature_set"])] = d
    glm = {}
    for e in ("hurdle", "reference"):
        p = KOUT / f"glmhmm_{e}_summary.json"
        if p.exists():
            glm[e] = {int(k): v for k, v in json.loads(p.read_text())["per_K"].items()}
    emissions = [e for e in ("hurdle", "reference") if e in glm]
    Ks = sorted({k for e in emissions for k in glm[e]})

    # Figure (2 x 2): pooled R2 all / within session; information recovered / GLM-HMM likelihood.
    # Hurdle solid, reference dashed.
    fig, axes = plt.subplots(2, 2, figsize=(120 * MM, 90 * MM))
    styles = {"hurdle": "-", "reference": "--"}
    for ax, (metric, title) in zip([axes[0, 0], axes[0, 1], axes[1, 0]],
                                   [("pooled_r2", "Pooled R², all trials"),
                                    ("pooled_r2_within", "Pooled R², within session"),
                                    ("info_recovered", "Information recovered")]):
        for e in emissions:
            for s in SETS:
                ks = [k for k in Ks if (e, k, s) in parts]
                ax.plot(ks, [parts[(e, k, s)][metric] for k in ks], styles[e], marker="o", color=COLORS[s],
                        label=f"{LABELS[s]}" if e == "hurdle" else None)
        ax.set_ylim(0, 1)
        ax.set_xticks(Ks)
        ax.set_title(title)
        ax.set_xlabel("K (number of states)")
    ax = axes[1, 1]
    for e in emissions:
        ax.plot(Ks, [glm[e][k]["glmhmm_likelihood_pilot"] for k in Ks], styles[e], marker="o", color="k",
                label=f"{e}, pilot trials")
        ax.plot(Ks, [glm[e][k]["logged_train_mice_eval_likelihood"] for k in Ks], styles[e], marker="o",
                color="0.6", label=f"{e}, logged (all 614 mice)")
    ax.set_xticks(Ks)
    ax.set_title("GLM-HMM predictive likelihood")
    ax.set_xlabel("K (number of states)")
    ax.legend(frameon=False, fontsize=4.5)
    axes[0, 0].legend(frameon=False, fontsize=5, loc="lower left", title="solid hurdle, dashed reference",
                      title_fontsize=5)
    fig.tight_layout()
    fig.savefig(KOUT / "fig_kscan.svg")

    table = []
    for e in emissions:
        for k in Ks:
            row = {"emission": e, "K": k, "run_id": glm[e][k]["run_id"],
                   "glmhmm_likelihood_pilot": glm[e][k]["glmhmm_likelihood_pilot"],
                   "target_entropy": glm[e][k]["target_entropy"],
                   "within_session_share": glm[e][k]["within_session_share_pooled"]}
            for s in SETS:
                d = parts.get((e, k, s))
                if d:
                    for m in ("pooled_r2", "pooled_r2_within", "info_recovered", "mean_state_r2", "kl", "kl_const"):
                        row[f"{s}.{m}"] = d[m]
                    row[f"{s}.max_logistic_iters"] = max(d["logistic_iters"])
            table.append(row)
    (KOUT / "kscan_results.json").write_text(json.dumps({
        "_meta": {"kind": "scratch pilot K scan (study 30 not scaffolded)",
                  "produced_by": [str(p) for p in sorted((PILOT / "scripts").glob("kscan_*"))],
                  "wrapper_git_sha": sha("/home/han.hou/code/wt/wrapper-study30-pilot"),
                  "dispatcher_git_sha": sha("/home/han.hou/code/wt/dispatcher-study30-pilot"),
                  "aind_dynamic_foraging_models_version": metadata.version("aind-dynamic-foraging-models"),
                  "gru_run_id": "nxd-3way-20260707-075401-80a6d102", "gru_checkpoint": "step_90000",
                  "glmhmm_groups": {"hurdle": "hurdle@20261027-212905", "reference": "reference@20261027-213006"},
                  "fold_seed": 0, "fold_of_mouse_source": str(OUT / "probe_results.json"),
                  "pilot_mice_source": str(OUT / "pilot_meta.json")},
        "glmhmm": glm, "table": table,
        "parts": {f"{e}|K{k}|{s}": d for (e, k, s), d in parts.items()}}, indent=1))

    print("emission K run  glmLik  H  within% | " + " | ".join(f"{s}: R2 R2w info" for s in SETS))
    for row in table:
        cells = []
        for s in SETS:
            if f"{s}.pooled_r2" in row:
                cells.append(f"{row[f'{s}.pooled_r2']:.2f} {row[f'{s}.pooled_r2_within']:.2f} {row[f'{s}.info_recovered']:.2f}"
                             + ("*" if row[f"{s}.max_logistic_iters"] >= 1000 else ""))
            else:
                cells.append("  -    -    - ")
        print(f"{row['emission'][:3]} {row['K']:2d} {row['run_id']} {row['glmhmm_likelihood_pilot']:.4f} "
              f"{row['target_entropy']:.2f} {100 * row['within_session_share']:.0f} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
