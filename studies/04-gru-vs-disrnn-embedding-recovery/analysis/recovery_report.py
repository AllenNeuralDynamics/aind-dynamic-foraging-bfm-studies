#!/usr/bin/env python
"""Single producer for the 04-gru-vs-disrnn-embedding-recovery study.

Reads the committed curated grids (offline, no W&B needed):
  - analysis/ladder_results.csv           (headline per stage)
  - analysis/stage4b_recovery_grid.csv    (per-run Stage-4b GRU recovery)
  - analysis/disrnn_stage4a_recovery_grid.csv (per-run disRNN Stage-4a recovery)

Writes:
  - analysis/recovery_summary.json  (with a _meta provenance block)
  - analysis/recovery_summary.csv
  - analysis/fig_ladder.png             (the baseline-flip ladder)
  - analysis/fig_disrnn_stage4a.png     (disRNN replication)
and regenerates the <!-- BEGIN result-N --> / <!-- END result-N --> blocks in
reports/r1-gru-ladder.md and reports/r2-disrnn-replication.md.

Idempotent: `make -C studies/04-gru-vs-disrnn-embedding-recovery` twice == empty diff.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent          # analysis/
STUDY = HERE.parent                              # studies/<study>/
sys.path.insert(0, str(STUDY.parent / "util"))   # studies/util
from _meta import build_meta                      # noqa: E402
from plot_style import apply_presentation_style   # noqa: E402

WANDB_PROJECT = "embedding_recovery"
WANDB_GROUPS = [
    "gru-stage1", "gru-stage2", "gru-stage2b", "gru-stage3", "gru-stage4a", "gru-stage4b",
    "disrnn-stage4a",
    "baseline-rl-stage1", "baseline-rl-stage2", "baseline-rl-stage2b",
    "baseline-bari-stage3", "baseline-hattori-stage3", "baseline-ctt-stage3",
    "baseline-bari-stage4a", "baseline-ctt-stage4a", "baseline-losscounting-stage4a",
    "baseline-bari-stage4b", "baseline-ctt-stage4b", "baseline-losscounting-stage4b",
]
FAMCOL = {"QLearning": "#2166ac", "CompareToThreshold": "#b2182b", "LossCounting": "#1b7837"}


def _pca2(X):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def main() -> None:
    ladder = pd.read_csv(HERE / "ladder_results.csv", dtype=str)
    s4b = pd.read_csv(HERE / "stage4b_recovery_grid.csv")
    dis = pd.read_csv(HERE / "disrnn_stage4a_recovery_grid.csv")

    # ---- curated summary JSON ----
    summary = {
        "_meta": build_meta("analysis/recovery_report.py", WANDB_GROUPS, study_root=STUDY),
        "wandb_project": WANDB_PROJECT,
        "ladder": ladder.to_dict(orient="records"),
        "stage4b_gru": s4b.to_dict(orient="records"),
        "disrnn_stage4a": dis.to_dict(orient="records"),
    }
    (HERE / "recovery_summary.json").write_text(json.dumps(summary, indent=2))
    ladder.to_csv(HERE / "recovery_summary.csv", index=False)

    apply_presentation_style()

    # ---- Figure 1: the baseline-flip ladder ----
    _fig_ladder(ladder)
    # ---- Figure 2: disRNN Stage-4a replication ----
    _fig_disrnn(dis)

    # ---- regenerate report blocks ----
    _update_reports(ladder, s4b, dis)
    print("recovery_report: wrote summary + 2 figures + report blocks")


def _fig_ladder(ladder):
    # relative-LL: GRU vs baseline across the ladder (numeric midpoints of the ranges)
    def mid(s):
        s = str(s).replace("~", "").split("(")[0].strip()
        if "\u2013" in s:
            a, b = s.split("\u2013"); return (float(a) + float(b)) / 2
        try: return float(s)
        except ValueError: return np.nan
    L = ladder[ladder.stage != "4a-disRNN"].copy()
    L["gru"] = L.model_rel_LL.map(mid); L["base"] = L.baseline_rel_LL.map(mid)
    x = np.arange(len(L))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x, L.gru, marker="o", color="#1a1a1a", label="GRU (data-driven)")
    ax.plot(x, L.base, marker="s", color="#b2182b", label="correct-model baseline")
    ax.axhline(1.0, color="#cccccc", ls=":", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels("S" + L.stage, rotation=0)
    ax.set_ylim(0.35, 1.03); ax.set_xlabel("ladder stage"); ax.set_ylabel("relative likelihood")
    ax.set_title("The baseline flip: correct-model baseline breaks under\nextrapolation + mixture; the GRU embedding does not", fontsize=13, loc="left")
    ax.annotate("baseline flip\n(2b: extrapolation)", xy=(2, L.base.iloc[2]), xytext=(2.1, 0.62),
                fontsize=11, color="#b2182b", arrowprops=dict(arrowstyle="->", color="#b2182b"))
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(HERE / "fig_ladder.png"); plt.close(fig)


def _fig_disrnn(dis):
    gru_lik = {4: 0.990552868070854, 8: 0.990559486606228, 16: 0.9881624486414816}
    mk = {"none": "o", "scalar": "s"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axA = axes[0]
    for enc in ["none", "scalar"]:
        sub = dis[dis.enc == enc].sort_values("embed")
        axA.plot(sub.embed, sub.family_acc_subject, marker=mk[enc], color="#762a83", ms=9,
                 mfc="#762a83" if enc == "scalar" else "white", mec="#762a83", mew=1.6, label=f"disRNN ({enc})")
    axA.axhline(1.0, color="#1a1a1a", lw=1.4); axA.text(16.2, 1.0, "GRU=1.00", va="center", fontsize=11)
    axA.axhline(0.70, color="#b2182b", ls="--", lw=1.3); axA.text(16.2, 0.70, "baseline=0.70", va="center", fontsize=11, color="#b2182b")
    axA.axhline(1/3, color="#cccccc", ls=":", lw=1.1); axA.text(16.2, 1/3, "chance", va="center", fontsize=10, color="#999")
    axA.set_xticks([4, 8, 16]); axA.set_xlim(3.4, 22); axA.set_ylim(0.28, 1.06)
    axA.set_xlabel("subject embedding size"); axA.set_ylabel("family-decoding accuracy")
    axA.set_title("0.95\u20130.98 with session cond.;\n0.75\u20130.90 without", fontsize=12, loc="left")
    axA.legend(frameon=False, fontsize=11, loc="lower left", bbox_to_anchor=(0.02, 0.02))
    axB = axes[1]
    x = np.array([4, 8, 16])
    dn = dis[dis.enc == "none"].set_index("embed").reindex(x)
    ds = dis[dis.enc == "scalar"].set_index("embed").reindex(x)
    axB.plot(x, [gru_lik[e] for e in x], marker="D", color="#1a1a1a", label="GRU (none)")
    axB.plot(x, dn.rel_LL.values, marker="o", color="#762a83", mfc="white", mec="#762a83", mew=1.6, label="disRNN (none)")
    axB.plot(x, ds.rel_LL.values, marker="s", color="#762a83", label="disRNN (scalar)")
    axB.set_xticks([4, 8, 16]); axB.set_xlim(3.4, 17)
    axB.set_xlabel("subject embedding size"); axB.set_ylabel("relative likelihood")
    axB.set_title("disRNN costs ~4\u20136 pts likelihood\nfor its sparse latent", fontsize=12, loc="left")
    axB.legend(frameon=False, fontsize=11, loc="center right")

    # Panel C: embedding-space PCA of the scalar-D16 disRNN subject embedding,
    # colored by true model family (the interpretable clusters). Reads the
    # committed emb CSV; the true family per subject is joined from the grid.
    axC = axes[2]
    FAMCOL = {"QLearning": "#2166ac", "CompareToThreshold": "#b2182b", "LossCounting": "#1b7837"}
    embp = HERE / "emb_disrnn_scalar_D16.csv"
    if embp.exists():
        emb = pd.read_csv(embp)
        ecols = [c for c in emb.columns if c.startswith("emb_")]
        X = emb[ecols].to_numpy(float); X = X - X.mean(0)
        # PCA via numpy SVD (sklearn absent in default env)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        pcs = X @ Vt[:2].T
        for f, col in FAMCOL.items():
            m = emb.dominant_family.values == f
            axC.scatter(pcs[m, 0], pcs[m, 1], s=26, c=col, edgecolor="white", lw=0.4,
                        label=f.replace("CompareToThreshold", "CompareThresh"), alpha=0.9)
        axC.set_xlabel("PC1 \u2192"); axC.set_ylabel("PC2 \u2192")
        axC.set_xticks([]); axC.set_yticks([])
        axC.legend(frameon=False, fontsize=10, loc="lower left")
        axC.set_title("disRNN embedding separates\nthe three families (scalar D16)", fontsize=12, loc="left")
    else:
        axC.axis("off")
        axC.text(0.5, 0.5, "emb CSV not staged\n(panel skipped)", ha="center", va="center",
                 fontsize=10, color="#999", transform=axC.transAxes)
    fig.tight_layout(); fig.savefig(HERE / "fig_disrnn_stage4a.png"); plt.close(fig)




def _replace_block(text: str, n: int, body: str) -> str:
    pat = re.compile(rf"(<!-- BEGIN result-{n} -->).*?(<!-- END result-{n} -->)", re.DOTALL)
    return pat.sub(lambda m: m.group(1) + "\n" + body + "\n" + m.group(2), text)


def _update_reports(ladder, s4b, dis):
    # r1 — GRU ladder
    r1 = HERE / "reports" / "r1-gru-ladder.md"
    tbl = "| stage | generator | GRU rel-LL | baseline rel-LL | recovery | value |\n|---|---|---|---|---|---|\n"
    for _, r in ladder[ladder.stage != "4a-disRNN"].iterrows():
        tbl += f"| {r.stage} | {r.generator} | {r.model_rel_LL} | {r.baseline_rel_LL} | {r.recovery} | {r.recovery_value} |\n"
    body1 = ("[regenerated by `analysis/recovery_report.py` — do not edit by hand]\n\n"
             "![baseline-flip ladder](../fig_ladder.png)\n\n"
             "*Relative likelihood (model NL / ground-truth NL, ceiling 1.0) of the GRU vs the "
             "correct-model-class baseline across the ladder. N=200.*\n\n" + tbl +
             "\n- **The baseline flip is the study's spine.** A correctly-specified baseline matches the "
             "GRU on stationary (S1) and interpolable (S2) data, then breaks under extrapolation "
             "(S2b: 0.94 vs GRU >0.987) and under mixed structure (S3 model-selection 47%, S4a 70%), "
             "while the GRU embedding recovers the true structure at 97.5\u2013100%.\n"
             "- **Embedding dimension is the identifiability knob** \u2014 recovery scales with D, not "
             "hidden-unit count; higher-diversity mixtures (S3/S4) need D=16.\n"
             "- **Stage-4b** (per-session family switching): recovery lives at the SUBJECT level "
             "(mixture-weight R\u00b2 0.55 @D16), not the session level \u2014 session-conditioning adds "
             "nothing over subject identity for decoding a session's family (0.62 vs 0.63), because "
             "Dirichlet(0.5) subjects are concentrated (mean dominant weight 0.70).\n")
    # Per-stage recovery / parameter-recovery / embedding-space figures, embedded in
    # order. The S3/S4a "combined" figures already carry the embedding-space PCA
    # panels (colored by true type/family + key params); the S3 standalone
    # embedding-space and baseline-confusion figures remain in figures/ as subsets.
    stage_figs = [
        ('recovery_ground_truth_schematic.png', 'recovery_ground_truth_schematic.png',
         '***Setup.** Each subject occupies a parameter subregion; sessions drift within it. '
         'The model must recover the subject-embedding table (between-subject) and the '
         'session-conditioning MLP (within-subject drift).*'),
        ('stage1_recovery_vs_baseline.png', 'stage1_recovery_vs_baseline.png',
         '***Stage 1 — static.** Fit quality relative to ground truth, all near the ceiling '
         '(a); mean parameter-recovery R² vs #subjects (b); per-parameter recovery R² at '
         'n=200 (c), baseline_rl vs GRU embed-2 vs embed-4. Fit likelihood saturates '
         'regardless of embedding size, but recovery separates them: embed-4 matches the '
         'correct-model baseline while embed-2 is under-capacity — embedding size, not '
         'network width, is the identifiability knob.*'),
        ('stage1_recovery_rho.png', 'stage1_recovery_rho.png',
         '***Stage 1 — rank-correlation companion to panel (c) above.** Same three bars '
         '(baseline_rl / GRU embed-2 / GRU embed-4) at n=200, reporting Spearman ρ instead of '
         'R². embed-4 and baseline_rl both sit at ρ≥0.98 for all three parameters, matching '
         "their near-ceiling R². embed-2's rank correlation (ρ=0.48–0.85) is consistently "
         'higher than its R² (0.20–0.69) — e.g. biasL goes from R²=0.20 to ρ=0.48 — so '
         "embed-2's under-capacity shows up more as an absolute-scale problem than a complete "
         'loss of subject ordering, though the ordering itself is also degraded relative to '
         "embed-4's ρ≥0.98.*"),
        ('stage1 embedding-size sweep', 'stage1_recovery_preliminary.png',
         '***Stage 1 — embedding-size sweep.** Recovery R² vs cohort size for embedding size '
         '4 (solid) vs 2 (dashed) at hidden_size=16 (a); recovered-vs-true scatter for the '
         '200-subject / embed-4 cell, one square panel per parameter with the y=x identity '
         'line (b). Size-4 embeddings recover all three parameters near ceiling and are '
         'insensitive to network width; size-2 embeddings sit below the identifiability '
         'threshold. Single seed (42) per cell — no error bars. Produced offline by '
         '`analysis/stage1_recovery_figure.py` from committed CSVs.*'),
        ('stage2_recovery_vs_baseline.png', 'stage2_recovery_vs_baseline.png',
         '***Stage 2 — mild drift.** Fit quality relative to ground truth (a, all near '
         "ceiling), and PER-SESSION parameter recovery R² — how well each model's per-session "
         'prediction tracks the true drifting parameter — as the mean over parameters vs '
         '#subjects (b) and per-parameter at n=200 (c), for baseline_rl, GRU session-blind, '
         'and GRU session-conditioned (markers: baseline = square, GRU = circle; color: light '
         'blue = 4-d subject-only, dark blue = 4-d + session conditioning, black = baseline). '
         'Fixed per-subject estimates (baseline, session-blind) are broadcast to every '
         'session; only the session-conditioned GRU predicts a per-session value, and it '
         'recovers best (0.84–0.91) while the two fixed models trail together (0.76–0.85) — '
         'all three are moderate-to-high because the per-session parameter is dominated by '
         'the subject centroid, so session conditioning adds the drift-tracking edge. '
         '(Recovery of the drift POSITION itself, where session-blind is 0 by construction, '
         'is a separate story — see stage2_session_trajectory.png.) baseline_rl '
         'softmax-temperature uses a ROBUST R² (fitted inverse-temperature winsorized at 20, '
         'true ceiling ~18.6; Spearman 0.89–0.93). Single seed (42) per cell — no error bars.*'),
        ('stage2_recovery_rho.png', 'stage2_recovery_rho.png',
         '***Stage 2 — rank-correlation companion to panel (c) above.** Same per-session, '
         'per-parameter grouping at n=200, Spearman ρ instead of R². All 9 cells sit in a '
         "tighter 0.92–0.99 band than the R² panel's 0.72–0.98 — softmax_temp in particular "
         "goes from baseline_rl's disclosed-robust R²=0.72 to ρ=0.92, confirming the "
         'winsorization was correcting a scale problem, not a ranking one.*'),
        ('stage2_session_trajectory.png', 'stage2_session_trajectory.png',
         '***Stage 2 — session trajectory.** Per-session parameter recovery R² at Stage 2 '
         '(n=200) for baseline_rl / GRU session-blind / GRU session-conditioned (a) — reads '
         "the same source values as the combined figure's panel c, so the bars agree exactly; "
         'session-position recovery, session-conditioned vs subject-only (b) — subject-only '
         'is 0 by construction, session conditioning recovers it at 0.94; embedding-space '
         'drift paths for 8 example subjects, colored by session position (c) — reconstructed '
         "via the training code's own `compute_session_conditioned_context_dataframe`, frozen "
         'to CSV once. Color: black = baseline, light blue = session-blind/subject-only, dark '
         'blue = session-conditioned; in (c), viridis = session position. Offline from '
         "committed CSVs.*\n\n*Every stage's likelihood is a held-out-session likelihood "
         '(`eval_every_n=2`)\n— Stages 1–2 just use an `interleaved` split (every 2nd session, '
         'scattered),\nwhich a static per-subject fit can interpolate through, so it '
         'looks\nuninformative. Stage 2b switches to a `tail` split (last 20% of '
         "each\nsubject's sessions, contiguous, `heldout_session_mode=tail`) that "
         'forces\nextrapolation — that switch, not held-out evaluation appearing for the '
         'first\ntime, is what makes the likelihood axis discriminating below.*'),
        ('stage2b_likelihood_flip.png', 'stage2b_likelihood_flip.png',
         '***Stage 2b — the baseline flip.** Fit quality at N=200 by stage: static Q-learning '
         'collapses (0.939) under S2b held-out-tail extrapolation while both GRUs stay >0.987 '
         '(a); the S2b relative-LL gaps show where model separation lives (b). House palette: '
         'black = baseline, light blue = GRU session-blind, dark blue = GRU '
         'session-conditioned. Offline from committed CSV.*'),
        ('stage2b_session_trajectory.png', 'stage2b_session_trajectory.png',
         '***Stage 2b — session trajectory.** Per-session parameter recovery R² at Stage 2b '
         '(n=200) for baseline_rl / GRU session-blind / GRU session-conditioned (a) — the '
         "same 3-model, 3-bar format as the Stage-2 combined figure's panel c, for direct "
         'side-by-side comparison; session-position recovery, Stage 2 vs Stage 2b, '
         'subject-only vs session-conditioned (b) — subject-only is 0 by construction, and '
         'the session delta recovers position at 0.94 (S2) → 0.47 (S2b) as the drift turns '
         'non-monotonic (baseline_rl has no per-session position estimate, so it is not shown '
         'in b); embedding-space drift paths for 8 example subjects, colored by session '
         "position (c) — reconstructed via the training code's own "
         '`compute_session_conditioned_context_dataframe` (the same function (b) uses), '
         'frozen to CSV once. Color: black = baseline, light blue = '
         'session-blind/subject-only, dark blue = session-conditioned; in (b), stage is '
         'distinguished by saturation; in (c), viridis = session position. Offline from '
         'committed CSVs.*'),
        ('stage2b_recovery_rho.png', 'stage2b_recovery_rho.png',
         '***Stage 2b — rank-correlation companion to panel (a) above.** Same 3-model, '
         '3-parameter bars at n=200, Spearman ρ instead of R². learn_rate and softmax_temp '
         "both sit above ρ=0.75 for every model despite panel (a)'s R² dropping as low as "
         '0.32 (baseline_rl softmax_temp) — the non-monotonic S2b drift degrades '
         'absolute-scale tracking more than it degrades rank ordering.*'),
        ('stage3_recovery_combined.png', 'stage3_recovery_combined.png',
         '***Stage 3 — QL-variant mixture.** Embedding-space PCA colored by true type (a), '
         'biasL (b), learn_rate (c); type decoded at 97.5–99.5% (d), confusion (e), '
         'within-family parameter recovery (f). Model TYPE → cluster; PARAMETERS → position '
         "within — but (b)/(c)'s within-cluster gradient strength is a 2D-PCA-projection "
         "artifact, not a measure of recovery quality: each type's true biasL-encoding "
         'direction in the full 4-d embedding aligns with the global top-2 PCA axes by chance '
         "(checked directly — RescorlaWagner's biasL direction is 82% aligned with local PC1, "
         'vs 7–11% for Bari/Hattori), so it looks clean for RescorlaWagner and scattered for '
         "Bari/Hattori despite (f)'s R² being uniformly high (0.89–0.96) for all three. (f) "
         'is the authoritative recovery number; (b)/(c) only show what any single 2D view '
         'happens to catch.*'),
        ('stage3_baseline_vs_gru_confusion.png', 'stage3_baseline_vs_gru_confusion.png',
         '***Stage 3 — model-identity confusion, baseline vs GRU.** GRU embedding decoding '
         '(a, 98.5% correct — same confusion data as stage3_recovery_combined.png panel e, '
         'isolated here for direct comparison) vs fixed-baseline model selection (b, 62.5% '
         'correct, n=200, MATCHED comparison): fit exactly the 3 true generative fitters '
         '(Bari/Hattori/RescorlaWagner) per subject, assign to whichever gives the best '
         'held-out likelihood — CompareToThreshold, which has no matching true preset in this '
         'stage, is dropped entirely rather than left in as a 4th off-target competitor. '
         'Accuracy rises from 51.0% (4-way, CTT still competing) to 62.5% once CTT stops '
         'siphoning off Bari/Hattori subjects as a spurious best fit — a large jump driven '
         "mostly by cleaner Bari/Hattori self-assignment (55/67, 62/67). RescorlaWagner's own "
         'fitter (baseline-rw-stage3 sweep, wandb run `qy9lof3x`) still only recovers 8/66 '
         'true-RW subjects even with no off-target competitor in the mix (36/66 misassigned '
         "to Bari, 22/66 to Hattori) — a genuine weak-identifiability property of RW's fit "
         'under this criterion, not an artifact of CTT crowding the earlier comparison. GRU '
         'embedding decoding still wins by a wide margin (98.5%). Two other selectable '
         'comparisons via `--baselines`: `historical` (Bari/Hattori/CTT, no RW fitter at all, '
         '47.0%) and `plus_rw` (adds RW alongside CTT, 4-way, 51.0%) — both retained for '
         'reference (`s3_baseline_modelselection.csv`, '
         '`s3_baseline_modelselection_4way.csv`). Real per-subject data throughout, replacing '
         'an earlier version of this figure whose baseline panel had been synthesized to '
         'match a remembered accuracy number rather than read from real per-subject fits.*'),
        ('stage3_recovery_vs_baseline.png', 'stage3_recovery_vs_baseline.png',
         '***Stage 3 — fit quality and per-session parameter recovery, baseline vs GRU.** '
         'Fills the same gap stages 2/2b already close: (a) relative held-out likelihood — '
         "the 6 GRU cells (0.978–0.990) vs baseline_rl's realistic best-of-4 model-selection "
         "likelihood (0.963, now including RescorlaWagner's own fitter) and its ceiling if "
         'the true model type were known (matched, now n=200 across all 3 families, 0.920). '
         'The matched ceiling DROPPED from 0.958 (n=134, Bari/Hattori only, RW excluded for '
         "lack of a fitter) to 0.920 once RW's own fitter is included — RW is "
         'fit-quality-poor even against its own true generative subjects (matched '
         "eval_likelihood 0.67 vs Bari/Hattori's 0.75–0.78), a weak-identifiability finding "
         'distinct from the earlier "RW has no fitter at all" gap; GRU still leads by a wide '
         'margin either way. (b,c) per-session parameter recovery (baseline_rl / GRU '
         'session-blind / GRU session-conditioned broadcasting a fixed per-subject estimate '
         'to every session, except session-conditioned which predicts a genuinely per-session '
         "value), mean over each family's own parameter set (b) and per-parameter (c). "
         "RescorlaWagner NOW has baseline_rl bars (RW's own fitter, wandb run `qy9lof3x`) and "
         'they are markedly negative (biasL/learn_rate/epsilon all R²<−1; Spearman rank '
         'correlation 0.26–0.88 remains positive, so subject ordering is partly preserved '
         "even though point estimates are off scale) — consistent with panel a's "
         'weak-identifiability finding, and a much worse per-session predictor than '
         "Bari/Hattori's fixed fits. Several baseline parameters (forget_rate_unchosen, "
         'learn_rate, learn_rate_rew, choice_kernel_relative_weight) show weak '
         'identifiability even after winsorizing near-degenerate MLE fits at the true '
         "parameter ceiling — consistent with this family's own previously-reported "
         'within-subject session-mean R² (forget_rate_unchosen was already negative there). '
         'GRU recovers every parameter session-conditioned > session-blind > baseline. Single '
         'seed (42) per cell — no error bars.*'),
        ('stage3_recovery_rho.png', 'stage3_recovery_rho.png',
         '***Stage 3 — rank-correlation companion to panel (c) above.** Same 12 '
         'family:parameter rows, Spearman ρ instead of R². All rows are positive '
         "(ρ=0.22–0.94) except Bari2019's forget_rate_unchosen under the two GRU conditions "
         "(ρ=−0.04 session-blind, −0.10 session-conditioned; baseline_rl's own fit stays "
         'weakly positive at ρ=0.22) — including every parameter that shows R²<−1 in the R² '
         'panel, the estimators are otherwise consistently rank-faithful even where they are '
         'absolute-scale-poor. forget_rate_unchosen under GRU is the one case where the '
         'parameter is not just off-scale but genuinely unrecovered.*'),
        ('stage3_baseline_initial_param_scatter.png', 'stage3_baseline_initial_param_scatter.png',
         '***Stage 3 — baseline_rl: true initial (session-0) parameter vs fitted static '
         'value.** Isolates whether the negative per-session recovery R² above comes from '
         'drift-tracking failure (baseline_rl broadcasts one static estimate across all 40 '
         'sessions, so it structurally cannot track drift) or from a genuine point-estimate '
         'fit-quality problem that already exists at session 0, before any drift has '
         'accumulated. Most weakly-recovered parameters (choice-kernel weight, '
         "forget-rate-unchosen, RescorlaWagner's own three parameters) already show "
         'ceiling-hugging degenerate MLE fits at session 0 — red triangles mark fits '
         'winsorized at the true parameter ceiling — so this is at least partly a static '
         'fit-quality problem, not purely a drift-confusion artifact. biasL is the clean '
         'exception (R²=0.74 for both Bari and Hattori). Several parameters keep a strong '
         'Spearman rank correlation even when R² is negative (e.g. RescorlaWagner biasL '
         'ρ=0.93 vs R²=-0.19, winsorized R²=0.33), pointing to a scale/calibration '
         'miscalibration rather than pure noise for those cases.*'),
        ('stage4a_baseline_initial_param_scatter.png', 'stage4a_baseline_initial_param_scatter.png',
         '***Stage 4a — baseline_rl: true initial (session-0) parameter vs fitted static '
         'value.** Direct analog of the Stage-3 baseline scatter, restricted to the diagonal '
         "(each family's fixed fitter scored only against subjects whose true generative "
         "family matches it). Unlike Stage 3's missing-RW-fitter gap, every Stage-4a family "
         "has a correctly-specified baseline — CompareToThreshold's is decent across all 4 "
         'parameters after winsorizing degenerate MLE excursions (R²=0.42–0.83, with 8/67 '
         "subjects' learn_rate fits clipped at the true ceiling) — comparable in relative "
         "frequency but far less severe than QLearning's, which shows the same severe "
         'ceiling-hugging degeneracy Stage 3 found for Bari2019 '
         '(choice_kernel_relative_weight, forget_rate_unchosen, softmax_inverse_temperature '
         'all pinned at the winsor ceiling for a substantial fraction of subjects). '
         "LossCounting's own fit is not uniformly clean either: `loss_count_threshold_std` "
         'shows a severe raw R²=-2.31 (winsorized -0.31, 31/66 subjects clipped) — the worst '
         "negative R² of any Stage-4a baseline parameter, comparable to QLearning's worst "
         'cases despite LossCounting otherwise looking well-behaved on biasL and '
         'loss_count_threshold_mean.'),
        ('stage4a_gru_initial_param_scatter.png', 'stage4a_gru_initial_param_scatter.png',
         '***Stage 4a — GRU session-blind: true initial (session-0) parameter vs '
         "within-family CV prediction.** Companion figure using GRU's held-out (5-fold "
         "GroupKFold over subjects) linear-readout prediction instead of baseline_rl's "
         "per-subject MLE. CompareToThreshold's `learn_rate` and "
         '`softmax_inverse_temperature` panels are the one place in this comparison where the '
         'correctly-specified baseline_rl fitter beats the GRU embedding (baseline '
         'R²=0.42/0.72 vs GRU R²=-2.20/-0.13). The two negative R²s are not the same failure, '
         "though: `softmax_inverse_temperature`'s negative R² is almost entirely a single "
         'leverage point — one subject with the largest embedding-norm outlier across all 200 '
         'subjects (~4.5× the population median) that single-handedly wrecks a 5-fold linear '
         'fit; excluding that one subject alone recovers R²=0.62, essentially matching '
         "baseline_rl. `learn_rate`'s negative R² is not primarily an outlier artifact — even "
         'excluding the same subject it stays clearly negative (R²=-0.58) because the '
         '*remaining* 66 predictions collapse into a narrow band (SD 0.11 vs true SD 0.25), a '
         'genuine variance-shrinkage failure of the cross-validated linear readout on this '
         'specific family/parameter, not a single bad point.'),
        ('stage3_gru_initial_param_scatter.png', 'stage3_gru_initial_param_scatter.png',
         '***Stage 3 — GRU session-blind: true initial (session-0) parameter vs within-family '
         'CV prediction.** Same session-0 true value on the x-axis as the baseline scatter '
         "above, but the y-axis is GRU session-blind's held-out prediction (5-fold GroupKFold "
         'cross-validation over subjects — every prediction comes from a fold that excluded '
         "that subject) instead of baseline_rl's per-subject independent MLE. A different "
         "failure mode from baseline_rl's ceiling-hugging degenerate fits: GRU predictions "
         'occasionally overshoot the true plausible range too, but only mildly — at most ~48% '
         "of the parameter's own range (Bari's choice-kernel weight) — versus baseline_rl's "
         "degenerate MLE excursions, which reach 90–400% of range (e.g. Bari's "
         'softmax_inverse_temperature fitted to 100 against a true ceiling of 15). No '
         'winsorization is applied to the GRU points because none of its overshoots are of '
         'that same catastrophic-outlier character, but two parameters (Bari and '
         "RescorlaWagner's learn_rate) show a genuine systematic shrinkage bias instead "
         '(predictions run ~0.25 too high on average, with compressed variance) rather than '
         "outlier-driven negative R². Bari's forget_rate_unchosen is essentially unrecovered "
         'by either estimator (ρ≈0, R²<0 for both) — a parameter neither approach can pull '
         "out of this task's likelihood surface."),
        ('stage3_trajectory_recovery.png', 'stage3_trajectory_recovery.png',
         '***Stage 3 — true parameter drift vs baseline/GRU recovery, one representative '
         'subject per family.** Qualitative complement to the aggregate R² numbers above: for '
         'one well-fit representative subject per true family, plots the true drifting '
         'parameter across all 40 sessions against baseline_rl (flat static line — '
         'necessarily flat, since it fits one point estimate per subject), GRU session-blind '
         '(flat static line — same limitation, despite its own embedding), and GRU '
         'session-conditioned (a genuine per-session curve, reconstructed the same way as the '
         "stage2/stage2b trajectory figures). Only session-conditioned tracks the parameter's "
         "actual shape — most visible for Bari/Hattori's non-monotonic sinusoidal "
         "inverse-temperature drift and RescorlaWagner's linear learn_rate ramp — while both "
         'flat baselines sit pinned at a single value throughout, including into the held-out '
         'tail (grey band).*\n\n**Why is GRU session-blind (also a single static value '
         'broadcast per subject) so much better than baseline_rl at the SAME task?** Both '
         'estimators produce one fixed per-subject value and broadcast it to every session — '
         "neither tracks drift — so the gap here isn't a drift-tracking story (that's the "
         "session-blind vs session-conditioned gap, discussed above). It's an estimation "
         "story: baseline_rl fits each subject's ~32 training sessions independently via "
         'differential evolution, with no information shared across subjects — so on '
         'weakly-identifiable parameters (forget_rate_unchosen, learn_rate_rew, '
         'softmax_inverse_temperature, choice_kernel_relative_weight) the per-subject '
         'optimizer can land on degenerate boundary values (fitted '
         'softmax_inverse_temperature up to ~100 vs a true ceiling of 15; '
         'forget_rate_unchosen at 1.0 vs a true ceiling of 0.4) — visible directly in '
         '`stage3_baseline_initial_param_scatter.png` above, where several parameters are '
         "already ceiling-hugging at session 0, before any drift accumulates. GRU's subject "
         'embedding, by contrast, is decoded by a readout trained JOINTLY across all 200 '
         "subjects, which acts as implicit regularization/shrinkage toward what's typical and "
         "well-constrained across the cohort — it can't run off to the same individual-fit "
         "extremes. biasL is the control case: it's strongly and cleanly identifiable from a "
         "single subject's choices alone, so baseline_rl and GRU session-blind are near-tied "
         'there (Bari 0.75 vs 0.72; Hattori 0.76 vs 0.78) — the gap opens specifically on the '
         'parameters where cross-subject pooling has the most to offer and independent '
         'per-subject MLE is most exposed. `stage3_trajectory_recovery.png` above makes the '
         'session-blind vs session-conditioned half of this story visible directly: both '
         'baseline_rl and GRU session-blind are flat lines by construction, and only '
         "session-conditioning tracks the drift's actual shape session-by-session."),
        ('stage4a_recovery_combined.png', 'stage4a_recovery_combined.png',
         '***Stage 4a — family mixture.** Embedding-space PCA separating the three families '
         '(a,b); GRU embedding decodes family at 100% (c) vs 70% fixed-baseline model '
         'selection (d).*'),
        ('stage4a_persession_recovery.png', 'stage4a_persession_recovery.png',
         '***Stage 4a — per-session parameter recovery, baseline vs GRU.** (a) mean '
         "per-session recovery R2 over each family's drifting params: baseline_rl fails for "
         'QLearning (-0.60, correctly-specified generative model notwithstanding) and is '
         'flat/negative for CompareToThreshold (-0.03), while GRU session-conditioning helps '
         'every family (0.58, 0.83, 0.83). (b) per-parameter breakdown, including '
         "CompareToThreshold's static threshold (no drift block — shown as a subject-level "
         "recovery check, not drift tracking; excluded from panel a's mean). "
         "CompareToThreshold's softmax_inverse_temperature and learn_rate are weakly "
         'identified already at the session-blind level (R2 -0.04, -1.16) and '
         'session-conditioning makes them WORSE, not merely unhelpful (biasL 0.72→0.25, '
         'softmax -0.04→-1.85) — a bias/inverse-temperature confound was checked directly in '
         "the baseline_rl fits and ruled out (corr=0.13); 60.5% of CTT's 200 fitted threshold "
         "values also land outside the true range [0.2, 0.6], despite threshold's own "
         'recovery R2 being good (0.79–0.85) — R2 tracks preserved relative scale across '
         'subjects, which survives even biased point estimates. Likely explanation: the '
         "CompareToThreshold agent's lack of a choice-kernel term leaves its likelihood "
         'surface less constrained for the DE optimizer than QLearning/LossCounting have, not '
         'something specific to session-conditioning. biasL recovery drops under conditioning '
         'for BOTH QLearning (0.67→0.55) and CompareToThreshold (0.72→0.25) — a modest drop '
         "is not unique to CTT, but the magnitude is ~4x larger, and CTT's softmax drop is "
         'far more severe still (delta -1.81, the largest conditioning-induced degradation in '
         'the figure). learn_rate and threshold both improve with conditioning in every '
         'family that has them. Net: conditioning helps on average but a real minority of '
         "cells (both biasL columns, CTT's already-weak softmax) get worse, most severely "
         'where the underlying signal was already marginal.*'),
        ('stage4a_persession_rho.png', 'stage4a_persession_rho.png',
         '***Stage 4a — rank-correlation companion to panels (a,b) above.** Same '
         'family/parameter grouping, Spearman ρ instead of R². Every one of the 27 (family, '
         'condition, parameter) cells has positive ρ (0.48–0.94) — including '
         "CompareToThreshold's session-conditioned softmax_inverse_temperature, whose "
         'R²=−1.85 is the largest degradation in the R² panel but whose ρ=0.56 stays clearly '
         "rank-informative. QLearning's baseline_rl learn_rate shows the same pattern "
         "(R²=−1.76, ρ=0.49): both baseline_rl's and GRU's worst R² cells are scale failures "
         'on an estimator that still orders subjects better than chance, not estimators that '
         'have lost the signal entirely.*'),
        ('stage4b_recovery.png', 'stage4b_recovery.png',
         '***Stage 4b — per-session family switching.** Mixture-weight recovery vs embedding '
         'size (a); subject-vs-session dissociation null (b); per-session family confusion '
         '(c).*'),
    ]
    gallery = "\n### Per-stage figures\n\n"
    for alt, fn, text in stage_figs:
        gallery += f"![{alt}](../figures/{fn})\n\n{text}\n\n"
    body1 = body1 + gallery.rstrip("\n")
    r1.write_text(_replace_block(r1.read_text(), 1, body1))
    # r2 — disRNN replication
    r2 = HERE / "reports" / "r2-disrnn-replication.md"
    dtbl = "| enc | D | rel-LL | family-decoding acc |\n|---|---|---|---|\n"
    for _, r in dis.sort_values(["enc", "embed"]).iterrows():
        dtbl += f"| {r.enc} | {r.embed} | {r.rel_LL:.4f} | {r.family_acc_subject:.3f} |\n"
    body2 = ("[regenerated by `analysis/recovery_report.py` — do not edit by hand]\n\n"
             "![disRNN Stage-4a replication](../fig_disrnn_stage4a.png)\n\n"
             "*disRNN family decoding (left) and likelihood cost (right) vs the GRU (=1.00 family, "
             "~0.99 rel-LL) and the correct-baseline ceiling (0.70). N=200, Stage-4a generator.*\n\n" + dtbl +
             "\n- **Interpretability is nearly free for recovery.** The information-bottlenecked disRNN "
             "decodes model family at 0.95\u20130.98 (with session conditioning), nearly matching the GRU's "
             "1.00 and far above the 0.70 baseline ceiling.\n"
             "- **The bottleneck costs ~4\u20136 pts of likelihood** (rel-LL ~0.93\u20130.95 vs GRU ~0.99) \u2014 "
             "smallest for `none` D4 (4.2 pts), largest for `scalar` D4 (6.4 pts).\n"
             "- **Session conditioning matters more for the disRNN.** Without it (`none`), decoding is "
             "lower and more variable (0.75\u20130.90); the GRU's raw embedding already hit 1.00.")
    r2.write_text(_replace_block(r2.read_text(), 1, body2))


if __name__ == "__main__":
    main()
