# Changelog — 13-hb-subject-vs-gru-heldout

## 2026-10-27

- k̂ rescue and refit (ADR 0004). Every subject's production draws were rescored
  (`hattori@20261027-181430`, `hattori@20261027-201810`; plain scores reproduce production exactly)
  and the 262 sessions failing Pareto k̂ were rescored by importance sampling; 44 still fail. The
  headline moves 0.721295 → 0.721297. The one allowed refit (centred, `hattori@20261027-181415`)
  was worse on every gate and is not used. r1 is `live`; r2–r4 read the final run table
  (`hattori_final_runs.json`), numbers unchanged beyond 1e-4 except r3's Spearman ρ (−0.1064 →
  −0.1078).

## 2026-10-26

- Figures redrawn in the paper style (studies/util/figure_style.py, #44): r2, r3 and r4 now write
  editable SVGs at print size (sized by content, 78–183 mm wide, 5–7 pt Arial) instead of PNGs; figure titles
  moved into captions in the result blocks. Numbers unchanged.

## 2026-10-25

- Study created.
