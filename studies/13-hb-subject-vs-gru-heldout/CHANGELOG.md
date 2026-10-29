# Changelog — 13-hb-subject-vs-gru-heldout

## 2026-10-28

- Variant `ctt` (HB-CompareToThreshold, foraging-models v0.26.0, wrapper `hb_model`): production
  `ctt@20261028-003745` and refit of the 42 gate-failing subjects `ctt@20261028-020002` (as in
  ADR 0004; divergences 1,189 → 54, 137 of 149 pass every gate). r5: HB-CTT 0.7225.
- r6: the HB-CTT ladder beside r4's (controls `freeze_plugin_ctt.py` → `ctt_plugin.json`).

- Non-centred refit (approved by Han after the centred one failed) of the 46 gate-failing subjects,
  `hattori@20261027-223626`: divergences 304 → 45, 36 of 46 now pass every gate (139 of 149 overall).
  The final set is those 46 plus the 103 rescored production fits; the headline is 0.721295
  (production 0.721295). r2–r4 read the new final set.

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
