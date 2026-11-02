# Fit programs at the session level, with session facts and split diagnostics

Programs are fitted with a session level during the search: each session's parameters vary around
the subject's typical values, with no population level, and a held-out session is scored by the
session marginal. Programs also receive each session's training stage, task variant and index, and
every score and feedback line is split by stage group and task variant. Without a session level the
search could only win back the likelihood that sessions differ by writing parameter inference into
the program, folding training stage, task and day-to-day drift into "mechanisms" (decided with Han,
2026-10-31).

The evidence: within a subject, sessions differ. Of the 22,555 sessions of the 763 cohort subjects in
snapshot 20260603, 55% are at the final stage or graduated, 17% at a recorded earlier stage and 28%
have no stage recorded, and 83% of subjects ran more than one task variant (read from the session
table with `aind_dynamic_foraging_database.select_sessions`, before the loader's filters). Study 13
r4 showed that inferring each held-out session's parameters from its own earlier trials raised the
held-out likelihood for every one of the 149 subjects, and letting sessions sit away from the
subject's typical values raised it again for 95% of them.

## Considered options

- **One parameter set per subject**, with session facts and split diagnostics, re-scoring the elites
  at the session level afterwards. Cheaper, but the search itself stays biased. Kept as the fallback
  if the session-level scorer fails its parity gate.
- **Discovery on final-stage sessions of one task family.** The cleanest steady-state mechanism, but
  every comparator would need re-scoring on the subset, and early learning drops out. A later variant.
- **A population level.** Study 08's three-level fit did not mix and scored below the per-subject
  fits for most subjects.

## Consequences

- The headline compares against HB-CTT, the comparator fitted the same way.
- Each elite also gets the both-regime test, to flag mechanisms that only absorb drift between
  sessions. A flagged mechanism is reported and ablated in the interpretation report (removed, the
  program refitted at the session level); no program is excluded automatically, since the signal is
  not yet calibrated.
