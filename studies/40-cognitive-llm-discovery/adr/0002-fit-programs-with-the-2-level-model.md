# Fit programs with the 2-level model, with session facts and split diagnostics

Programs are fitted with the 2-level model during the search (2-level EM, MEvolve ADR 0004): each
session's parameters vary around the subject's typical values, with no population level, and a
held-out session is scored by the session marginal. Programs also receive each session's training stage, task variant and index, and
every score and feedback line is split by stage group and task variant. If sessions could not vary around the subject,
the search could only win back the likelihood that sessions differ by writing parameter inference into
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
  as 2-level EM afterwards. Cheaper, but the search itself stays biased. Kept as the fallback
  if the 2-level EM scorer fails its parity gate.
- **Discovery on final-stage sessions of one task family.** The cleanest steady-state mechanism, but
  every comparator would need re-scoring on the subset, and early learning drops out. A later variant.
- **A population level.** Study 08's three-level fit did not mix and scored below the per-subject
  fits for most subjects.

## Consequences

- The headline compares against HB-CTT, the comparator with the same 2-level model (2-level HB,
  sampled by NUTS rather than fitted by Laplace-EM).
- Each elite also gets the both-regime test, to flag mechanisms that only absorb drift between
  sessions. A flagged mechanism is reported and ablated in the interpretation report (removed, the
  program refitted as 2-level EM); no program is excluded automatically, since the signal is
  not yet calibrated.
