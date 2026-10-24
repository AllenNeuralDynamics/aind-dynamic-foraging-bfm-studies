# split-smoke (infrastructure, not science)

A one-cell live launch for dispatcher #175 / #124: the first real launch of a sweep that lives in
the studies repository, through the dispatcher's `launch_beaker_resumable.py`. It checks that the
W&B run carries `meta.studies_commit` (from `BFM_META_STUDIES_COMMIT`), that the launch record
carries the studies, dispatcher, wrapper and foraging-models SHAs, and that the run finishes.

Tiny `gru-stage1`-style cell: 10 synthetic subjects, H=16, embedding 2, 300 steps. Its numbers
mean nothing and feed no report. Results and the provenance check: `launch_record_*/results.md`.
