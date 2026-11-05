# Screenshots of Shinka's portal

Screenshots of ShinkaEvolve's web portal (`shinka_visualize`), for r1 and the `seeded-islands-smoke`
notes. They are pictures of a tool, not figures drawn by this study's code, so they are PNG and sit
outside the paper-style figure rule; the numbers in the reports come from the frozen JSON, not from
these pictures.

| file prefix | run | W&B archive (artifact, digest) |
|---|---|---|
| `gate4-seeded_` | `gate4-recovery@20261104-132600` | `NeuralDynamics/dfbfm-mevolve/run-gate4-recovery-20261104-132600:v0`, `fc7aa68e6cbabd3e8130a94451d162c2` |
| `gate4-noseed_` | `gate4-noseed@20261104-141503` | `NeuralDynamics/dfbfm-mevolve/run-gate4-noseed-20261104-141503:v0`, `2f93969227496d022e93973980589299` |
| `smoke_` | `seeded-islands-smoke@20261105-011815` | `NeuralDynamics/dfbfm-mevolve/run-seeded-islands-smoke-20261105-011815:v0`, `19f17a3875d4ea7c7dfe5507bccc6335` |

Three views per run:

- `_tree`: the program tree. One node per program, an edge from parent to child; the colour is the
  score, the shape the kind of edit (diamond: initial program, circle: full rewrite, square: diff,
  cross: crossover), red a program that failed. The badge above a node names its island. The orange
  line runs from the root to the best program.
- `_score`: the score trace. Each proposal's validation score (red dots), the best so far (blue), the
  mean so far (green) and the best program's line of ancestors (orange); score axis from 72.0.
- `_path`: the best program's line of ancestors, oldest first, each with the hypothesis Claude wrote
  for its edit.

**How they were made** (2026-11-05, on a compute node): each archive was downloaded at its digest; the
portal from MEvolve's `mevolve-search` environment (ShinkaEvolve pinned at 8adc053a) served them through
`serve_portal.py` (read-only, 127.0.0.1 only); `capture_portal.py <db_path> <out dir> 72.0` took the
pictures in headless Chromium at twice the pixel density; the trees were cropped to their content and
every picture reduced to 256 colours.
