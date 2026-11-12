"""Pick example sessions with many clear GLM-HMM state transitions (held-out fold only).

For K = 4 (hurdle, predictive posterior, states sorted by p(ignore)): label each trial with its argmax state
when max pi >= CONF, else unlabeled; collapse labeled trials into runs; keep runs of >= MIN_RUN trials;
a transition is a change between consecutive kept runs. Rank sessions by transitions, then distinct states;
one session per mouse; sessions of 300-800 trials. Writes out/selected_sessions.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe/out")
K, CONF, MIN_RUN, TEST_FOLD, N = 4, 0.8, 20, 2, 4
j = pd.read_parquet(OUT / "joined_trials.parquet", columns=["subject_id", "ses_idx", "choice"])
j["subject_id"], j["ses_idx"] = j["subject_id"].astype(str), j["ses_idx"].astype(str)
P = pd.read_parquet(OUT / "kscan" / f"glmhmm_hurdle_K{K}.parquet", columns=[f"pred_{k}" for k in range(K)]).to_numpy()
c = j["choice"].to_numpy()
P = P[:, np.argsort(-((P * (c == 2)[:, None]).sum(0) / P.sum(0)))]
fom = json.loads((OUT / "probe_results.json").read_text())["fold_of_mouse"]
lab = np.where(P.max(1) >= CONF, P.argmax(1), -1)
rows = []
for (m, s), idx in j.groupby(["subject_id", "ses_idx"], sort=False).indices.items():
    if fom[m] != TEST_FOLD or not 300 <= len(idx) <= 800:
        continue
    l = lab[idx]
    l = l[l >= 0]
    if len(l) == 0:
        continue
    starts = np.flatnonzero(np.r_[True, l[1:] != l[:-1]])
    lens = np.diff(np.r_[starts, len(l)])
    kept = l[starts][lens >= MIN_RUN]
    kept = kept[np.r_[True, kept[1:] != kept[:-1]]] if len(kept) else kept
    rows.append({"mouse": m, "session": s, "n_trials": int(len(idx)), "transitions": int(max(len(kept) - 1, 0)),
                 "states_visited": int(len(set(kept.tolist()))), "frac_confident": float((lab[idx] >= 0).mean())})
df = pd.DataFrame(rows).sort_values(["transitions", "states_visited", "frac_confident"], ascending=False)
pick = df.groupby("mouse", sort=False).head(1).head(N)
print(df.head(15).to_string(index=False))
print("\npicked:\n", pick.to_string(index=False))
(OUT / "selected_sessions.json").write_text(json.dumps({
    "criterion": {"K": K, "conf": CONF, "min_run": MIN_RUN, "fold": TEST_FOLD + 1, "per_mouse": 1},
    "sessions": pick["session"].tolist(),
    "tags": [f"{r.transitions} clear K = 4 transitions, {r.states_visited} states" for r in pick.itertuples()],
    "table_top15": df.head(15).to_dict("records")}, indent=1))
