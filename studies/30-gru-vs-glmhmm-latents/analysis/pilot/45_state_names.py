"""Describe the global hurdle K=3 states on the pilot trials (smoothed-posterior-weighted)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
OUT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe/out")
j = pd.read_parquet(OUT / "joined_trials.parquet", columns=["choice", "reward", "pred_0", "pred_1", "pred_2",
                    "smoothed_0", "smoothed_1", "smoothed_2", "viterbi", "reward_probabilityL", "reward_probabilityR"])
c = j["choice"].to_numpy(); r = j["reward"].to_numpy()
resp = c != 2
better_right = j["reward_probabilityR"].to_numpy() > j["reward_probabilityL"].to_numpy()
desc = []
for k in range(3):
    w = j[f"smoothed_{k}"].to_numpy()
    wr = w * resp
    desc.append({"state": k, "occupancy_smoothed": float(w.mean()), "occupancy_pred": float(j[f"pred_{k}"].mean()),
                 "ignore_rate": float((w * (c == 2)).sum() / w.sum()),
                 "p_right_given_respond": float((wr * (c == 1)).sum() / wr.sum()),
                 "reward_rate_given_respond": float((wr * r).sum() / wr.sum()),
                 "p_choose_better_side": float((wr * ((c == 1) == better_right)).sum() / wr.sum())})
print(json.dumps(desc, indent=1))
(OUT / "state_description.json").write_text(json.dumps(desc, indent=1))
