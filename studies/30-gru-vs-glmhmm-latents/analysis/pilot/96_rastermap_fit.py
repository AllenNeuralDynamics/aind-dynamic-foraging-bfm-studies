"""Rastermap unit ordering of the GRU latents, fitted on the 40 training-fold mice only (runs in env-rastermap).

Two fits: raw standardised latents, and latents smoothed with a centred 20-trial moving average within each
session (slow structure). Saves out/rastermap_order.npz (isort and embedding for each). Units are the same
network in every mouse, so one ordering applies to every held-out session.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rastermap import Rastermap

OUT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe/out")
TEST_FOLD, SMOOTH = 2, 20
cols = [c for c in pq.ParquetFile(OUT / "joined_trials.parquet").schema.names if c.startswith("latent_")]
j = pd.read_parquet(OUT / "joined_trials.parquet", columns=["subject_id", "ses_idx"] + cols)
fom = json.loads((OUT / "probe_results.json").read_text())["fold_of_mouse"]
tr = j["subject_id"].astype(str).map(fom).to_numpy() != TEST_FOLD
sc = np.load(OUT / "kscan" / "weights" / f"hurdle_K2_fold{TEST_FOLD}.npz")
Z = ((j[cols].to_numpy() - sc["mean"]) / sc["scale"])[tr].astype(np.float32)
ses = j["ses_idx"].astype(str).to_numpy()[tr]
Zs = pd.DataFrame(Z).groupby(ses).transform(lambda v: v.rolling(SMOOTH, center=True, min_periods=1).mean()).to_numpy(np.float32)
res = {}
for name, data in [("raw", Z), ("smooth", Zs)]:
    for seed in (0, 1):
        model = Rastermap(n_clusters=40, n_PCs=64, locality=0.0, time_lag_window=0, grid_upsample=10,
                          mean_time=True, verbose=False, random_state=seed).fit(data.T)
        res[f"{name}_s{seed}_isort"] = model.isort
        res[f"{name}_s{seed}_embedding"] = model.embedding.ravel()
        print(name, seed, "done", flush=True)
for name in ("raw", "smooth"):
    a, b = res[f"{name}_s0_embedding"], res[f"{name}_s1_embedding"]
    from scipy.stats import spearmanr
    print(name, "seed stability (Spearman of embeddings):", round(float(abs(spearmanr(a, b)[0])), 3))
np.savez(OUT / "rastermap_order.npz", **res)
