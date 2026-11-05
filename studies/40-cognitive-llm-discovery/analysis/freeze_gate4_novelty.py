#!/usr/bin/env python
"""Freeze the post hoc novelty and repeat tables of gate 4's two searches (``make freeze-gate4-novelty``).

Reads the committed ``analysis/gate4/runs.json`` (islands, times, parents) and each run's archived
database for the code (checked against ``archives.json``), and writes ``analysis/gate4/novelty.json``:

* per proposal, the highest cosine similarity of its code's embedding to every program on its
  island when its novelty check would run, as ShinkaEvolve computes it (``compute_similarity``:
  every program on the island, the parent included). The check runs once the code is written and
  before it is scored (``checked_at``, Shinka's ``sampling_finished_at``); a program is on the
  island then if its row was written before (``timestamp``, set after scoring), and island
  membership follows the migrations. Also its similarity to its parent and its exact-repeat key;
* the embedding model and revision, and the MEvolve commit whose ``code_key`` defines a repeat.

The embedding is a free local model, Alibaba-NLP/gte-modernbert-base at a pinned revision: the
CLS vector of the first 10,000 characters of the code (ShinkaEvolve's cap), L2-normalised. It needs
torch and transformers (the model is in ``HF_HOME``), and MEvolve's ``search/shinka_eval.py`` on
``PYTHONPATH`` for ``code_key`` (it imports only the standard library at module level), and the
run directories on /allen; so it does not run in CI. r1's producer reads only the committed file.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

RUNS = HERE / "gate4" / "runs.json"
ARCHIVES = HERE / "gate4" / "archives.json"
OUT = HERE / "gate4" / "novelty.json"
MODEL, REVISION = "Alibaba-NLP/gte-modernbert-base", "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
MAX_CHARS = 10000


def island_at(p: dict, t: float) -> int:
    island = p["island_proposed"]
    for m in p["migrations"]:
        if m["timestamp"] < t:
            island = m["to"]
    return island


def main() -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    import mevolve.search.shinka_eval as shinka_eval

    mevolve_sha = subprocess.run(["git", "-C", str(Path(shinka_eval.__file__).parent), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, check=True).stdout.strip()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION, attn_implementation="sdpa", reference_compile=False,
                                      use_safetensors=True, torch_dtype=torch.float32).eval()
    assert model.config._commit_hash == REVISION, model.config._commit_hash

    def embed(texts):
        batch = tok([t[:MAX_CHARS] for t in texts], padding=True, truncation=True, max_length=8192, return_tensors="pt")
        with torch.inference_mode():
            out = model(**batch)
        return F.normalize(out.last_hidden_state[:, 0], p=2, dim=1).numpy()

    import hashlib

    from freeze_gate4 import RUNS as RUN_DIRS, rows as db_rows

    runs = json.loads(RUNS.read_text())["runs"]
    archives = json.loads(ARCHIVES.read_text())["archives"]
    arms, sources = {}, []
    for arm, run in runs.items():
        db = RUN_DIRS / run["variant"] / run["launch_id"] / "search" / "programs.sqlite"
        for suffix in ("", "-wal"):
            sha = hashlib.sha256(Path(f"{db}{suffix}").read_bytes()).hexdigest()
            assert sha == archives[arm]["sha256"][f"search/programs.sqlite{suffix}"], f"{db}{suffix} is not archived"
            sources.append({"type": "file", "id": f"{db}{suffix}", "sha256": sha})
        sources.append({"type": "url", "id": f"wandb-artifact://{archives[arm]['artifact']}"})
        code = {r["id"]: r["code"] for r in db_rows(db)[0]}
        P = sorted(run["programs"], key=lambda p: p["timestamp"])
        X = embed([code[p["id"]] for p in P])
        assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-4)
        index = {p["id"]: k for k, p in enumerate(P)}
        rows = []
        for k, p in enumerate(P):
            t = p["checked_at"] if p["checked_at"] is not None else p["timestamp"]     # generation 0: placed, not checked
            prior = [j for j, q in enumerate(P) if j != k and q["timestamp"] < t
                     and island_at(q, t) == p["island_proposed"]]
            parent = index.get(p["parent_id"]) if p["generation"] > 0 else None
            rows.append({"generation": p["generation"], "id": p["id"], "island": p["island_proposed"],
                         "n_compared": len(prior),
                         "max_similarity": float(max(X[k] @ X[j] for j in prior)) if prior else None,
                         "parent_similarity": float(X[k] @ X[parent]) if parent is not None else None,
                         "repeat_key": shinka_eval.code_key(code[p["id"]])})
        arms[arm] = rows
    write_artifact(OUT, {
        "_meta": build_meta("analysis/freeze_gate4_novelty.py", [], study_root=STUDY, kind="freeze", code=[__file__, HERE / "freeze_gate4.py"],
                            inputs=[RUNS, ARCHIVES],
                            sources=[*sources, {"type": "url", "id": f"https://huggingface.co/{MODEL}/tree/{REVISION}"},
                                     {"type": "url", "id": "https://github.com/AllenNeuralDynamics/"
                                      f"aind-dynamic-foraging-mevolve/blob/{mevolve_sha}/src/mevolve/search/shinka_eval.py"}]),
        "about": "Post hoc: each program's highest code-embedding similarity to the programs on its island just "
                 "before it (ShinkaEvolve's novelty check), its similarity to its parent, and MEvolve's repeat key.",
        "embedding": {"model": MODEL, "revision": REVISION, "max_chars": MAX_CHARS, "vector": "CLS, L2-normalised"},
        "repeat_key": {"function": "mevolve.search.shinka_eval.code_key", "mevolve_git_sha": mevolve_sha},
        "arms": arms})
    print(f"wrote {sum(len(r) for r in arms.values())} rows to {OUT}")


if __name__ == "__main__":
    main()
