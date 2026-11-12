"""Read-only probe: find the D614 GRU artifact through other query paths. No wandb.init."""
import traceback, wandb
print("wandb", wandb.__version__)
api = wandb.Api(timeout=60)
rid = "nxd-3way-20260707-075401-80a6d102"
name = f"AIND-disRNN/mice_ignore_scaling/gru-output-{rid}"
for label, fn in [
    ("collection", lambda: list(api.artifact_collection("training-output", name).artifacts())),
    ("versions", lambda: list(api.artifacts("training-output", name))),
    ("run.logged", lambda: list(api.run(f"AIND-disRNN/mice_ignore_scaling/{rid}").logged_artifacts())),
]:
    try:
        arts = fn(); print(label, "OK", [(a.name, a.size) for a in arts][:5])
        a = arts[0]; files = [f.name for f in a.files()]
        print(len(files), [f for f in files if "step_" not in f][:40])
        print(sorted({f.split('/')[1] for f in files if f.startswith('checkpoints/step_')}))
        break
    except Exception as e:
        print(label, "FAIL", repr(e)[:300])
