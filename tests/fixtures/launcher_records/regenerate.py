"""Regenerate these fixtures with the dispatcher launchers' own save_record.

usage: python regenerate.py <dispatcher_root> [<hpc conda env prefix>]

Run it from an environment with the dispatcher's dependencies (requests, PyYAML), at a clean
dispatcher checkout, then update the dispatcher commit named in test_launcher_record_contract.py.

Beaker records pin snapshot 20260603 (manifest fetched over HTTPS, as the launchers do) and the
image the resumable route runs (a stub client returns its real Beaker id); the HPC record uses
synthetic data and the real dynamic-foraging-bfm-cpu env. Temp paths are normalized and ids fixed.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

DISPATCHER, OUT = Path(sys.argv[1]).resolve(), Path(__file__).resolve().parent
HPC_ENV = Path(sys.argv[2] if len(sys.argv) > 2 else
               "/allen/aind/scratch/han.hou/miniforge3/envs/dynamic-foraging-bfm-cpu")
sys.path.insert(0, str(DISPATCHER / "code"))
sys.path.insert(0, str(DISPATCHER / "tests"))
import yaml  # noqa: E402
import launch_beaker  # noqa: E402
import launch_beaker_resumable as resumable  # noqa: E402
import launch_hpc  # noqa: E402
import provenance  # noqa: E402
from launch_inputs import data_record, image_record_beaker, image_record_hpc  # noqa: E402
from test_provenance_helpers import EXPERIMENT, SWEEP, make_studies_repo  # noqa: E402

IMAGE, IMAGE_ID = "han-hou/disrnn-wrapper-main-20260712", "01KXCF2EASQ8NV463684PZJ0ZP"
EXP_ID, GROUP = "01KEXAMPLEEXPERIMENT0000000", "stan-arm@20261023-000000"
NOTE, FIXED_TMP = "baseline launch: does the example converge?", "/tmp/studies-fixture"
SHAS = {"aind-dynamic-foraging-bfm-wrapper": "a" * 40, "aind-dynamic-foraging-models": "b" * 40}


def stub_client(_workspace):
    return types.SimpleNamespace(image=types.SimpleNamespace(get=lambda name: types.SimpleNamespace(id=IMAGE_ID)))


def write(kind, sweep_text, save):
    with tempfile.TemporaryDirectory(prefix="fixture_") as t:
        _write(kind, sweep_text, save, Path(t))


def _write(kind, sweep_text, save, tmp):
    root, sweep, _ = make_studies_repo(tmp)
    if sweep_text != SWEEP:      # re-commit so the record sees a clean, tracked sweep
        sweep.write_text(sweep_text)
        subprocess.run(["git", "-C", str(root), "commit", "-qam", "snapshot sweep"], check=True)
    exp = tmp / "experiment.yaml"
    exp.write_text(EXPERIMENT.replace("{beaker: img}", "{beaker: %s}" % IMAGE))
    prov = provenance.launch_provenance(sweep, DISPATCHER, SHAS)
    data = data_record(yaml.safe_load(sweep_text), DISPATCHER / "code" / "config")
    out = tmp / "out"
    name = save(sweep, exp, prov, data, out)
    dest = OUT / kind
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for f in out.iterdir():
        text = f.read_text().replace(str(tmp), FIXED_TMP)
        (dest / f.name).write_text(text)
    print(kind, json.dumps({k: json.loads((dest / name).read_text()).get(k) for k in ("data", "image")}))


def resumable_save(sweep, exp, prov, data, out):
    spec = resumable.build_spec(str(sweep), str(exp), provenance=prov)
    prov = {**prov, "data": data, "image": image_record_beaker(spec, None, stub_client)}
    resumable.save_record(str(sweep), EXP_ID, 1, out, provenance=prov, runtime_shas=SHAS,
                          wandb_group=GROUP, label="arm", note=NOTE)
    return "beaker_resumable.json"


def native_save(sweep, exp, prov, data, out):
    spec = yaml.safe_load(exp.read_text())
    prov = {**prov, "data": data, "image": image_record_beaker(spec, None, stub_client)}
    launch_beaker.save_record(str(sweep), "AIND-disRNN/test/abc123", EXP_ID, out, provenance=prov,
                              runtime_shas=SHAS, wandb_group=GROUP, label="arm", note=NOTE, n_tasks=4)
    return "beaker_sweep.json"


def hpc_save(sweep, exp, prov, data, out):
    site_packages = sorted(HPC_ENV.glob("lib/python3.*/site-packages"))   # as launch_hpc globs it
    prov = {**prov, "data": data, "image": image_record_hpc(site_packages)}
    launch_hpc._save_record(out, sweep_file=sweep, sweep_path="AIND-disRNN/test/xyz789", group=GROUP,
                            array_jobid="12345", autostop_jobid="12346", n_array_tasks=2, agent_count=1,
                            mode="cpu", provenance=prov, dirty_override=None, label="arm", note=NOTE)
    return "hpc_sweep.json"


SNAPSHOT_SWEEP = SWEEP.replace("data=synthetic", "data=mice_snapshot")
write("beaker_resumable", SNAPSHOT_SWEEP, resumable_save)
write("beaker_native", SNAPSHOT_SWEEP, native_save)
write("hpc", SWEEP, hpc_save)
