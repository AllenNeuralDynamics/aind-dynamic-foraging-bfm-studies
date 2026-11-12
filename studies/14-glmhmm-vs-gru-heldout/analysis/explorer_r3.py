"""Interactive explorer for r3 (``make r3-explorer``): every setting, every K, both emissions.

Reads the freeze's W&B archive (``archives_r3.json``: the artifact and the sha256 of its
``merged/`` files). Files already in ``_cache_r3/`` with the recorded sha256 are used as they
are; others are downloaded from the artifact and checked. Before drawing, it checks that the
archived weights reproduce the frozen axes and coordinates (``check_anchor``). Writes
``interactive/r3_state_spread.html`` and its data files ``interactive/r3_data/*.js`` (one per
emission, labelling and K, loaded on demand); both are git-ignored and rebuilt by this target.
Needs the W&B SDK and a key for ``AIND-disRNN`` unless the cache is already complete.
"""
import base64
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from r3_scores import COHORTS, MIN_TRIALS, named_features, pca_axes  # noqa: E402
from r3_state_spread import ANCHOR, EMISSIONS, FROZEN, LABELLINGS, STATE_COLORS, cut_rows  # noqa: E402

ARCHIVES = HERE / "archives_r3.json"
CACHE = HERE / "_cache_r3"
CELLS = {(e, lab): CACHE / f"frozen_r3_cells_{e}_{lab}.npz" for e in EMISSIONS for lab in LABELLINGS}
WEIGHTS = {(e, lab): CACHE / f"frozen_r3_weights_{e}_{lab}.npz" for e in EMISSIONS for lab in LABELLINGS}
REFITS = {e: CACHE / f"frozen_r3_refits_{e}.npz" for e in EMISSIONS}
INTERACTIVE = HERE / "interactive" / "r3_state_spread.html"
PLOTLY = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.34.0/plotly.min.js"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch():
    """Every archived file in ``_cache_r3/``, checked against its recorded sha256."""
    archive = json.loads(ARCHIVES.read_text())["archive"]
    expected = archive["merged"]["sha256"]
    if expected != json.loads(FROZEN.read_text())["data"]["archive_files"]:
        raise RuntimeError("archives_r3.json and frozen_r3.json record different files")
    CACHE.mkdir(exist_ok=True)
    missing = [n for n, s in expected.items() if not (CACHE / n).exists() or _sha256(CACHE / n) != s]
    if missing:
        import wandb

        art = wandb.Api().artifact(archive["artifact"])
        if art.digest != archive["digest"]:
            raise RuntimeError(f"{archive['artifact']} has digest {art.digest}, not {archive['digest']}")
        with tempfile.TemporaryDirectory(dir=CACHE) as tmp:
            for name in missing:
                path = Path(art.get_entry(f"merged/{name}").download(root=tmp))
                shutil.move(str(path), CACHE / name)
        bad = [n for n in missing if _sha256(CACHE / n) != expected[n]]
        if bad:
            raise RuntimeError(f"downloaded files do not match archives_r3.json: {bad}")
    return expected


def load():
    """The frozen runs, each with every setting's kept cells (state ranks, cohorts, coordinates).

    States are ranked by the anchor setting's trials (state 1 = most used) in every setting.
    Coordinates: PC1-3 of the anchor's training cells, then the six named weights.
    """
    frozen = json.loads(FROZEN.read_text())
    cohort_of = np.array([s["cohort"] for s in frozen["subjects"]])
    cells = {k: np.load(p) for k, p in CELLS.items()}
    refits = {e: np.load(p) for e, p in REFITS.items()}
    runs = []
    for run in frozen["runs"]:
        emission, K = run["emission"], run["num_states"]
        rank = np.asarray(run["state_rank"])
        settings = {}
        for key, entry in run["settings"].items():
            z, pre = cells[(emission, entry["labelling"])], f"K{K}_{key}_"
            n = z[pre + "n_trials"]
            keep = n >= MIN_TRIALS
            subject = z[pre + "subject"].astype(int)[keep]
            settings[key] = {**entry, "state": rank[z[pre + "state"].astype(int)[keep]],
                             "subject": subject, "cohort": cohort_of[subject], "n_trials": n[keep],
                             "coords": z[pre + "coords"].astype(np.float64)[keep]}
        mean, axes = np.asarray(run["pca_mean"]), np.asarray(run["pca_axes"])
        G = np.asarray(run["global_weights"])[np.argsort(rank)]          # ranked order
        R = refits[emission][f"K{K}_W"].astype(np.float64)                 # (mice, K, D), global order
        refit_subject = np.repeat(run["refit"]["subject"], K)
        runs.append({**run, "settings": settings, "mean": mean, "axes": axes,
                     "star_coords": coordinates(emission, run["feature_names"], G, mean, axes),
                     "refit_coords": coordinates(emission, run["feature_names"], R.reshape(-1, R.shape[2]),
                                                 mean, axes),
                     "refit_state": np.tile(rank, len(R)), "refit_subject": refit_subject})
    return frozen, runs


def coordinates(emission, names, W, mean, axes):
    """PC coordinates, then the six named weights, of weight vectors ``W``."""
    return np.concatenate([(W - mean) @ axes.T,
                           np.stack([f for _, f in named_features(emission, names, W)], 1)], 1)


def check_anchor(run, frozen):
    """The archived weights reproduce the uncut settings' frozen axes and coordinates.

    The freeze scored and projected the cells from these float16 weights; recomputing here
    guards against the cells file and the weights file drifting apart.
    """
    K, emission = run["num_states"], run["emission"]
    cohort_of = np.array([s["cohort"] for s in frozen["subjects"]])
    for labelling in LABELLINGS:
        key = f"{labelling}_p0_m1"
        z, pre = np.load(CELLS[(emission, labelling)]), f"K{K}_{key}_"
        W = np.load(WEIGHTS[(emission, labelling)])[f"K{K}"].astype(np.float64)
        if len(W) != len(z[pre + "n_trials"]):
            raise RuntimeError(f"{emission} K={K} {key}: weights and cells differ in length")
        coords = coordinates(emission, run["feature_names"], W, run["mean"], run["axes"])
        if not np.allclose(coords, z[pre + "coords"].astype(np.float64), rtol=2e-3, atol=2e-3 * np.abs(coords).max()):
            raise RuntimeError(f"{emission} K={K} {key}: coordinates do not follow from the weights")
        if labelling == "global":
            fit = (z[pre + "n_trials"] >= MIN_TRIALS) & (cohort_of[z[pre + "subject"].astype(int)] == "training")
            mean, axes, _ = pca_axes(W[fit], n=len(run["axes"]))
            if not (np.allclose(mean, run["mean"]) and np.allclose(axes, run["axes"])):
                raise RuntimeError(f"{emission} K={K}: PCA axes do not follow from the anchor weights")


def pack(values, dtype):
    """Base64 of a numeric array as ``dtype`` (little-endian)."""
    return base64.b64encode(np.ascontiguousarray(values, dtype=dtype).tobytes()).decode()


def pack_cols(X):
    """Columns of ``X`` quantised to uint16 between their min and max: (base64, lo, step) each."""
    out = []
    for col in X.T:
        lo, hi = (float(col.min()), float(col.max())) if len(col) else (0.0, 0.0)
        step = (hi - lo) / 65535 or 1.0
        out.append([pack(np.round((col - lo) / step), "<u2"), round(lo, 6), step])
    return out


def pack_cells(state, subject, n, coords):
    return {"count": int(len(state)), "state": pack(state, "u1"), "subject": pack(subject, "<u2"),
            "n": pack(n, "<u4"), "cols": pack_cols(coords)}


def interactive_files(runs):
    """One data file per (emission, labelling, K), plus the refit dots per emission.

    Each is a script that hands its data to the page (``R3.load``), so the page also works when
    opened as a local file. Coordinates are quantised to 1/65535 of each column's range.
    """
    files = {}
    for emission in EMISSIONS:
        by_k = {r["num_states"]: r for r in runs if r["emission"] == emission}
        for labelling in LABELLINGS:
            for K, run in by_k.items():
                files[f"{emission}_{labelling}_K{K}"] = {
                    key: pack_cells(s["state"], s["subject"], s["n_trials"], s["coords"])
                    for key, s in run["settings"].items() if s["labelling"] == labelling}
        files[f"{emission}_refits"] = {
            str(K): pack_cells(run["refit_state"], run["refit_subject"],
                               np.zeros(len(run["refit_state"]), int), run["refit_coords"])
            for K, run in by_k.items()}
    return files


def interactive_page(frozen, runs):
    """The page's own data: settings, stars, scores; the cells load from the data files."""
    data = {"subjects": [s["subject_id"] for s in frozen["subjects"]],
            "cohort": [int(s["cohort"] == "heldout") for s in frozen["subjects"]],
            "stateColors": STATE_COLORS, "minTrials": MIN_TRIALS, "anchor": ANCHOR,
            "cuts": frozen["data"]["cuts"], "emissions": {}}
    keep = ("n_cells", "n_trials_passing", "silhouette", "silhouette_null_95", "silhouette_pc12",
            "eta2_state", "eta2_subject", "eta2_state_null_mean", "kmeans_ari")
    for emission in EMISSIONS:
        by_k = {r["num_states"]: r for r in runs if r["emission"] == emission}
        names = by_k[1]["feature_names"]
        features = [label for label, _ in named_features(emission, names, np.zeros((1, len(names))))]
        data["emissions"][emission] = {
            "features": features,
            "K": {str(K): {"run": r["wandb_run_id"], "pcShare": [round(x, 4) for x in r["pca_variance_share"]],
                           "stars": np.round(r["star_coords"], 4).T.tolist(),
                           "round2": {k: r["round2"][k] for k in ("wandb_run_id", "prior_sigma_subject",
                                                                  "transition_strength_subject")}}
                  for K, r in by_k.items()},
            "scores": {key: {c: [{k: x.get(k) for k in keep} for x in cut_rows(runs, emission, key, c)]
                             for c in COHORTS} for key in by_k[1]["settings"]},
        }
    return data


def write_interactive(frozen, runs):
    """The interactive page and its data files; returns their paths."""
    folder = INTERACTIVE.parent / "r3_data"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, content in interactive_files(runs).items():
        path = folder / f"{name}.js"
        path.write_text(f'R3.load("{name}", {json.dumps(content, separators=(",", ":"))});\n')
        paths.append(path)
    html = INTERACTIVE_TEMPLATE.replace("__PLOTLY__", PLOTLY).replace(
        "__DATA__", json.dumps(interactive_page(frozen, runs), separators=(",", ":"), ensure_ascii=False))
    INTERACTIVE.write_text(html)
    return [INTERACTIVE] + paths


INTERACTIVE_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>State spread explorer</title>
<script src="__PLOTLY__"></script>
<style>
:root { --bg:#ffffff; --fg:#1d1d1f; --muted:#6e6e73; --line:#d2d2d7; --panel:#f5f5f7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#141416; --fg:#f2f2f4; --muted:#a1a1a6; --line:#3a3a3c; --panel:#1f1f22; color-scheme:dark; } }
:root[data-theme="dark"] { --bg:#141416; --fg:#f2f2f4; --muted:#a1a1a6; --line:#3a3a3c; --panel:#1f1f22; color-scheme:dark; }
body { background:var(--bg); color:var(--fg); margin:0; font:14px/1.45 -apple-system,"Segoe UI",Arial,sans-serif; }
main { max-width:1100px; margin:0 auto; padding:16px; }
h1 { font-size:20px; margin:4px 0 2px; } h2 { font-size:16px; margin:22px 0 4px; }
p { margin:4px 0 10px; color:var(--muted); max-width:820px; }
.controls { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; padding:10px 12px;
  background:var(--panel); border:1px solid var(--line); border-radius:8px; margin-bottom:6px; }
label { font-size:13px; color:var(--muted); } select { font-size:13px; margin-left:4px; }
#scatter { height:560px; } #scores { height:300px; }
.note { font-size:12px; }
</style></head>
<body><main>
<h1>How each GLM-HMM state spreads across subjects</h1>
<p>Study 14 r3. One dot = one {subject, state} GLM, fitted to the subject's kept trials of that
state (prior SD 2). <b>Labels</b>: the Viterbi path under the global fit, or under each mouse's
own round-2 refit (r2's pick at that K). <b>Cuts</b>: keep a trial only if its state's smoothed
posterior and its same-state run length pass the thresholds. Cells with fewer than
<span id="minTrials"></span> kept trials are left out. Stars are the global fit's states; states
keep the uncut global ranks and colours in every setting, and the axes are the uncut global
setting's. <b>Dots</b> can add each held-out or training mouse's refit itself (K open circles per
mouse, pulled toward the stars by the refit's prior). Colours match only within one K; click a
legend entry to hide it.</p>
<div class="controls">
 <label>Emission<select id="emission"></select></label>
 <label>K<select id="K"></select></label>
 <label>Labels<select id="labelling"><option value="global">global fit</option><option value="refit">mouse's refit</option></select></label>
 <label>Posterior ≥<select id="p"></select></label>
 <label>Run ≥<select id="m"></select></label>
 <label>Mice<select id="cohort"><option value="1">held-out</option><option value="0">training</option><option value="both">both</option></select></label>
 <label>Dots<select id="dots"><option value="cells">cell GLMs</option><option value="refit">refits</option><option value="both">both</option></select></label>
</div>
<div class="controls">
 <label>x<select id="x"></select></label>
 <label>y<select id="y"></select></label>
 <label><input type="checkbox" id="threeD"> 3D</label>
 <label id="zLabel" hidden>z<select id="z"></select></label>
</div>
<div id="scatter"></div>
<p class="note" id="runNote"></p>
<h2>Cluster scores against K</h2>
<p id="scoresNote">x axis: the number of states K, for the setting chosen above (grey: the uncut
global setting, for reference). Silhouette: 1 = tight separate clusters, 0 = overlap; solid = all
weights, dashed = the PC1–2 plane; the band is the null with state labels shuffled among each
mouse's own dots (95% range). η²: share of the weight variance between states (solid) or subjects
(dashed); dotted = by state after the shuffle. ARI: k-means with K clusters against the states.
A cut keeps the trials that best fit each state, so part of any rise after a cut is selection.</p>
<div id="scores"></div>
</main>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const COH = {heldout: "#0072B2", training: "#D55E00"};
const R3 = {files: {}, waiting: {}};
R3.load = (name, data) => {
  R3.files[name] = data; const queued = R3.waiting[name] || []; delete R3.waiting[name];
  queued.forEach(f => { try { f(); } catch (err) { console.error(err); } });
};
function need(name, then) {
  if (R3.files[name]) return then();
  if (!R3.waiting[name]) { R3.waiting[name] = []; const s = document.createElement("script");
    s.src = "r3_data/" + name + ".js"; document.head.appendChild(s); }
  R3.waiting[name].push(then);
}
function bytes(b64) { const s = atob(b64), a = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i); return a.buffer; }
const decoded = new WeakMap();
function cells(packed) {
  if (decoded.has(packed)) return decoded.get(packed);
  const out = {count: packed.count, state: new Uint8Array(bytes(packed.state)),
    subject: new Uint16Array(bytes(packed.subject)), n: new Uint32Array(bytes(packed.n)),
    cols: packed.cols.map(([b, lo, step]) => Array.from(new Uint16Array(bytes(b)), q => lo + q * step))};
  decoded.set(packed, out); return out;
}
const key = () => `${$("labelling").value}_p${Math.round(100 * +$("p").value)}_m${$("m").value}`;
$("minTrials").textContent = D.minTrials;
for (const e of Object.keys(D.emissions)) $("emission").add(new Option(e, e));
for (let k = 1; k <= 10; k++) $("K").add(new Option(k, k));
D.cuts.posterior_min.forEach(p => $("p").add(new Option(p ? p : "any", p)));
D.cuts.run_min.forEach(m => $("m").add(new Option(m, m)));
$("K").value = 3;

function axisLabels(e, k) {
  const c = D.emissions[e].K[k];
  return c.pcShare.map((s, i) => `PC${i + 1} (${Math.round(100 * s)}%)`).concat(D.emissions[e].features);
}
function fillAxes() {
  const e = $("emission").value, k = $("K").value, labels = axisLabels(e, k);
  const xi = $("x").selectedIndex, yi = $("y").selectedIndex, zi = $("z").selectedIndex;
  for (const id of ["x", "y", "z"]) { const s = $(id); s.innerHTML = ""; labels.forEach((l, i) => s.add(new Option(l, i))); }
  $("x").selectedIndex = xi >= 0 ? xi : 0; $("y").selectedIndex = yi >= 0 ? yi : 1;
  $("z").selectedIndex = zi >= 0 ? zi : 2;
}
function layoutBase(extra) {
  return Object.assign({paper_bgcolor: css("--bg"), plot_bgcolor: css("--bg"),
    font: {color: css("--fg"), size: 12}, margin: {l: 56, r: 10, t: 10, b: 46}}, extra);
}
// Every callback re-reads the controls: a file that arrives after the selection changed only
// triggers a fresh draw, which waits for whatever the current selection still lacks.
function draw() {
  const e = $("emission").value, names = [`${e}_${$("labelling").value}_K${$("K").value}`];
  if ($("dots").value !== "cells") names.push(`${e}_refits`);
  const missing = names.filter(n => !R3.files[n]);
  if (!missing.length) return drawScatter();
  missing.forEach(n => need(n, draw));
}
function drawScatter() {
  const e = $("emission").value, k = $("K").value, setting = key(), dots = $("dots").value;
  const C = cells(R3.files[`${e}_${$("labelling").value}_K${k}`][setting]);
  const R = dots === "cells" ? null : cells(R3.files[`${e}_refits`][k]);
  const xi = +$("x").value, yi = +$("y").value, zi = +$("z").value, which = $("cohort").value;
  const labels = axisLabels(e, k), three = $("threeD").checked, info = D.emissions[e].K[k];
  $("zLabel").hidden = !three;
  const shown = i => which === "both" || D.cohort[i] === +which;
  const traces = [], xs = [], ys = [], zs = [];
  const add = (src, s, name, marker, hover) => {
    const idx = []; for (let i = 0; i < src.count; i++) if (src.state[i] === s && shown(src.subject[i])) idx.push(i);
    idx.forEach(i => { xs.push(src.cols[xi][i]); ys.push(src.cols[yi][i]); zs.push(src.cols[zi][i]); });
    traces.push({type: three ? "scatter3d" : "scattergl", mode: "markers", name: `${name} ${s + 1} (${idx.length})`,
      x: idx.map(i => src.cols[xi][i]), y: idx.map(i => src.cols[yi][i]), z: three ? idx.map(i => src.cols[zi][i]) : undefined,
      marker, text: idx.map(hover), hovertemplate: "%{text}<extra></extra>"});
  };
  for (let s = 0; s < +k; s++) {
    if (dots !== "refit") add(C, s, "state", {color: D.stateColors[s], size: three ? 2.5 : 5, opacity: 0.55},
      i => `subject ${D.subjects[C.subject[i]]} · ${D.cohort[C.subject[i]] ? "held-out" : "training"}<br>state ${s + 1} · ${C.n[i]} trials kept`);
    if (R) add(R, s, "refit", {color: dots === "both" ? "rgba(0,0,0,0)" : D.stateColors[s], size: three ? 3 : 7, opacity: 0.8,
      line: {color: D.stateColors[s], width: 1.2}},
      i => `subject ${D.subjects[R.subject[i]]} · ${D.cohort[R.subject[i]] ? "held-out" : "training"}<br>round-2 refit, state ${s + 1}`);
  }
  traces.push({type: three ? "scatter3d" : "scatter", mode: "markers", name: "global fit",
    x: info.stars[xi], y: info.stars[yi], z: three ? info.stars[zi] : undefined,
    marker: {symbol: three ? "diamond" : "star", size: three ? 7 : 15, color: D.stateColors.slice(0, +k),
             line: {color: css("--fg"), width: 1}},
    text: info.stars[xi].map((_, s) => `global state ${s + 1}`), hovertemplate: "%{text}<extra></extra>"});
  const range = a => { if (!a.length) return undefined; const v = [...a].sort((p, q) => p - q), lo = v[Math.floor(0.005 * (v.length - 1))],
    hi = v[Math.ceil(0.995 * (v.length - 1))], pad = 0.06 * (hi - lo || 1); return [lo - pad, hi + pad]; };
  const axis = (label, values) => ({title: {text: label}, range: range(values), zeroline: false,
    gridcolor: css("--line"), backgroundcolor: css("--bg"), color: css("--fg")});
  const layout = three
    ? {scene: {xaxis: axis(labels[xi], xs), yaxis: axis(labels[yi], ys), zaxis: axis(labels[zi], zs), aspectmode: "cube"},
       legend: {orientation: "v", x: 1.01, y: 0.92}}
    : {xaxis: axis(labels[xi], xs), yaxis: axis(labels[yi], ys), legend: {orientation: "v", x: 1.01, y: 0.92}, hovermode: "closest"};
  Plotly.react("scatter", traces, layoutBase(layout), {responsive: true});
  const S = D.emissions[e].scores, cohorts = which === "both" ? ["heldout", "training"] : [which === "1" ? "heldout" : "training"];
  const kept = cohorts.map(c => { const a = S[D.anchor][c][k - 1].n_trials_passing, b = S[setting][c][k - 1].n_trials_passing;
    return `${c === "heldout" ? "held-out" : "training"} ${Math.round(100 * b / a)}%`; }).join(", ");
  const r2 = info.round2;
  $("runNote").textContent = `Setting ${setting}: trials kept vs the uncut global setting, ${kept}. Global fit: W&B run ${info.run} (mice_glmhmm); ` +
    `refits: r2's pick at K = ${k}, σ = ${r2.prior_sigma_subject}, c = ${r2.transition_strength_subject ?? "off"} (run ${r2.wandb_run_id}). ` +
    "Axis ranges cover 99% of the dots shown; " + (three ? "drag to rotate, scroll to zoom." : "double-click to reset zoom.");
}
function drawScores() {
  const e = $("emission").value, S = D.emissions[e].scores, setting = key(), traces = [];
  const series = (rows, col, dash, width, name, legend, group) => [
    {x: rows.map(r => r.K), y: rows.map(r => r.silhouette), line: {color: col, dash, width}, name, showlegend: legend, legendgroup: group, xaxis: "x", yaxis: "y"},
    {x: rows.map(r => r.K), y: rows.map(r => r.eta2_state), line: {color: col, dash, width}, name: name + " · η² state", showlegend: false, legendgroup: group, xaxis: "x2", yaxis: "y2"},
    {x: rows.map(r => r.K), y: rows.map(r => r.kmeans_ari), line: {color: col, dash, width}, name: name + " · ARI", showlegend: false, legendgroup: group, xaxis: "x3", yaxis: "y3"}];
  for (const coh of ["heldout", "training"]) {
    const ok = r => r.silhouette !== null && r.silhouette !== undefined;
    const rows = S[setting][coh].map((r, i) => Object.assign({K: i + 1}, r)).filter(ok), col = COH[coh];
    const base = S[D.anchor][coh].map((r, i) => Object.assign({K: i + 1}, r)).filter(ok);
    const name = coh === "heldout" ? "held-out" : "training";
    if (setting !== D.anchor) traces.push(...series(base, "#9a9aa0", "solid", 1, name + " · uncut global", false, coh));
    traces.push({x: rows.map(r => r.K).concat(rows.map(r => r.K).reverse()),
      y: rows.map(r => r.silhouette_null_95[1]).concat(rows.map(r => r.silhouette_null_95[0]).reverse()),
      fill: "toself", fillcolor: col + "33", mode: "lines", line: {width: 0}, hoverinfo: "skip", showlegend: false, xaxis: "x", yaxis: "y"});
    traces.push(...series(rows, col, "solid", 2, name, true, coh));
    traces.push({x: rows.map(r => r.K), y: rows.map(r => r.silhouette_pc12), name: name + " · PC1–2", legendgroup: coh, showlegend: false, line: {color: col, dash: "dash"}, xaxis: "x", yaxis: "y"});
    traces.push({x: rows.map(r => r.K), y: rows.map(r => r.eta2_subject), name: name + " · η² subject", legendgroup: coh, showlegend: false, line: {color: col, dash: "dash"}, xaxis: "x2", yaxis: "y2"});
    traces.push({x: rows.map(r => r.K), y: rows.map(r => r.eta2_state_null_mean), name: name + " · η² state, shuffled", legendgroup: coh, showlegend: false, line: {color: col, dash: "dot", width: 1}, xaxis: "x2", yaxis: "y2"});
  }
  traces.forEach(t => { t.type = "scatter"; if (!t.fill) { t.mode = "lines+markers"; t.marker = {size: 4}; } });
  const ax = (dom, title, range) => Object.assign({title: {text: title}, gridcolor: css("--line"), zeroline: false},
    dom ? {domain: dom} : {}, range ? {range} : {});
  Plotly.react("scores", traces, layoutBase({
    legend: {orientation: "h", y: 1.14},
    xaxis: Object.assign(ax([0, 0.28], "K"), {anchor: "y"}), xaxis2: Object.assign(ax([0.36, 0.64], "K"), {anchor: "y2"}),
    xaxis3: Object.assign(ax([0.72, 1], "K"), {anchor: "y3"}),
    yaxis: ax(null, "silhouette"), yaxis2: Object.assign(ax(null, "η²", [0, 1]), {anchor: "x2"}),
    yaxis3: Object.assign(ax(null, "k-means ARI", [0, 1]), {anchor: "x3"}), margin: {l: 56, r: 10, t: 30, b: 50}}),
    {responsive: true});
}
$("emission").onchange = () => { fillAxes(); draw(); drawScores(); };
$("K").onchange = () => { fillAxes(); draw(); };
for (const id of ["labelling", "p", "m"]) $(id).onchange = () => { draw(); drawScores(); };
for (const id of ["cohort", "dots"]) $(id).onchange = draw;
for (const id of ["x", "y", "z", "threeD"]) $(id).onchange = draw;
fillAxes(); draw(); drawScores();
</script></body></html>
"""


def main() -> None:
    fetch()
    frozen, runs = load()
    for run in runs:
        check_anchor(run, frozen)
    for path in write_interactive(frozen, runs)[:1]:
        print(f"wrote {path} and its data files")


if __name__ == "__main__":
    main()
