"""Producer for r4: the 3-way GRU's 3-way and engagement likelihood per H × D.

Reads ``frozen_runs.json`` (the kept runs r1 and r2 use, one per (D, H, seed)) and writes, per cell,
the seeds, mean and SEM of two held-out scores that r1 and r2 do not publish:

- ``three_way``: ``eval_likelihood``, the likelihood of the actual outcome (left, right or ignore) on
  every held-out trial (chance 1/3);
- ``engage``: ``eval_likelihood_engage``, the likelihood of respond vs ignore (chance 1/2).

The atlas leaderboard "with ignored trials" reads ``r4_three_way_engagement.json`` at
``/grid/<metric>/<H>/<D>/seeds``; renaming those keys fails its build.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'studies' / 'util'))
from _meta import build_meta, write_artifact  # noqa: E402
import figure_style as fs  # noqa: E402
from plot_style import HCOLOR  # noqa: E402

FROZEN = HERE / 'frozen_runs.json'
REPORT = HERE / 'reports' / 'r4-three-way-engagement.md'
FIG = HERE / 'fig_r4_three_way_engagement.svg'
HS, DS = [16, 64, 128, 256], [10, 30, 100, 614]
METRICS = {
    'three_way': ('eval_likelihood', '3-way likelihood (L / R / ignore)',
                  'likelihood of the actual outcome (left, right or ignore) on every held-out trial; chance 1/3'),
    'engage': ('eval_likelihood_engage', 'Engagement likelihood',
               'likelihood of respond vs ignore on every held-out trial; chance 1/2'),
}


def compute() -> tuple[dict, list[str]]:
    runs = json.loads(FROZEN.read_text())['runs']
    groups = sorted({r['wandb_group'] for r in runs})
    grid = {}
    for name, (key, _, _) in METRICS.items():
        cells = defaultdict(list)
        for r in runs:
            cells[(r['H'], r['D'])].append(float(r[key]))
        grid[name] = {}
        for h in HS:
            grid[name][str(h)] = {}
            for d in DS:
                v = np.array(cells[(h, d)])
                grid[name][str(h)][str(d)] = {
                    'mean': float(v.mean()), 'sem': float(v.std(ddof=1) / np.sqrt(len(v))),
                    'n': len(v), 'seeds': [round(x, 6) for x in v.tolist()]}
    return grid, groups


def figure(grid: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=fs.mm(2 * (fs.SQUARE_PANEL_MM + fs.AXIS_MM), 55),
                             layout='constrained')
    for ax, (name, (_, ylabel, _)) in zip(axes, METRICS.items()):
        for h in HS:
            cells = [grid[name][str(h)][str(d)] for d in DS]
            for d, c in zip(DS, cells):
                ax.scatter([d * 0.93] * c['n'], c['seeds'], s=3, color=HCOLOR[h], alpha=0.35,
                           edgecolors='none', zorder=2)
            ax.errorbar(DS, [c['mean'] for c in cells], yerr=[c['sem'] for c in cells], color=HCOLOR[h],
                        marker='o', ms=2.5, lw=0.8, zorder=3, label=f'H = {h}')
        ax.set_xscale('log')
        fs.plain_log_axis(ax, 'x')
        ax.set_xticks(DS, [str(d) for d in DS])
        ax.set_xlabel('Training mice D')
        ax.set_ylabel(ylabel)
    axes[0].legend(loc='lower right')
    fs.panel_label(axes[0], 'a')
    fs.panel_label(axes[1], 'b')
    fs.save_figure(fig, FIG)


def _table(grid: dict, name: str) -> str:
    rows = ['| H | ' + ' | '.join(f'D = {d}' for d in DS) + ' | Δ (D = 10 → 614) |',
            '|---|' + '---:|' * (len(DS) + 1)]
    for h in HS:
        c = grid[name][str(h)]
        cells = [f"{c[str(d)]['mean']:.4f} ± {c[str(d)]['sem']:.4f}" for d in DS]
        rows.append(f'| {h} | ' + ' | '.join(cells) + f" | {c['614']['mean'] - c['10']['mean']:+.4f} |")
    return '\n'.join(rows)


def render(grid: dict) -> str:
    return '\n'.join([
        f'![3-way and engagement likelihood](../{FIG.name})',
        '',
        'The 3-way GRU (left, right or ignore) on the 149 held-out mice, against training mice D, per hidden '
        'size H. Mean ± SEM over 3 seeds; small dots are single seeds. Same kept runs as r1 and r2.',
        '',
        f"**a. 3-way likelihood**: {METRICS['three_way'][2]}.",
        '',
        _table(grid, 'three_way'),
        '',
        f"**b. Engagement likelihood**: {METRICS['engage'][2]}.",
        '',
        _table(grid, 'engage'),
    ])


def _rewrite(text: str, block: str, body: str) -> str:
    pat = re.compile(rf'(<!-- BEGIN {block} -->\n).*?(\n<!-- END {block} -->)', re.S)
    return pat.sub(lambda m: m.group(1) + body + m.group(2), text)


def main() -> None:
    fs.apply_paper_style()
    grid, groups = compute()
    figure(grid)
    REPORT.write_text(_rewrite(REPORT.read_text(), 'result-1', render(grid)))
    meta = build_meta('analysis/r4_three_way_engagement.py', groups, study_root=HERE.parent,
                      code=[__file__], inputs=[FROZEN])
    result = {'Hs': HS, 'Ds': DS,
              'metrics': {k: {'key': v[0], 'label': v[1], 'description': v[2]} for k, v in METRICS.items()},
              'grid': grid}
    write_artifact(HERE / 'r4_three_way_engagement.json', {'_meta': meta, **result},
                   outputs=[FIG, f'{REPORT}#result-1'])


if __name__ == '__main__':
    main()
