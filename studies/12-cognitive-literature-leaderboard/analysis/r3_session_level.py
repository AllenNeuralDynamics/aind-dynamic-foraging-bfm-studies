"""Producer for r3: TODO describe what it computes from which frozen inputs."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'studies' / 'util'))
from _meta import build_meta, write_artifact  # noqa: E402
# Figures: `import figure_style as fs`, `fs.apply_paper_style()`, draw at final size
# (`figsize=fs.mm(183, 80)`), then `fs.save_figure(fig, HERE / 'fig_<slug>.svg')`.

WANDB_GROUPS: list[str] = []   # exact groups read, if any (via a committed freeze)


def main() -> None:
    inputs: list[Path] = []      # committed frozen inputs read; _meta v2 needs at least one
    result = {}                  # TODO compute
    meta = build_meta('analysis/r3_session_level.py', WANDB_GROUPS, study_root=HERE.parent,
                      code=[__file__], inputs=inputs)
    # TODO: rewrite the result block in reports/r3-session-level.md, then list it in outputs as
    #       HERE / 'reports' / 'r3-session-level.md#result-1' alongside any figures.
    write_artifact(HERE / 'r3_session_level.json', {'_meta': meta, **result}, outputs=[])


if __name__ == '__main__':
    main()
