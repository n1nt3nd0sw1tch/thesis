"""Stages figures and tables for the thesis project.

    python scripts/publish.py
    python scripts/publish.py --overleaf ~/Git/thesis-overleaf

Analysis writes to two places in this repository and neither of them is the
thesis. Overleaf is a separate git repository, so a figure regenerated here does
not reach the document until it is copied across, and a number in the thesis can
drift from the number in the data without anything failing.

Without --overleaf this collects everything into results/overleaf/, laid out the
way the Overleaf project expects, so the whole folder can be dragged in at once.
With it, the files are copied straight into a local clone, which is the version
to use once there is one, because it leaves a commit behind.

What goes where in this repository:

    results/language/*.jsonl    per-reply measures, one file per model.
                                Gitignored by results/**/*.jsonl, which is
                                right: they are rebuilt from the collected
                                replies in about a minute, so losing them costs
                                a command rather than money.

    results/tables/*.tex        LaTeX tables. Committed, since only .jsonl is
                                ignored under results/.

    figures/*.pdf, *.png        Figures. Committed. The PDF is what the thesis
                                includes; the PNG is for reading in a notebook
                                or pasting into a message.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from settings import RESULTS_DIR
from utils import section

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / 'figures'
TABLES = RESULTS_DIR / 'tables'
STAGING = RESULTS_DIR / 'overleaf'


# Define function to copy one kind of file into the layout the document expects,
# reporting what moved rather than doing it silently
def stage(source, target, pattern):
    target.mkdir(parents=True, exist_ok=True)
    moved = []
    for path in sorted(source.glob(pattern)):
        destination = target / path.name
        if destination.exists() and destination.stat().st_mtime >= path.stat().st_mtime:
            continue
        shutil.copy2(path, destination)
        moved.append(path.name)
    return moved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overleaf', default='',
                        help='a local clone of the Overleaf project')
    arguments = parser.parse_args()

    destination = Path(arguments.overleaf).expanduser() if arguments.overleaf \
        else STAGING
    if arguments.overleaf and not destination.exists():
        raise SystemExit(f'{destination} does not exist')

    section('Publish')
    figures = stage(FIGURES, destination / 'figures', '*.pdf')
    tables = stage(TABLES, destination / 'tables', '*.tex')

    for name, moved, held in [('figures', figures, FIGURES.glob('*.pdf')),
                              ('tables', tables, TABLES.glob('*.tex'))]:
        total = len(list(held))
        print(f'  {name:<10} {len(moved):>2} copied, {total} in total')
        for item in moved:
            print(f'    {item}')

    print(f'\nStaged in {destination}')
    if not arguments.overleaf:
        print('Drag the two folders into the Overleaf project, or pass '
              '--overleaf with a path to a clone.')
    print('\nInclude them as:')
    print(r'    \includegraphics[width=\linewidth]{language_by_age}')
    print(r'    \input{tables/language_by_age}')
    print('\nThe first needs no path because the preamble sets '
          r'\graphicspath{{figures/}}.')


if __name__ == '__main__':
    main()
