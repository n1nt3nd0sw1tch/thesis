"""Renumber dialogue message positions as turns.

python scripts/renameturns.py --check
python scripts/renameturns.py --apply

The dialogue collection numbers every message, so a three-exchange dialogue runs
from 1 to 6: the user speaks at 1, 3 and 5 and the model answers at 2, 4 and 6.
The thesis defines a turn as one user message together with the reply to it, so
those six positions are three turns. This brings every stored file onto that
definition.

    position 1, 2 -> turn 1    the opening, carried over from the adaptation run
    position 3, 4 -> turn 2    the first pressure and its reply
    position 5, 6 -> turn 3    the second pressure and its reply

The rule is turn = (position + 1) // 2, which is why it cannot be a simple
lookup on the even positions alone: plan.csv and turns.csv hold the user side as
well, and a mapping that only knew 2, 4 and 6 would leave those rows numbered in
the old scheme beside renumbered ones.

After this runs, turn no longer identifies a row on its own in the two files
that carry both sides. A row there is a turn and a role, and anything keying on
turn alone must be changed to key on both.

Idempotent: a file whose turn values already sit inside 1 to 3 is skipped, so
running twice is safe.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCOPES = (ROOT / 'tables', ROOT / 'results')

OLD = {4, 5, 6}
HEADING = re.compile(r'\bTurn ([1-6])\b')


def turn(position):
    return (position + 1) // 2


def heading(text):
    return HEADING.sub(lambda m: f'Turn {turn(int(m.group(1)))}', text)


def column(rows, name='turn'):
    """Renumber a turn column in place. Returns the substitutions made."""
    header = [cell.lower() for cell in rows[0]]
    if name not in header:
        return []
    index = header.index(name)
    values = []
    for row in rows[1:]:
        if index < len(row):
            try:
                values.append(int(row[index]))
            except ValueError:
                pass
    if not set(values) & OLD:            # already renumbered
        return []
    seen = set()
    for row in rows[1:]:
        if index >= len(row):
            continue
        try:
            position = int(row[index])
        except ValueError:
            continue
        seen.add(f'turn {position} -> {turn(position)}')
        row[index] = str(turn(position))
    return sorted(seen)


def do_csv(path, apply):
    rows = list(csv.reader(path.open()))
    if not rows:
        return []
    changes = []
    fresh = [heading(cell) for cell in rows[0]]
    if fresh != rows[0]:
        changes += [f'{a} -> {b}' for a, b in zip(rows[0], fresh) if a != b]
        rows[0] = fresh
    changes += column(rows)
    if changes and apply:
        with path.open('w', newline='') as handle:
            csv.writer(handle).writerows(rows)
    return changes


def do_jsonl(path, apply):
    records, positions = [], set()
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if 'turn' in record:
                positions.add(int(record['turn']))
            records.append(record)
    if not positions & OLD:
        return []
    seen = set()
    for record in records:
        if 'turn' not in record:
            continue
        position = int(record['turn'])
        seen.add(f'turn {position} -> {turn(position)}')
        # The field is a string in the classification output and an integer in
        # the transcripts. Each keeps the type it had.
        record['turn'] = (str(turn(position)) if isinstance(record['turn'], str)
                          else turn(position))
    if apply:
        with path.open('w') as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return sorted(seen)


def main(args):
    touched = 0
    for scope in SCOPES:
        if not scope.exists():
            raise SystemExit(f'{scope} not found. Run this from the repository root.')
        for path in sorted(list(scope.rglob('*.csv')) + list(scope.rglob('*.jsonl'))):
            changes = (do_csv if path.suffix == '.csv' else do_jsonl)(path, args.apply)
            if changes:
                touched += 1
                print(path.relative_to(ROOT))
                for change in changes:
                    print(f'    {change}')

    verb = 'rewritten' if args.apply else 'would be rewritten'
    print(f'\n{touched} files {verb}.')
    if not args.apply and touched:
        print('Nothing was written. Re-run with --apply.')
    if args.apply and touched:
        print('notebooks/18_dialogue.ipynb still writes message positions. '
              'Re-running it will undo this on the files it produces.')


def parser():
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = cli.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true', help='report only')
    group.add_argument('--apply', action='store_true', help='write the changes')
    return cli


if __name__ == '__main__':
    main(parser().parse_args())
