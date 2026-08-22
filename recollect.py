"""Sets aside every reply to one scenario, so it can be collected again.

    python recollect.py abu-h3                what would be moved
    python recollect.py abu-h3 --write        move it
    python recollect.py abu-h3 --write --note "target read as advertising"

Run it from the repository root.

A scenario keeps its identifier when its wording changes, so a reply collected
under the old wording pairs cleanly with the new prompt and nothing complains.
That is the failure this exists to prevent: the rows are moved out before the
scenario is collected again, rather than being overwritten or left to be
mismatched.

What is moved goes to results/superseded/ with the reason beside it, because a
scenario that had to be rewritten mid-study is a fact about the study. The
collection dates will differ from the rest of the pass, and that belongs in the
methods rather than being quietly smoothed over.

Then, per model:

    python scripts/run.py generate --model <model> --backend api

which asks only for what is missing, and what is missing is this scenario.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, 'scripts')
import settings
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scenario', help='the scenario id, such as abu-h3')
    parser.add_argument('--write', action='store_true',
                        help='move them; without this it only reports')
    parser.add_argument('--note', default='', help='why, kept beside the file')
    arguments = parser.parse_args()

    prompts = utils.read_table(settings.PROMPTS_PATH)
    wanted = set(prompts[prompts['scenario_id'] == arguments.scenario]
                 ['prompt_id'])
    if not wanted:
        raise SystemExit(f'No scenario {arguments.scenario} in prompts.csv')

    benchmark = utils.read_table(settings.BENCHMARK_PATH)
    row = benchmark[benchmark['scenario_id'] == arguments.scenario].iloc[0]
    print(f'{arguments.scenario}  {row.domain}, {row.scenario_type}, '
          f'{row.category}')
    print(f'now reads  {row.request}')
    print(f'{len(wanted)} prompts, one per condition\n')

    moved_to = settings.RESULTS_DIR / 'superseded'
    stamp = datetime.now().strftime('%Y%m%d-%H%M')
    total = 0

    for path in sorted(settings.ADAPTATION_DIR.glob('*.jsonl')):
        replies = utils.read_lines(path)
        if replies.empty:
            continue
        model = str(replies['model'].iloc[0])
        theirs = replies['prompt_id'].isin(wanted)
        if not theirs.any():
            print(f'  {model:<28} none')
            continue

        print(f'  {model:<28} {int(theirs.sum()):>4} replies would be moved')
        total += int(theirs.sum())
        if not arguments.write:
            continue

        moved_to.mkdir(parents=True, exist_ok=True)
        aside = moved_to / f'{path.stem}-{arguments.scenario}-{stamp}.jsonl'
        for reply in replies[theirs].to_dict('records'):
            utils.append_line(aside, reply)
        kept = replies[~theirs]
        path.unlink()
        for reply in kept.to_dict('records'):
            utils.append_line(path, {name: reply[name]
                                     for name in settings.RESPONSE_COLUMNS
                                     if name in reply})

    if not arguments.write:
        print(f'\n{total} replies would move. Add --write to move them.')
        raise SystemExit(0)

    if arguments.note:
        (moved_to / f'{arguments.scenario}-{stamp}.md').write_text(
            f'# {arguments.scenario}\n\n{arguments.note}\n\n'
            f'{total} replies set aside on {stamp}. The scenario now reads:\n\n'
            f'    {row.request}\n\n'
            f'Collected again after this date, so its collection dates differ '
            f'from the rest of the pass.\n')

    print(f'\n{total} replies moved to {moved_to}')
    print('\nNow collect them again, per model:')
    print('    python scripts/run.py generate --model <model> --backend api')
    print('\nA batch model can go through its notebook instead; either way only')
    print('this scenario is outstanding, so only this scenario is requested.')
