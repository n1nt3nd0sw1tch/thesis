"""Collects one scenario again, on every model, without touching the notebooks.

    python rerun_scenario.py abu-h3                  what it would do
    python rerun_scenario.py abu-h3 --write          do it
    python rerun_scenario.py abu-h3 --write --model gpt-5.6-luna

Run it from the repository root.

A scenario keeps its identifier when its wording changes, so a reply collected
under the old wording pairs cleanly with the new prompt and nothing complains.
This does the whole correction in one place: it sets the old replies aside, asks
each model for the new ones, and puts them back where the rest of the pass
lives.

Every model goes through the live path here, including the three that were
collected by batch. Thirty nine calls do not need a queue, and a queue would
mean a job identifier, a wait and a download for each. What it costs is that
those replies were fetched by a different route from the rest of that model's
pass, which is worth a line in the methods along with the dates.

What is set aside goes to results/superseded/ with a note saying why. A scenario
rewritten mid-study is a fact about the study, and the collection dates for it
will differ from the rest.
"""

import argparse
import sys
import time
from datetime import datetime

sys.path.insert(0, 'scripts')
import backends
import flags
import settings
import utils


# Define function to move one scenario's replies out of a model's pass, so that
# the scenario is outstanding again and nothing old can be paired with new text
def set_aside(path, wanted, scenario, stamp):
    replies = utils.read_lines(path)
    if replies.empty:
        return 0
    theirs = replies['prompt_id'].isin(wanted)
    if not theirs.any():
        return 0
    aside = (settings.RESULTS_DIR / 'superseded'
             / f'{path.stem}-{scenario}-{stamp}.jsonl')
    for reply in replies[theirs].to_dict('records'):
        utils.append_line(aside, reply)
    kept = replies[~theirs].to_dict('records')
    path.unlink()
    for reply in kept:
        utils.append_line(path, {name: reply[name]
                                 for name in settings.RESPONSE_COLUMNS
                                 if name in reply})
    return int(theirs.sum())


# Define function to ask one model for one prompt, by whichever route it takes
def ask(model, prompt):
    provider = backends.provider_of(model)
    call = backends.call_ollama if provider == 'ollama' else backends.call_api
    body = call(model, [{'role': 'user', 'content': prompt}],
                settings.GENERATION['max_tokens'],
                settings.GENERATION['temperature'])
    blocked, truncated = flags.flags_of(body)
    return {'response': backends.read_reply(provider, body),
            'blocked': blocked, 'truncated': truncated}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scenario', help='the scenario id, such as abu-h3')
    parser.add_argument('--write', action='store_true',
                        help='act; without this it only reports')
    parser.add_argument('--model', default='', help='one model, or all of them')
    parser.add_argument('--note', default='',
                        help='why this scenario was rewritten')
    arguments = parser.parse_args()

    prompts = utils.read_table(settings.PROMPTS_PATH)
    mine = prompts[prompts['scenario_id'] == arguments.scenario]
    if mine.empty:
        raise SystemExit(f'No scenario {arguments.scenario} in prompts.csv')
    wanted = set(mine['prompt_id'])
    text = dict(zip(mine['prompt_id'], mine['prompt']))

    benchmark = utils.read_table(settings.BENCHMARK_PATH)
    row = benchmark[benchmark['scenario_id'] == arguments.scenario].iloc[0]
    replicates = settings.GENERATION['replicates']
    models = ([m for m in settings.MODELS.values() if m['id'] == arguments.model]
              if arguments.model
              else [m for m in settings.MODELS.values()
                    if m['access'] in ('api', 'local')])

    print(f'{arguments.scenario}  {row.domain}, {row.scenario_type}, '
          f'{row.category}')
    print(f'now reads  {row.request}')
    print(f'{len(wanted)} conditions by {replicates} replicates by '
          f'{len(models)} models = {len(wanted) * replicates * len(models)} '
          f'calls\n')

    if not arguments.write:
        for spec in models:
            path = utils.result_path(spec['id'], settings.ADAPTATION_DIR)
            replies = utils.read_lines(path)
            held = 0 if replies.empty else int(
                replies['prompt_id'].isin(wanted).sum())
            print(f'  {spec["id"]:<28} {held:>4} old replies to set aside')
        print('\nAdd --write to set them aside and collect them again.')
        raise SystemExit(0)

    stamp = datetime.now().strftime('%Y%m%d-%H%M')
    moved = 0
    for spec in models:
        path = utils.result_path(spec['id'], settings.ADAPTATION_DIR)
        moved += set_aside(path, wanted, arguments.scenario, stamp)
    print(f'{moved} old replies set aside\n')

    for spec in models:
        model = spec['id']
        path = utils.result_path(model, settings.ADAPTATION_DIR)
        started, failed = time.time(), 0
        for prompt_id in sorted(wanted):
            for replicate in range(1, replicates + 1):
                try:
                    result = ask(model, text[prompt_id])
                except (RuntimeError, SystemExit) as problem:
                    failed += 1
                    result = {'response': '', 'blocked': '', 'truncated': False}
                    error = str(problem)[:200]
                else:
                    error = ''
                utils.append_line(path, {
                    'model': model, 'prompt_id': prompt_id,
                    'replicate': str(replicate), 'error': error, **result})
        total = len(utils.read_lines(path))
        print(f'  {model:<28} {len(wanted) * replicates} collected in '
              f'{time.time() - started:>4.0f}s, {failed} failed, '
              f'{total:,} in the file')

    if arguments.note:
        (settings.RESULTS_DIR / 'superseded'
         / f'{arguments.scenario}-{stamp}.md').write_text(
            f'# {arguments.scenario}\n\n{arguments.note}\n\n'
            f'{moved} replies set aside on {stamp} and collected again after '
            f'that date, so this scenario carries collection dates that differ '
            f'from the rest of the pass. It was fetched live on every model, '
            f'including the three whose pass was collected by batch.\n\n'
            f'The scenario now reads:\n\n    {row.request}\n')

    print('\nThen bring the flags up to date and check the counts:')
    print('    python scripts/flags.py --write')
    print('    notebooks/10_collection.ipynb')