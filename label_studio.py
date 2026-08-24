"""Move the labelling sheets in and out of Label Studio.

CSV import cannot carry pre-annotations: Label Studio reads every column into
`data`, so the label columns would show as text beside the reply rather than as
selected radio buttons. Tasks therefore go in as JSON, with the machine labels
under `predictions`.

Predictions rather than annotations, deliberately. An annotation is yours; a
prediction is a suggestion you accept or override, and Label Studio keeps both,
so the export records where you disagreed with the pre-annotator. That is worth
having: it is the only measure of how much the pre-annotation moved you.

    python label_studio.py export  results/annotation  results/label_studio
    python label_studio.py import  ~/Downloads/project.json  results/annotation

The second reads a Label Studio JSON export back into the same CSV shape the
pipeline writes, with judge set to human.
"""

import json
import sys
from pathlib import Path

import pandas as pd

FIELDS = ['answer', 'delivery_response', 'alternative_response', 'risk_statement',
          'legal_statement', 'eligibility_statement', 'social_signpost',
          'expert_signpost', 'service_signpost', 'system_identity',
          'boundary_identity', 'limitation_identity', 'companion_identity']

CARRIED = ['model', 'prompt_id', 'replicate', 'prompt', 'response', 'note']
MODEL_VERSION = 'claude-pre'


def to_tasks(frame):
    tasks = []
    for row in frame.to_dict('records'):
        stratum = {'h': 'Harmful', 'a': 'Age Restricted',
                   'r': 'Rights', 'b': 'Benign'}.get(
            row['prompt_id'].split('-')[1][0], '')
        data = {key: str(row.get(key, '')) for key in CARRIED}
        data['meta'] = f"{row['prompt_id']}  ·  {stratum}  ·  {row['model']}"
        if not data['note'].strip():
            data['note'] = ''
        result = [{'from_name': field, 'to_name': 'response', 'type': 'choices',
                   'value': {'choices': [str(row[field])]}}
                  for field in FIELDS if str(row.get(field, '')).strip()]
        task = {'data': data}
        if result:
            task['predictions'] = [{'model_version': MODEL_VERSION,
                                    'result': result}]
        tasks.append(task)
    return tasks


def export(source, target):
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob('*.csv')):
        if path.stem == 'blocked':
            continue
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        tasks = to_tasks(frame)
        out = target / f'{path.stem}.json'
        out.write_text(json.dumps(tasks, indent=1, ensure_ascii=False))
        predicted = sum(1 for task in tasks if 'predictions' in task)
        print(f'  {path.stem:<32} {len(tasks):>4} tasks, '
              f'{predicted} carry a prediction  ->  {out.name}')


def from_export(path, target):
    tasks = json.loads(Path(path).read_text())
    rows = []
    for task in tasks:
        data = task.get('data', {})
        row = {key: data.get(key, '') for key in CARRIED}
        row['judge'] = 'human'
        annotations = task.get('annotations') or []
        # Label Studio keeps every annotation ever submitted on a task. The last
        # one is the current state, and a cancelled or skipped one is not a
        # label at all.
        chosen = [item for item in annotations
                  if not item.get('was_cancelled')]
        result = chosen[-1]['result'] if chosen else []
        for field in FIELDS:
            row[field] = ''
        row['uncertain'], row['comment'] = '', ''
        for item in result:
            name, value = item.get('from_name'), item.get('value', {})
            if name in FIELDS and value.get('choices'):
                row[name] = value['choices'][0]
            elif name == 'uncertain' and value.get('choices'):
                row['uncertain'] = 'unsure'
            elif name == 'comment' and value.get('text'):
                row['comment'] = ' '.join(value['text'])
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit('No tasks in that export')
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    columns = (['model', 'prompt_id', 'replicate', 'judge', 'prompt', 'response']
               + FIELDS + ['uncertain', 'comment', 'note'])
    for model, group in frame.groupby('model'):
        slug = str(model).replace('.', '_').replace(':', '-').replace('/', '-')
        out = target / f'{slug}_human_labels.csv'
        group[columns].to_csv(out, index=False)
        done = int((group['answer'].str.strip() != '').sum())
        unsure = int((group['uncertain'] == 'unsure').sum())
        print(f'  {slug:<32} {done:>4} of {len(group)} labelled, '
              f'{unsure} marked unsure  ->  {out.name}')


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] not in ('export', 'import'):
        raise SystemExit(__doc__)
    if sys.argv[1] == 'export':
        export(sys.argv[2], sys.argv[3])
    else:
        from_export(sys.argv[2], sys.argv[3])
