"""Measures what the judge pass will actually cost before committing to it.

    python scripts/bench.py
    python scripts/bench.py --workers 8 16 24 --how-many 60

Two questions, one run.

How far does concurrency scale? Throughput rises with workers only while the
endpoint has capacity free. Past that the requests queue and the wall clock stops
improving, so the useful worker count is the one after which the line flattens
rather than the largest one that runs.

Is the policy being cached? The policy is around 2,100 tokens and is byte
identical on every call, which is 85% of the input for a typical reply. If the
prefix is cached, the second pass over the same replies is markedly faster than
the first. If the two passes take the same time, the policy is being recomputed
46,640 times and the only lever left is making it shorter.

One caveat for a cloud model. gpt-oss:120b-cloud runs on the provider's hardware
with the local daemon as a relay, so OLLAMA_NUM_PARALLEL and OLLAMA_KEEP_ALIVE
govern local queueing rather than remote caching. The test still answers the
question; it just means an unfavourable answer cannot be fixed by a setting.

Nothing is written. The verdicts are discarded, so this can be run against the
real corpus without touching results/.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate
from settings import ADAPTATION_DIR, BENCHMARK_PATH, JUDGE, PROMPTS_PATH
from utils import read_all, read_table, section


# Define function to draw a fixed sample, seeded so that every pass and every
# worker count scores exactly the same replies and the times are comparable
def sample(how_many, seed=7):
    replies = read_all(ADAPTATION_DIR)
    if replies.empty:
        raise SystemExit(f'Nothing collected in {ADAPTATION_DIR}')
    replies = replies[replies['response'].astype(str).str.strip() != '']
    return replies.sample(n=min(how_many, len(replies)), random_state=seed)


# Define function to score a sample once and return how long it took
def timed(rows, requests, scenarios, judge, backend, workers):
    def score(row):
        try:
            return evaluate.judge_reply(
                judge=judge, reply=row.response, backend=backend,
                request=requests[scenarios[row.prompt_id]])
        except Exception:
            return None

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = list(pool.map(score, rows))
    elapsed = time.perf_counter() - started
    good = sum(1 for v in verdicts if v)
    return elapsed, good


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', default='ollama')
    parser.add_argument('--judge', default='')
    parser.add_argument('--workers', type=int, nargs='+', default=[8, 16, 24])
    parser.add_argument('--how-many', type=int, default=60)
    parser.add_argument('--total', type=int, default=46640,
                        help='replies in the full pass, for the estimate')
    arguments = parser.parse_args()
    if not arguments.judge:
        arguments.judge = JUDGE['id']

    prompts = read_table(PROMPTS_PATH)
    benchmark = read_table(BENCHMARK_PATH)
    scenarios = dict(zip(prompts['prompt_id'], prompts['scenario_id']))
    requests = dict(zip(benchmark['scenario_id'], benchmark['request']))
    rows = list(sample(arguments.how_many).itertuples())

    section('Benchmark')
    print(f'{arguments.judge} on {arguments.backend}, {len(rows)} replies a pass')
    print(f'policy {len(evaluate.build_policy()):,} characters')
    for name in ('OLLAMA_NUM_PARALLEL', 'OLLAMA_KEEP_ALIVE'):
        print(f'{name} = {os.environ.get(name, "unset")}')
    print()

    print(f'  {"workers":>8}{"cold":>9}{"warm":>9}{"gain":>8}'
          f'{"replies/s":>11}{"full pass":>12}')
    best = None
    for workers in arguments.workers:
        cold, ok = timed(rows, requests, scenarios, arguments.judge,
                         arguments.backend, workers)
        warm, _ = timed(rows, requests, scenarios, arguments.judge,
                        arguments.backend, workers)
        rate = len(rows) / warm
        hours = arguments.total / rate / 3600
        gain = (cold - warm) / cold
        print(f'  {workers:>8}{cold:>8.0f}s{warm:>8.0f}s{gain:>7.0%}'
              f'{rate:>11.2f}{hours:>11.1f}h'
              + ('' if ok == len(rows) else f'   {len(rows) - ok} failed'))
        if best is None or rate > best[1]:
            best = (workers, rate, gain)

    workers, rate, gain = best
    section('Reading this')
    print(f'Fastest at {workers} workers: {rate:.2f} replies a second, '
          f'{arguments.total / rate / 3600:.1f} hours for the pass.')
    print()
    if gain > 0.25:
        print(f'The warm pass is {gain:.0%} faster, so the prefix is being '
              f'cached. Do not touch judge.yml from here: one changed character '
              f'invalidates it and the policy is paid for again.')
    else:
        print(f'The warm pass is only {gain:.0%} faster, so the policy is being '
              f'recomputed on every call. The remaining lever is its length: '
              f'examples are about half of it, and cutting them from twelve to '
              f'six removes roughly a fifth of the input.')
    print()
    print('If throughput stopped rising with workers, the endpoint is saturated '
          'and more concurrency only lengthens the queue. Use the count where '
          'the line flattened, not the largest one that ran.')


if __name__ == '__main__':
    main()
