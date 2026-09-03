# Does Age Matter?

Code and data for a benchmark that asks whether a chatbot changes what it will
answer when it is told how old the user is. The canonical request is
byte-identical across the disclosure conditions and only the age material around
it varies, so the age signal is the only thing that differs between two prompts
for the same scenario. 200 scenarios, 13 disclosure conditions, 3 replicates,
6 models, 46,800 requests.

## Layout

```
config/
  settings.yml     the design, the corpora, and the model panel
  scenarios.yml    the 200 scenarios, the one part written by hand

scripts/
  settings.py      what the design states, and where everything lives
  utils.py         identifiers, files, validation, and the loop the stages share
  backends.py      one reply, through whichever runtime is available
  download.py      1  the corpora        -> data/process/corpora/
  build.py         2  the benchmark      -> data/
  run.py           3  the replies        -> results/adaptation/
  judge.py         4  the classifications -> results/classification/
  analysis.py         the corpus, the statistics and the shared paths
  language.py         the fifteen language measures, computed from reply text
  figures.py       5  the readability figures -> figures/
  safety_figures.py 5 the safety figures      -> figures/

data/              the corpora downloaded, and the benchmark built from them
notebooks/         what the pipeline produced, read back
jobs/              cluster submission
archive/           an approach tried and abandoned, kept because Chapter 3 reports it
```

## Setup

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"
cp .env.example .env        # then add the api keys the panel needs
```

## Running it

Four stages, in order. Each reads the design from `config/`, writes into `data/`
or `results/`, and reports what it did.

| Stage | Command | Writes |
|---|---|---|
| 1. Corpora | `python scripts/download.py` | `data/process/corpora/` |
| 2. Benchmark | `python scripts/build.py` | `drafts.csv`, `benchmark.csv`, `prompts.csv`, `scores.csv` |
| 3. Replies | `python scripts/run.py generate --model <id> --backend <runtime>` | `results/adaptation/<model>.jsonl` |
| 4. Classification | `python scripts/judge.py --backend <runtime>` | `results/classification/<model>.jsonl` |
| 5. Figures | `python scripts/figures.py` and `python scripts/safety_figures.py` | `figures/*.pdf` |

Stage 5 runs after the notebooks, not before them. `figures.py` recomputes its
own reductions from the language table, but `safety_figures.py` reads the
intervals out of `tables/machine/register_safety.csv` rather than recomputing
them, so that a figure and the table beside it in the thesis come from one
result. It refuses to run if that file is absent and says why.

Both write straight into `figures/` and neither touches `tables/`. That split is
the point: a figure can be redrawn without regenerating twenty-two CSVs, which
is what rerunning a notebook to change one line of drawing code used to do.

Stage 2 is the whole build: it reads the corpora and `config/scenarios.yml`,
opens a draft per usable source record, fills the 200 scenario slots from the
drafts marked to keep, expands each across the thirteen disclosure conditions,
and scores every request variant for readability. `drafts.csv` is the only file edited by
hand, and work already in it is preserved on every run, so the stage can be
rerun after any revision to the design.

Stages 3 and 4 append one line at a time and skip whatever is already collected,
so a run that stops part way resumes where it left off. Give either a `--limit`
first: the rate it reports is enough to estimate a full pass before committing a
night to it.

The persistence extension is built once replies exist:

```
python scripts/build.py turns       # results/dialogue/plan.csv and turns.csv
```

## Before a run

```
python scripts/run.py check
python scripts/run.py check --model qwen3:8b --backend ollama \
    --prompt-id sub-h1-age09
python scripts/evaluate.py --policy
```

The first looks every identifier up against its provider, so a renamed or
retired model fails here rather than part way through generation, and reports
how many calls the panel implies. Naming a model puts one prompt through
generation, scoring and comparison as well; `--reply` scores a supplied text
instead, which exercises the scoring path without loading a large model. The
last prints the policy as the classifier sees it.

## On a cluster

Compute nodes have no outbound network, so a model is fetched from a login node
first and the job runs offline against the cache.

```
bash jobs/fetch.sh Qwen/Qwen3.6-27B
qsub -v MODEL=Qwen/Qwen3.6-27B jobs/run.sh
qsub -v STAGE=judge jobs/run.sh
```

## The modules

Three files are imported rather than run. `settings.py` reads the two
configuration files and exposes what they state along with everything that
follows from it; it does no work. `utils.py` holds the machinery that acts on
those values: identifiers, file reading and writing, validation, reporting, and
the resumable loop that generation and judging both run. `backends.py` generates
one reply through vLLM on a GPU, Ollama or MLX locally, or transformers
anywhere. `evaluate.py` is both: it holds the policy, the verdict reading and
the language measures, which `build.py` also uses to score the requests, and
running it scores the replies.

## Licences

The code is under the licence in `LICENSE`. The source corpora keep their own,
recorded with a checksum per file in `data/sources.md`. They are
downloaded rather than redistributed here.
