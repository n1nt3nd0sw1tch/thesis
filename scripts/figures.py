"""Readability figures: what the tables cannot show.

    python scripts/figures.py
    python scripts/figures.py --only lexical

Writes into figures/:

    readability_lexical.pdf       structure moves with age, vocabulary does not
    readability_correlations.pdf  the fifteen measures, and how far they repeat
    readability_coverage.pdf      what the floor removed, and from where
    readability_signals.pdf       reading level at each level of the age signal

----------------------------------------------------------------------------
Why these four
----------------------------------------------------------------------------

Each is a claim in Chapter 4 that a table states and cannot demonstrate.

The lexical figure carries the mechanism. Section 4.3.3 concludes that models
simplify syntactic form more than lexical choice, and its evidence is a pair of
correlations: sentence length moves with grade level at .60 and vocabulary
difficulty at .02. A correlation states that; a trajectory shows it. All three
are drawn as movement away from age seven in units of their own spread, because
they are in three different units and any other scaling either flattens one or
inflates another.

The correlation figure carries the tiering. Table G.7 is a fifteen by fifteen
matrix at scriptsize, which is unreadable as evidence even though it is exact.
The claim it supports is that the five readability formulas are one measurement
reported five times, and a block structure is what a reader should see rather
than compute from 225 cells.

The coverage figure carries the caveat the whole chapter rests on. The floor
removes replies selected on the outcome, and Section 4.3.1 says so in two
paragraphs and two tables. A model by scenario type grid shows the selection in
one look: the loss concentrates where refusal is commonest, which is where the
age condition acts.

The signals figure carries the ordering. Section 4.3.4 reports five levels of
the age signal in a fixed order, and the point is the spacing between them, not
the numbers: a stated minor age moves the measure several times further than a
cue does.

----------------------------------------------------------------------------
Style
----------------------------------------------------------------------------

Serif throughout, matching the document, and every label black rather than the
grey used for rules and ticks. A label carries a word a reader has to read; a
tick mark carries a position they only have to see, so the two do not take the
same weight. Axis labels and legends are set in Title Case, as table and figure
headings are everywhere else in this thesis.

Panel colours are the saturated values from analysis.PANEL, so a model is the
same colour here as in a table row, an inline tag and the reading-level ladder.

Label sizes are computed from the width a figure will be included at, for the
reason set out in scripts/wordclouds.py: matplotlib sets text in points of the
figure it is drawn on, and a figure included at a fraction of the text width is
scaled by that fraction. Pass --display to match the \\includegraphics width.

Diverging quantities use a diverging map centred on zero, and sequential ones a
sequential map. A correlation runs from minus one to one and has a meaningful
middle; a share of replies removed runs from zero and does not.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps, colors

sys.path.insert(0, str(Path(__file__).resolve().parent))

import language
from analysis import COLOUR, INK, MUTED, NAME, ORDER, PALE
from settings import ROOT

FIGURES = ROOT / 'figures'
TEXT_WIDTH_CM = 16.0

LADDER = [7, 9, 11, 13, 15, 17, 18, 21]
MINOR = [7, 9, 11, 13, 15, 17]
ADULT = [18, 21]
TYPES = ['Harmful', 'Age Restricted', 'Rights', 'Benign']

# The fifteen measures in the order Table G.7 numbers them: the five formulas
# first so their block sits in the top left of the matrix, then the vocabulary
# measures, then the structural ones.
MEASURES = [
    ('fkgl', 'FKGL'), ('fre', 'FRE'), ('gunning_fog', 'Gunning Fog'),
    ('ari', 'ARI'), ('smog', 'SMOG'),
    ('mean_aoa', 'Mean AoA'), ('p90_aoa', 'P90 AoA'), ('max_aoa', 'Max AoA'),
    ('difficult_share', 'Difficult Share'), ('aoa_coverage', 'AoA Coverage'),
    ('response_length', 'Response Length'), ('sentence_length', 'Sentence Length'),
    ('word_length', 'Word Length'), ('ttr', 'TTR'), ('mtld', 'MTLD'),
]

# The five levels of the age signal, weakest evidence of minority to strongest,
# as Section 4.3.4 orders them.
SIGNALS = [('neutral', 'Control'), ('adult_cue', 'Adult Cue'),
           ('adult_age', 'Adult Age'), ('minor_cue', 'Minor Cue'),
           ('minor_age', 'Minor Age')]


# Define function to set the type and the sizes for one figure
#
# Returns the label size in points. Everything else is set on rcParams, which is
# reset between figures so one drawing cannot leak into the next.
def styled(display):
    scale = display * TEXT_WIDTH_CM / (6.5 * 2.54)
    points = 9.0 / scale
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': points,
        'axes.labelsize': points,
        'axes.titlesize': points * 1.05,
        'xtick.labelsize': points * 0.9,
        'ytick.labelsize': points * 0.9,
        'legend.fontsize': points * 0.9,
        'axes.edgecolor': MUTED,
        'axes.labelcolor': 'black',
        'text.color': 'black',
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.labelcolor': 'black',
        'ytick.labelcolor': 'black',
        'axes.linewidth': 0.7,
        'grid.color': PALE,
        'grid.linewidth': 0.5,
    })
    return points


# Define function to write a figure and report it
def save(figure, name):
    FIGURES.mkdir(exist_ok=True)
    written = FIGURES / name
    figure.savefig(written, bbox_inches='tight')
    plt.close(figure)
    print(f'  {written.name}')
    return written


# ----------------------------------------------------------------------------
# Structure moves with age, vocabulary does not
# ----------------------------------------------------------------------------

# Define function to draw what moves with age and what does not
#
# Three measures on one axis, each shown as movement away from its own value at
# age seven, in units of its own standard deviation over the whole corpus.
#
# The normalisation is the whole point. Grade level is in school grades, mean
# age of acquisition in years and sentence length in words, so plotted raw they
# cannot be compared and plotted on separate axes each fills its panel whatever
# it does. Expressing all three as standard deviations of themselves asks one
# question of each: how far does this measure move across the age ladder,
# relative to how far it varies in the corpus at all. A line that ends near zero
# has not moved.
#
# What Section 4.3.3 claims is that the adaptation is structural rather than
# lexical, and the evidence for it in Table G.7 is that sentence length
# correlates .60 with grade level and mean age of acquisition .02. This figure
# is that correlation as a trajectory: if the claim holds, sentence length
# tracks grade level up the ladder and vocabulary difficulty lags behind both.
#
# One caution belongs with it. The 4.97 to 5.22 quoted for mean age of
# acquisition in Section 4.3.3 is its range ACROSS MODELS, not across ages, so
# it is not the quantity drawn here and the two must not be read as one.
def draw_lexical(frame, display):
    points = styled(display)
    figure, axis = plt.subplots(figsize=(6.5, 3.1))

    tracks = [('fkgl', 'Reading Grade Level', '-', 'o'),
              ('sentence_length', 'Sentence Length', '-', 's'),
              ('mean_aoa', 'Vocabulary Difficulty', '--', '^')]
    tones = ['black', colormaps['viridis'](0.42),
             colormaps['viridis'](0.68)]

    for (measure, label, style, mark), tone in zip(tracks, tones):
        levels = frame.groupby('age')[measure].mean().reindex(LADDER)
        spread = frame[measure].std()
        moved = (levels - levels.iloc[0]) / spread
        axis.plot(LADDER, moved.values, style, marker=mark, markersize=3.4,
                  linewidth=1.6, color=tone, label=label)

    axis.axhline(0, color=PALE, linewidth=0.8, zorder=0)
    axis.set_xlabel('Age')
    axis.set_ylabel('Movement From Age 7 (SD)')
    axis.set_xticks(LADDER)
    axis.grid(axis='y', linewidth=0.5)
    axis.set_axisbelow(True)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)
    axis.legend(loc='upper left', frameon=False, fontsize=points * 0.9)

    figure.tight_layout()

    # Printed so the figure can be checked against the claim rather than taken
    # on trust. If vocabulary difficulty moves as far as sentence length does,
    # the mechanism paragraph in Section 4.3.3 does not hold and the figure
    # should not be used to support it.
    print('  movement from age 7 to age 21, in standard deviations:')
    for measure, label, _, _ in tracks:
        levels = frame.groupby('age')[measure].mean().reindex(LADDER)
        moved = (levels.iloc[-1] - levels.iloc[0]) / frame[measure].std()
        print(f'    {label:<24}{moved:+.2f}')
    return save(figure, 'readability_lexical.pdf')


# ----------------------------------------------------------------------------
# The fifteen measures, and how far they repeat one another
# ----------------------------------------------------------------------------

# Define function to draw the correlation matrix as a heatmap
#
# Lower triangle only. A correlation matrix is symmetric, so the upper half is
# the same information mirrored, and printing it twice doubles what a reader has
# to scan for nothing.
def draw_correlations(frame, display):
    points = styled(display)
    columns = [key for key, _ in MEASURES if key in frame.columns]
    labels = [name for key, name in MEASURES if key in frame.columns]
    matrix = frame[columns].corr()

    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    shown = np.ma.masked_where(mask, matrix.values)

    figure, axis = plt.subplots(figsize=(6.5, 5.6))
    image = axis.imshow(shown, cmap='RdBu_r', vmin=-1, vmax=1)

    axis.set_xticks(range(len(labels)), labels, rotation=45, ha='right')
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    axis.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    axis.grid(which='minor', color='white', linewidth=1.0)
    axis.tick_params(which='minor', length=0)
    for side in axis.spines.values():
        side.set_visible(False)

    # The value in each cell, in whichever of black or white reads on it.
    for row in range(len(labels)):
        for column in range(row + 1):
            value = matrix.values[row, column]
            text = f'{value:.2f}'.replace('0.', '.').replace('-.', '$-$.')
            axis.text(column, row, text, ha='center', va='center',
                      fontsize=points * 0.62,
                      color='white' if abs(value) > 0.62 else 'black')

    bar = figure.colorbar(image, ax=axis, shrink=0.55, pad=0.02)
    bar.set_label('Pearson Correlation', size=points * 0.9,
                  color='black')
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=points * 0.8, length=2)

    figure.tight_layout()
    return save(figure, 'readability_correlations.pdf')


# ----------------------------------------------------------------------------
# What the floor removed, and from where
# ----------------------------------------------------------------------------

# Define function to draw the floor loss as a model by scenario type heatmap
#
# Sequential rather than diverging: a share removed runs from zero and has no
# meaningful middle, so a map with a light centre would put a false break in it.
def draw_coverage(frame, floor, display):
    points = styled(display)
    short = frame['response_length'] < floor
    loss = (frame.assign(short=short)
            .pivot_table(index='label', columns='scenario_type',
                         values='short', aggfunc='mean')
            .reindex(index=ORDER, columns=TYPES) * 100)

    figure, axis = plt.subplots(figsize=(6.5, 3.0))
    image = axis.imshow(loss.values, cmap='YlOrBr', vmin=0,
                        vmax=float(np.nanmax(loss.values)))

    axis.set_xticks(range(len(TYPES)), TYPES)
    axis.set_yticks(range(len(ORDER)), ORDER)
    axis.set_xticks(np.arange(len(TYPES) + 1) - 0.5, minor=True)
    axis.set_yticks(np.arange(len(ORDER) + 1) - 0.5, minor=True)
    axis.grid(which='minor', color='white', linewidth=1.2)
    axis.tick_params(which='minor', length=0)
    for side in axis.spines.values():
        side.set_visible(False)

    ceiling = float(np.nanmax(loss.values))
    for row in range(len(ORDER)):
        for column in range(len(TYPES)):
            value = loss.values[row, column]
            axis.text(column, row, f'{value:.1f}', ha='center', va='center',
                      fontsize=points * 0.85,
                      color='white' if value > 0.6 * ceiling else 'black')

    bar = figure.colorbar(image, ax=axis, shrink=0.8, pad=0.02)
    bar.set_label(f'Below the {floor} Word Floor (\\%)',
                  size=points * 0.9, color='black')
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=points * 0.8, length=2)

    figure.tight_layout()
    return save(figure, 'readability_coverage.pdf')


# ----------------------------------------------------------------------------
# Reading level at each level of the age signal
# ----------------------------------------------------------------------------

# Define function to label each reply with the level of its age signal
def signal_level(frame):
    level = pd.Series('', index=frame.index, dtype=object)
    level[frame['condition'].eq('neutral')] = 'neutral'
    level[frame['signal'].eq('cue') & frame['condition'].str.contains('adult')] = 'adult_cue'
    level[frame['signal'].eq('cue') & frame['condition'].str.contains('minor')] = 'minor_cue'
    level[frame['age'].isin(ADULT)] = 'adult_age'
    level[frame['age'].isin(MINOR)] = 'minor_age'
    return level


# Define function to draw the five signal levels as a dot plot
#
# A dot plot rather than bars. The quantity is a grade level, which does not
# start at zero, and a bar drawn from zero to a grade of eight would give
# nine-tenths of its length to the part of the scale that carries no
# information. What the figure is for is the spacing between five levels.
def draw_signals(frame, display):
    points = styled(display)
    frame = frame.assign(level=signal_level(frame))
    frame = frame[frame['level'] != '']

    figure, axis = plt.subplots(figsize=(6.5, 2.8))
    keys = [key for key, _ in SIGNALS]

    for name in ORDER:
        part = frame[frame['label'] == name]
        levels = part.groupby('level')['fkgl'].mean().reindex(keys)
        axis.plot(range(len(keys)), levels.values, marker='o', markersize=4.5,
                  linewidth=1.0, alpha=0.85, color=COLOUR[name], label=name)

    macro = frame.groupby('level')['fkgl'].mean().reindex(keys)
    axis.plot(range(len(keys)), macro.values, marker='D', markersize=5.5,
              linewidth=2.0, color='black', zorder=5,
              label='Panel Mean')

    axis.set_xticks(range(len(keys)), [name for _, name in SIGNALS])
    axis.set_ylabel('Reading Grade Level')
    axis.grid(axis='y', linewidth=0.5)
    axis.set_axisbelow(True)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)

    axis.legend(loc='upper right', ncol=2, frameon=False,
                fontsize=points * 0.82)
    figure.tight_layout()
    return save(figure, 'readability_signals.pdf')


def main(arguments):
    raw = language.load()
    raw['label'] = raw['model'].map(NAME)

    # The floor is applied here, as it is in the notebook, so a figure and a
    # table computed from the same corpus report the same denominator.
    frame = raw.copy()
    short = frame['response_length'] < arguments.floor
    frame.loc[short, ['fkgl', 'fre', 'gunning_fog', 'ari', 'smog']] = np.nan

    print(f'{len(raw):,} replies, {int(short.sum()):,} below the '
          f'{arguments.floor} word floor\n')

    drawn = {'lexical': lambda: draw_lexical(frame[frame['age'].notna()],
                                            arguments.display),
             'correlations': lambda: draw_correlations(frame,
                                                       arguments.display),
             'coverage': lambda: draw_coverage(raw, arguments.floor,
                                               arguments.display),
             'signals': lambda: draw_signals(frame, arguments.display)}

    wanted = drawn if arguments.only == 'all' else {arguments.only:
                                                    drawn[arguments.only]}
    for name, build in wanted.items():
        try:
            build()
        except Exception as failure:
            print(f'  {name} FAILED, {type(failure).__name__}: {failure}')

    print(f'\nWritten to {FIGURES.relative_to(ROOT)}')
    print('Upload them to Overleaf: the tooling writes text only.')


def parser():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--floor', type=int, default=50,
                        help='word floor applied before the formula measures '
                             'are read')
    parser.add_argument('--display', type=float, default=1.0,
                        help='fraction of the text width the figures will be '
                             'included at, which sets the label sizes')
    parser.add_argument('--only', default='all',
                        choices=['all', 'lexical', 'correlations', 'coverage',
                                 'signals'],
                        help='draw one figure rather than all four')
    return parser


if __name__ == '__main__':
    main(parser().parse_args())
