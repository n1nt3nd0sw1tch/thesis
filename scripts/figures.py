"""Readability figures, built on one panel grammar and one decomposition rule.

    python scripts/figures.py
    python scripts/figures.py --set main
    python scripts/figures.py --only ladder

Main text, one figure a question, every one on the model grid:

    readability_ladder.pdf         Flesch-Kincaid grade level across stated age
    readability_distribution.pdf   minor against adult, as distributions
    readability_signals.pdf        three measures by strength of the age signal
    readability_coverage.pdf       what the floor removed, by age and stratum
    readability_correlations.pdf   how far the fifteen measures repeat

Supplement, the same three questions decomposed by scenario type, four figures
each, plus the coverage detail:

    readability_ladder_{harmful,age_restricted,rights,benign}.pdf
    readability_distribution_{...}.pdf
    readability_signals_{...}.pdf
    readability_coverage_grid.pdf

----------------------------------------------------------------------------
One grammar, and why there is no four by six figure
----------------------------------------------------------------------------

Every figure here is a two by three grid of models, in the order GPT, Claude,
Gemini above and DeepSeek, Mistral, Gemma below. A reader learns the layout once
and then reads position as model everywhere.

A scenario type is a separate figure, never a fourth and fifth row. Twenty-four
panels on one page collide their titles, merge their ticks and shrink their
annotations past reading, and the decomposition is worth more than the page it
would save: on this corpus the age-restricted stratum behaves unlike the other
three, which is a result and not a detail. Four figures of six panels say it;
one figure of twenty-four buries it.

The main figure is the overall result and the four that follow it are the
decomposition. That is the same relation the word-cloud families use.

----------------------------------------------------------------------------
What encodes what
----------------------------------------------------------------------------

Model is colour, always, from analysis.PANEL, so a model is the same colour here
as in a table row and an inline tag. Because a panel already names its model,
nothing inside a panel needs colour to say which model it is, and colour is free
to carry something else.

Minor against adult is a filled shape against a dashed outline, within the
model's colour. Scenario type, where several appear in one panel, is line style
and marker rather than hue, so the distinction survives a monochrome print.

Sequential quantities take a single-hue blue. The correlation figure shows the
absolute value in blue and keeps the sign in the printed number, because the
question it answers is how far two measures repeat one another and that is a
question about magnitude.

----------------------------------------------------------------------------
The unit a figure plots
----------------------------------------------------------------------------

The distribution figures plot scenario-level means, not replies. The primary
contrast averages replicates within a scenario and age, then the six minor ages
and the two adult ones, then differences scenarios; a histogram of individual
replies would show a different distribution from the one the reported effect is
computed over, and the annotation would then belong to a quantity the picture
does not draw. Plotting the scenario means puts the figure, the annotation, the
interval and the primary table on one observational unit.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps, colors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))

import language
from analysis import (COLOUR, INK, MARKER, MUTED, NAME, ORDER, PALE, PASTEL,
                      bootstrap_paired)
from settings import ROOT

FIGURES = ROOT / 'figures'
TEXT_WIDTH_CM = 16.0
LABEL_POINTS = 9.0

LADDER = [7, 9, 11, 13, 15, 17, 18, 21]
MINOR = [7, 9, 11, 13, 15, 17]
ADULT = [18, 21]
TYPES = ['Harmful', 'Age Restricted', 'Rights', 'Benign']
SLUG = {kind: kind.lower().replace(' ', '_') for kind in TYPES}

# The target of equation eq:target, defined for minor ages only. It is drawn to
# seventeen and stops: continuing it through eighteen and twenty-one would
# assert an adult target grade the methodology explicitly declines to set.
TARGET = {age: min(age - 5, 12) for age in MINOR}

# The kind of signal in brackets under the block it belongs to, so the pairing
# is read from the layout: Adult and Minor name the block and Cue and Age name
# how it was given.
SIGNALS = [('neutral', 'Control', ''), ('adult_cue', 'Adult', '(Cue)'),
           ('adult_age', 'Adult', '(Age)'), ('minor_cue', 'Minor', '(Cue)'),
           ('minor_age', 'Minor', '(Age)')]

# Line style, marker and depth of blue a scenario type. Style and marker carry
# it in greyscale and the blue carries it in colour, which is the same
# redundancy the heatmaps use and the reason a model colour is not needed here:
# the panel already names the model.
TYPE_STYLE = {'Harmful': ('-', 'o', 0.92), 'Age Restricted': ('--', 's', 0.72),
              'Rights': (':', '^', 0.52), 'Benign': ('-.', 'D', 0.34)}

MEASURE_GROUPS = [
    ('Readability', [('fkgl', 'FKGL'), ('fre', 'FRE'),
                     ('gunning_fog', 'Gunning Fog'), ('ari', 'ARI'),
                     ('smog', 'SMOG')]),
    ('Vocabulary', [('mean_aoa', 'Mean AoA'), ('p90_aoa', 'P90 AoA'),
                    ('max_aoa', 'Max AoA'),
                    ('difficult_share', 'Difficult Share'),
                    ('aoa_coverage', 'AoA Coverage')]),
    ('Structure', [('response_length', 'Response Length'),
                   ('sentence_length', 'Sentence Length'),
                   ('word_length', 'Word Length'), ('ttr', 'TTR'),
                   ('mtld', 'MTLD')]),
]
MEASURES = [pair for _, group in MEASURE_GROUPS for pair in group]

PANEL_FILL = '#F5F5F5'
MINOR_BAND = '#E8EDF2'
ADULT_BAND = '#F2EDE8'


# ----------------------------------------------------------------------------
# The reduction, shared with the notebook
# ----------------------------------------------------------------------------

# Define function to reduce to scenario level before averaging
def by_scenario(part, measure, keys):
    cell = part.groupby(keys + ['scenario_id', 'condition'],
                        observed=True)[measure].mean()
    scenario = cell.groupby(keys + ['scenario_id'], observed=True).mean()
    return scenario.groupby(keys, observed=True).mean()


# Define function to reduce one subset to a minor and an adult mean a scenario
#
# Complete case across all eight stated ages, as the primary contrast is: a
# scenario missing a measurable reply at any one of them is dropped whole.
# Returns the two series and their difference, so a figure and its annotation
# are computed from one object.
def blocks(part, column='fkgl'):
    wide = part.pivot_table(index='scenario_id', columns='age', values=column,
                            aggfunc='mean')
    wide = wide.reindex(columns=MINOR + ADULT).dropna()
    if wide.empty:
        empty = pd.Series(dtype=float)
        return empty, empty, empty
    return wide[MINOR].mean(axis=1), wide[ADULT].mean(axis=1), None


def contrast(part, column='fkgl', interval=True):
    minor, adult, _ = blocks(part, column)
    if minor.empty:
        return float('nan'), float('nan'), float('nan'), 0
    difference = minor - adult
    if not interval:
        return float(difference.mean()), float('nan'), float('nan'), len(difference)
    point, low, high = bootstrap_paired(difference)
    return point, low, high, len(difference)


# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------

def styled(display, width_inches=7.4):
    scale = display * TEXT_WIDTH_CM / (width_inches * 2.54)
    points = LABEL_POINTS / scale
    plt.rcParams.update({
        'font.size': points, 'axes.labelsize': points,
        'axes.titlesize': points * 1.02, 'xtick.labelsize': points * 0.88,
        'ytick.labelsize': points * 0.88, 'legend.fontsize': points * 0.9,
        'axes.edgecolor': MUTED, 'axes.labelcolor': 'black',
        'text.color': 'black', 'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.labelcolor': 'black', 'ytick.labelcolor': 'black',
        'axes.linewidth': 0.7,
    })
    return points


def panel(axis, title=None, points=9.0):
    axis.set_facecolor(PANEL_FILL)
    axis.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.55,
              color='white')
    axis.set_axisbelow(True)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)
    if title:
        axis.set_title(title, pad=points * 0.5, color='black')


def save(figure, name):
    FIGURES.mkdir(exist_ok=True)
    written = FIGURES / name
    figure.savefig(written, bbox_inches='tight')
    plt.close(figure)
    print(f'  {written.name}')
    return written


# Define function to annotate a panel with the contrast it draws
#
# The interval rather than the count: an interval says how precisely the effect
# is known and a count does not, and the complete-scenario counts are in the
# primary table where a reader can compare them.
def annotate(axis, point, low, high):
    if not np.isfinite(point):
        return
    text = (rf'$\Delta = {point:+.2f}$' if not np.isfinite(low)
            else rf'$\Delta = {point:+.2f}$' '\n' rf'$[{low:+.2f}, {high:+.2f}]$')
    # In a white box in the corner, so it sits over the panel fill without
    # competing with the title above it or the data behind it. The complete
    # scenario count belongs in the caption and the primary table, where a
    # reader can compare it across models, not inside every panel.
    axis.text(0.97, 0.95, text, transform=axis.transAxes, ha='right', va='top',
              color='black', fontsize=plt.rcParams['font.size'] * 0.8,
              linespacing=1.25,
              bbox=dict(facecolor='white', alpha=0.78, edgecolor='none',
                        pad=2.0))


# Define function to shade the minor and adult stretches of the age axis
def age_bands(axis):
    axis.axvspan(6.4, 17.5, color=MINOR_BAND, zorder=0)
    axis.axvspan(17.5, 21.6, color=ADULT_BAND, zorder=0)


# Define function to pick the text colour for a heatmap cell
#
# White on the darker cells and black on the lighter, which is how a heatmap
# is read and what the confusion matrices this style follows do.
#
# The cut-off is a luminance rather than a position on the ramp, so it lands in
# the right place whatever the colormap: Blues reaches a given lightness at a
# different position from Reds or viridis, and a fixed fraction of the ramp
# would switch too early on one and too late on another. On Blues, 0.42 falls
# at about three quarters along, so the darkest quarter of the cells carry
# white numerals and the rest carry black.
def readable_on(rgba):
    red, green, blue = rgba[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return 'white' if luminance < 0.42 else 'black'


# Define function to label a small-multiple figure once rather than per panel
#
# A shared label and label_outer between them are what stop six panels carrying
# six copies of the same axis name. Without it the left column sets its y label
# twice and the two collide, which is what the earlier versions of these figures
# did.
def outer_labels(figure, axes, xlabel, ylabel, points):
    for axis in axes.flat:
        axis.label_outer()
    if xlabel:
        figure.supxlabel(xlabel, color='black', fontsize=points)
    if ylabel:
        figure.supylabel(ylabel, color='black', fontsize=points)


# ----------------------------------------------------------------------------
# Grade level across age
# ----------------------------------------------------------------------------

# Define function to draw one model's trajectory against the target
#
# The target is drawn to seventeen and stops. Continuing it through eighteen and
# twenty-one would assert an adult target grade the methodology declines to set.
def ladder_panel(axis, part, name, points, title=None, small=False):
    levels = by_scenario(part, 'fkgl', ['age']).reindex(LADDER)
    age_bands(axis)
    axis.plot(list(TARGET), list(TARGET.values()), linestyle=':',
              linewidth=1.3 if small else 1.5, color=MUTED, zorder=2)
    axis.plot(LADDER, levels.values, marker=MARKER[name],
              markersize=3.2 if small else 4.4, markerfacecolor=PASTEL[name],
              markeredgecolor=COLOUR[name], markeredgewidth=1.0,
              linewidth=1.4 if small else 1.7, color=COLOUR[name], zorder=3)
    panel(axis, title, points)
    axis.set_facecolor('none')
    axis.set_xticks(LADDER)
    axis.set_xlim(6.4, 21.6)

def draw_ladder(frame, display, kind=None):
    points = styled(display)
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=True,
                                sharey=True, constrained_layout=True)
    for index, name in enumerate(ORDER):
        axis = axes[index // 3][index % 3]
        part = frame[frame['label'] == name]
        ladder_panel(axis, part, name, points, name)
        annotate(axis, *contrast(part)[:3])
    outer_labels(figure, axes, 'Age', 'Flesch-Kincaid Grade Level',
                 points)

    handles = [Patch(facecolor=MINOR_BAND), Patch(facecolor=ADULT_BAND),
               Line2D([0], [0], color=MUTED, linestyle=':', linewidth=1.5)]
    figure.legend(handles, ['Minor', 'Adult', 'Target'],
                  loc='lower center', ncol=3, frameon=False,
                  bbox_to_anchor=(0.5, -0.04), fontsize=points * 0.9)
    name = ('readability_ladder.pdf' if kind is None
            else f'readability_ladder_{SLUG[kind]}.pdf')
    return save(figure, name)


# ----------------------------------------------------------------------------
# Minor against adult, as distributions
# ----------------------------------------------------------------------------

def distribution_panel(axis, part, name, edges, points, title=None,
                       label=False):
    minor, adult, _ = blocks(part)
    axis.hist(minor, bins=edges, density=True, alpha=0.28,
              facecolor=COLOUR[name], edgecolor=COLOUR[name], linewidth=1.1,
              label='Minor' if label else None)
    axis.hist(adult, bins=edges, density=True, histtype='step',
              linestyle='--', linewidth=2.0, color=COLOUR[name],
              label='Adult' if label else None)
    panel(axis, title, points)
    return minor, adult


def overlap(minor, adult, edges):
    if minor.empty or adult.empty:
        return float('nan')
    a, _ = np.histogram(minor, bins=edges, density=True)
    b, _ = np.histogram(adult, bins=edges, density=True)
    return float(np.minimum(a, b).sum() * (edges[1] - edges[0]))


def draw_distribution(frame, display, kind=None):
    points = styled(display)
    edges = np.linspace(2, 14, 19)
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=True,
                                sharey=True, constrained_layout=True)

    print(f'  overlap, minor against adult'
          f'{"" if kind is None else ", " + kind}:')
    for index, name in enumerate(ORDER):
        axis = axes[index // 3][index % 3]
        part = frame[frame['label'] == name]
        minor, adult = distribution_panel(axis, part, name, edges, points,
                                          name, label=(index == 0))
        annotate(axis, *contrast(part)[:3])
        print(f'    {name:<24}{overlap(minor, adult, edges):.0%}')
    outer_labels(figure, axes, 'Flesch-Kincaid Grade Level', 'Density', points)

    handles = [Patch(facecolor=PALE, alpha=0.5, edgecolor=MUTED, linewidth=1.1),
               Line2D([0], [0], color=MUTED, linewidth=2.0, linestyle='--')]
    figure.legend(handles, ['Minor', 'Adult'], loc='lower center', ncol=2,
                  frameon=False, bbox_to_anchor=(0.5, -0.03),
                  fontsize=points * 0.95)
    name = ('readability_distribution.pdf' if kind is None
            else f'readability_distribution_{SLUG[kind]}.pdf')
    return save(figure, name)


# ----------------------------------------------------------------------------
# Grade level by strength of the age signal
# ----------------------------------------------------------------------------

def signal_level(frame):
    level = pd.Series('', index=frame.index, dtype=object)
    level[frame['condition'].eq('neutral')] = 'neutral'
    level[frame['signal'].eq('cue')
          & frame['condition'].str.contains('adult', na=False)] = 'adult_cue'
    level[frame['signal'].eq('cue')
          & frame['condition'].str.contains('minor', na=False)] = 'minor_cue'
    level[frame['age'].isin(ADULT)] = 'adult_age'
    level[frame['age'].isin(MINOR)] = 'minor_age'
    return level


# The three measures the signal figure tracks. Grade level is in school grades
# and the two vocabulary measures in years, so they cannot share an axis raw;
# each is drawn as movement away from the control condition in units of its own
# standard deviation, which asks one question of all three: how far does this
# measure move when the model is told something about the user, relative to how
# far it varies at all.
SIGNAL_TRACKS = [('fkgl', 'FKGL', '-', 'o', 1.0),
                 ('mean_aoa', 'Mean AoA', '--', '^', 0.85),
                 ('p90_aoa', 'P90 AoA', ':', 's', 0.7)]


def draw_signals(frame, display, kind=None):
    points = styled(display)
    frame = frame.assign(level=signal_level(frame))
    frame = frame[frame['level'] != '']
    keys = [key for key, _, _ in SIGNALS]
    names = [top if not sub else f'{top}\n{sub}' for _, top, sub in SIGNALS]

    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), sharex=True,
                                sharey=True, constrained_layout=True)
    for index, name in enumerate(ORDER):
        axis = axes[index // 3][index % 3]
        axis.axvspan(0.5, 2.5, color=ADULT_BAND, zorder=0)
        axis.axvspan(2.5, 4.5, color=MINOR_BAND, zorder=0)
        axis.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
        part = frame[frame['label'] == name]

        for measure, label, style, mark, weight in SIGNAL_TRACKS:
            levels = by_scenario(part, measure, ['level']).reindex(keys)
            spread = part[measure].std()
            moved = (levels - levels.loc['neutral']) / spread
            axis.plot(range(len(keys)), moved.values, style, marker=mark,
                      markersize=4.2, markerfacecolor='white',
                      markeredgecolor=COLOUR[name], markeredgewidth=1.0,
                      linewidth=1.7 * weight, color=COLOUR[name],
                      alpha=weight, zorder=3,
                      label=label if index == 0 else None)

        panel(axis, name, points)
        axis.set_facecolor('none')
        axis.set_xticks(range(len(keys)), names)
        axis.set_xlim(-0.5, 4.5)
        axis.margins(y=0.16)

    outer_labels(figure, axes, '', 'Movement From Control (SD)', points)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
                  bbox_to_anchor=(0.5, -0.03), fontsize=points * 0.9)
    name = ('readability_signals.pdf' if kind is None
            else f'readability_signals_{SLUG[kind]}.pdf')
    return save(figure, name)


# ----------------------------------------------------------------------------
# What the floor removed
# ----------------------------------------------------------------------------

def draw_coverage(frame, floor, display):
    points = styled(display)
    frame = frame.assign(short=frame['response_length'] < floor)

    blues = colormaps['Blues']
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), sharex=True,
                                sharey=True, constrained_layout=True)
    for index, name in enumerate(ORDER):
        axis = axes[index // 3][index % 3]
        part = frame[frame['label'] == name]
        for kind in TYPES:
            style, mark, depth = TYPE_STYLE[kind]
            loss = (part[part['scenario_type'] == kind]
                    .groupby('age')['short'].mean().reindex(LADDER) * 100)
            # Plotted at the real ages, so the one year between seventeen and
            # eighteen is drawn as one year and not as the two that separate
            # every other pair on the ladder.
            axis.plot(LADDER, loss.values, style, marker=mark, markersize=4.0,
                      linewidth=1.6, color=blues(depth),
                      label=kind if index == 0 else None)
        panel(axis, name, points)
        axis.set_xticks(LADDER)
        axis.set_xlim(6.4, 21.6)

    outer_labels(figure, axes, 'Age',
                 f'Below the {floor} Word Floor (\\%)', points)
    handles, names = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, names, loc='lower center', ncol=4, frameon=False,
                  bbox_to_anchor=(0.5, -0.03), fontsize=points * 0.9)
    return save(figure, 'readability_coverage.pdf')


def draw_coverage_grid(frame, floor, display):
    points = styled(display, 9.6)
    frame = frame.assign(short=frame['response_length'] < floor)
    ceiling = float(frame.groupby(['label', 'scenario_type', 'age'])['short']
                    .mean().max() * 100)
    blues = colormaps['Blues']

    figure, axes = plt.subplots(2, 3, figsize=(13.0, 8.0))
    figure.subplots_adjust(hspace=0.42, wspace=0.10, right=0.88, top=0.90,
                           bottom=0.14)
    for index, name in enumerate(ORDER):
        axis = axes[index // 3][index % 3]
        loss = (frame[frame['label'] == name]
                .pivot_table(index='scenario_type', columns='age',
                             values='short', aggfunc='mean')
                .reindex(index=TYPES, columns=LADDER) * 100)
        image = axis.imshow(loss.values, cmap='Blues', vmin=0, vmax=ceiling,
                            aspect='auto')
        axis.set_xticks(range(len(LADDER)),
                        [str(a) for a in LADDER] if index // 3 == 1 else [])
        axis.set_yticks(range(len(TYPES)), TYPES if index % 3 == 0 else [])
        axis.set_title(name, pad=points * 0.6, color='black')
        # No rules between the cells and no frame around them. A confusion
        # matrix reads as a continuous field of colour, and a white grid over it
        # turns each cell into an object to be counted rather than a value to be
        # compared against its neighbours.
        for side in axis.spines.values():
            side.set_visible(False)
        axis.tick_params(length=0)
        for row in range(len(TYPES)):
            for column in range(len(LADDER)):
                value = loss.values[row, column]
                if np.isfinite(value):
                    axis.text(column, row, f'{value:.0f}', ha='center',
                              va='center', fontsize=points * 0.6,
                              color='white' if value > 0.55 * ceiling
                              else 'black')

    figure.supxlabel('Age', color='black', fontsize=points, y=0.04)
    figure.supylabel('Scenario Type', color='black', fontsize=points)
    # The colourbar carries its scale in the ticks and its meaning in the
    # caption. A label written down its side competes with the shared axis
    # titles for the same edge of the figure.
    # Spanning both rows of panels, top to bottom, rather than shrunk to the
    # middle. The scale is shared by all six, so it should stand beside all six.
    top = axes[0][-1].get_position()
    bottom = axes[-1][-1].get_position()
    cax = figure.add_axes([top.x1 + 0.02, bottom.y0, 0.014,
                           top.y1 - bottom.y0])
    bar = figure.colorbar(image, cax=cax)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=points * 0.85, length=0, labelcolor='black')
    return save(figure, 'readability_coverage_grid.pdf')


# ----------------------------------------------------------------------------
# How far the fifteen measures repeat one another
# ----------------------------------------------------------------------------

# Define function to draw the correlation matrix as absolute value in blue
#
# Colour is the absolute value, because the question the figure answers is how
# far two measures repeat one another and that is a question about magnitude:
# FKGL and FRE at minus .93 duplicate each other exactly as thoroughly as two
# measures at plus .93 would. The sign stays in the printed number, where it can
# be read without being mistaken for strength.
#
# Lower triangle, since the matrix is symmetric and the exact figures are in
# Table G.7. The figure's job is the block structure, not every cell.
def draw_correlations(frame, display):
    points = styled(display, 10.5)
    columns = [key for key, _ in MEASURES if key in frame.columns]
    labels = [name for key, name in MEASURES if key in frame.columns]
    matrix = frame[columns].corr()

    # The whole square, not a triangle. The matrix is symmetric, so a triangle
    # holds the same numbers, but the three family blocks the figure exists to
    # show are squares, and half a square does not read as one.
    figure, axis = plt.subplots(figsize=(10.5, 9.0))
    image = axis.imshow(np.abs(matrix.values), cmap='Blues', vmin=0, vmax=1)
    blues = colormaps['Blues']
    axis.set_xticks(range(len(labels)), labels, rotation=90)
    axis.set_yticks(range(len(labels)), labels)
    # Contiguous cells, no frame, no grid, as a confusion matrix has. The only
    # rules drawn are the two that separate the measure families, which carry
    # information the colour cannot.
    for side in axis.spines.values():
        side.set_visible(False)
    axis.tick_params(length=0)

    # Separators after each family, so the readability block reads as a block.
    edge = 0
    for _, group in MEASURE_GROUPS[:-1]:
        edge += len(group)
        axis.axhline(edge - 0.5, color='white', linewidth=2.2)
        axis.axvline(edge - 0.5, color='white', linewidth=2.2)

    for row in range(len(labels)):
        for column in range(len(labels)):
            value = matrix.values[row, column]
            text = f'{value:.2f}'.replace('0.', '.').replace('-.', '$-$.')
            axis.text(column, row, text, ha='center', va='center',
                      fontsize=points * 0.58,
                      color=readable_on(blues(abs(value))))

    # The colourbar is made the exact height of the matrix rather than left to
    # shrink, so the scale reads against the cells it belongs to. A bar that
    # spans half the plot invites a reader to match a cell against the part of
    # the ramp beside it, which is not what it means.
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axis)
    cax = divider.append_axes('right', size='3.5%', pad=0.18)
    bar = figure.colorbar(image, cax=cax)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=points * 0.85, length=0, labelcolor='black')
    figure.tight_layout()
    return save(figure, 'readability_correlations.pdf')


# ----------------------------------------------------------------------------

def main(arguments):
    raw = language.load()
    raw['label'] = raw['model'].map(NAME)
    frame = raw.copy()
    short = frame['response_length'] < arguments.floor
    frame.loc[short, ['fkgl', 'fre', 'gunning_fog', 'ari', 'smog']] = np.nan
    stated = frame[frame['signal'].eq('stated')]
    d = arguments.display

    print(f'{len(raw):,} replies, {int(short.sum()):,} below the '
          f'{arguments.floor} word floor\n')

    figures = {'ladder': (lambda: draw_ladder(stated, d), 'main'),
               'distribution': (lambda: draw_distribution(stated, d), 'main'),
               'signals': (lambda: draw_signals(frame, d), 'main'),
               'coverage': (lambda: draw_coverage(raw, arguments.floor, d),
                            'main'),
               'correlations': (lambda: draw_correlations(frame, d), 'main')}
    for kind in TYPES:
        slug = SLUG[kind]
        figures[f'ladder_{slug}'] = (
            lambda k=kind: draw_ladder(stated[stated['scenario_type'] == k],
                                       d, k), 'supplement')
        figures[f'distribution_{slug}'] = (
            lambda k=kind: draw_distribution(
                stated[stated['scenario_type'] == k], d, k), 'supplement')
        figures[f'signals_{slug}'] = (
            lambda k=kind: draw_signals(frame[frame['scenario_type'] == k],
                                        d, k), 'supplement')
    figures['coverage_grid'] = (
        lambda: draw_coverage_grid(raw, arguments.floor, 1.0), 'supplement')

    for name, (build, tier) in figures.items():
        if arguments.only not in ('all', name):
            continue
        if arguments.set not in ('both', tier):
            continue
        try:
            build()
        except Exception as failure:
            print(f'  {name} FAILED, {type(failure).__name__}: {failure}')

    print(f'\nWritten to {FIGURES.relative_to(ROOT)}')
    print('Upload them to Overleaf: the tooling writes text only.')


def parser():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--floor', type=int, default=50,
                        help='word floor applied before the formulas are read')
    parser.add_argument('--display', type=float, default=1.0,
                        help='fraction of the text width a figure will be '
                             'included at, which sets the label sizes')
    parser.add_argument('--only', default='all',
                        help='draw one figure by name rather than all of them')
    parser.add_argument('--set', default='both',
                        choices=['both', 'main', 'supplement'],
                        help='draw only the main-text or only the supplement')
    return parser


if __name__ == '__main__':
    main(parser().parse_args())
