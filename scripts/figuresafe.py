"""Safety figures for the thesis.

python scripts/figuresafe.py
python scripts/figuresafe.py --set main
python scripts/figuresafe.py --only trajectory
python scripts/figuresafe.py --display 0.8

The readability counterpart is scripts/figureread.py and this file follows its
grammar: a styled() call that fixes the typography, a local save() that reports
what it wrote, one draw function a figure, a registry mapping a short name to a
builder and a tier, and a CLI that can run one figure or one tier. They differ
in where their data comes from. figureread.py reads the language table through
language.load() and computes its own reductions; the safety figures read the
classified corpus and the test register through analysis, so that a figure and
the table beside it in the thesis are produced by one code path and cannot
disagree.

Typography. A figure is drawn on a canvas wider than the text block and LaTeX
then shrinks it to \\textwidth, so type set at nine points on the canvas does not
arrive at nine points on the page. styled() undoes that: it divides the target
size by the scale LaTeX will apply, so every figure in the family lands at
LABEL_POINTS whatever canvas it was drawn on. width_inches is therefore the
figure's real canvas width and nothing else, and each draw function passes its
own figsize[0] rather than a nominal constant. The three canvas sizes below are
the whole geometry of this file; a figure that needs a fourth should add it
there rather than inventing a width in place.

The register is read from tables/machine/register_safety.csv rather than
rebuilt, so this script must run after the notebook. It says so if the file is
absent.

Rates are scenario weighted throughout, through analysis.by_scenario and
analysis.rate_by_model, which is the reducer every rate in Section 4.2 uses.
Averaging the returned rows directly is not the same thing wherever a provider
blocked, and a figure that did so disagreed with Table 4.3 on the one model that
blocks.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis
from analysis import (CELL_ORDER, COLOUR, FOCUS, INK, LINEWIDTH, MACRO, MARKER,
                      MARKERSIZE, MUTED, ORDER, PALE, STATED, STATED_ADULT,
                      STATED_AGE, STATED_MINOR, STRATA, THRESHOLD,
                      IMPLICIT_MINOR, SIGNAL, THRESHOLD_CONTRAST, TRAJECTORY,
                      by_scenario, rate_by_model, save_figure, write_captions)
from settings import ROOT

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

FIGURES = ROOT / "figures"
REGISTER_PATH = analysis.MACHINE / "register_safety.csv"

# The text block of the thesis, from the geometry options in style/preamble.tex.
# Every figure is included at \textwidth, so this is the width each one is
# scaled to and the denominator of the typography correction.
TEXT_WIDTH_CM = 16.0
TEXT_WIDTH_PT = TEXT_WIDTH_CM / 2.54 * 72.0

# The size body type should arrive at on the page, matching figureread.py.
LABEL_POINTS = 11

# The canvas sizes this family draws on. Three, chosen once, rather than a width
# invented per figure: six ad hoc canvases gave six different rendered type
# sizes, from 4.8 to 9.4 points, because nothing compensated for the scaling.
WIDE_SIZE = (10.5, 5.6)      # panels side by side, or a two by two grid
COMPACT_SIZE = (8.5, 5.2)    # a single axis

# Sizes that are not body type, as multiples of the base. Kept here so that the
# hierarchy is one list rather than a scatter of literals, and so that a figure
# cannot quietly set eight points on a canvas that will be halved.
ANNOTATION = 0.85
VALUE = 0.80
SUPTITLE = 1.00

# Legends sit outside the axes but inside the figure. The constrained layout
# engine reserves the strip for them, which an anchored axes legend does not do
# and which is why these used to be clipped off the bottom of the canvas.
LEGEND_BELOW = dict(loc="outside lower center", frameon=False,
                    handlelength=2.2, handletextpad=0.7, columnspacing=1.5)

MONO = {}
AGES = [STATED_AGE[name] for name in STATED]

# The boolean column behind each outcome cell, named as
# notebooks/15_safety.ipynb names it, so that a figure of the outcome and
# Table 4.3 read the same columns through the same reducer.
CELL_COLUMNS = {cell: cell.lower().replace(" ", "_") for cell in CELL_ORDER}

# The three primary contrasts, in the order Table 4.8 reports them. Named here
# rather than imported so that this file states what it draws, and so that a
# contrast added to the notebook does not silently appear in a figure.
PRIMARY = [TRAJECTORY, THRESHOLD_CONTRAST, SIGNAL]

# What each of those panels is titled. A contrast name runs to forty-eight
# characters and a panel is a third of the text block, so the full name only
# ever collided with its neighbours. The names themselves are in the caption.
# Set on two lines: a panel is about 205 points wide once the model names are
# allowed for, and one line of this at body size is not.
PANEL = {
    TRAJECTORY: "(a) Minor Ages\nvs Adult Ages",
    THRESHOLD_CONTRAST: "(b) Age 17\nvs Age 18",
    SIGNAL: "(c) Minor (Age)\nvs Minor (Cue)",
}

# Per-stratum axis limits for the four-panel trajectory. Benign and Rights sit
# under nine per cent and Age Restricted and Harmful run the full range, so a
# shared axis would draw two flat lines against two curves and say nothing about
# either. The scales are fixed here rather than left to matplotlib so that the
# supplement figure does not rescale when the corpus is regenerated.
SCALE = {"Benign": (-0.5, 9.0), "Rights": (-0.5, 9.0),
         "Age Restricted": (-6.0, 106.0), "Harmful": (-6.0, 106.0)}
TICKS = {"Benign": range(0, 9, 2), "Rights": range(0, 9, 2),
         "Age Restricted": range(0, 101, 20), "Harmful": range(0, 101, 20)}

# Label offsets for the mismatch scatter, as multiples of the base type size
# rather than as absolute points, so that they hold their proportion when the
# canvas changes. Six points in a small axis collide on at least two pairs
# whatever the data, so these are set by hand and are the one place in this file
# that would need revisiting if a model's position moved substantially.
MISMATCH_OFFSET = {
    "GPT-5.6 Luna": (0.9, 0.6), "Claude Haiku 4.5": (0.9, -1.2),
    "Gemini 3.5 Flash Lite": (0.9, 0.6), "DeepSeek-V4 Flash": (0.9, 0.6),
    "Mistral Small 4": (-0.9, 0.9), "Gemma 4 31B": (0.9, -1.2),
}

OUTCOME_FILLS = [INK, MUTED, PALE, "0.93"]

# The narrowest bar segment that can carry its own value at body size. Below it
# the label overhangs the segment and collides with its neighbour, so the
# smaller cells are read from Table 4.3 instead, which sits beside the figure.
VALUE_FLOOR = 5.0

def display_label(value):
    """Canonical display spelling for figure labels."""
    return "Macro-Average" if value == MACRO else value
PANEL_FILL = "#F5F5F5"
GRID_COLOUR = "#D7DCE2"
SPINE_COLOUR = "#BFC5CC"


# ---------------------------------------------------------------------
# Typography and output
# ---------------------------------------------------------------------

# Define function to set the type for one figure against the width it is drawn.
#
# scale is the factor LaTeX will apply when the exported PDF is included at
# \textwidth, so dividing the target size by it means the type arrives at
# LABEL_POINTS. width_inches must be the figure's real canvas width: passing a
# nominal value instead makes the correction wrong by whatever the two differ
# by, and the whole point of the correction is that it is exact.
def styled(display, width_inches, label_points=None):
    scale = display * TEXT_WIDTH_CM / (width_inches * 2.54)
    points = (LABEL_POINTS if label_points is None else label_points) / scale
    plt.rcParams.update({
        # A tight bounding box grows the file around anything that overhangs
        # the axes, and inflated type overhangs a lot, so the written PDF came
        # out wider than the canvas and LaTeX then shrank it further than
        # styled() had assumed. Writing the canvas itself keeps the arithmetic
        # exact; the constrained layout below is what stops that leaving
        # whitespace or clipping a legend.
        "savefig.bbox": "standard",
        "font.size": points,
        "axes.labelsize": points,
        "axes.titlesize": points * 1.05,
        "xtick.labelsize": points * 0.95,
        "ytick.labelsize": points * 0.95,
        "legend.fontsize": points * 0.95,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": "black",
        "axes.linewidth": 0.7,
        "text.color": "black",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
    })
    return points


# Define function to read the width of a written PDF, in points.
#
# savefig crops to the tight bounding box, so the file is not the figsize it was
# asked for and the true reduction is not the one styled() assumed. Reading the
# box back is what turns the report below into a check rather than a restatement
# of the intention.
def written_width(path):
    box = re.search(rb"/MediaBox\s*\[([^\]]*)\]", path.read_bytes())
    if box is None:
        return None
    edges = [float(value) for value in box.group(1).split()]
    return edges[2] - edges[0]


# Define function to write one figure and report what it will look like.
#
# analysis.save_figure does the writing and the caption registration, so the
# missing-entry check still covers this file. What is added here is the line
# printed for each output: the canvas it was drawn on, the base size set on it,
# and the size that base will arrive at once LaTeX has scaled the file. That
# last column is the one to read. It should sit at LABEL_POINTS on every row,
# and a figure that reports five points is a figure nobody will be able to read.
def save(figure, name, points):
    width_inches = figure.get_figwidth()
    path = save_figure(figure, name)
    plt.close(figure)

    width_points = written_width(path)
    if width_points is None:
        arrival = "unknown"
    else:
        arrival = f"{points * TEXT_WIDTH_PT / width_points:5.1f} pt"
    print(f"  {path.name:<30} {width_inches:5.1f} in  "
          f"{points:5.1f} pt set  {arrival} on the page")
    return path


# Define function to render a contrast name as the thesis writes it.
#
# The register stores a contrast as its machine key, which reads "vs". Every
# table and every sentence in the thesis writes "against". The key is not
# rewritten, since tables/machine is the source of truth and other code joins on
# it; only the drawn label is.
def phrase(name):
    return name.replace(" vs ", " against ")


def panel(axes, grid_axis="x"):
    """Shared publication-style panel used by all safety figures."""
    axes.set_facecolor(PANEL_FILL)
    axes.grid(axis=grid_axis, linestyle="-", linewidth=0.6,
              alpha=0.32, color=GRID_COLOUR)
    axes.set_axisbelow(True)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_color(SPINE_COLOUR)
    axes.spines["bottom"].set_color(SPINE_COLOUR)
    axes.tick_params(color=SPINE_COLOUR)



# ---------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------

# Define function to give the refusal rate at each stated age for one model,
# scenario weighted through by_scenario.
def ladder(panel):
    return [by_scenario(panel, "refusal", [name]).mean() * 100
            for name in STATED]


# Define function to draw one forest panel of model effects with intervals.
#
# Used by both the primary contrast figure and the cue figure, which differ only
# in how many panels they carry. The macro-average is drawn in ink with a
# diamond and separated by a rule, because it is a summary of the six rows above
# it and not a seventh model.
def forest(axes, part, rows, positions):
    """Draw one compact model-level forest panel."""
    for position, model in zip(positions, rows):
        if model not in part.index:
            continue

        colour = INK if model == MACRO else COLOUR[model]
        effect = float(part.at[model, "effect"])
        low = float(part.at[model, "low"])
        high = float(part.at[model, "high"])

        axes.plot(
            [low, high], [position, position],
            color=colour,
            linewidth=LINEWIDTH,
            solid_capstyle="butt",
            zorder=2,
        )
        axes.plot(
            effect, position,
            marker="D" if model == MACRO else MARKER[model],
            color=colour,
            markersize=MARKERSIZE + 0.5,
            linestyle="none",
            zorder=3,
        )

    axes.axvline(
        0, color=MUTED, linewidth=1.0,
        linestyle="--", alpha=0.70, zorder=1
    )

    if MACRO in rows:
        axes.axhline(0.5, color="0.80", linewidth=0.8)

    axes.set_yticks(positions)
    axes.set_yticklabels([display_label(row) for row in rows])
    axes.set_ylim(-0.7, len(rows) - 0.3)
    axes.set_xlabel("Difference in refusal rate (pp)")
    axes.grid(axis="y", visible=False)
    panel(axes, grid_axis="x")


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def draw_trajectory(focus, display):
    points = styled(display, COMPACT_SIZE[0])
    figure, axes = plt.subplots(figsize=COMPACT_SIZE, layout="constrained")

    for label in ORDER:
        axes.plot(AGES, ladder(focus[focus["label"] == label]),
                  marker=MARKER[label], markersize=MARKERSIZE,
                  linewidth=LINEWIDTH, color=COLOUR[label], label=label)

    axes.axvline(THRESHOLD, color=PALE, linewidth=1.0, linestyle="--")
    axes.annotate("Age 18 threshold", (THRESHOLD, 96),
                  fontsize=points * ANNOTATION, color=MUTED, ha="right",
                  xytext=(-4, 0), textcoords="offset points")
    axes.set_xticks(AGES)
    axes.set_xlim(6.0, 22.0)
    axes.set_ylim(-6, 106)
    axes.set_yticks(range(0, 101, 20))
    axes.set_xlabel("Age")
    axes.set_ylabel("Refusal rate (%)")
    axes.set_title("Age Restricted scenarios", loc="left")
    panel(axes, grid_axis="y")
    figure.legend(ncol=3, fontsize=points * 0.95,
                  **LEGEND_BELOW)
    return save(figure, "safety_trajectory", points)


def draw_trajectory_all(returned, display):
    # Two by two rather than one by four. At body size a quarter of the text
    # block is about 113 points across, which will not hold a stratum title, a
    # y-axis label and eight tick positions, and the six-model legend is wider
    # than the block on one line whatever the canvas. The pairing is not
    # arbitrary: the two rows are the two axis scales, so a row is read across
    # without rescaling.
    points = styled(display, WIDE_SIZE[0])
    figure, panels = plt.subplots(2, 2, figsize=WIDE_SIZE, layout="constrained")

    for axes, stratum, letter in zip(panels.flat, STRATA, "abcd"):
        part = returned[returned["scenario_type"].eq(stratum)]
        for label in ORDER:
            axes.plot(AGES, ladder(part[part["label"] == label]),
                      marker=MARKER[label], markersize=MARKERSIZE - 0.7,
                      linewidth=LINEWIDTH - 0.3, color=COLOUR[label],
                      label=label)
        if stratum == FOCUS:
            axes.axvline(THRESHOLD, color=PALE, linewidth=1.0, linestyle="--")
        axes.set_xticks(AGES)
        axes.set_xticklabels([str(age) if age in (7, 11, 15, 18, 21) else ""
                              for age in AGES])
        axes.set_xlim(6.0, 22.0)
        axes.set_ylim(*SCALE[stratum])
        axes.set_yticks(list(TICKS[stratum]))
        axes.set_title(f"({letter}) {stratum}", loc="left")
        panel(axes, grid_axis="y")

    # One label a side rather than one a panel, as figureread.py does on its
    # grids. The axis text is identical in all four, so four copies of it spend
    # a quarter of the figure saying the same thing.
    figure.supxlabel("Age", fontsize=points * 1.08)
    figure.supylabel("Refusal rate (%)", fontsize=points * 1.15)

    # Below body size, and the one place in this file that is. Six released
    # names on three columns is 795 points at 0.95 against a 756 point canvas,
    # and the alternative, two columns, spends a fifth of the figure height on
    # a legend.
    handles, labels = panels.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3,
                  fontsize=points * 0.80, frameon=False)
    return save(figure, "safety_trajectory_all", points)


def draw_primary(register, display):
    points = styled(display, WIDE_SIZE[0])
    figure, panels = plt.subplots(1, 3, figsize=WIDE_SIZE, sharey=True,
                                  sharex=True, layout="constrained")
    rows = [MACRO] + ORDER[::-1]
    positions = np.arange(len(rows))
    primary = register[register["contrast"].isin(PRIMARY)]
    low = min(0.0, float(primary["low"].min()))
    high = float(primary["high"].max())
    span = high - low

    for axes, name in zip(panels, PRIMARY):
        forest(axes, register[register["contrast"] == name].set_index("model"),
               rows, positions)
        axes.set_xlim(low - 0.06 * span, high + 0.06 * span)
        axes.set_title(PANEL[name], loc="left")
        axes.set_xlabel("")

    figure.supxlabel("Difference in refusal rate (pp)", fontsize=points * 1.08)
    figure.suptitle(
        "Age Restricted scenarios",
        x=0.015, ha="left", fontsize=points * SUPTITLE)
    return save(figure, "safety_primary", points)


def draw_outcome(returned, display):
    # Scenario weighted, through the same rate_by_model every rate in the
    # chapter goes through: replicates are averaged within a scenario and
    # condition and the conditions are then averaged with equal weight.
    #
    # An earlier version took value_counts over the returned rows instead. That
    # weights a scenario by how many of its requests came back, which differs
    # only where a provider blocked and differed there: it drew Gemini 3.5
    # Flash Lite at 21.7 per cent Strong Refusal against the 21.9 of
    # Table 4.3, so a figure and the table it is read beside disagreed on the
    # one model whose blocking the design already treats as a qualification.
    share = pd.DataFrame({cell: rate_by_model(returned, CELL_COLUMNS[cell])
                          for cell in CELL_ORDER}).reindex(ORDER)

    points = styled(display, COMPACT_SIZE[0])
    figure, axes = plt.subplots(figsize=COMPACT_SIZE, layout="constrained")
    left = np.zeros(len(ORDER))
    for cellname, fill in zip(CELL_ORDER, OUTCOME_FILLS):
        axes.barh(np.arange(len(ORDER)), share[cellname], left=left,
                  height=0.60, color=fill, edgecolor=INK, linewidth=0.5,
                  label=cellname)
        for position, (value, start) in enumerate(zip(share[cellname], left)):
            if value >= VALUE_FLOOR:
                axes.text(start + value / 2, position, f"{value:.1f}",
                          ha="center", va="center", fontsize=points * VALUE,
                          color="white" if fill in (INK, MUTED) else INK)
        left = left + share[cellname].to_numpy()

    axes.set_yticks(np.arange(len(ORDER)))
    axes.set_yticklabels(ORDER)
    axes.invert_yaxis()
    axes.set_xlim(0, 104)
    axes.set_xticks(range(0, 101, 20))
    axes.set_ylim(len(ORDER) - 0.4, -0.6)
    axes.set_xlabel("Share of returned replies (%)")
    axes.set_title("Outcome distribution", loc="left")
    axes.grid(axis="y", visible=False)
    panel(axes, grid_axis="x")
    figure.legend(ncol=2, fontsize=points * 0.95,
                  **LEGEND_BELOW)
    return save(figure, "safety_outcome", points)


def draw_cues(register, display):
    points = styled(display, COMPACT_SIZE[0])
    cues = register[register["family"] == "implicit cue"].set_index("model")
    rows = [MACRO] + ORDER[::-1] if MACRO in cues.index else ORDER[::-1]
    positions = np.arange(len(rows))
    span = float(cues["high"].max())

    figure, axes = plt.subplots(figsize=COMPACT_SIZE, layout="constrained")
    forest(axes, cues, rows, positions)
    axes.set_xlim(-0.06 * span, span * 1.10)
    axes.set_title("Minor (Cue) vs Adult (Cue)", loc="left")
    return save(figure, "safety_cues", points)


def draw_mismatch(returned, display):
    # Each axis carries its own range. Forcing them equal, which an earlier
    # version did to make a diagonal meaningful, put five of six models into one
    # corner: Weak Refusal never exceeds half a point while Minimal Compliance
    # reaches seventeen, so a shared scale is a shared scale with nothing on
    # most of it.
    #
    # Both axes are scenario weighted for the reason given in draw_outcome, so
    # a point here sits at the value Table 4.3 prints for that model.
    mismatch = pd.DataFrame({
        "minimal": rate_by_model(returned, CELL_COLUMNS["Minimal Compliance"]),
        "weak": rate_by_model(returned, CELL_COLUMNS["Weak Refusal"])}
    ).reindex(ORDER)

    points = styled(display, COMPACT_SIZE[0])
    figure, axes = plt.subplots(figsize=COMPACT_SIZE, layout="constrained")
    for label in ORDER:
        offset = MISMATCH_OFFSET[label]
        axes.plot(mismatch.at[label, "minimal"], mismatch.at[label, "weak"],
                  marker=MARKER[label], markersize=MARKERSIZE + 3.0,
                  color=COLOUR[label], linestyle="none")
        axes.annotate(label, (mismatch.at[label, "minimal"],
                              mismatch.at[label, "weak"]),
                      textcoords="offset points",
                      xytext=(offset[0] * points, offset[1] * points),
                      fontsize=points * ANNOTATION, color=INK,
                      ha="right" if label == "Mistral Small 4" else "left",
                      )

    axes.set_xlim(-1.2, float(mismatch["minimal"].max()) * 1.22)
    axes.set_ylim(-0.06, float(mismatch["weak"].max()) * 1.35)
    axes.set_xlabel("Minimal Compliance: agreed, supplied nothing (%)")
    axes.set_ylabel("Weak Refusal: declined, supplied anyway (%)")
    axes.set_title("Decision and delivery mismatches", loc="left")
    panel(axes, grid_axis="both")
    return save(figure, "safety_mismatch", points)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def registry(returned, focus, register, display):
    return {
        "trajectory": (lambda: draw_trajectory(focus, display), "main"),
        "primary": (lambda: draw_primary(register, display), "main"),
        "outcome": (lambda: draw_outcome(returned, display), "main"),
        "cues": (lambda: draw_cues(register, display), "main"),
        "mismatch": (lambda: draw_mismatch(returned, display), "main"),
        "trajectory_all": (lambda: draw_trajectory_all(returned, display),
                           "supplement"),
    }


def load_register():
    if not REGISTER_PATH.exists():
        raise SystemExit(
            f"{REGISTER_PATH.relative_to(ROOT)} is missing. Run "
            f"notebooks/15_safety.ipynb first: the intervals drawn here are "
            f"the ones it registered, and recomputing them separately would "
            f"let a figure and Table 4.8 disagree.")
    return pd.read_csv(REGISTER_PATH)


def main(args):
    mpl.rcParams.update(analysis.STYLE)
    FIGURES.mkdir(parents=True, exist_ok=True)

    frame = analysis.load_corpus()
    returned = frame[frame["responded"]]
    focus = returned[returned["scenario_type"].eq(FOCUS)]
    register = load_register()

    counts = frame.attrs["fingerprint"]
    print(f"{counts['requests']:,} requests, {counts['blocked']:,} blocked, "
          f"{counts['returned']:,} returned, policy {counts['policy']}\n")

    for name, (build, tier) in registry(returned, focus, register,
                                        args.display).items():
        if args.only not in {"all", name} or args.set not in {"both", tier}:
            continue
        try:
            build()
        except Exception as exc:
            print(f"  {name} FAILED, {type(exc).__name__}: {exc}")

    # Figures and tables are described in one config file and written by two
    # programs, so the missing-entry check is scoped to the kind this run
    # produced. Without the scope every run would report the other program's
    # outputs as described but not written.
    write_captions(kind="figure")
    print(f"\nWritten to {FIGURES.relative_to(ROOT)}, "
          f"body type at {LABEL_POINTS} points")


def parser():
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--display", type=float, default=1.0)
    cli.add_argument("--only", default="all")
    cli.add_argument("--set", default="both",
                     choices=["both", "main", "supplement"])
    return cli


if __name__ == "__main__":
    main(parser().parse_args())
