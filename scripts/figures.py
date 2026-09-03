"""Readability figures for the thesis.

python scripts/figures.py
python scripts/figures.py --set main
python scripts/figures.py --only ladder
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import language
from analysis import COLOUR, INK, MARKER, MUTED, NAME, ORDER, PALE, bootstrap_paired
from settings import ROOT


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

FIGURES = ROOT / "figures"
TEXT_WIDTH_CM = 16.0
LABEL_POINTS = 11
GRID_SIZE = (13.5, 7.8)

LADDER = [7, 9, 11, 13, 15, 17, 18, 21]
MINOR = [7, 9, 11, 13, 15, 17]
ADULT = [18, 21]
TYPES = ["Harmful", "Age Restricted", "Rights", "Benign"]
SLUG = {x: x.lower().replace(" ", "_") for x in TYPES}
TARGET = {age: min(age - 5, 12) for age in MINOR}

SIGNALS = [
    ("neutral", "Neutral", ""),
    ("adult_cue", "Adult", "(Cue)"),
    ("adult_age", "Adult", "(Age)"),
    ("minor_cue", "Minor", "(Cue)"),
    ("minor_age", "Minor", "(Age)"),
]
SIGNAL_TRACKS = [
    ("fkgl", "FKGL", "-", "o", 1.00),
    ("mean_aoa", "Mean AoA", "--", "^", 0.85),
    ("p90_aoa", "P90 AoA", ":", "s", 0.70),
]
TYPE_STYLE = {
    "Harmful": ("-", "o", 0.92),
    "Age Restricted": ("--", "s", 0.72),
    "Rights": (":", "^", 0.52),
    "Benign": ("-.", "D", 0.34),
}

MEASURE_GROUPS = [
    ("Readability", [("fkgl", "FKGL"), ("fre", "FRE"), ("gunning_fog", "Gunning Fog"),
                     ("ari", "ARI"), ("smog", "SMOG")]),
    ("Vocabulary", [("mean_aoa", "Mean AoA"), ("p90_aoa", "P90 AoA"),
                    ("max_aoa", "Max AoA"), ("difficult_share", "Difficult Share"),
                    ("aoa_coverage", "AoA Coverage")]),
    ("Structure", [("response_length", "Response Length"),
                   ("sentence_length", "Sentence Length"), ("word_length", "Word Length"),
                   ("ttr", "TTR"), ("mtld", "MTLD")]),
]
MEASURES = [x for _, group in MEASURE_GROUPS for x in group]

PANEL_FILL = "#F5F5F5"
MINOR_BAND = "#E8EDF2"
ADULT_BAND = "#F2EDE8"

FORMULAE = ["fkgl", "fre", "gunning_fog", "ari", "smog"]
LEGEND = dict(
    loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.075),
    handlelength=2.2, handletextpad=0.7, columnspacing=1.5
)


# ---------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------

def by_scenario(df, measure, keys):
    cell = df.groupby(keys + ["scenario_id", "condition"], observed=True)[measure].mean()
    scenario = cell.groupby(keys + ["scenario_id"], observed=True).mean()
    return scenario.groupby(keys, observed=True).mean()


def blocks(df, column="fkgl"):
    wide = (df.pivot_table(index="scenario_id", columns="age", values=column, aggfunc="mean")
              .reindex(columns=MINOR + ADULT).dropna())
    if wide.empty:
        empty = pd.Series(dtype=float)
        return empty, empty
    return wide[MINOR].mean(axis=1), wide[ADULT].mean(axis=1)


def contrast(df, column="fkgl"):
    minor, adult = blocks(df, column)
    if minor.empty:
        return np.nan, np.nan, np.nan, 0
    diff = minor - adult
    point, low, high = bootstrap_paired(diff)
    return point, low, high, len(diff)


# ---------------------------------------------------------------------
# Shared figure grammar
# ---------------------------------------------------------------------

def styled(display, width_inches=7.4, label_points=None):
    scale = display * TEXT_WIDTH_CM / (width_inches * 2.54)
    points = (LABEL_POINTS if label_points is None else label_points) / scale
    plt.rcParams.update({
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


def grid(figsize=GRID_SIZE, **kwargs):
    return plt.subplots(2, 3, figsize=figsize, **kwargs)


def panel(ax, title=None, points=9):
    ax.set_facecolor(PANEL_FILL)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.25, color=MUTED)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, pad=points * 0.5, color="black")


def outer_labels(fig, axes, xlabel, ylabel, points):
    for ax in axes.flat:
        ax.label_outer()
    if xlabel:
        fig.supxlabel(xlabel, color="black", fontsize=points * 1.08)
    if ylabel:
        fig.supylabel(ylabel, color="black", fontsize=points * 1.15)


def legend(fig, handles, labels, points, ncol):
    fig.legend(handles, labels, ncol=ncol, fontsize=points * 1.05, **LEGEND)


def filename(prefix, kind):
    return f"{prefix}.pdf" if kind is None else f"{prefix}_{SLUG[kind]}.pdf"


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")
    return path


def annotate_contrast(ax, point, low, high):
    if not np.isfinite(point):
        return
    text = rf"$\Delta = {point:+.2f}$" + "\n" + rf"$[{low:+.2f}, {high:+.2f}]$"
    ax.text(
        0.97, 0.95, text, transform=ax.transAxes, ha="right", va="top",
        fontsize=plt.rcParams["font.size"] * 0.90, linespacing=1.25,
        bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=2)
    )


def age_bands(ax):
    ax.axvspan(6.4, 17.5, color=MINOR_BAND, zorder=0)
    ax.axvspan(17.5, 21.6, color=ADULT_BAND, zorder=0)


def readable_on(rgba):
    r, g, b = rgba[:3]
    return "white" if 0.2126*r + 0.7152*g + 0.0722*b < 0.42 else "black"


# ---------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------

def ladder_panel(ax, df, model, points, title=None, small=False):
    values = by_scenario(df, "fkgl", ["age"]).reindex(LADDER)
    age_bands(ax)

    ax.axvline(17.5, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.65, zorder=1)
    ax.plot(list(TARGET), list(TARGET.values()), ":", color=MUTED,
            linewidth=1.2 if small else 1.4, alpha=0.75, zorder=2)
    ax.plot(
        LADDER, values.values, marker=MARKER[model],
        markersize=3.5 if small else 4.8, markerfacecolor="white",
        markeredgecolor=COLOUR[model], markeredgewidth=1.1,
        linewidth=1.5 if small else 1.9, color=COLOUR[model], zorder=3
    )

    # Representative ages: child, adolescent, pre-threshold, adult threshold.
    for age, offset in {7: (0, 7), 13: (0, 7), 17: (-7, 7), 18: (7, 7)}.items():
        value = values.loc[age]
        if np.isfinite(value):
            ax.annotate(
                f"{value:.1f}", (age, value), xytext=offset,
                textcoords="offset points", ha="center", va="bottom",
                fontsize=points * 0.72, color=COLOUR[model], zorder=4
            )

    panel(ax, title, points)
    ax.set_xticks(LADDER)
    ax.set_xlim(6.4, 21.6)
    ax.yaxis.set_major_locator(MultipleLocator(1))


def draw_ladder(frame, display, kind=None):
    points = styled(display)
    fig, axes = grid(figsize=(13.5, 7.6), sharex=True, sharey=True,
                     constrained_layout=True)

    for ax, model in zip(axes.flat, ORDER):
        part = frame[frame["label"] == model]
        ladder_panel(ax, part, model, points, model)
        annotate_contrast(ax, *contrast(part)[:3])

    outer_labels(fig, axes, "Age", "Flesch-Kincaid Grade Level", points)
    legend(
        fig,
        [Patch(facecolor=MINOR_BAND), Patch(facecolor=ADULT_BAND),
         Line2D([0], [0], color=MUTED, linestyle=":", linewidth=1.5)],
        ["Minor", "Adult", "Target"], points, 3
    )
    return save(fig, filename("readability_ladder", kind))


# ---------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------

def overlap(minor, adult, edges):
    if minor.empty or adult.empty:
        return np.nan
    a, _ = np.histogram(minor, bins=edges, density=True)
    b, _ = np.histogram(adult, bins=edges, density=True)
    return float(np.minimum(a, b).sum() * np.diff(edges)[0])


def draw_distribution(frame, display, kind=None):
    points = styled(display)
    edges = np.linspace(2, 14, 19)
    fig, axes = grid(figsize=(13.5, 7.6), sharex=True, sharey=True,
                     constrained_layout=True)

    suffix = "" if kind is None else f", {kind}"
    print(f"  overlap, minor against adult{suffix}:")

    for ax, model in zip(axes.flat, ORDER):
        part = frame[frame["label"] == model]
        minor, adult = blocks(part)

        ax.hist(minor, bins=edges, density=True, alpha=0.40,
                facecolor=COLOUR[model], edgecolor=COLOUR[model], linewidth=1.2)
        ax.hist(adult, bins=edges, density=True, histtype="step",
                linestyle="--", linewidth=2.0, color=COLOUR[model])
        panel(ax, model, points)
        annotate_contrast(ax, *contrast(part)[:3])
        print(f"    {model:<24}{overlap(minor, adult, edges):.0%}")

    outer_labels(fig, axes, "Flesch-Kincaid Grade Level", "Density", points)
    legend(
        fig,
        [Patch(facecolor=PALE, alpha=0.5, edgecolor=MUTED, linewidth=1.1),
         Line2D([0], [0], color=MUTED, linewidth=2.0, linestyle="--")],
        ["Minor", "Adult"], points, 2
    )
    return save(fig, filename("readability_distribution", kind))


# ---------------------------------------------------------------------
# Signal strength
# ---------------------------------------------------------------------

def signal_level(frame):
    level = pd.Series("", index=frame.index, dtype=object)
    cue = frame["signal"].eq("cue")
    condition = frame["condition"].astype(str)

    level[frame["condition"].eq("neutral")] = "neutral"
    level[cue & condition.str.contains("adult", na=False)] = "adult_cue"
    level[cue & condition.str.contains("minor", na=False)] = "minor_cue"
    level[frame["age"].isin(ADULT)] = "adult_age"
    level[frame["age"].isin(MINOR)] = "minor_age"
    return level


def draw_signals(frame, display, kind=None):
    points = styled(display)
    frame = frame.assign(level=signal_level(frame))
    frame = frame[frame["level"].ne("")]

    keys = [key for key, _, _ in SIGNALS]
    labels = [top if not sub else f"{top}\n{sub}" for _, top, sub in SIGNALS]
    fig, axes = grid(sharex=True, sharey=True, constrained_layout=True)

    for index, (ax, model) in enumerate(zip(axes.flat, ORDER)):
        part = frame[frame["label"] == model]
        ax.axhline(0, color=INK, linewidth=1.0, alpha=0.75, zorder=2)

        for track, (measure, name, style, marker, weight) in enumerate(SIGNAL_TRACKS):
            levels = by_scenario(part, measure, ["level"]).reindex(keys)
            moved = (levels - levels.loc["neutral"]) / part[measure].std()

            ax.plot(
                range(len(keys)), moved.values, style, marker=marker,
                markersize=4.8, markerfacecolor="white",
                markeredgecolor=COLOUR[model], markeredgewidth=1.1,
                linewidth=1.9 * weight, color=COLOUR[model],
                alpha=max(weight, 0.78), zorder=3,
                label=name if index == 0 else None
            )

            # Exact values for the two minor-associated conditions.
            for x, xoff in ((3, -4), (4, 4)):
                value = moved.iloc[x]
                if np.isfinite(value):
                    ax.annotate(
                        f"{value:+.2f}", (x, value),
                        xytext=(xoff, (9, -12, 7)[track]),
                        textcoords="offset points", ha="center", va="center",
                        fontsize=points * 0.68, color=COLOUR[model], zorder=5
                    )

        panel(ax, model, points)
        ax.set_xticks(range(len(keys)), labels)
        ax.set_xlim(-0.5, 4.5)
        ax.margins(y=0.18)
        ax.yaxis.set_major_locator(MultipleLocator(0.25))

    outer_labels(fig, axes, "", "Change From Neutral (SD)", points)
    handles, names = axes.flat[0].get_legend_handles_labels()
    legend(fig, handles, names, points, 3)
    return save(fig, filename("readability_signals", kind))


# ---------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------

def draw_coverage(frame, floor, display):
    points = styled(display)
    frame = frame.assign(short=frame["response_length"] < floor)
    blues = colormaps["Blues"]
    fig, axes = grid(sharex=True, sharey=True, constrained_layout=True)

    for index, (ax, model) in enumerate(zip(axes.flat, ORDER)):
        part = frame[frame["label"] == model]
        for kind in TYPES:
            style, marker, depth = TYPE_STYLE[kind]
            loss = (part[part["scenario_type"] == kind]
                    .groupby("age")["short"].mean().reindex(LADDER) * 100)
            ax.plot(
                LADDER, loss.values, style, marker=marker,
                markersize=4.0, linewidth=1.6, color=blues(depth),
                label=kind if index == 0 else None
            )
        panel(ax, model, points)
        ax.set_xticks(LADDER)
        ax.set_xlim(6.4, 21.6)

    outer_labels(fig, axes, "Age", f"Below the {floor} Word Floor (\\%)", points)
    handles, names = axes.flat[0].get_legend_handles_labels()
    legend(fig, handles, names, points, 4)
    return save(fig, "readability_coverage.pdf")


def draw_coverage_grid(frame, floor, display):
    points = styled(display, 9.6, label_points=9.0)
    frame = frame.assign(short=frame["response_length"] < floor)
    ceiling = float(
        frame.groupby(["label", "scenario_type", "age"])["short"].mean().max() * 100
    )

    fig, axes = grid(figsize=(13.0, 8.0))
    fig.subplots_adjust(hspace=0.42, wspace=0.10, right=0.88, top=0.90, bottom=0.14)

    for index, (ax, model) in enumerate(zip(axes.flat, ORDER)):
        loss = (frame[frame["label"] == model]
                .pivot_table(index="scenario_type", columns="age",
                             values="short", aggfunc="mean")
                .reindex(index=TYPES, columns=LADDER) * 100)

        image = ax.imshow(loss.values, cmap="Blues", vmin=0, vmax=ceiling, aspect="auto")
        ax.set_xticks(range(len(LADDER)),
                      [str(x) for x in LADDER] if index // 3 == 1 else [])
        ax.set_yticks(range(len(TYPES)), TYPES if index % 3 == 0 else [])
        ax.set_title(model, pad=points * 0.6, color="black")
        ax.tick_params(length=0)

        for spine in ax.spines.values():
            spine.set_visible(False)

        for row, col in np.ndindex(loss.shape):
            value = loss.iat[row, col]
            if np.isfinite(value):
                ax.text(
                    col, row, f"{value:.0f}", ha="center", va="center",
                    fontsize=points * 0.6,
                    color="white" if value > 0.55 * ceiling else "black"
                )

    fig.supxlabel("Age", color="black", fontsize=points, y=0.04)
    top, bottom = axes[0, -1].get_position(), axes[-1, -1].get_position()
    cax = fig.add_axes([top.x1 + 0.02, bottom.y0, 0.014, top.y1 - bottom.y0])
    bar = fig.colorbar(image, cax=cax)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=points * 0.85, length=0, labelcolor="black")
    return save(fig, "readability_coverage_grid.pdf")


# ---------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------

def draw_correlations(frame, display):
    points = styled(display, 10.5, label_points=9.0)
    columns = [key for key, _ in MEASURES if key in frame.columns]
    labels = [label for key, label in MEASURES if key in frame.columns]
    matrix = frame[columns].corr()
    blues = colormaps["Blues"]

    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    image = ax.imshow(np.abs(matrix.values), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    edge = 0
    for _, group in MEASURE_GROUPS[:-1]:
        edge += len(group)
        ax.axhline(edge - 0.5, color="white", linewidth=2.2)
        ax.axvline(edge - 0.5, color="white", linewidth=2.2)

    for row, col in np.ndindex(matrix.shape):
        value = matrix.iat[row, col]
        text = f"{value:.2f}".replace("0.", ".").replace("-.", "$-$.")
        ax.text(
            col, row, text, ha="center", va="center",
            fontsize=points * 0.58, color=readable_on(blues(abs(value)))
        )

    cax = make_axes_locatable(ax).append_axes("right", size="3.5%", pad=0.18)
    bar = fig.colorbar(image, cax=cax)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=points * 0.85, length=0, labelcolor="black")
    fig.tight_layout()
    return save(fig, "readability_correlations.pdf")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def registry(raw, frame, stated, floor, display):
    jobs = {
        "ladder": (lambda: draw_ladder(stated, display), "main"),
        "distribution": (lambda: draw_distribution(stated, display), "main"),
        "signals": (lambda: draw_signals(frame, display), "main"),
        "coverage": (lambda: draw_coverage(raw, floor, display), "main"),
        "correlations": (lambda: draw_correlations(frame, display), "main"),
    }

    for kind in TYPES:
        slug = SLUG[kind]
        s = stated[stated["scenario_type"] == kind]
        f = frame[frame["scenario_type"] == kind]
        jobs[f"ladder_{slug}"] = (lambda data=s, k=kind: draw_ladder(data, display, k),
                                  "supplement")
        jobs[f"distribution_{slug}"] = (
            lambda data=s, k=kind: draw_distribution(data, display, k), "supplement"
        )
        jobs[f"signals_{slug}"] = (
            lambda data=f, k=kind: draw_signals(data, display, k), "supplement"
        )

    jobs["coverage_grid"] = (
        lambda: draw_coverage_grid(raw, floor, 1.0), "supplement"
    )
    return jobs


def main(args):
    raw = language.load()
    raw["label"] = raw["model"].map(NAME)

    frame = raw.copy()
    short = frame["response_length"] < args.floor
    frame.loc[short, FORMULAE] = np.nan
    stated = frame[frame["signal"].eq("stated")]

    print(f"{len(raw):,} replies, {int(short.sum()):,} below the {args.floor} word floor\n")

    for name, (build, tier) in registry(raw, frame, stated, args.floor, args.display).items():
        if args.only not in {"all", name} or args.set not in {"both", tier}:
            continue
        try:
            build()
        except Exception as exc:
            print(f"  {name} FAILED, {type(exc).__name__}: {exc}")

    print(f"\nWritten to {FIGURES.relative_to(ROOT)}")


def parser():
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--floor", type=int, default=50)
    cli.add_argument("--display", type=float, default=1.0)
    cli.add_argument("--only", default="all")
    cli.add_argument("--set", default="both", choices=["both", "main", "supplement"])
    return cli


if __name__ == "__main__":
    main(parser().parse_args())