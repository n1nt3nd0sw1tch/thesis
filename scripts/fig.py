"""Joint safety-readability figures for the thesis.

This script is project-native: it imports all paths, model colours, scenario
colours, markers and Matplotlib defaults from analysis.py. It reads only the
frozen tables written by notebook 17.

Main figures
------------
joint_association.pdf
    Pooled concordance versus within-type concordance for each model, with
    bootstrap confidence intervals and numeric coefficients.
joint_boundary_gap.pdf
    Difference between the refusal boundary share and the FKGL boundary share.

Supplementary figures
---------------------
joint_strata.pdf
    Scenario-stratified forest/grid for Rights, Age Restricted, Harmful and
    Within-Type concordance.
joint_concordance.pdf
    Pooled scenario-level concordance with bootstrap intervals.
joint_concordance_intervals.pdf
    Pooled, scenario-specific and within-type concordance with bootstrap
    intervals in a common 2 x 3 grid.

Examples
--------
python scripts/figurejoint.py
python scripts/figurejoint.py --set main
python scripts/figurejoint.py --only association
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis


# ---------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------

FIGURES = analysis.FIGURES / "joint"
ASSOCIATION_PATH = analysis.MAIN / "joint_04_association.csv"
SHAPE_PATH = analysis.MAIN / "joint_02_shape.csv"
CONCORDANCE_PATH = analysis.SUPPLEMENT / "joint_03_concordance.csv"

ORDER = list(analysis.ORDER)
MACRO = analysis.MACRO
COLOUR = analysis.COLOUR
MARKER = analysis.MARKER
SCENARIO_COLOUR = analysis.SCENARIO_COLOUR
INK = analysis.INK
MUTED = analysis.MUTED
PALE = analysis.PALE

TEXT_WIDTH_CM = 16.0
LABEL_POINTS = 11.0
PANEL_FILL = "#F5F5F5"

MAIN_ASSOCIATION_SIZE = (13.2, 6.1)
BOUNDARY_SIZE = (12.6, 5.7)
STRATA_SIZE = (16.0, 7.6)
POOLED_SIZE = (10.8, 5.3)
INTERVALS_SIZE = (16.0, 7.6)

MODEL_AXIS = {model: model for model in ORDER}
MODEL_AXIS[MACRO] = "Macro-Average"

STRATA_ORDER = ["Rights", "Age Restricted", "Harmful"]
STRATA_AXIS = {
    "Rights": "Rights",
    "Age Restricted": "Age Restricted",
    "Harmful": "Harmful",
    "Within-Type": "Within-Type",
}
INTERVAL_PANELS = [
    "Pooled",
    "Benign",
    "Rights",
    "Age Restricted",
    "Harmful",
    "Within-Type",
]

FIGURESPEC = {
    "association": (
        "main",
        "joint_association.pdf",
        "Pooled versus within-type concordance",
    ),
    "boundary_gap": (
        "main",
        "joint_boundary_gap.pdf",
        "Boundary concentration gap",
    ),
    "strata": (
        "supplement",
        "joint_strata.pdf",
        "Scenario-stratified concordance",
    ),
    "concordance": (
        "supplement",
        "joint_concordance.pdf",
        "Pooled scenario-level concordance",
    ),
    "concordance_intervals": (
        "supplement",
        "joint_concordance_intervals.pdf",
        "Pooled, scenario-specific and within-type concordance",
    ),
}

STALE = [
    "joint_ladders.pdf",
    "joint_progression.pdf",
    "joint_phases.pdf",
    "joint_boundary.pdf",
    "joint_boundary_concentration.pdf",
]


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


def panel(ax, title=None, points=9):
    ax.set_facecolor(PANEL_FILL)
    # Keep the exact panel grammar used by figureread.py and figuresafe.py.
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.25, color=MUTED)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, pad=points * 0.5, color="black")


def save(fig, filename):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")
    return path


def clean_stale():
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name in STALE:
        path = FIGURES / name
        if path.exists():
            path.unlink()


def ordered_rows(frame, models):
    out = frame[frame["Model"].isin(models)].copy()
    out["Model"] = pd.Categorical(out["Model"], categories=models, ordered=True)
    return out.sort_values("Model").reset_index(drop=True)


def annotate_value(ax, x, y, text, colour, side="right", dy=0.0, points=9):
    dx = 6 if side == "right" else -6
    ha = "left" if side == "right" else "right"
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va="center",
        fontsize=points * 0.74,
        fontweight="bold",
        color=colour,
    )


# ---------------------------------------------------------------------
# Frozen data
# ---------------------------------------------------------------------

def read_table(path, required):
    if not path.exists():
        raise SystemExit(
            f"{path.relative_to(analysis.ROOT)} is missing. "
            "Run notebooks/17_joint.ipynb before generating joint figures."
        )
    frame = pd.read_csv(path)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    return frame


def load_association():
    return read_table(
        ASSOCIATION_PATH,
        [
            "Model",
            "Rights",
            "Age Restricted",
            "Harmful",
            "Within-Type",
            "95% CI Lower",
            "95% CI Upper",
            "Standardised Check",
        ],
    )


def load_concordance():
    return read_table(
        CONCORDANCE_PATH,
        [
            "Scenario Type",
            "Model",
            "n",
            "rho",
            "95% CI Lower",
            "95% CI Upper",
            "Reason",
        ],
    )


def load_shape():
    return read_table(
        SHAPE_PATH,
        [
            "Measure",
            "Model",
            "Step as Share of Range (%)",
            "Step at the Boundary (17 to 18)",
            "Full Range (7 to 21)",
        ],
    )


# ---------------------------------------------------------------------
# Derived tables
# ---------------------------------------------------------------------

def association_table():
    concordance = load_concordance()
    pooled = concordance[concordance["Scenario Type"].eq("Pooled")][
        ["Model", "rho", "95% CI Lower", "95% CI Upper", "n"]
    ].rename(columns={
        "rho": "Pooled",
        "95% CI Lower": "Pooled Low",
        "95% CI Upper": "Pooled High",
        "n": "Pooled n",
    })

    association = load_association()
    columns = [
        "Model",
        "Within-Type",
        "95% CI Lower",
        "95% CI Upper",
        "Standardised Check",
    ]
    if "Valid Draws" in association.columns:
        columns.append("Valid Draws")
    within = association[columns].rename(columns={
        "95% CI Lower": "Within Low",
        "95% CI Upper": "Within High",
    })

    return ordered_rows(pooled.merge(within, on="Model", how="inner"), ORDER)


def boundary_gap_table():
    shape = load_shape().copy()
    shape = shape[shape["Model"].ne("Panel Ratio")]
    shares = shape.pivot(
        index="Model",
        columns="Measure",
        values="Step as Share of Range (%)",
    ).rename(columns={
        "Refusal Rate (pp)": "Refusal",
        "Grade Level": "FKGL",
    })
    shares["Gap"] = shares["Refusal"] - shares["FKGL"]
    return ordered_rows(shares.reset_index(), ORDER + [MACRO])


def strata_table():
    concordance = load_concordance()
    base = concordance[concordance["Scenario Type"].isin(STRATA_ORDER)][
        ["Model", "Scenario Type", "rho", "95% CI Lower", "95% CI Upper", "n"]
    ].rename(columns={
        "rho": "Estimate",
        "95% CI Lower": "Low",
        "95% CI Upper": "High",
    })

    within = load_association()[
        ["Model", "Within-Type", "95% CI Lower", "95% CI Upper"]
    ].copy()
    within["Scenario Type"] = "Within-Type"
    within = within.rename(columns={
        "Within-Type": "Estimate",
        "95% CI Lower": "Low",
        "95% CI Upper": "High",
    })
    within["n"] = np.nan

    out = pd.concat([base, within], ignore_index=True)
    out["Model"] = pd.Categorical(out["Model"], categories=ORDER, ordered=True)
    out["Scenario Type"] = pd.Categorical(
        out["Scenario Type"],
        categories=STRATA_ORDER + ["Within-Type"],
        ordered=True,
    )
    return out.sort_values(["Model", "Scenario Type"]).reset_index(drop=True)


def pooled_concordance_table():
    frame = load_concordance()
    return ordered_rows(frame[frame["Scenario Type"].eq("Pooled")], ORDER)


# ---------------------------------------------------------------------
# Main 1: pooled to within-type concordance
# ---------------------------------------------------------------------

def draw_association(display):
    frame = association_table()
    points = styled(display, MAIN_ASSOCIATION_SIZE[0], label_points=10.4)
    fig, ax = plt.subplots(figsize=MAIN_ASSOCIATION_SIZE)
    panel(ax, None, points)
    ax.axvline(0, color=MUTED, linewidth=0.85, linestyle="--", zorder=1)

    y = np.arange(len(frame))
    offset = 0.13
    for yi, row in frame.iterrows():
        model = row["Model"]
        colour = COLOUR[model]
        pooled = float(row["Pooled"])
        pooled_low = float(row["Pooled Low"])
        pooled_high = float(row["Pooled High"])
        within = float(row["Within-Type"])
        within_low = float(row["Within Low"])
        within_high = float(row["Within High"])

        yp, yw = yi - offset, yi + offset

        ax.plot([pooled, within], [yp, yw], color=colour, linewidth=0.95,
                alpha=0.35, zorder=1)
        ax.plot([pooled_low, pooled_high], [yp, yp], color=colour,
                linewidth=1.7, alpha=0.58, solid_capstyle="butt", zorder=2)
        ax.plot([within_low, within_high], [yw, yw], color=colour,
                linewidth=2.3, solid_capstyle="butt", zorder=3)

        ax.plot(pooled, yp, marker=MARKER[model], markersize=5.7,
                markerfacecolor="white", markeredgecolor=colour,
                markeredgewidth=1.3, linestyle="none", zorder=4)
        ax.plot(within, yw, marker=MARKER[model], markersize=6.2,
                color=colour, linestyle="none", zorder=5)

        # Numeric change is aligned outside the data region rather than printed
        # on top of intervals or markers.
        ax.text(
            1.025, yi,
            f"{pooled:+.2f}  " + r"$\rightarrow$" + f"  {within:+.2f}",
            transform=ax.get_yaxis_transform(),
            ha="left", va="center", clip_on=False,
            fontsize=points * 0.72, fontweight="bold", color=colour,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_AXIS[m] for m in frame["Model"]])
    ax.set_ylim(-0.52, len(frame) - 0.48)
    ax.invert_yaxis()
    ax.set_xlim(-0.85, 0.80)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))

    # Keep the axis label and legend in separate vertical bands. The label is
    # centred on the plotting area; the legend is centred on the full figure.
    ax.set_xlabel(r"Spearman $\rho$", labelpad=13)
    fig.legend(
        [
            Line2D([0], [0], color=INK, marker="o", markerfacecolor="white",
                   markeredgecolor=INK, linewidth=1.6, markersize=5.5),
            Line2D([0], [0], color=INK, marker="o", markerfacecolor=INK,
                   markeredgecolor=INK, linewidth=2.2, markersize=5.5),
        ],
        ["Pooled", "Within-Type"],
        ncol=2,
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(0.5, 0.035),
        fontsize=points * 0.90,
        handlelength=2.0,
        handletextpad=0.6,
        columnspacing=1.5,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.18, right=0.84, top=0.97, bottom=0.30)
    return save(fig, FIGURESPEC["association"][1])


# ---------------------------------------------------------------------
# Main 2: boundary concentration gap
# ---------------------------------------------------------------------

def draw_boundary_gap(display):
    frame = boundary_gap_table()
    points = styled(display, BOUNDARY_SIZE[0], label_points=10.4)
    fig, ax = plt.subplots(figsize=BOUNDARY_SIZE)
    panel(ax, None, points)
    ax.axvline(0, color=MUTED, linewidth=0.85, linestyle="--", zorder=1)

    y = np.arange(len(frame))
    for yi, row in frame.iterrows():
        model = row["Model"]
        gap = float(row["Gap"])
        colour = INK if model == MACRO else COLOUR[model]
        marker = "D" if model == MACRO else MARKER[model]
        linewidth = 2.3 if model == MACRO else 1.8

        ax.plot([0, gap], [yi, yi], color=colour, linewidth=linewidth,
                alpha=0.72, solid_capstyle="butt", zorder=2)
        ax.plot(gap, yi, marker=marker, markersize=6.5, color=colour,
                linestyle="none", zorder=3)

        ax.annotate(
            f"{gap:+.1f}",
            (gap, yi),
            xytext=(7 if gap >= 0 else -7, 0),
            textcoords="offset points",
            ha="left" if gap >= 0 else "right",
            va="center",
            fontsize=points * 0.78,
            fontweight="bold",
            color=colour,
        )

    if MACRO in frame["Model"].values:
        ax.axhline(len(ORDER) - 0.5, color="#D0D0D0", linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_AXIS[m] for m in frame["Model"]])
    ax.set_ylim(-0.52, len(frame) - 0.48)
    ax.invert_yaxis()
    ax.set_xlim(-12, 82)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("Boundary Concentration Gap (pp)", labelpad=14)
    fig.subplots_adjust(left=0.22, right=0.99, top=0.965, bottom=0.20)
    return save(fig, FIGURESPEC["boundary_gap"][1])


# ---------------------------------------------------------------------
# Supplement 1: scenario-stratified forest/grid
# ---------------------------------------------------------------------

def draw_strata(display):
    frame = strata_table()
    points = styled(display, STRATA_SIZE[0], label_points=9.8)
    fig, axes = plt.subplots(2, 3, figsize=STRATA_SIZE, sharex=True, sharey=True)

    rows = STRATA_ORDER + ["Within-Type"]
    ypos = np.arange(len(rows))
    value_x = 1.04

    for ax, model in zip(axes.flat, ORDER):
        panel(ax, None, points)
        ax.set_title(model, fontsize=points * 0.94, pad=points * 0.45,
                     color="black", fontweight="normal")
        ax.axvline(0, color=MUTED, linewidth=0.8, linestyle="--", zorder=1)
        sub = frame[frame["Model"].eq(model)]

        for yi, label in enumerate(rows):
            row = sub[sub["Scenario Type"].eq(label)]
            if row.empty:
                continue
            row = row.iloc[0]
            colour = INK if label == "Within-Type" else SCENARIO_COLOUR[label]

            if pd.isna(row["Estimate"]):
                ax.text(value_x, yi, "constant", ha="right", va="center",
                        fontsize=points * 0.62, color=MUTED)
                continue

            value = float(row["Estimate"])
            low = float(row["Low"])
            high = float(row["High"])
            marker = "D" if label == "Within-Type" else "o"
            ax.plot([low, high], [yi, yi], color=colour, linewidth=2.0,
                    solid_capstyle="butt", zorder=2)
            ax.plot(value, yi, marker=marker, markersize=5.3, color=colour,
                    linestyle="none", zorder=3)
            ax.text(value_x, yi, f"{value:+.2f}", ha="right", va="center",
                    fontsize=points * 0.68, fontweight="bold", color=colour)

        ax.set_yticks(ypos)
        ax.set_yticklabels([STRATA_AXIS[r] for r in rows])
        ax.set_xlim(-1.00, 1.12)
        ax.set_xticks([-0.8, -0.4, 0.0, 0.4, 0.8])
        ax.set_ylim(-0.55, len(rows) - 0.45)
        ax.invert_yaxis()

    # Row labels already encode the scenario colours, so a second legend would
    # be redundant. Keep only the common centred axis label.
    for ax in axes.flat:
        ax.label_outer()
    fig.supxlabel(r"Spearman $\rho$", x=0.5, y=0.045,
                  ha="center", va="center",
                  color="black", fontsize=points * 1.04)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.94, bottom=0.20,
                        wspace=0.16, hspace=0.30)
    return save(fig, FIGURESPEC["strata"][1])


# ---------------------------------------------------------------------
# Supplement 2: pooled concordance
# ---------------------------------------------------------------------

def draw_concordance(display):
    frame = pooled_concordance_table()
    points = styled(display, POOLED_SIZE[0], label_points=10.2)
    fig, ax = plt.subplots(figsize=POOLED_SIZE)
    panel(ax, None, points)
    ax.axvline(0, color=MUTED, linewidth=0.85, linestyle="--", zorder=1)

    y = np.arange(len(frame))
    value_x = 0.74
    for yi, row in frame.iterrows():
        model = row["Model"]
        colour = COLOUR[model]
        rho = float(row["rho"])
        low = float(row["95% CI Lower"])
        high = float(row["95% CI Upper"])
        ax.plot([low, high], [yi, yi], color=colour, linewidth=2.0,
                solid_capstyle="butt", zorder=2)
        ax.plot(rho, yi, marker=MARKER[model], markersize=5.8,
                color=colour, linestyle="none", zorder=3)
        ax.text(value_x, yi, f"{rho:+.2f}", ha="right", va="center",
                fontsize=points * 0.72, fontweight="bold", color=colour)

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_AXIS[m] for m in frame["Model"]])
    ax.set_ylim(-0.52, len(frame) - 0.48)
    ax.invert_yaxis()
    ax.set_xlim(-0.82, 0.82)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.set_xlabel(r"Spearman $\rho$", labelpad=14)
    fig.subplots_adjust(left=0.22, right=0.99, top=0.97, bottom=0.21)
    return save(fig, FIGURESPEC["concordance"][1])


# ---------------------------------------------------------------------
# Supplement 3: all concordance intervals
# ---------------------------------------------------------------------

def draw_concordance_intervals(display):
    concordance = load_concordance()
    association = load_association().set_index("Model").reindex(ORDER)
    points = styled(display, INTERVALS_SIZE[0], label_points=9.6)
    fig, axes = plt.subplots(2, 3, figsize=INTERVALS_SIZE, sharex=True, sharey=True)
    y = np.arange(len(ORDER))
    value_x = 1.04

    for ax, scenario in zip(axes.flat, INTERVAL_PANELS):
        panel(ax, None, points)
        ax.set_title(scenario, fontsize=points * 0.96, pad=points * 0.42,
                     color="black", fontweight="normal")
        ax.axvline(0, color=MUTED, linewidth=0.8, linestyle="--", zorder=1)

        if scenario == "Within-Type":
            for yi, model in enumerate(ORDER):
                row = association.loc[model]
                value = float(row["Within-Type"])
                low = float(row["95% CI Lower"])
                high = float(row["95% CI Upper"])
                colour = COLOUR[model]
                ax.plot([low, high], [yi, yi], color=colour, linewidth=2.0,
                        solid_capstyle="butt", zorder=2)
                ax.plot(value, yi, marker=MARKER[model], markersize=5.3,
                        color=colour, linestyle="none", zorder=3)
                ax.text(value_x, yi, f"{value:+.2f}", ha="right", va="center",
                        fontsize=points * 0.68, fontweight="bold", color=colour)
        else:
            sub = (concordance[concordance["Scenario Type"].eq(scenario)]
                   .set_index("Model").reindex(ORDER))
            for yi, model in enumerate(ORDER):
                row = sub.loc[model]
                colour = COLOUR[model]
                if pd.isna(row["rho"]):
                    ax.text(value_x, yi, "constant", ha="right", va="center",
                            fontsize=points * 0.60, color=MUTED)
                    continue
                value = float(row["rho"])
                low = float(row["95% CI Lower"])
                high = float(row["95% CI Upper"])
                ax.plot([low, high], [yi, yi], color=colour, linewidth=2.0,
                        solid_capstyle="butt", zorder=2)
                ax.plot(value, yi, marker=MARKER[model], markersize=5.3,
                        color=colour, linestyle="none", zorder=3)
                ax.text(value_x, yi, f"{value:+.2f}", ha="right", va="center",
                        fontsize=points * 0.68, fontweight="bold", color=colour)

        ax.set_xlim(-1.00, 1.12)
        ax.set_xticks([-0.8, -0.4, 0.0, 0.4, 0.8])
        ax.set_yticks(y)
        ax.set_yticklabels([MODEL_AXIS[m] for m in ORDER])
        ax.set_ylim(-0.55, len(ORDER) - 0.45)
        ax.invert_yaxis()

    for ax in axes.flat:
        ax.label_outer()
    fig.supxlabel(r"Spearman $\rho$", x=0.5, y=0.045,
                  ha="center", va="center",
                  color="black", fontsize=points * 1.04)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.93, bottom=0.20,
                        wspace=0.16, hspace=0.30)
    return save(fig, FIGURESPEC["concordance_intervals"][1])


# ---------------------------------------------------------------------
# Registry and CLI
# ---------------------------------------------------------------------

def registry(display):
    return {
        "association": (lambda: draw_association(display), "main"),
        "boundary_gap": (lambda: draw_boundary_gap(display), "main"),
        "strata": (lambda: draw_strata(display), "supplement"),
        "concordance": (lambda: draw_concordance(display), "supplement"),
        "concordance_intervals": (
            lambda: draw_concordance_intervals(display),
            "supplement",
        ),
    }


def print_manifest():
    print("Joint Figure Set")
    print("-" * 92)
    for key, (tier, filename, purpose) in FIGURESPEC.items():
        print(f"{key:<24}{tier:<12}{filename:<36}{purpose}")
    print("-" * 92)


def main(args):
    mpl.rcParams.update(analysis.STYLE)
    clean_stale()

    load_association()
    load_concordance()
    load_shape()

    print_manifest()
    jobs = registry(args.display)

    print("\nGenerating")
    print("-" * 92)
    built = 0
    for name, (build, tier) in jobs.items():
        if args.only not in {"all", name}:
            continue
        if args.set not in {"both", tier}:
            continue
        print(f"{name:<24}[{tier}] {FIGURESPEC[name][2]}")
        build()
        built += 1

    print(
        f"\n{built} figure{'s' if built != 1 else ''} "
        f"written to {FIGURES.relative_to(analysis.ROOT)}/."
    )


def parser():
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--display", type=float, default=1.0)
    cli.add_argument("--set", default="both", choices=("both", "main", "supplement"))
    cli.add_argument("--only", default="all", choices=("all", *FIGURESPEC.keys()))
    return cli


if __name__ == "__main__":
    main(parser().parse_args())
