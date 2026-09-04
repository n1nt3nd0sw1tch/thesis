"""Joint safety-readability figures for the thesis.

The script reads only the frozen outputs of notebooks/17_joint.ipynb and uses
exactly the same project-wide figure grammar as figureread.py and figuresafe.py:
one model palette, horizontal labels, restrained grey panels, no figure titles,
and quantitative annotations only where they help interpretation.

Main figures
------------
joint_ladders.pdf
    Raw age trajectories for refusal in Age Restricted scenarios and FKGL over
    all stated-age replies.
joint_progression.pdf
    Within-model cumulative location of the 7-21 age shift on the complete
    four-age cohort. This is the clearest shape comparison between safety and
    readability.
joint_boundary.pdf
    17-18 step as a share of the full 7-21 shift for both outcomes.
joint_concordance.pdf
    Scenario-level Spearman concordance between safety adaptation and linguistic
    simplification.

Supplementary figures
---------------------
joint_phases.pdf
    Full three-part decomposition of the 7-21 shift.
joint_concordance_intervals.pdf
    Bootstrap intervals for the pooled and Age Restricted concordance estimates.

Examples
--------
python scripts/figurejoint.py
python scripts/figurejoint.py --set main
python scripts/figurejoint.py --only progression
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis
from settings import ROOT


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

FIGURES = ROOT / "figures" / "joint"

LADDERS_PATH = ROOT / "tables" / "main" / "joint_01_ladders.csv"
SHAPE_PATH = ROOT / "tables" / "main" / "joint_02_shape.csv"
CONCORDANCE_PATH = ROOT / "tables" / "supplement" / "joint_03_concordance.csv"

TEXT_WIDTH_CM = 16.0
LABEL_POINTS = 10.2
GRID_SIZE = (13.5, 7.8)
WIDE_SIZE = (13.5, 6.2)
FOREST_SIZE = (12.4, 5.9)
MAP_SIZE = (9.6, 5.8)

PANEL_FILL = "#F5F5F5"
MINOR_BAND = "#E8EDF2"
ADULT_BAND = "#F2EDE8"

INK = analysis.INK
MUTED = analysis.MUTED
COLOUR = analysis.COLOUR
MARKER = analysis.MARKER
MODEL_ORDER = list(analysis.ORDER)
MACRO = analysis.MACRO

LEGEND = dict(
    loc="lower center",
    frameon=False,
    bbox_to_anchor=(0.5, -0.065),
    handlelength=2.1,
    handletextpad=0.6,
    columnspacing=1.35,
)

AGES = [7, 9, 11, 13, 15, 17, 18, 21]
SHAPE_AGES = [7, 17, 18, 21]

SAFETY_LADDER = "Refusal Rate (%)"
READABILITY_LADDER = "Grade Level"
SAFETY_SHAPE = "Refusal Rate (pp)"
READABILITY_SHAPE = "Grade Level"

PHASES = [
    "Across Childhood (7 to 17)",
    "Step at the Boundary (17 to 18)",
    "Above the Boundary (18 to 21)",
]
PHASE_AXIS = ["7-17", "17-18", "18-21"]
PHASE_COLOUR = {
    PHASES[0]: "#AAB7C4",
    PHASES[1]: "#4F5B66",
    PHASES[2]: "#D8C9BC",
}

MODEL_AXIS = {
    "GPT-5.6 Luna": "GPT-5.6\nLuna",
    "Claude Haiku 4.5": "Claude\nHaiku 4.5",
    "Gemini 3.5 Flash Lite": "Gemini 3.5\nFlash Lite",
    "DeepSeek-V4 Flash": "DeepSeek-V4\nFlash",
    "Mistral Small 4": "Mistral\nSmall 4",
    "Gemma 4 31B": "Gemma 4\n31B",
    MACRO: "Macro-\nAverage",
}

CONCORDANCE_SCENARIOS = ["Pooled", "Rights", "Age Restricted", "Harmful"]
CONCORDANCE_AXIS = {
    "Pooled": "Pooled",
    "Rights": "Rights",
    "Age Restricted": "Age\nRestricted",
    "Harmful": "Harmful",
}

FIGURESPEC = {
    "ladders": (
        "main",
        "joint_ladders.pdf",
        "Raw safety and readability trajectories across age",
    ),
    "progression": (
        "main",
        "joint_progression.pdf",
        "Cumulative location of the age shift within each model",
    ),
    "boundary": (
        "main",
        "joint_boundary.pdf",
        "Refusal and readability concentration at the 17-18 boundary",
    ),
    "concordance": (
        "main",
        "joint_concordance.pdf",
        "Scenario-level safety-readability concordance",
    ),
    "phases": (
        "supplement",
        "joint_phases.pdf",
        "Three-part decomposition of the 7-21 age shift",
    ),
    "concordance_intervals": (
        "supplement",
        "joint_concordance_intervals.pdf",
        "Bootstrap intervals for pooled and Age Restricted concordance",
    ),
}

STALE = [
    "joint_profiles.pdf",
    "joint_boundary_concentration.pdf",
    "joint_phase_map.pdf",
    "joint_focus_forest.pdf",
    "joint_threshold_scatter.pdf",
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


def map_panel(ax, title=None, points=9):
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_axisbelow(False)
    if title:
        ax.set_title(title, pad=points * 0.5, color="black")


def outer_labels(fig, axes, xlabel, ylabel, points):
    for ax in np.asarray(axes).flat:
        ax.label_outer()
    if xlabel:
        fig.supxlabel(xlabel, color="black", fontsize=points * 1.05)
    if ylabel:
        fig.supylabel(ylabel, color="black", fontsize=points * 1.10)


def legend(fig, handles, labels, points, ncol):
    fig.legend(handles, labels, ncol=ncol, fontsize=points * 1.05, **LEGEND)


def save(fig, filename):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")
    return path


def clean_stale():
    FIGURES.mkdir(parents=True, exist_ok=True)
    for filename in STALE:
        path = FIGURES / filename
        if path.exists():
            path.unlink()


def age_bands(ax):
    ax.axvspan(6.4, 17.5, color=MINOR_BAND, zorder=0)
    ax.axvspan(17.5, 21.6, color=ADULT_BAND, zorder=0)
    ax.axvline(17.5, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.70, zorder=1)


def readable_on(rgba):
    r, g, b = rgba[:3]
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.42 else "black"


# ---------------------------------------------------------------------
# Frozen data
# ---------------------------------------------------------------------


def read_table(path, required):
    if not path.exists():
        raise SystemExit(
            f"{path.relative_to(ROOT)} is missing. Run notebooks/17_joint.ipynb "
            "before generating the joint figures."
        )
    frame = pd.read_csv(path)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    return frame


def load_ladders():
    return read_table(
        LADDERS_PATH,
        ["Measure", "Model", *[f"Age {age}" for age in AGES]],
    )


def load_shape():
    return read_table(
        SHAPE_PATH,
        [
            "Measure",
            "Model",
            *PHASES,
            "Full Range (7 to 21)",
            "Step as Share of Range (%)",
            "n",
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


# ---------------------------------------------------------------------
# Reductions derived only from the frozen tables
# ---------------------------------------------------------------------


def ordered_rows(frame, models):
    out = frame[frame["Model"].isin(models)].copy()
    out["Model"] = pd.Categorical(out["Model"], categories=models, ordered=True)
    return out.sort_values("Model").reset_index(drop=True)


def shape_rows(measure, include_macro=False):
    models = MODEL_ORDER + ([MACRO] if include_macro else [])
    frame = load_shape()
    return ordered_rows(frame[frame["Measure"].eq(measure)], models)


def phase_shares(measure, include_macro=False):
    out = shape_rows(measure, include_macro=include_macro).copy()
    full = out["Full Range (7 to 21)"].astype(float).replace(0, np.nan)
    for phase in PHASES:
        out[phase] = 100.0 * out[phase].astype(float) / full
    return out


def progression_table(measure):
    out = phase_shares(measure, include_macro=False)
    rows = []
    for _, row in out.iterrows():
        first = float(row[PHASES[0]])
        second = float(row[PHASES[1]])
        rows.append({
            "Model": row["Model"],
            7: 0.0,
            17: first,
            18: first + second,
            21: 100.0,
            "Boundary": second,
        })
    return pd.DataFrame(rows)


def boundary_table():
    safety = shape_rows(SAFETY_SHAPE, include_macro=True)[
        ["Model", "Step as Share of Range (%)"]
    ].rename(columns={"Step as Share of Range (%)": "Safety"})
    reading = shape_rows(READABILITY_SHAPE, include_macro=True)[
        ["Model", "Step as Share of Range (%)"]
    ].rename(columns={"Step as Share of Range (%)": "Readability"})
    return safety.merge(reading, on="Model", how="inner")


def ci_excludes_zero(row):
    low = row["95% CI Lower"]
    high = row["95% CI Upper"]
    if pd.isna(low) or pd.isna(high):
        return False
    return bool(low > 0 or high < 0)


# ---------------------------------------------------------------------
# Main 1: raw ladders
# ---------------------------------------------------------------------


def draw_ladders(display):
    frame = load_ladders()
    points = styled(display, WIDE_SIZE[0])
    fig, axes = plt.subplots(1, 2, figsize=WIDE_SIZE)

    specs = [
        (SAFETY_LADDER, "Refusal", "Refusal Rate (%)", (0, 85), 10),
        (READABILITY_LADDER, "FKGL", "Flesch-Kincaid Grade Level", (4.0, 10.2), 1),
    ]

    for ax, (measure, title, ylabel, ylim, tick) in zip(axes, specs):
        age_bands(ax)
        sub = frame[frame["Measure"].eq(measure)]

        for model in MODEL_ORDER:
            row = sub[sub["Model"].eq(model)]
            if row.empty:
                continue
            values = row[[f"Age {age}" for age in AGES]].iloc[0].astype(float).to_numpy()
            ax.plot(
                AGES,
                values,
                color=COLOUR[model],
                marker=MARKER[model],
                markersize=4.6,
                markerfacecolor="white",
                markeredgecolor=COLOUR[model],
                markeredgewidth=1.05,
                linewidth=1.8,
                zorder=3,
            )
            # The predeclared legal boundary is the comparison of interest.
            for age in (17, 18):
                idx = AGES.index(age)
                ax.plot(
                    age,
                    values[idx],
                    marker=MARKER[model],
                    markersize=6.1,
                    color=COLOUR[model],
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    linestyle="none",
                    zorder=4,
                )

        panel(ax, title, points)
        ax.set_xticks(AGES)
        ax.set_xlim(6.4, 21.6)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(MultipleLocator(tick))
        ax.set_xlabel("Age")
        ax.set_ylabel(ylabel)

    handles = [
        Line2D(
            [0], [0], color=COLOUR[model], marker=MARKER[model],
            markerfacecolor="white", markeredgecolor=COLOUR[model],
            linewidth=1.8, markersize=4.8,
        )
        for model in MODEL_ORDER
    ]
    legend(fig, handles, MODEL_ORDER, points, 3)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.29, wspace=0.24)
    return save(fig, FIGURESPEC["ladders"][1])


# ---------------------------------------------------------------------
# Main 2: cumulative location of the age shift
# ---------------------------------------------------------------------


def boundary_badge(ax, model, safety, reading, points):
    text = f"17-18:  Refusal {safety:.1f}%\n          FKGL {reading:.1f}%"
    ax.text(
        0.035,
        0.955,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=points * 0.78,
        fontweight="semibold",
        color=COLOUR[model],
        linespacing=1.15,
        zorder=7,
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor="white",
            alpha=0.92,
            edgecolor="#D0D0D0",
            linewidth=0.55,
        ),
    )


def draw_progression(display):
    safety = progression_table(SAFETY_SHAPE).set_index("Model")
    reading = progression_table(READABILITY_SHAPE).set_index("Model")

    points = styled(display)
    fig, axes = grid(
        figsize=GRID_SIZE,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax, model in zip(axes.flat, MODEL_ORDER):
        age_bands(ax)
        s = safety.loc[model, SHAPE_AGES].astype(float).to_numpy()
        r = reading.loc[model, SHAPE_AGES].astype(float).to_numpy()

        ax.plot(
            SHAPE_AGES,
            s,
            color=COLOUR[model],
            marker="o",
            markersize=5.1,
            linewidth=2.0,
            zorder=3,
        )
        ax.plot(
            SHAPE_AGES,
            r,
            color=COLOUR[model],
            linestyle="--",
            marker="s",
            markersize=4.8,
            markerfacecolor="white",
            markeredgecolor=COLOUR[model],
            markeredgewidth=1.2,
            linewidth=1.8,
            zorder=3,
        )

        panel(ax, model, points)
        boundary_badge(
            ax,
            model,
            float(safety.loc[model, "Boundary"]),
            float(reading.loc[model, "Boundary"]),
            points,
        )

        ax.set_xticks(SHAPE_AGES)
        ax.set_xlim(6.4, 21.6)
        ax.set_ylim(-3, 106)
        ax.yaxis.set_major_locator(MultipleLocator(20))

    outer_labels(fig, axes, "Age", "Cumulative Share of 7-21 Shift (%)", points)
    legend(
        fig,
        [
            Line2D([0], [0], color=INK, marker="o", linewidth=2.0),
            Line2D(
                [0], [0], color=INK, linestyle="--", marker="s",
                markerfacecolor="white", markeredgecolor=INK, linewidth=1.8,
            ),
        ],
        ["Refusal", "FKGL"],
        points,
        2,
    )
    return save(fig, FIGURESPEC["progression"][1])


# ---------------------------------------------------------------------
# Main 3: 17-18 boundary concentration
# ---------------------------------------------------------------------


def draw_boundary(display):
    frame = boundary_table()
    order = MODEL_ORDER + [MACRO]
    frame = ordered_rows(frame, order)

    points = styled(display, width_inches=10.0)
    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    panel(ax, None, points)
    ax.grid(axis="x", linestyle="-", linewidth=0.6, alpha=0.25, color=MUTED)
    ax.grid(axis="y", visible=False)

    y = np.arange(len(frame))
    for yi, row in frame.iterrows():
        model = row["Model"]
        safety = float(row["Safety"])
        reading = float(row["Readability"])
        colour = INK if model == MACRO else COLOUR[model]
        marker = "D" if model == MACRO else MARKER[model]
        linewidth = 2.3 if model == MACRO else 1.6

        ax.plot(
            [reading, safety], [yi, yi],
            color=colour, linewidth=linewidth, alpha=0.72, zorder=2,
        )
        ax.plot(
            safety, yi, marker=marker, markersize=6.4,
            color=colour, linestyle="none", zorder=4,
        )
        ax.plot(
            reading, yi, marker="s", markersize=6.0,
            markerfacecolor="white", markeredgecolor=colour,
            markeredgewidth=1.5, linestyle="none", zorder=4,
        )

        # Exact values are the point of this figure. Stagger only the one pair
        # whose values almost coincide.
        if abs(safety - reading) < 5:
            ax.annotate(
                f"{safety:.1f}", (safety, yi), xytext=(6, 7),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=points * 0.73, fontweight="bold", color=colour,
            )
            ax.annotate(
                f"{reading:.1f}", (reading, yi), xytext=(6, -8),
                textcoords="offset points", ha="left", va="top",
                fontsize=points * 0.73, fontweight="bold", color=colour,
            )
        else:
            ax.annotate(
                f"{reading:.1f}", (reading, yi), xytext=(-6, 0),
                textcoords="offset points", ha="right", va="center",
                fontsize=points * 0.73, fontweight="bold", color=colour,
            )
            ax.annotate(
                f"{safety:.1f}", (safety, yi), xytext=(6, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=points * 0.73, fontweight="bold", color=colour,
            )

    # Visually separate the panel summary without changing model order.
    ax.axhline(len(MODEL_ORDER) - 0.5, color="#D0D0D0", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_AXIS[model] for model in frame["Model"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 76)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("Boundary Share of 7-21 Shift (%)")

    legend(
        fig,
        [
            Line2D([0], [0], marker="o", color=INK, linestyle="none", markersize=6.2),
            Line2D(
                [0], [0], marker="s", markerfacecolor="white",
                markeredgecolor=INK, linestyle="none", markersize=6.0,
            ),
        ],
        ["Refusal", "FKGL"],
        points,
        2,
    )
    return save(fig, FIGURESPEC["boundary"][1])


# ---------------------------------------------------------------------
# Main 4: scenario concordance
# ---------------------------------------------------------------------


def draw_concordance(display):
    frame = load_concordance().copy()
    frame = frame[frame["Scenario Type"].isin(CONCORDANCE_SCENARIOS)]
    frame["Model"] = pd.Categorical(frame["Model"], categories=MODEL_ORDER, ordered=True)
    frame["Scenario Type"] = pd.Categorical(
        frame["Scenario Type"], categories=CONCORDANCE_SCENARIOS, ordered=True,
    )

    table = frame.pivot(index="Model", columns="Scenario Type", values="rho").reindex(
        index=MODEL_ORDER,
        columns=CONCORDANCE_SCENARIOS,
    )
    values = table.to_numpy(dtype=float)

    points = styled(display, MAP_SIZE[0], label_points=9.4)
    fig, ax = plt.subplots(figsize=MAP_SIZE, constrained_layout=True)
    map_panel(ax, None, points)

    cmap = colormaps["RdBu_r"].copy()
    cmap.set_bad("#FFFFFF")
    image = ax.imshow(values, cmap=cmap, vmin=-0.8, vmax=0.8, aspect="auto")

    ax.set_xticks(
        range(len(CONCORDANCE_SCENARIOS)),
        [CONCORDANCE_AXIS[value] for value in CONCORDANCE_SCENARIOS],
        rotation=0,
    )
    ax.set_yticks(range(len(MODEL_ORDER)), [MODEL_AXIS[model] for model in MODEL_ORDER])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row_i, model in enumerate(MODEL_ORDER):
        for col_i, scenario in enumerate(CONCORDANCE_SCENARIOS):
            row = frame[
                frame["Model"].eq(model) & frame["Scenario Type"].eq(scenario)
            ]
            if row.empty:
                continue
            row = row.iloc[0]
            value = row["rho"]
            reason = str(row["Reason"])

            if pd.isna(value):
                text = "const." if "constant" in reason else "n.e."
                colour = MUTED
            else:
                text = f"{float(value):.2f}"
                if ci_excludes_zero(row):
                    text += "*"
                if "small stratum" in reason:
                    text += "†"
                colour = readable_on(cmap((float(value) + 0.8) / 1.6))

            ax.text(
                col_i, row_i, text,
                ha="center", va="center",
                fontsize=points * 0.80,
                fontweight="bold" if not pd.isna(value) else "normal",
                color=colour,
            )

    for x in np.arange(-0.5, len(CONCORDANCE_SCENARIOS), 1):
        ax.axvline(x, color="white", linewidth=1.0)
    for y in np.arange(-0.5, len(MODEL_ORDER), 1):
        ax.axhline(y, color="white", linewidth=1.0)

    cax = make_axes_locatable(ax).append_axes("right", size="3.5%", pad=0.18)
    bar = fig.colorbar(image, cax=cax)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=points * 0.82, length=0, labelcolor="black")
    bar.ax.set_ylabel(r"Spearman $\rho$", rotation=90)

    fig.text(
        0.02, 0.012,
        "* 95% bootstrap CI excludes zero    † n < 20    const. = constant safety shift",
        ha="left", va="bottom", fontsize=points * 0.67, color=MUTED,
    )
    return save(fig, FIGURESPEC["concordance"][1])


# ---------------------------------------------------------------------
# Supplement 1: phase decomposition
# ---------------------------------------------------------------------


def draw_phases(display):
    points = styled(display, WIDE_SIZE[0], label_points=9.4)
    fig, axes = plt.subplots(1, 2, figsize=WIDE_SIZE, sharey=True)

    for ax, measure, title in zip(
        axes,
        [SAFETY_SHAPE, READABILITY_SHAPE],
        ["Refusal", "FKGL"],
    ):
        sub = phase_shares(measure, include_macro=False)
        y = np.arange(len(MODEL_ORDER))
        left = np.zeros(len(sub), dtype=float)

        for phase in PHASES:
            values = sub[phase].astype(float).to_numpy()
            ax.barh(
                y,
                values,
                left=left,
                height=0.62,
                color=PHASE_COLOUR[phase],
                edgecolor="white",
                linewidth=0.7,
                label=phase,
                zorder=2,
            )
            for yi, start, value in zip(y, left, values):
                if not np.isfinite(value) or abs(value) < 6:
                    continue
                xpos = start + value / 2
                rgba = mpl.colors.to_rgba(PHASE_COLOUR[phase])
                ax.text(
                    xpos, yi, f"{value:.1f}",
                    ha="center", va="center",
                    fontsize=points * 0.66,
                    fontweight="bold",
                    color=readable_on(rgba),
                    zorder=3,
                )
            left += values

        panel(ax, title, points)
        ax.grid(axis="x", linestyle="-", linewidth=0.6, alpha=0.25, color=MUTED)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.set_yticks(y)
        ax.set_yticklabels([MODEL_AXIS[model] for model in MODEL_ORDER])

    axes[0].invert_yaxis()
    fig.supxlabel("Share of 7-21 Shift (%)", color="black", fontsize=points * 1.05, y=0.105)
    legend(
        fig,
        [Patch(facecolor=PHASE_COLOUR[phase]) for phase in PHASES],
        PHASE_AXIS,
        points,
        3,
    )
    fig.subplots_adjust(left=0.11, right=0.99, top=0.93, bottom=0.28, wspace=0.14)
    return save(fig, FIGURESPEC["phases"][1])


# ---------------------------------------------------------------------
# Supplement 2: uncertainty on the two most informative concordances
# ---------------------------------------------------------------------


def draw_concordance_intervals(display):
    frame = load_concordance().copy()
    points = styled(display, FOREST_SIZE[0])
    fig, axes = plt.subplots(1, 2, figsize=FOREST_SIZE, sharey=True)

    for ax, scenario in zip(axes, ["Pooled", "Age Restricted"]):
        sub = ordered_rows(frame[frame["Scenario Type"].eq(scenario)], MODEL_ORDER)
        y = np.arange(len(MODEL_ORDER))

        ax.axvline(0, color=MUTED, linewidth=0.9, linestyle="--", zorder=1)
        for yi, row in sub.iterrows():
            if pd.isna(row["rho"]):
                continue
            model = row["Model"]
            rho = float(row["rho"])
            low = float(row["95% CI Lower"])
            high = float(row["95% CI Upper"])
            colour = COLOUR[model]

            ax.plot(
                [low, high], [yi, yi],
                color=colour, linewidth=2.0, solid_capstyle="butt", zorder=2,
            )
            ax.plot(
                rho, yi,
                marker=MARKER[model], color=colour,
                markersize=6.0, linestyle="none", zorder=3,
            )

            note = "†" if "small stratum" in str(row["Reason"]) else ""
            ax.text(
                0.88, yi, f"{rho:.2f}{note}",
                ha="right", va="center",
                fontsize=points * 0.72,
                fontweight="semibold",
                color=colour,
            )

        panel(ax, CONCORDANCE_AXIS[scenario].replace("\n", " "), points)
        ax.grid(axis="x", linestyle="-", linewidth=0.6, alpha=0.25, color=MUTED)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(-0.9, 0.9)
        ax.xaxis.set_major_locator(MultipleLocator(0.3))
        ax.set_yticks(y)
        ax.set_yticklabels([MODEL_AXIS[model] for model in MODEL_ORDER])

    axes[0].invert_yaxis()
    axes[1].tick_params(labelleft=False)
    fig.supxlabel(r"Spearman $\rho$", color="black", fontsize=points * 1.05, y=0.095)
    fig.text(
        0.02, 0.018,
        "† n < 20; interval shown but interpreted cautiously.",
        ha="left", va="bottom", fontsize=points * 0.68, color=MUTED,
    )
    fig.subplots_adjust(left=0.20, right=0.99, top=0.90, bottom=0.25, wspace=0.14)
    return save(fig, FIGURESPEC["concordance_intervals"][1])


# ---------------------------------------------------------------------
# Registry and CLI
# ---------------------------------------------------------------------


def registry(display):
    return {
        "ladders": (lambda: draw_ladders(display), "main"),
        "progression": (lambda: draw_progression(display), "main"),
        "boundary": (lambda: draw_boundary(display), "main"),
        "concordance": (lambda: draw_concordance(display), "main"),
        "phases": (lambda: draw_phases(display), "supplement"),
        "concordance_intervals": (
            lambda: draw_concordance_intervals(display), "supplement"
        ),
    }


def print_manifest():
    print("Joint Figure Set")
    print("-" * 98)
    for key, (tier, filename, purpose) in FIGURESPEC.items():
        print(f"{key:<24}{tier:<12}{filename:<36}{purpose}")
    print("-" * 98)


def main(args):
    mpl.rcParams.update(analysis.STYLE)
    clean_stale()

    # Validate all three frozen notebook outputs before building anything.
    load_ladders()
    load_shape()
    load_concordance()

    print_manifest()
    jobs = registry(args.display)

    print("\nGenerating")
    print("-" * 98)

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
        f"written to {FIGURES.relative_to(ROOT)}/."
    )


def parser():
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--display", type=float, default=1.0)
    cli.add_argument("--set", default="both", choices=("both", "main", "supplement"))
    cli.add_argument("--only", default="all", choices=("all", *FIGURESPEC.keys()))
    return cli


if __name__ == "__main__":
    main(parser().parse_args())
