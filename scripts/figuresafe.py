"""Safety figures for the thesis and paper.

The figure grammar deliberately mirrors scripts/figureread.py:
- the same typography scaling;
- model panels where model is the natural unit;
- outer axis labels rather than repeated labels;
- bottom legends;
- restrained panel titles only;
- scenario-weighted rates throughout.

Examples
--------
python scripts/figuresafe.py
python scripts/figuresafe.py --set main
python scripts/figuresafe.py --set supplement
python scripts/figuresafe.py --only outcomes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis
from settings import ROOT, measure_column


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

FIGURES = ROOT / "figures" / "safety"
REGISTER = analysis.MACHINE / "register_safety.csv"

TEXT_WIDTH_CM = 16.0
LABEL_POINTS = 10.2
GRID_SIZE = (13.5, 7.8)
WIDE_SIZE = (13.5, 6.2)

PANEL_FILL = "#F5F5F5"
MINOR_BAND = "#E8EDF2"
ADULT_BAND = "#F2EDE8"

LEGEND = dict(
    loc="lower center",
    frameon=False,
    bbox_to_anchor=(0.5, -0.075),
    handlelength=2.2,
    handletextpad=0.7,
    columnspacing=1.5,
)

MODEL_AXIS = {
    "GPT-5.6 Luna": "GPT-5.6\nLuna",
    "Claude Haiku 4.5": "Claude Haiku\n4.5",
    "Gemini 3.5 Flash Lite": "Gemini 3.5\nFlash Lite",
    "DeepSeek-V4 Flash": "DeepSeek-V4\nFlash",
    "Mistral Small 4": "Mistral Small\n4",
    "Gemma 4 31B": "Gemma 4\n31B",
    analysis.MACRO: "Macro\nAverage",
}

SCENARIOS = list(analysis.STRATA)
SCENARIO_AXIS = {
    "Benign": "Benign",
    "Rights": "Rights",
    "Age Restricted": "Age\nRestricted",
    "Harmful": "Harmful",
}

# Same scenario grammar as readability coverage.
TYPE_STYLE = {
    "Harmful": ("-", "o", 0.92),
    "Age Restricted": ("--", "s", 0.72),
    "Rights": (":", "^", 0.52),
    "Benign": ("-.", "D", 0.34),
}

AGE_MAP = {
    condition: int(condition.replace("age", ""))
    for condition in analysis.CONDITION_ORDER
    if condition.startswith("age")
}
AGE_CONDITIONS = sorted(AGE_MAP, key=AGE_MAP.get)
AGES = [AGE_MAP[condition] for condition in AGE_CONDITIONS]
AGE_CONDITION = {AGE_MAP[condition]: condition for condition in AGE_CONDITIONS}
MINOR_AGE_CONDITIONS = [condition for condition in AGE_CONDITIONS if AGE_MAP[condition] < 18]
ADULT_AGE_CONDITIONS = [condition for condition in AGE_CONDITIONS if AGE_MAP[condition] >= 18]

LABEL_GROUPS = [
    ("Neutral", [analysis.NEUTRAL]),
    ("Adult (Cue)", list(analysis.IMPLICIT_ADULT)),
    ("Adult (Age)", ADULT_AGE_CONDITIONS),
    ("Minor (Cue)", list(analysis.IMPLICIT_MINOR)),
    ("Minor (Age)", MINOR_AGE_CONDITIONS),
]
SIGNAL_KEYS = [label for label, _ in LABEL_GROUPS]
SIGNAL_AXIS = {
    "Neutral": "Neutral",
    "Adult (Cue)": "Adult\n(Cue)",
    "Adult (Age)": "Adult\n(Age)",
    "Minor (Cue)": "Minor\n(Cue)",
    "Minor (Age)": "Minor\n(Age)",
}

CELL_COLUMNS = {
    cell: cell.lower().replace(" ", "_")
    for cell in analysis.CELL_ORDER
}

# Main descriptive figure: keep the fields that carry the safety argument.
# Identity / boundary fields remain in the dedicated supplementary field maps.
RATE_FIELDS = [
    ("refusal", "Refusal"),
    ("delivered", "Delivery"),
    (measure_column("Alternative Response"), "Alternative"),
    (measure_column("Risk Statement"), "Risk"),
    (measure_column("Legal Statement"), "Legal"),
    (measure_column("Social Signpost"), "Social"),
    (measure_column("Expert Signpost"), "Expert"),
    (measure_column("Service Signpost"), "Service"),
]

# Main conditional-safeguard comparison: enough fields to answer the question
# without turning every panel into a wall of labels.
SAFEGUARD_MAIN = [
    (measure_column("Risk Statement"), "Risk"),
    (measure_column("Legal Statement"), "Legal"),
    (measure_column("Social Signpost"), "Social"),
    (measure_column("Expert Signpost"), "Expert"),
    (measure_column("Service Signpost"), "Service"),
    (measure_column("System Identity"), "System"),
]

SAFEGUARD_ALL = [
    (measure_column("Risk Statement"), "Risk"),
    (measure_column("Legal Statement"), "Legal"),
    (measure_column("Eligibility Statement"), "Eligibility"),
    (measure_column("Social Signpost"), "Social"),
    (measure_column("Expert Signpost"), "Expert"),
    (measure_column("Service Signpost"), "Service"),
    (measure_column("System Identity"), "System"),
    (measure_column("Boundary Identity"), "Boundary"),
    (measure_column("Limitation Identity"), "Limitation"),
    (measure_column("Companion Identity"), "Companion"),
]

PRIMARY = [
    analysis.TRAJECTORY,
    analysis.THRESHOLD_CONTRAST,
    analysis.SIGNAL,
]
PRIMARY_LABEL = {
    analysis.TRAJECTORY: "Minor (Age) - Adult (Age)",
    analysis.THRESHOLD_CONTRAST: "Age 17 - Age 18",
    analysis.SIGNAL: "Minor (Age) - Minor (Cue)",
}

FIGURESPEC = {
    "rates": ("main", "safety_rates.pdf", "Safety and response rates by scenario"),
    "outcomes": ("main", "safety_outcomes.pdf", "Outcome cells by scenario and model"),
    "safeguards": ("main", "safety_safeguards.pdf", "Safeguards within refusal and compliance"),
    "trajectory": ("main", "safety_trajectory.pdf", "Refusal trajectory across exact ages"),
    "primary": ("main", "safety_primary.pdf", "Primary age-conditioning contrasts"),
    "controls": ("supplement", "safety_controls.pdf", "Age trajectories by scenario"),
    "cues": ("supplement", "safety_cues.pdf", "Refusal by age-signal strength"),
    "fields_age": ("supplement", "safety_fields_age.pdf", "Safeguards by age"),
    "fields_labels": ("supplement", "safety_fields_labels.pdf", "Safeguards by age signal"),
    "fields_scenarios": ("supplement", "safety_fields_scenarios.pdf", "Safeguards by scenario"),
    "failures": ("supplement", "safety_failures.pdf", "Directional safety failures"),
}

STALE = [
    "safety_basics.pdf",
    "safety_main_metrics.pdf",
    "safety_compliance_fields.pdf",
    "safety_characteristics.pdf",
    "safety_mismatch.pdf",
    "safety_age.pdf",
    "safety_labels.pdf",
    "safety_scenarios.pdf",
    "safety_failures_protective.pdf",
    "safety_failures_overrestriction.pdf",
]
for cell in analysis.CELL_ORDER:
    slug = cell.lower().replace(" ", "_")
    STALE += [
        f"safety_age_{slug}.pdf",
        f"safety_labels_{slug}.pdf",
        f"safety_scenarios_{slug}.pdf",
        f"safety_safeguards_{slug}.pdf",
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
        "axes.edgecolor": analysis.MUTED,
        "axes.labelcolor": "black",
        "axes.linewidth": 0.7,
        "text.color": "black",
        "xtick.color": analysis.MUTED,
        "ytick.color": analysis.MUTED,
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
    })
    return points


def grid(figsize=GRID_SIZE, **kwargs):
    return plt.subplots(2, 3, figsize=figsize, **kwargs)


def panel(ax, title=None, points=9):
    ax.set_facecolor(PANEL_FILL)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.25, color=analysis.MUTED)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, pad=points * 0.5, color="black")


def map_panel(ax, title=None, points=9):
    if title:
        ax.set_title(title, pad=points * 0.5, color="black")


def outer_labels(fig, axes, xlabel, ylabel, points):
    for ax in np.asarray(axes).flat:
        ax.label_outer()
    if xlabel:
        fig.supxlabel(xlabel, color="black", fontsize=points * 1.08)
    if ylabel:
        fig.supylabel(ylabel, color="black", fontsize=points * 1.15)


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


def add_grid_colorbar(fig, axes, image, points):
    top = axes[0, -1].get_position()
    bottom = axes[-1, -1].get_position()
    cax = fig.add_axes([
        top.x1 + 0.020,
        bottom.y0,
        0.014,
        top.y1 - bottom.y0,
    ])
    bar = fig.colorbar(image, cax=cax)
    bar.outline.set_visible(False)
    bar.ax.tick_params(
        labelsize=points * 0.85,
        length=0,
        labelcolor="black",
    )
    return bar


def add_single_colorbar(fig, ax, image, points):
    cax = make_axes_locatable(ax).append_axes(
        "right",
        size="3.5%",
        pad=0.18,
    )
    bar = fig.colorbar(image, cax=cax)
    bar.outline.set_visible(False)
    bar.ax.tick_params(
        labelsize=points * 0.85,
        length=0,
        labelcolor="black",
    )
    return bar


def readable_on(rgba):
    rgb = np.asarray(rgba[:3])
    luminance = float(np.dot(rgb, [0.2126, 0.7152, 0.0722]))
    return "white" if luminance < 0.48 else "black"


def nice_ceiling(maximum):
    if not np.isfinite(maximum) or maximum <= 0:
        return 1.0
    if maximum <= 1:
        return max(0.5, np.ceil(maximum * 10) / 10)
    if maximum <= 3:
        return np.ceil(maximum * 2) / 2
    if maximum <= 10:
        return np.ceil(maximum)
    if maximum <= 25:
        return np.ceil(maximum / 2) * 2
    if maximum <= 60:
        return np.ceil(maximum / 5) * 5
    return 100.0


# ---------------------------------------------------------------------
# Reduction helpers
# ---------------------------------------------------------------------

def load_register():
    if not REGISTER.exists():
        raise SystemExit(
            f"{REGISTER.relative_to(ROOT)} is missing. "
            "Run notebooks/15_safety.ipynb first."
        )
    raw = pd.read_csv(REGISTER)
    return analysis.adjust(raw) if hasattr(analysis, "adjust") else raw


def by_scenario(frame, measure, conditions=None):
    return analysis.by_scenario(frame, measure, conditions=conditions)


def rate_by_model(frame, measure, conditions=None):
    return analysis.rate_by_model(frame, measure, conditions=conditions).reindex(analysis.ORDER)


def available_fields(frame, candidates):
    return [(measure, label) for measure, label in candidates if measure in frame.columns]


def age_table(frame, intervals=False):
    rows = []
    for age in AGES:
        values = by_scenario(frame, "refusal", [AGE_CONDITION[age]])
        if values.empty:
            rows.append((age, np.nan, np.nan, np.nan))
            continue

        point = float(values.mean()) * 100
        if intervals:
            _, low, high = analysis.bootstrap_paired(values)
            low, high = 100 * low, 100 * high
        else:
            low = high = np.nan
        rows.append((age, point, low, high))

    return pd.DataFrame(rows, columns=["age", "point", "low", "high"]).set_index("age")


def register_rows(register, contrast):
    rows = analysis.ORDER[::-1] + [analysis.MACRO]
    part = register[register["contrast"].eq(contrast)].copy()
    part["order"] = pd.Categorical(part["model"], categories=rows, ordered=True)
    return part.sort_values("order")


def groups_for(kind, frame):
    if kind == "age":
        return [
            (f"Age {age}", frame[frame["condition"].eq(AGE_CONDITION[age])])
            for age in AGES
        ]
    if kind == "labels":
        return [
            (label, frame[frame["condition"].isin(conditions)])
            for label, conditions in LABEL_GROUPS
        ]
    if kind == "scenarios":
        return [
            (scenario, frame[frame["scenario_type"].eq(scenario)])
            for scenario in SCENARIOS
        ]
    raise ValueError(kind)


# ---------------------------------------------------------------------
# Heatmap helper
# ---------------------------------------------------------------------

def draw_heatmap(ax, table, points, vmax, annotate=True):
    blues = colormaps["Blues"]
    values = table.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if annotate:
        for row, col in np.ndindex(values.shape):
            value = values[row, col]
            if np.isfinite(value):
                rgba = blues(value / vmax if vmax else 0)
                ax.text(
                    col, row, f"{value:.1f}",
                    ha="center", va="center",
                    fontsize=points * 0.60,
                    fontweight="bold",
                    color=readable_on(rgba),
                )
    return image


# ---------------------------------------------------------------------
# Main 1: rates by scenario
# ---------------------------------------------------------------------

def scenario_rate_table(part, fields):
    return pd.DataFrame(
        {
            SCENARIO_AXIS[scenario]: [
                float(
                    by_scenario(
                        part[part["scenario_type"].eq(scenario)],
                        measure,
                    ).mean()
                ) * 100
                for measure, _ in fields
            ]
            for scenario in SCENARIOS
        },
        index=[label for _, label in fields],
    )


def draw_rates(returned, display):
    """Scenario-specific safety rates, one panel per model."""
    points = styled(display, 9.6, label_points=9.0)
    fields = available_fields(returned, RATE_FIELDS)

    tables = {
        model: scenario_rate_table(
            returned[returned["label"].eq(model)],
            fields,
        )
        for model in analysis.ORDER
    }
    vmax = nice_ceiling(
        max(
            np.nanmax(table.to_numpy(dtype=float))
            for table in tables.values()
        )
    )

    fig, axes = grid(figsize=(13.0, 8.0))
    fig.subplots_adjust(
        hspace=0.42,
        wspace=0.10,
        right=0.88,
        top=0.90,
        bottom=0.14,
    )

    image = None
    for index, (ax, model) in enumerate(
        zip(axes.flat, analysis.ORDER)
    ):
        table = tables[model]
        values = table.to_numpy(dtype=float)

        image = ax.imshow(
            values,
            cmap="Blues",
            vmin=0,
            vmax=vmax,
            aspect="auto",
        )

        ax.set_xticks(
            range(len(table.columns)),
            table.columns if index // 3 == 1 else [],
        )
        ax.set_yticks(
            range(len(table.index)),
            table.index if index % 3 == 0 else [],
        )
        ax.set_title(
            model,
            pad=points * 0.60,
            color="black",
        )
        ax.tick_params(length=0)

        for spine in ax.spines.values():
            spine.set_visible(False)

        blues = colormaps["Blues"]
        for row, col in np.ndindex(values.shape):
            value = values[row, col]
            if not np.isfinite(value):
                continue
            ax.text(
                col,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=points * 0.52,
                fontweight="bold",
                color=readable_on(
                    blues(value / vmax if vmax else 0)
                ),
            )

    add_grid_colorbar(fig, axes, image, points)
    return save(fig, FIGURESPEC["rates"][1])


# ---------------------------------------------------------------------
# Main 2: outcomes by scenario and model
# ---------------------------------------------------------------------

def draw_outcomes(returned, display):
    """Four outcome cells as grouped horizontal bars by scenario and model."""
    points = styled(display)
    blues = colormaps["Blues"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.2, 8.2),
        sharey=True,
        constrained_layout=True,
    )

    y = np.arange(len(analysis.ORDER))[::-1]
    height = 0.17
    offsets = np.linspace(-1.5 * height, 1.5 * height, len(SCENARIOS))

    for index, (ax, cell) in enumerate(
        zip(axes.flat, analysis.CELL_ORDER)
    ):
        panel(ax, cell, points)
        all_values = []

        for offset, scenario in zip(offsets, SCENARIOS):
            _, _, depth = TYPE_STYLE[scenario]
            part = returned[
                returned["scenario_type"].eq(scenario)
            ]
            values = rate_by_model(
                part,
                CELL_COLUMNS[cell],
            ).to_numpy(dtype=float)
            all_values.extend(values.tolist())

            bars = ax.barh(
                y + offset,
                values,
                height=height * 0.90,
                facecolor=blues(depth),
                edgecolor=blues(depth),
                linewidth=0.55,
                label=scenario if index == 0 else None,
                zorder=3,
            )

            maximum = max(values) if len(values) else 1.0
            for bar, value in zip(bars, values):
                if not np.isfinite(value):
                    continue
                ax.text(
                    value + max(0.04, maximum * 0.012),
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}",
                    ha="left",
                    va="center",
                    fontsize=points * 0.62,
                    color="black",
                )

        ax.grid(
            axis="x",
            linewidth=0.6,
            alpha=0.25,
            color=analysis.MUTED,
        )
        ax.grid(axis="y", visible=False)
        ax.set_yticks(
            y,
            [MODEL_AXIS[m] for m in analysis.ORDER],
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

        maximum = max(all_values) if all_values else 1.0
        minimum = min(all_values) if all_values else 0.0

        if cell == "Weak Refusal":
            ax.set_xlim(0, max(1.2, maximum * 1.40))
        elif cell == "Minimal Compliance":
            ax.set_xlim(0, max(7.0, maximum * 1.22))
        elif cell == "Strong Refusal":
            ax.set_xlim(0, min(100, maximum * 1.18))
        else:
            lower = max(0, minimum - 5.0)
            upper = min(100, maximum + 7.0)
            ax.set_xlim(lower, upper)

        if index % 2 == 1:
            ax.tick_params(labelleft=False)

    outer_labels(fig, axes, "Rate (%)", "", points)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend(fig, handles, labels, points, 4)
    return save(fig, FIGURESPEC["outcomes"][1])


# ---------------------------------------------------------------------
# Main 3: safeguards within Strong Refusal vs Total Compliance
# ---------------------------------------------------------------------

def conditional_field_rates(part, model, fields):
    model_part = part[part["label"].eq(model)]
    return pd.Series(
        {
            label: float(by_scenario(model_part, measure).mean()) * 100
            for measure, label in fields
        },
        dtype=float,
    )


def draw_safeguards(returned, display):
    """Safeguard fields within Strong Refusal and Total Compliance."""
    points = styled(display)
    fields = available_fields(returned, SAFEGUARD_MAIN)
    labels = [label for _, label in fields]
    x = np.arange(len(labels))
    width = 0.34

    strong = returned[
        returned[CELL_COLUMNS["Strong Refusal"]].eq(1.0)
    ]
    total = returned[
        returned[CELL_COLUMNS["Total Compliance"]].eq(1.0)
    ]

    fig, axes = grid(
        figsize=(13.5, 7.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for index, (ax, model) in enumerate(
        zip(axes.flat, analysis.ORDER)
    ):
        a = conditional_field_rates(
            strong,
            model,
            fields,
        ).reindex(labels)
        b = conditional_field_rates(
            total,
            model,
            fields,
        ).reindex(labels)

        ax.bar(
            x - width / 2,
            a.values,
            width=width,
            facecolor=analysis.COLOUR[model],
            edgecolor=analysis.COLOUR[model],
            linewidth=1.1,
            alpha=0.40,
            label="Strong Refusal" if index == 0 else None,
            zorder=3,
        )

        outline = ax.bar(
            x + width / 2,
            b.values,
            width=width,
            facecolor="none",
            edgecolor=analysis.COLOUR[model],
            linewidth=1.5,
            label="Total Compliance" if index == 0 else None,
            zorder=3,
        )
        for patch in outline:
            patch.set_linestyle("--")

        panel(ax, model, points)
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.55, len(labels) - 0.45)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(MultipleLocator(20))

        if index < 3:
            ax.tick_params(labelbottom=False)
        else:
            ax.tick_params(axis="x", labelsize=points * 0.78)
            ax.tick_params(axis="x", labelsize=points * 0.76)
            plt.setp(
                ax.get_xticklabels(),
                rotation=0,
                ha="center",
            )

    outer_labels(
        fig,
        axes,
        "",
        "Conditional Rate (%)",
        points,
    )
    handles, names = axes.flat[0].get_legend_handles_labels()
    legend(fig, handles, names, points, 2)
    return save(fig, FIGURESPEC["safeguards"][1])


# ---------------------------------------------------------------------
# Main 4: exact-age trajectory
# ---------------------------------------------------------------------

def age_bands(ax):
    ax.axvspan(6.4, 17.5, color=MINOR_BAND, zorder=0)
    ax.axvspan(17.5, 21.6, color=ADULT_BAND, zorder=0)
    ax.axvline(
        analysis.THRESHOLD,
        color=analysis.MUTED,
        linestyle=":",
        linewidth=1.0,
        zorder=1,
    )


def value_note(ax, x, y, text, colour, points, dx, dy, va):
    note = ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=points * 0.78,
        fontweight="bold",
        color=colour,
        zorder=6,
    )
    note.set_path_effects([
        pe.withStroke(linewidth=2.0, foreground="white"),
        pe.Normal(),
    ])


def draw_trajectory(focus, display):
    points = styled(display)
    fig, axes = grid(
        figsize=(13.5, 7.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax, model in zip(axes.flat, analysis.ORDER):
        part = focus[focus["label"].eq(model)]
        estimates = age_table(part, intervals=True)

        age_bands(ax)
        errors = np.vstack([
            estimates["point"] - estimates["low"],
            estimates["high"] - estimates["point"],
        ])

        ax.errorbar(
            AGES,
            estimates["point"],
            yerr=errors,
            marker=analysis.MARKER[model],
            markersize=4.8,
            markerfacecolor="white",
            markeredgecolor=analysis.COLOUR[model],
            markeredgewidth=1.1,
            linewidth=1.9,
            elinewidth=0.75,
            capsize=1.8,
            color=analysis.COLOUR[model],
            zorder=3,
        )

        panel(ax, model, points)
        ax.set_xticks(AGES)
        ax.set_xlim(6.4, 21.6)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(MultipleLocator(20))

        base_specs = {
            7: ((0, 11), "bottom"),
            17: ((-12, 12), "bottom"),
            18: ((12, -14), "top"),
        }
        for age, (offset, va) in base_specs.items():
            value = estimates.at[age, "point"]
            if not np.isfinite(value):
                continue

            dx, dy = offset
            if value > 94:
                dy, va = -14, "top"

            value_note(
                ax,
                age,
                value,
                f"{value:.1f}",
                analysis.COLOUR[model],
                points,
                dx,
                dy,
                va,
            )

    outer_labels(
        fig,
        axes,
        "Age",
        "Refusal Rate (%)",
        points,
    )
    legend(
        fig,
        [
            Patch(facecolor=MINOR_BAND),
            Patch(facecolor=ADULT_BAND),
            Line2D(
                [0],
                [0],
                color=analysis.MUTED,
                linestyle=":",
                linewidth=1.5,
            ),
            Line2D(
                [0],
                [0],
                color=analysis.MUTED,
                linewidth=1.2,
            ),
        ],
        ["Minor", "Adult", "Age 18", "95% CI"],
        points,
        4,
    )
    return save(fig, FIGURESPEC["trajectory"][1])


# ---------------------------------------------------------------------
# Main 5: primary contrasts
# ---------------------------------------------------------------------

def forest(ax, table):
    y = np.arange(len(table))

    for position, (_, row) in enumerate(table.iterrows()):
        model = row["model"]
        colour = analysis.INK if model == analysis.MACRO else analysis.COLOUR[model]
        marker = "D" if model == analysis.MACRO else analysis.MARKER[model]

        ax.plot(
            [row["low"], row["high"]],
            [position, position],
            color=colour,
            linewidth=analysis.LINEWIDTH,
            solid_capstyle="butt",
            zorder=2,
        )
        ax.plot(
            row["effect"],
            position,
            marker=marker,
            color=colour,
            markersize=analysis.MARKERSIZE + 0.5,
            linestyle="none",
            zorder=3,
        )

    ax.axvline(0, color=analysis.PALE, linewidth=1.0, linestyle=":", zorder=1)
    ax.set_yticks(y, [MODEL_AXIS[model] for model in table["model"]])
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_primary(register, display):
    points = styled(display)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=WIDE_SIZE,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    extent = register[register["contrast"].isin(PRIMARY)]
    low = min(0.0, float(extent["low"].min()))
    high = max(0.0, float(extent["high"].max()))
    span = max(high - low, 1.0)

    for index, (ax, contrast) in enumerate(zip(axes, PRIMARY)):
        part = register_rows(register, contrast)
        forest(ax, part)
        panel(ax, PRIMARY_LABEL[contrast], points)
        ax.set_xlim(low - 0.06 * span, high + 0.06 * span)

        if index > 0:
            ax.tick_params(labelleft=False)

    fig.supxlabel(
        "Effect on Refusal Rate (percentage points)",
        color="black",
        fontsize=points * 1.08,
    )
    return save(fig, FIGURESPEC["primary"][1])


# ---------------------------------------------------------------------
# Supplement 1: age trajectories for all scenario types
# ---------------------------------------------------------------------

def dynamic_ylim(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0, 1

    low = float(values.min())
    high = float(values.max())
    span = max(high - low, 1.0)

    if high <= 10:
        return 0, high + max(1.0, 0.18 * span)

    return max(0, low - 0.14 * span), min(100, high + 0.14 * span)


def draw_controls(returned, display):
    """
    Four scenario panels, so small Benign/Rights movements are not compressed
    by the much larger Age Restricted/Harmful ranges.
    """
    points = styled(display)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 8.0),
        sharex=True,
        constrained_layout=True,
    )

    for index, (ax, scenario) in enumerate(zip(axes.flat, SCENARIOS)):
        all_values = []

        for model in analysis.ORDER:
            part = returned[
                returned["label"].eq(model)
                & returned["scenario_type"].eq(scenario)
            ]
            estimates = age_table(part, intervals=False)
            all_values.extend(estimates["point"].dropna().tolist())

            ax.plot(
                AGES,
                estimates["point"],
                marker=analysis.MARKER[model],
                markersize=3.8,
                markerfacecolor="white",
                markeredgewidth=0.9,
                linewidth=1.45,
                color=analysis.COLOUR[model],
                label=model if index == 0 else None,
            )

        panel(ax, scenario, points)
        ax.set_xticks(AGES)
        ax.set_xlim(6.4, 21.6)

        lower, upper = dynamic_ylim(all_values)
        ax.set_ylim(lower, upper)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        if scenario == analysis.FOCUS:
            ax.axvline(
                analysis.THRESHOLD,
                color=analysis.MUTED,
                linestyle=":",
                linewidth=1.0,
                zorder=1,
            )

    outer_labels(fig, axes, "Age", "Refusal Rate (%)", points)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend(fig, handles, labels, points, 3)
    return save(fig, FIGURESPEC["controls"][1])


# ---------------------------------------------------------------------
# Supplement 2: signal ladder
# ---------------------------------------------------------------------

def draw_cues(returned, display):
    """Refusal across the five age-signal levels, one model per panel."""
    points = styled(display)
    x = np.arange(len(LABEL_GROUPS))

    fig, axes = grid(
        figsize=(13.5, 7.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for index, (ax, model) in enumerate(
        zip(axes.flat, analysis.ORDER)
    ):
        part = returned[
            returned["label"].eq(model)
            & returned["scenario_type"].eq(analysis.FOCUS)
        ]

        values = pd.Series(
            [
                float(
                    by_scenario(
                        part,
                        "refusal",
                        conditions,
                    ).mean()
                ) * 100
                for _, conditions in LABEL_GROUPS
            ],
            index=[label for label, _ in LABEL_GROUPS],
            dtype=float,
        )

        ax.plot(
            x,
            values.values,
            "-",
            marker=analysis.MARKER[model],
            markersize=4.8,
            markerfacecolor="white",
            markeredgecolor=analysis.COLOUR[model],
            markeredgewidth=1.1,
            linewidth=1.9,
            color=analysis.COLOUR[model],
            zorder=3,
        )

        panel(ax, model, points)
        ax.set_xticks(
            x,
            [SIGNAL_AXIS[label] for label in values.index],
        )
        ax.set_xlim(-0.45, 4.45)
        ax.set_ylim(0, 85)
        ax.yaxis.set_major_locator(MultipleLocator(20))

        # Label the explicit adult baseline and the two minor conditions.
        for xpos in (2, 3, 4):
            value = values.iloc[xpos]
            if not np.isfinite(value):
                continue
            note = ax.annotate(
                f"{value:.1f}",
                xy=(xpos, value),
                xytext=(0, 10 if value < 74 else -12),
                textcoords="offset points",
                ha="center",
                va="bottom" if value < 74 else "top",
                fontsize=points * 0.72,
                fontweight="bold",
                color=analysis.COLOUR[model],
                zorder=6,
            )
            note.set_path_effects([
                pe.withStroke(
                    linewidth=2.0,
                    foreground="white",
                ),
                pe.Normal(),
            ])

        if index < 3:
            ax.tick_params(labelbottom=False)
        else:
            plt.setp(
                ax.get_xticklabels(),
                rotation=0,
                ha="center",
            )

    outer_labels(
        fig,
        axes,
        "",
        "Refusal Rate (%)",
        points,
    )
    return save(fig, FIGURESPEC["cues"][1])


# ---------------------------------------------------------------------
# Supplement 3-5: safeguard field maps
# ---------------------------------------------------------------------

def draw_field_slice(returned, display, kind):
    points = styled(
        display,
        width_inches=12.0,
        label_points=8.1,
    )
    fields = available_fields(returned, SAFEGUARD_ALL)
    groups = groups_for(kind, returned)

    table = pd.DataFrame(
        {
            label: {
                group_label: float(
                    analysis.rate_by_model(
                        group_frame,
                        measure,
                    ).mean()
                )
                for group_label, group_frame in groups
            }
            for measure, label in fields
        }
    )

    values = table.to_numpy(dtype=float)
    vmax = nice_ceiling(np.nanmax(values))

    height = {
        "age": 6.2,
        "labels": 5.0,
        "scenarios": 4.6,
    }[kind]

    fig, ax = plt.subplots(
        figsize=(12.0, height),
        constrained_layout=True,
    )

    image = ax.imshow(
        values,
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        aspect="auto",
    )
    ax.set_xticks(
        range(len(table.columns)),
        table.columns,
    )
    ax.set_yticks(
        range(len(table.index)),
        table.index,
    )
    ax.tick_params(length=0)
    ax.tick_params(axis="x", labelsize=points * 0.78)

    # Explicitly keep every field label horizontal.
    plt.setp(
        ax.get_xticklabels(),
        rotation=0,
        ha="center",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    blues = colormaps["Blues"]
    for row, col in np.ndindex(values.shape):
        value = values[row, col]
        if not np.isfinite(value):
            continue
        ax.text(
            col,
            row,
            f"{value:.1f}",
            ha="center",
            va="center",
            fontsize=points * 0.56,
            fontweight="bold",
            color=readable_on(
                blues(value / vmax if vmax else 0)
            ),
        )

    add_single_colorbar(
        fig,
        ax,
        image,
        points,
    )
    return save(fig, FIGURESPEC[f"fields_{kind}"][1])


# ---------------------------------------------------------------------
# Supplement 6: directional failures
# ---------------------------------------------------------------------

def failure_series(returned):
    harmful = returned[returned["scenario_type"].eq("Harmful")]
    age_restricted = returned[returned["scenario_type"].eq("Age Restricted")]
    benign_rights = returned[returned["scenario_type"].isin(["Benign", "Rights"])]

    return {
        "Harmful Delivery": rate_by_model(harmful, "delivered"),
        "Minor (Age) Delivery": rate_by_model(
            age_restricted,
            "delivered",
            MINOR_AGE_CONDITIONS,
        ),
        "Adult (Age) Refusal": rate_by_model(
            age_restricted,
            "refusal",
            ADULT_AGE_CONDITIONS,
        ),
        "Benign / Rights Refusal": rate_by_model(
            benign_rights,
            "refusal",
        ),
    }


def paired_failure_panel(
    ax,
    first,
    second,
    points,
    title,
    first_label,
    second_label,
):
    y = np.arange(len(analysis.ORDER))[::-1]
    height = 0.30
    dark = colormaps["Blues"](0.82)
    light = colormaps["Blues"](0.48)

    bars_a = ax.barh(
        y + height / 2,
        first.to_numpy(dtype=float),
        height=height,
        facecolor=dark,
        edgecolor=dark,
        linewidth=0.6,
        label=first_label,
        zorder=3,
    )
    bars_b = ax.barh(
        y - height / 2,
        second.to_numpy(dtype=float),
        height=height,
        facecolor=light,
        edgecolor=light,
        linewidth=0.6,
        label=second_label,
        zorder=3,
    )

    panel(ax, title, points)
    ax.set_yticks(
        y,
        [MODEL_AXIS[m] for m in analysis.ORDER],
    )
    ax.grid(
        axis="x",
        linewidth=0.6,
        alpha=0.25,
        color=analysis.MUTED,
    )
    ax.grid(axis="y", visible=False)

    maximum = max(
        float(first.max()),
        float(second.max()),
    )
    ax.set_xlim(0, maximum * 1.22 + 0.2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    for bars in (bars_a, bars_b):
        for bar in bars:
            value = bar.get_width()
            ax.text(
                value + max(0.05, maximum * 0.012),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}",
                ha="left",
                va="center",
                fontsize=points * 0.68,
                color="black",
            )

    ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=points * 0.78,
    )


def draw_failures(returned, display):
    points = styled(
        display,
        width_inches=10.5,
        label_points=9.2,
    )
    series = failure_series(returned)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.5),
        sharey=True,
        constrained_layout=True,
    )

    paired_failure_panel(
        axes[0],
        series["Harmful Delivery"],
        series["Minor (Age) Delivery"],
        points,
        "Protective Failure",
        "Harmful Delivery",
        "Minor (Age) Delivery",
    )
    paired_failure_panel(
        axes[1],
        series["Adult (Age) Refusal"],
        series["Benign / Rights Refusal"],
        points,
        "Over-Restriction",
        "Adult (Age) Refusal",
        "Benign / Rights Refusal",
    )

    axes[1].tick_params(labelleft=False)
    fig.supxlabel(
        "Rate (%)",
        color="black",
        fontsize=points * 1.02,
    )
    return save(fig, FIGURESPEC["failures"][1])


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def registry(returned, focus, register, display):
    return {
        "rates": (lambda: draw_rates(returned, display), "main"),
        "outcomes": (lambda: draw_outcomes(returned, display), "main"),
        "safeguards": (lambda: draw_safeguards(returned, display), "main"),
        "trajectory": (lambda: draw_trajectory(focus, display), "main"),
        "primary": (lambda: draw_primary(register, display), "main"),
        "controls": (lambda: draw_controls(returned, display), "supplement"),
        "cues": (lambda: draw_cues(returned, display), "supplement"),
        "fields_age": (lambda: draw_field_slice(returned, display, "age"), "supplement"),
        "fields_labels": (lambda: draw_field_slice(returned, display, "labels"), "supplement"),
        "fields_scenarios": (lambda: draw_field_slice(returned, display, "scenarios"), "supplement"),
        "failures": (lambda: draw_failures(returned, display), "supplement"),
    }


def print_manifest():
    print("Safety Figure Set")
    print("-" * 94)
    for key, (tier, filename, purpose) in FIGURESPEC.items():
        print(f"{key:<18}{tier:<12}{filename:<32}{purpose}")
    print("-" * 94)


def main(args):
    mpl.rcParams.update(analysis.STYLE)
    clean_stale()

    frame = analysis.load_corpus()
    returned = frame[frame["responded"]].copy()
    focus = returned[returned["scenario_type"].eq(analysis.FOCUS)].copy()
    register = load_register()

    fingerprint = frame.attrs["fingerprint"]
    print(
        f"{fingerprint['requests']:,} requests, "
        f"{fingerprint['blocked']:,} blocked, "
        f"{fingerprint['returned']:,} returned, "
        f"rubric {fingerprint['policy']}\n"
    )

    print_manifest()
    jobs = registry(returned, focus, register, args.display)

    print("\nGenerating")
    print("-" * 94)

    built = 0
    for name, (build, tier) in jobs.items():
        if args.only not in {"all", name}:
            continue
        if args.set not in {"both", tier}:
            continue

        print(f"{name:<18}[{tier}] {FIGURESPEC[name][2]}")
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
