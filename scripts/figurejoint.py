"""Joint-analysis figures for the thesis.

Designed to mirror the grammar of figureread.py and figuresafe.py:
- same typography scaling idea;
- restrained panel styling;
- full model names;
- bottom legends;
- outer labels where useful;
- no decorative titles.

Examples
--------
python scripts/figurejoint.py
python scripts/figurejoint.py --set main
python scripts/figurejoint.py --only shape
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------
# Optional project imports; fall back to local constants if unavailable.
# ---------------------------------------------------------------------

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analysis  # type: ignore
    from settings import ROOT  # type: ignore

    COLOUR = analysis.COLOUR
    INK = analysis.INK
    MUTED = analysis.MUTED
    MACRO = getattr(analysis, "MACRO", "Macro-average")
    SCENARIO_ORDER = list(getattr(analysis, "SCENARIO_ORDER", [
        "Benign", "Rights", "Age Restricted", "Harmful"
    ]))
    PROJECT_ROOT = ROOT
except Exception:
    analysis = None
    PROJECT_ROOT = Path(__file__).resolve().parent
    COLOUR = {
        "GPT-5.6 Luna": "#0B7A3B",
        "Claude Haiku 4.5": "#C58F16",
        "Gemini 3.5 Flash Lite": "#352A8F",
        "DeepSeek-V4 Flash": "#73BFE2",
        "Mistral Small 4": "#B13A9B",
        "Gemma 4 31B": "#A33D2D",
    }
    INK = "#1F1F1F"
    MUTED = "#6F6F6F"
    MACRO = "Macro-average"
    SCENARIO_ORDER = ["Benign", "Rights", "Age Restricted", "Harmful"]

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

FIGURES = PROJECT_ROOT / "figures" / "joint"
TEXT_WIDTH_CM = 16.0
LABEL_POINTS = 9.8
GRID_SIZE = (13.5, 7.8)
WIDE_SIZE = (13.5, 5.0)
FOREST_SIZE = (13.5, 7.0)

PANEL_FILL = "#F5F5F5"
MINOR_BAND = "#E8EDF2"
ADULT_BAND = "#F2EDE8"

MODEL_ORDER = [
    "GPT-5.6 Luna",
    "Claude Haiku 4.5",
    "Gemini 3.5 Flash Lite",
    "DeepSeek-V4 Flash",
    "Mistral Small 4",
    "Gemma 4 31B",
]
MODEL_AXIS = {
    "GPT-5.6 Luna": "GPT-5.6\nLuna",
    "Claude Haiku 4.5": "Claude\nHaiku 4.5",
    "Gemini 3.5 Flash Lite": "Gemini 3.5\nFlash Lite",
    "DeepSeek-V4 Flash": "DeepSeek-V4\nFlash",
    "Mistral Small 4": "Mistral\nSmall 4",
    "Gemma 4 31B": "Gemma 4\n31B",
    MACRO: "Macro-\nAverage",
    "Panel Ratio": "Panel\nRatio",
}

AGES = [7, 9, 11, 13, 15, 17, 18, 21]
SHAPE_COLUMNS = [
    "Across Childhood (7 to 17)",
    "Step at the Boundary (17 to 18)",
    "Above the Boundary (18 to 21)",
]
COMPONENT_LABEL = {
    "Across Childhood (7 to 17)": "Across childhood",
    "Step at the Boundary (17 to 18)": "Step at the boundary",
    "Above the Boundary (18 to 21)": "Above the boundary",
}
COMPONENT_COLOUR = {
    "Across Childhood (7 to 17)": "#B8C7D9",
    "Step at the Boundary (17 to 18)": "#4A6C8C",
    "Above the Boundary (18 to 21)": "#D9BE98",
}

CONCORDANCE_ORDER = ["Pooled", "Benign", "Rights", "Age Restricted", "Harmful"]
FIGURESPEC = {
    "ladders": ("main", "joint_ladders.pdf", "Age ladders for refusal and grade level"),
    "shape": ("main", "joint_shape.pdf", "Boundary-shape decomposition"),
    "concordance": ("main", "joint_concordance.pdf", "Concordance heatmap"),
    "concordance_forest": (
        "supplement",
        "joint_concordance_forest.pdf",
        "Concordance estimates with confidence intervals",
    ),
}

# ---------------------------------------------------------------------
# Shared figure grammar
# ---------------------------------------------------------------------

def styled(display, width_inches=7.4, label_points=None):
    scale = display * TEXT_WIDTH_CM / (width_inches * 2.54)
    points = (LABEL_POINTS if label_points is None else label_points) / scale
    plt.rcParams.update({
        "font.size": points,
        "axes.labelsize": points,
        "axes.titlesize": points * 1.03,
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


def panel(ax, title=None, points=9, ygrid=True):
    ax.set_facecolor(PANEL_FILL)
    if ygrid:
        ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.25, color=MUTED)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, pad=points * 0.45, color="black")


def heatmap_panel(ax, title=None, points=9):
    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, pad=points * 0.45, color="black")


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")
    return path


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def locate(filename: str) -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here / filename,
        Path.cwd() / filename,
        PROJECT_ROOT / filename,
        PROJECT_ROOT / "tables" / "main" / filename,
        Path("/mnt/data") / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(filename)


def load_ladders(path=None):
    path = locate("joint_01_ladders(2).csv") if path is None else Path(path)
    if not path.exists():
        path = locate("joint_01_ladders.csv")
    return pd.read_csv(path)


def load_shape(path=None):
    path = locate("joint_02_shape(2).csv") if path is None else Path(path)
    if not path.exists():
        path = locate("joint_02_shape.csv")
    return pd.read_csv(path)


def load_concordance(path=None):
    path = locate("joint_03_concordance(2).csv") if path is None else Path(path)
    if not path.exists():
        path = locate("joint_03_concordance.csv")
    return pd.read_csv(path)


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def figure_ladders():
    df = load_ladders()
    points = styled(display=1, width_inches=WIDE_SIZE[0])
    fig, axes = plt.subplots(1, 2, figsize=WIDE_SIZE)

    measures = ["Refusal Rate (%)", "Grade Level"]
    x = np.arange(len(AGES))

    for ax, measure in zip(axes, measures):
        panel(ax, measure, points=points)
        ax.axvspan(-0.5, 5.5, color=MINOR_BAND, alpha=0.55, zorder=0)
        ax.axvspan(5.5, 7.5, color=ADULT_BAND, alpha=0.6, zorder=0)
        ax.axvline(5.5, color=MUTED, linewidth=0.9, linestyle="--", zorder=1)

        sub = df[df["Measure"] == measure].copy()
        for model in MODEL_ORDER:
            row = sub[sub["Model"] == model]
            if row.empty:
                continue
            y = row[[f"Age {age}" for age in AGES]].iloc[0].astype(float).to_numpy()
            ax.plot(
                x, y,
                color=COLOUR[model],
                marker="o",
                linewidth=1.9,
                markersize=4.1,
                label=model,
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([str(age) for age in AGES], rotation=0)
        ax.set_xlabel("Stated age")
        ax.set_xlim(-0.3, 7.3)

        if measure == "Refusal Rate (%)":
            ax.set_ylabel("Refusal rate (%)")
            ax.set_ylim(0, max(82, np.nanmax(sub[[f"Age {age}" for age in AGES]].to_numpy()) + 4))
        else:
            ax.set_ylabel("Grade level")
            ax.set_ylim(4.0, max(10.2, np.nanmax(sub[[f"Age {age}" for age in AGES]].to_numpy()) + 0.35))

    handles = [Line2D([0], [0], color=COLOUR[m], marker='o', linewidth=1.9, markersize=4.5)
               for m in MODEL_ORDER]
    labels = MODEL_ORDER
    fig.legend(handles, labels, ncol=3, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, -0.02), handlelength=2.2,
               handletextpad=0.6, columnspacing=1.4)
    fig.subplots_adjust(bottom=0.25, wspace=0.25)
    return save(fig, FIGURESPEC["ladders"][1])


def figure_shape():
    df = load_shape()
    points = styled(display=1, width_inches=WIDE_SIZE[0])
    fig, axes = plt.subplots(1, 2, figsize=WIDE_SIZE)

    panel_rows = [
        ("Refusal Rate (pp)", "Change in refusal rate (pp)"),
        ("Grade Level", "Change in grade level"),
    ]

    for ax, (measure, xlabel) in zip(axes, panel_rows):
        panel(ax, measure, points=points)
        sub = df[(df["Measure"] == measure) & (df["Model"] != "Panel Ratio")].copy()
        order = MODEL_ORDER + [MACRO]
        sub["Model"] = pd.Categorical(sub["Model"], categories=order, ordered=True)
        sub = sub.sort_values("Model")
        y = np.arange(len(sub))

        left = np.zeros(len(sub), dtype=float)
        for column in SHAPE_COLUMNS:
            values = sub[column].astype(float).to_numpy()
            ax.barh(
                y,
                values,
                left=left,
                height=0.68,
                color=COMPONENT_COLOUR[column],
                edgecolor="white",
                linewidth=0.8,
                label=COMPONENT_LABEL[column],
            )
            left = left + values

        totals = sub["Full Range (7 to 21)"].astype(float).to_numpy()
        share = sub["Step as Share of Range (%)"].astype(float).to_numpy()
        for yi, total, frac in zip(y, totals, share):
            dx = 0.9 if total >= 0 else -0.9
            ha = "left" if total >= 0 else "right"
            ax.text(total + dx, yi, f"{frac:.1f}%", va="center", ha=ha, color=INK)

        ax.axvline(0, color=MUTED, linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([MODEL_AXIS.get(m, m) for m in sub["Model"]])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)

        data = sub[SHAPE_COLUMNS + ["Full Range (7 to 21)"]].astype(float).to_numpy().ravel()
        lim = max(abs(np.nanmin(data)), abs(np.nanmax(data))) + 6
        ax.set_xlim(-lim if measure == "Grade Level" else -2, lim if measure == "Refusal Rate (pp)" else 1.4)

    handles = [Patch(facecolor=COMPONENT_COLOUR[c], edgecolor="white", label=COMPONENT_LABEL[c])
               for c in SHAPE_COLUMNS]
    fig.legend(handles, [COMPONENT_LABEL[c] for c in SHAPE_COLUMNS], ncol=3,
               loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.02),
               handlelength=1.5, handletextpad=0.6, columnspacing=1.5)
    fig.subplots_adjust(bottom=0.22, wspace=0.32)
    return save(fig, FIGURESPEC["shape"][1])


def figure_concordance():
    df = load_concordance().copy()
    points = styled(display=1, width_inches=7.5)
    fig, ax = plt.subplots(figsize=(7.9, 4.9))
    heatmap_panel(ax, points=points)

    df["Model"] = pd.Categorical(df["Model"], categories=MODEL_ORDER, ordered=True)
    df["Scenario Type"] = pd.Categorical(df["Scenario Type"], categories=CONCORDANCE_ORDER, ordered=True)
    df = df.sort_values(["Model", "Scenario Type"])

    pivot = df.pivot(index="Model", columns="Scenario Type", values="rho").reindex(index=MODEL_ORDER, columns=CONCORDANCE_ORDER)
    values = pivot.to_numpy(dtype=float)
    cmap = colormaps["RdBu_r"].copy()
    cmap.set_bad("#FFFFFF")
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-0.85, vmax=0.85)

    ax.set_xticks(np.arange(len(CONCORDANCE_ORDER)))
    ax.set_xticklabels(["Pooled", "Benign", "Rights", "Age\nRestricted", "Harmful"], rotation=0)
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_AXIS[m] for m in MODEL_ORDER])

    for i, model in enumerate(MODEL_ORDER):
        for j, scenario in enumerate(CONCORDANCE_ORDER):
            row = df[(df["Model"] == model) & (df["Scenario Type"] == scenario)]
            if row.empty:
                continue
            reason = str(row["Reason"].iloc[0])
            rho = row["rho"].iloc[0]
            txt = ""
            colour = INK
            if pd.notna(rho):
                txt = f"{rho:.2f}"
                if "small stratum" in reason:
                    txt += "†"
            elif "constant" in reason:
                txt = "const."
            elif "not-estimable" in reason:
                txt = "n.e."
            if txt:
                if pd.notna(rho) and abs(float(rho)) > 0.45:
                    colour = "white"
                ax.text(j, i, txt, ha="center", va="center", color=colour)

    for x in np.arange(-0.5, len(CONCORDANCE_ORDER), 1):
        ax.axvline(x, color="#E5E5E5", linewidth=0.8, zorder=3)
    for y in np.arange(-0.5, len(MODEL_ORDER), 1):
        ax.axhline(y, color="#E5E5E5", linewidth=0.8, zorder=3)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.outline.set_linewidth(0.7)
    cbar.ax.set_ylabel("Spearman rho", rotation=90)

    ax.text(0.0, -0.18, "Cells marked † are small strata and should be interpreted cautiously.",
            transform=ax.transAxes, ha="left", va="top", color=INK)
    fig.subplots_adjust(bottom=0.24, right=0.88)
    return save(fig, FIGURESPEC["concordance"][1])


def figure_concordance_forest():
    df = load_concordance().copy()
    points = styled(display=1, width_inches=FOREST_SIZE[0])
    fig, axes = plt.subplots(2, 3, figsize=FOREST_SIZE, sharex=True)
    axes = axes.ravel()

    panels = ["Pooled", "Rights", "Age Restricted", "Harmful", "Benign"]
    xlim = (-0.9, 0.9)

    for ax, scenario in zip(axes, panels + [None]):
        if scenario is None:
            ax.axis("off")
            continue
        panel(ax, scenario, points=points, ygrid=False)
        sub = df[df["Scenario Type"] == scenario].copy()
        sub["Model"] = pd.Categorical(sub["Model"], categories=MODEL_ORDER, ordered=True)
        sub = sub.sort_values("Model")
        y = np.arange(len(sub))
        ax.axvline(0, color=MUTED, linewidth=0.9, linestyle="--")

        if scenario == "Benign":
            ax.set_xlim(*xlim)
            ax.set_ylim(-0.5, len(MODEL_ORDER) - 0.5)
            ax.set_yticks(y)
            ax.set_yticklabels([MODEL_AXIS[m] for m in sub["Model"]])
            for yi, reason in zip(y, sub["Reason"].tolist()):
                label = "Constant" if "constant" in str(reason) else ""
                ax.text(0, yi, label, ha="center", va="center", color=INK)
            ax.invert_yaxis()
            continue

        for yi, row in zip(y, sub.itertuples(index=False)):
            reason = str(row.Reason)
            rho = row.rho
            lo = getattr(row, "_4") if False else None
            # Safer column access by name for spaces in headers.
        for yi, (_, row) in zip(y, sub.iterrows()):
            reason = str(row["Reason"])
            rho = row["rho"]
            low = row["95% CI Lower"]
            high = row["95% CI Upper"]
            if pd.notna(rho):
                ax.hlines(yi, low, high, color=COLOUR[row["Model"]], linewidth=1.6)
                ax.plot(rho, yi, marker="o", color=COLOUR[row["Model"]], markersize=4.4)
                txt = f"{rho:.2f}"
                if "small stratum" in reason:
                    txt += "†"
                ax.text(xlim[1] - 0.03, yi, txt, ha="right", va="center", color=INK)
            elif "constant" in reason:
                ax.text(0, yi, "Constant", ha="center", va="center", color=INK)
            else:
                ax.text(0, yi, "n.e.", ha="center", va="center", color=INK)

        ax.set_xlim(*xlim)
        ax.set_yticks(y)
        ax.set_yticklabels([MODEL_AXIS[m] for m in sub["Model"]])
        ax.invert_yaxis()

    for ax in axes[:5]:
        ax.set_xlabel("Spearman rho")

    fig.text(0.02, 0.01, "† small stratum, interpret cautiously.", ha="left", va="bottom", color=INK)
    fig.subplots_adjust(bottom=0.10, wspace=0.28, hspace=0.34)
    return save(fig, FIGURESPEC["concordance_forest"][1])


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build(which):
    if which == "ladders":
        return figure_ladders()
    if which == "shape":
        return figure_shape()
    if which == "concordance":
        return figure_concordance()
    if which == "concordance_forest":
        return figure_concordance_forest()
    raise ValueError(which)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=["main", "supplement", "all"], default="all")
    parser.add_argument("--only", choices=list(FIGURESPEC), default=None)
    args = parser.parse_args(argv)

    if args.only:
        build(args.only)
        return

    for name, (group, _, _) in FIGURESPEC.items():
        if args.set != "all" and group != args.set:
            continue
        build(name)


if __name__ == "__main__":
    main()
