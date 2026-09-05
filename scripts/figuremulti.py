"""Dialogue figures from the frozen Notebook 19 tables.

Run: python scripts/figuremulti.py [--set main|supplement|all] [--only memory]
Requires the latest Notebook 19 exports, including dialogue_s14_boundary.csv.
No inference calls, classification, bootstrap reruns, or changes to source data.
Style and model/scenario colours are imported from figuresafe and analysis.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import analysis
import figuresafe as house

MODELS = list(analysis.ORDER) + [analysis.MACRO]
MODEL_LABELS = list(analysis.ORDER) + ["Macro-Average"]
STRATA = ["Age Restricted", "Harmful"]
TURNS = ["Turn 1", "Turn 2", "Turn 3"]
TURN_SHORT = {turn: f"T{i}" for i, turn in enumerate(TURNS, 1)}
TURN_LABELS = list(TURN_SHORT.values())
METHODS = ["Emotional Pushback", "Purpose Reverse", "Role Play"]
RADAR_STRATA = ["All Scenarios", "Age Restricted", "Harmful"]
CELLS = list(analysis.CELL_ORDER)
TURN_COLOURS = ["#8E8E93", "#376B8C", "#172F4D"]
METHOD_LABELS = ["Pushback", "Reverse", "Role Play"]
RADAR_LABEL = {
    "All Scenarios": "All",
    "Age Restricted": "Age Restricted",
    "Harmful": "Harmful",
}
RETIRED = [
    "dialogue_first_break",
    "dialogue_methods_age_restricted",
    "dialogue_methods_harmful",
    "dialogue_changes_age_restricted",
    "dialogue_changes_harmful",
    "dialogue_age_control",
    "dialogue_age_age_restricted",
    "dialogue_age_harmful",
    "dialogue_memory_gap",
    "dialogue_domains",
    "dialogue_trajectories_age_restricted",
    "dialogue_trajectories_harmful",
    "dialogue_failure_age_restricted",
    "dialogue_failure_harmful",
]
OUTCOME_COLOUR = {
    "Strong Refusal": "#006D77",
    "Weak Refusal": "#83C5BE",
    "Minimal Compliance": "#E9C46A",
    "Total Compliance": "#E76F51",
}
DIRECTION_COLOUR = {
    "Over-Permissive": "#D62828",
    "Over-Restrictive": "#4E342E",
}


class Figures:
    def __init__(self, tables, output, png=False, preview=True):
        self.tables, self.output, self.png = tables, output, png
        self.output.mkdir(parents=True, exist_ok=True)
        self.preview = preview
        self.manifest = []

    def clean_retired(self):
        for name in RETIRED:
            for suffix in (".pdf", ".png"):
                (self.output / f"{name}{suffix}").unlink(missing_ok=True)

    def read(self, name):
        paths = [self.tables / tier / f"{name}.csv" for tier in ("main", "supplement")]
        path = next((p for p in paths if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"Missing {name}.csv. Run the latest Notebook 19 first.")
        return pd.read_csv(path)

    def layout(self, rows=1, cols=2, width=13.5, height=6.4, maps=False):
        plt.rcParams.update(analysis.STYLE)
        # Exact house scaling; maps use smaller labels, as in the other scripts.
        points = house.styled(1.0, width, label_points=8.0 if maps else 9.0)
        fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False,
                                 layout="constrained")
        fig.get_layout_engine().set(w_pad=0.12, h_pad=0.12, wspace=0.12, hspace=0.16)
        return fig, axes, points

    def save(self, fig, key, tier, caption, source):
        forests = [ax for ax in fig.axes if getattr(ax, '_multi_forest', False)]
        if forests:
            limits = [ax.get_xlim() for ax in forests]
            shared = (min(v[0] for v in limits), max(v[1] for v in limits))
            for ax in forests:
                ax.set_xlim(shared)
        name = f"dialogue_{key}"
        target = self.output / f"{name}.pdf"
        for attempt in range(3):
            handle, temporary = tempfile.mkstemp(suffix=".pdf", dir=self.output)
            os.close(handle)
            fig.savefig(temporary, bbox_inches="tight")
            if Path(temporary).read_bytes()[-1024:].rstrip().endswith(b"%%EOF"):
                os.replace(temporary, target)
                break
            Path(temporary).unlink(missing_ok=True)
        else:
            raise OSError(f"Could not write a complete PDF after three attempts: {target}")
        if self.png:
            fig.savefig(self.output / f"{name}.png", dpi=120, bbox_inches="tight")
        self.manifest.append(dict(figure=name, tier=tier, source=source, caption=caption))
        plt.close(fig)
        print(f"  {name}.pdf")

    def close(self):
        if self.preview and self.manifest:
            pdfunite = shutil.which("pdfunite")
            if pdfunite:
                target = self.output / "dialogue_preview.pdf"
                handle, temporary = tempfile.mkstemp(suffix=".pdf", dir=self.output)
                os.close(handle)
                Path(temporary).unlink()
                inputs = [self.output / f"{item['figure']}.pdf" for item in self.manifest]
                subprocess.run([pdfunite, *map(str, inputs), temporary], check=True)
                os.replace(temporary, target)
            else:
                print("  dialogue_preview.pdf skipped: pdfunite is unavailable")
        (self.output / "captions.json").write_text(json.dumps(self.manifest, indent=2)+"\n")


def model_frame(frame):
    if frame.Model.duplicated().any():
        raise ValueError("A figure selection contains more than one row per model")
    result = frame.set_index("Model").reindex(MODELS)
    if result.isna().all(axis=1).any():
        raise ValueError("Missing model in figure selection")
    return result


def subset(d, **filters):
    for column, value in filters.items():
        d = d[d[column].eq(value)]
    return d


def heatmap(fig, ax, matrix, title, points, signed=False, limit=100, labels=None):
    a = np.asarray(matrix, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError(f"Non-finite heatmap cell in {title}")
    house.map_panel(ax, title, points)
    im = ax.imshow(a, aspect="auto", cmap="RdBu_r" if signed else "Blues",
                   vmin=-limit if signed else 0, vmax=limit)
    ax.set_xticks(range(a.shape[1]), labels or list(matrix.columns), rotation=0)
    ax.set_yticks(range(a.shape[0]), list(matrix.index))
    dense = a.shape[1] >= 4
    ax.tick_params(length=0, pad=7, labelsize=points * (.58 if dense else .83))
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (row, col), v in np.ndenumerate(a):
        if dense and abs(v) < .05:
            continue
        colour = house.readable_on(im.cmap(im.norm(v)))
        text = f"{v:+.1f}" if signed else f"{v:.1f}"
        ax.text(col, row, text, ha="center", va="center", color=colour,
                fontsize=points*(.68 if dense else .82), fontweight="bold")
    if a.shape[0] == 7:
        ax.axhline(5.5, color="white", linewidth=2.5)
    return im


def colourbar(fig, axes, image, label, points):
    bar = fig.colorbar(image, ax=list(np.asarray(axes).flat), shrink=.85, pad=.025,
                       fraction=.025, aspect=28)
    bar.set_label(label, fontsize=points*.85)
    bar.ax.tick_params(labelsize=points*.8)
    bar.outline.set_visible(False)


def forest(ax, frame, estimate, low, high, title, points, percent=False,
           show_labels=True, xlabel=None):
    ax._multi_forest = True
    d = model_frame(frame)
    x, lo, hi = (d[c].to_numpy(float) for c in (estimate, low, high))
    if not np.isfinite(np.r_[x, lo, hi]).all() or (lo > hi).any():
        raise ValueError(f"Invalid interval in {title}")
    house.panel(ax, title, points)
    ax.axvline(0, color=analysis.MUTED, linewidth=.8)
    for y, (label, value, lower, upper) in enumerate(zip(MODELS, x, lo, hi)):
        c = analysis.COLOUR.get(label, "#222222")
        ax.plot([lower, upper], [y, y], color=c, lw=2)
        ax.plot(value, y, marker="D" if label == analysis.MACRO else "o",
                color=c, markersize=6)
        # Dedicated right column prevents values colliding with confidence intervals.
        ax.text(1.025, y, f"{value:.1f}" if percent else f"{value:+.1f}",
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=points*.83, color=c, fontweight="bold", clip_on=False)
    ax.set_yticks(range(7), MODEL_LABELS if show_labels else [""] * 7,
                  fontsize=points*.78)
    ax.set_ylim(6.65, -.65)
    span = max(hi.max()-lo.min(), 10)
    ax.set_xlim((0, 100) if percent else (min(lo.min(), 0)-span*.07,
                                        max(hi.max(), 0)+span*.07))
    ax.axhline(5.5, color=analysis.MUTED, alpha=.35, lw=.7)
    if xlabel is None:
        xlabel = "Held both (%)" if percent else "Change (pp)"
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", length=0)


def zero_frame():
    return pd.DataFrame({
        "Model": MODELS,
        "Estimate": np.zeros(len(MODELS)),
        "Lower": np.zeros(len(MODELS)),
        "Upper": np.zeros(len(MODELS)),
    })


def radar_panel(ax, title, points):
    """Polar version of the shared figuresafe panel."""
    ax.set_facecolor(house.PANEL_FILL)
    ax.grid(color=analysis.MUTED, linewidth=.6, alpha=.25)
    ax.set_axisbelow(True)
    ax.spines["polar"].set_color(analysis.MUTED)
    ax.spines["polar"].set_linewidth(.7)
    ax.set_title(title, pad=points * 1.25, color="black")


def draw_methods_radar(f):
    """SafeDialBench-style model profiles across the three dialogue methods."""
    d = f.read("dialogue_05_methods")
    d = d[
        d["Scenario Type"].isin(RADAR_STRATA)
        & d["Method"].isin(METHODS)
        & d["Model"].isin(analysis.ORDER)
    ].copy()

    expected = len(RADAR_STRATA) * len(METHODS) * len(analysis.ORDER)
    if len(d) != expected:
        raise ValueError(
            f"Method radar expects {expected} stratum-method-model rows; found {len(d)}"
        )
    if d.duplicated(["Scenario Type", "Method", "Model"]).any():
        raise ValueError("Duplicate stratum-method-model row in method radar")

    # The three method branches share their opening. Enforce that design before
    # drawing Turn 1 as the common baseline in every radar.
    opening_spread = d.groupby(["Scenario Type", "Model"])[
        "Defect Turn 1 (%)"
    ].agg(lambda values: values.max() - values.min())
    if not np.allclose(opening_spread, 0, atol=.03):
        raise ValueError("Matched method branches do not share the same Turn 1 rate")

    plt.rcParams.update(analysis.STYLE)
    points = house.styled(1.0, 13.5, label_points=8.7)
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(13.5, 13.6),
        subplot_kw={"projection": "polar"},
        squeeze=False,
        layout="constrained",
    )
    fig.get_layout_engine().set(
        w_pad=.13, h_pad=.16, wspace=.10, hspace=.18
    )

    angles = np.linspace(0, 2 * np.pi, len(METHODS), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]

    for row, stratum in enumerate(RADAR_STRATA):
        part = subset(d, **{"Scenario Type": stratum})
        for col, turn in enumerate(TURNS):
            ax = axes[row, col]
            radar_panel(
                ax,
                f"{RADAR_LABEL[stratum]} ({TURN_SHORT[turn]})",
                points,
            )
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles, METHOD_LABELS)
            ax.tick_params(axis="x", pad=7, labelsize=points * .78, colors="black")
            ax.set_ylim(0, 100)
            ax.set_yticks([25, 50, 75, 100])
            ax.set_yticklabels(["25", "50", "75", "100"],
                               fontsize=points * .66, color=analysis.MUTED)
            ax.set_rlabel_position(205)

            for model in analysis.ORDER:
                model_part = subset(part, Model=model).set_index("Method").reindex(METHODS)
                if model_part.isna().all(axis=1).any():
                    raise ValueError(f"Missing method value for {stratum}: {model}")
                if turn == "Turn 1":
                    defect = model_part["Defect Turn 1 (%)"].to_numpy(float)
                else:
                    defect = (
                        model_part["Defect Turn 1 (%)"]
                        + model_part[f"Defect Change {turn} (pp)"]
                    ).to_numpy(float)
                alignment = 100 - defect
                if ((alignment < -.03) | (alignment > 100.03)).any():
                    raise ValueError(f"Action Alignment outside 0 to 100 for {stratum}: {model}")
                alignment = np.clip(alignment, 0, 100)
                closed = np.r_[alignment, alignment[0]]
                colour = analysis.COLOUR[model]
                ax.plot(
                    closed_angles,
                    closed,
                    color=colour,
                    marker=analysis.MARKER[model],
                    markersize=3.8,
                    linewidth=1.45,
                    label=model,
                    zorder=3,
                )
                ax.fill(closed_angles, closed, color=colour, alpha=.035, zorder=2)

    handles = [
        Line2D(
            [], [], color=analysis.COLOUR[model], marker=analysis.MARKER[model],
            linewidth=1.45, markersize=4.5, label=model
        )
        for model in analysis.ORDER
    ]
    fig.supylabel(r"Action Alignment, $A_t$ (%)",fontsize=points*.82)
    fig.legend(
        handles,
        analysis.ORDER,
        ncol=3,
        fontsize=points * .72,
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(.5, -.075),
        handlelength=2.0,
        handletextpad=.55,
        columnspacing=1.2,
    )
    f.save(
        fig,
        "methods_radar",
        "main",
        "Action Alignment, A_t, for all six models across Emotional Pushback, Purpose "
        "Reverse and Role Play. Panels show all scenarios, Age Restricted and Harmful at "
        "T1, T2 and T3. T1 is the matched common opening. Higher values indicate stronger "
        "benchmark-aligned action.",
        "dialogue_05_methods",
    )


def draw_outcomes(f):
    d = f.read("dialogue_01_outcomes")
    fig, axes, p = f.layout(2, 3, height=8.8)
    x = np.arange(len(TURNS))
    for ax, model in zip(axes.flat, analysis.ORDER):
        house.panel(ax, model, p)
        bottom = np.zeros(len(TURNS))
        total = np.zeros(len(TURNS))
        for cell in CELLS:
            row = subset(d, Model=model, Outcome=cell)
            if len(row) != 1:
                raise ValueError(f"Missing outcome row for {model}: {cell}")
            values = row[TURNS].iloc[0].to_numpy(float)
            total += values
            bars = ax.bar(x, values, bottom=bottom, width=.66,
                          color=OUTCOME_COLOUR[cell], edgecolor="white",
                          linewidth=.6, zorder=3)
            for bar, value, base in zip(bars, values, bottom):
                if value < 5:
                    continue
                note = ax.text(bar.get_x() + bar.get_width()/2, base + value/2,
                               f"{value:.1f}", ha="center", va="center",
                               fontsize=p*.62, color="white", fontweight="bold")
                note.set_path_effects([pe.withStroke(linewidth=1.2, foreground="#303030")])
            bottom += values
        if not np.allclose(total, 100, atol=.03):
            raise ValueError(f"Outcome distribution does not sum to 100 for {model}")
        ax.set_xticks(x, TURN_LABELS, rotation=0)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.label_outer()
    fig.supylabel("Replies (%)", fontsize=p*1.10, color="black")
    handles = [Patch(facecolor=OUTCOME_COLOUR[cell], edgecolor="white", label=cell)
               for cell in CELLS]
    house.legend(fig, handles, CELLS, p, ncol=4)
    f.save(fig, "outcomes", "main", "Four outcomes at each turn on the same complete cohort. "
           "Each model-turn distribution sums to 100%; scenarios and models are equally weighted "
           "at their respective averaging stages.", "dialogue_01_outcomes")


def draw_defects(f):
    d = f.read("dialogue_02_defects")
    fig, axes, p = f.layout(2, 3, height=8.8)
    for ax, label in zip(axes.flat, analysis.ORDER):
        house.panel(ax, label, p)
        series = {s: [subset(d, Model=label, **{"Scenario Type": s}).iloc[0]
                      [f"Defect {t} (%)"] for t in TURNS] for s in STRATA}
        for j, s in enumerate(STRATA):
            vals = series[s]
            ax.plot(range(3), vals, color=analysis.SCENARIO_COLOUR[s],
                    marker="s" if j==0 else "o", linestyle="--" if j==0 else "-", lw=2)
            # The higher series gets the upper label at each turn, including crossings.
            for i, v in enumerate(vals):
                above = v >= series[STRATA[1-j]][i] if j==0 else v > series[STRATA[0]][i]
                ax.annotate(f"{v:.1f}", (i, v), (0, 11 if above else -15),
                            textcoords="offset points", ha="center", va="center",
                            color=analysis.SCENARIO_COLOUR[s], fontsize=p*.66, fontweight="bold")
        ax.set_xticks(range(3), TURN_LABELS, rotation=0)
        ax.set_xlim(-.25, 2.25); ax.set_ylim(-7, 107)
        ax.set_yticks([0,25,50,75,100]); ax.label_outer()
    fig.supylabel(r"Action Defect, $D_t$ (%)", fontsize=p)
    handles=[Line2D([], [], color=analysis.SCENARIO_COLOUR[s], marker="s" if i==0 else "o",
                    ls="--" if i==0 else "-") for i,s in enumerate(STRATA)]
    house.legend(fig,handles,STRATA,p,ncol=2)
    f.save(fig,"defects","main","Action Defect by turn and stratum. Undefined age-neutral "
           "expectations are excluded; lines describe turn-specific rates, not cumulative breaks.",
           "dialogue_02_defects")


def draw_memory(f):
    d=f.read("dialogue_03_memory")
    fig, axes,p=f.layout(height=6.8,maps=True)
    for ax,s in zip(axes.flat,STRATA):
        part=model_frame(subset(d,**{"Scenario Type":s}))
        block=pd.DataFrame({
            "Turn 1":100.0,
            "Turn 2":part["Held Turn 2 (%)"],
            "Turn 3":part["Held Both (%)"],
        },index=part.index)
        block.index=MODEL_LABELS
        im=heatmap(fig,ax,block,s,p,labels=TURN_LABELS)
    colourbar(fig,axes,im,r"Safety Memory, $M_t$ (%)",p)
    f.save(fig,"memory","main","Safety Memory among dialogues that "
           "expected refusal and had Strong Refusal at Turn 1. Turn 1 is 100% by cohort "
           "definition; Turn 3 requires the boundary to hold at both later turns.",
           "dialogue_03_memory")


def draw_first_break(f):
    d=f.read("dialogue_03_memory")
    fig,axes,p=f.layout(height=6.6,maps=True)
    for ax,s in zip(axes.flat,STRATA):
        part=model_frame(subset(d,**{"Scenario Type":s}))
        partition=part[["First Break Turn 2 (%)","First Break Turn 3 (%)","Held Both (%)"]]
        if not np.allclose(partition.sum(axis=1),100,atol=.03):
            raise ValueError("First-break partition does not sum to 100")
        block=pd.DataFrame({
            "Turn 1":0.0,
            "Turn 2":part["First Break Turn 2 (%)"],
            "Turn 3":part["First Break Turn 2 (%)"]+part["First Break Turn 3 (%)"],
        },index=part.index)
        block.index=MODEL_LABELS
        im=heatmap(fig,ax,block,s,p,labels=TURNS)
    colourbar(fig,axes,im,"Cumulative first break (%)",p)
    f.save(fig,"first_break","main","Cumulative first loss of Strong Refusal among "
           "established protective boundaries. Turn 1 is 0% by cohort definition. A later "
           "return does not erase an earlier break.","dialogue_03_memory")


def draw_methods(f):
    d=f.read("dialogue_05_methods")
    for s in STRATA:
        fig,axes,p=f.layout(3,1,width=9.2,height=15.5,maps=True)
        part=subset(d,**{"Scenario Type":s})
        for ax,method in zip(axes.flat,METHODS):
            method_part=subset(part,Method=method).set_index("Model").reindex(MODELS)
            block=pd.DataFrame({
                "Turn 1":0.0,
                "Turn 2":method_part["Defect Change Turn 2 (pp)"],
                "Turn 3":method_part["Defect Change Turn 3 (pp)"],
            },index=MODELS)
            block.index=MODEL_LABELS
            im=heatmap(fig,ax,block,method,p,signed=True,limit=65,labels=TURNS)
        colourbar(fig,axes,im,"Action Defect change (pp)",p)
        f.save(fig,"methods_"+slug(s),"main",f"{s}: change in Action Defect from Turn 1, "
               "matched across all three methods on shared opening seeds. Positive values mean "
               "more defects. Purpose Reverse endpoint improvements do not establish recovery.",
               "dialogue_05_methods")


def draw_age(f):
    d=f.read("dialogue_04_age")
    fig,axes,p=f.layout(2,3,height=10.8)
    for row,s in enumerate(STRATA):
        for col,t in enumerate(TURNS):
            forest(
                axes[row,col],
                subset(d,Measure="Strong Refusal",Turn=t,**{"Scenario Type":s}),
                "Gap (pp)","Gap CI Lower","Gap CI Upper",
                f"{s} ({TURN_SHORT[t]})",p,show_labels=col==0,xlabel="",
            )
    fig.supxlabel(r"$G_t=S_{\mathrm{minor},t}-S_{18,t}$ (pp)",fontsize=p)
    f.save(
        fig,"age","main",
        "Strong-Refusal age gap, G_t, for Age Restricted and Harmful scenarios. "
        "S_minor,t is the mean Strong-Refusal rate across minor ages and S_18,t is the "
        "age-18 rate at turn t. Whiskers are paired scenario-bootstrap 95% intervals.",
        "dialogue_04_age",
    )


def draw_roleplay(f):
    d=f.read("dialogue_06_roleplay")
    fig,axes,p=f.layout(2,3,height=11.5)
    for row,arm in enumerate(["Age 9","Age 17"]):
        part=subset(d,Measure="Strong Refusal",Condition=arm+" minus Control")
        for col,t in enumerate(TURNS):
            frame=zero_frame() if t=="Turn 1" else part.rename(columns={
                f"Change {t} (pp)":"Estimate",
                f"Change {t} CI Lower":"Lower",
                f"Change {t} CI Upper":"Upper",
            })
            forest(axes[row,col],frame,"Estimate","Lower","Upper",
                   f"{arm} ({TURN_SHORT[t]})",p,show_labels=col==0,
                   xlabel="")
    fig.supxlabel(r"$\Delta_t G_a=G_{a,t}-G_{a,1}$ (pp)",fontsize=p)
    f.save(fig,"roleplay","main","Role Play: change from T1 in the age-control "
           "Strong-Refusal gap, where G_a,t = S_a,t - S_control,t. T1 is the zero "
           "reference; later whiskers are paired scenario-bootstrap 95% intervals. This is "
           "descriptive gap erosion, not proof of a displaced internal age state.",
           "dialogue_06_roleplay")


def draw_changes(f):
    d=f.read("dialogue_02_defects")
    for s in STRATA:
        fig,axes,p=f.layout(1,3,height=6.8)
        part=subset(d,**{"Scenario Type":s})
        for i,(ax,t) in enumerate(zip(axes.flat,TURNS)):
            if t=="Turn 1":
                frame=zero_frame()
                est,lo,hi="Estimate","Lower","Upper"
            else:
                frame=part
                est,lo,hi=f"Change {t} (pp)",f"Change {t} CI Lower",f"Change {t} CI Upper"
            forest(ax,frame,est,lo,hi,t,p,show_labels=i==0,
                   xlabel="Action Defect change (pp)")
        f.save(fig,"changes_"+slug(s),"supplement",f"{s}: Action Defect change from Turn 1 "
               "across all turns. Turn 1 is the zero reference; later whiskers are paired "
               "scenario-bootstrap 95% intervals.","dialogue_02_defects")


def draw_trajectories(f):
    d=f.read("dialogue_s01_trajectories")
    cols=["Aligned Aligned Aligned (%)","Aligned Defect Defect (%)",
          "Aligned Defect Aligned (%)","Aligned Aligned Defect (%)"]
    for s in STRATA:
        fig,axes,p=f.layout(3,1,width=10.5,height=16,maps=True)
        for ax,m in zip(axes.flat,METHODS):
            b=model_frame(subset(d,Method=m,**{"Scenario Type":s}))[cols];b.index=MODEL_LABELS
            if not np.allclose(b.sum(axis=1),100,atol=.03):raise ValueError("Route partition")
            im=heatmap(fig,ax,b,m,p,labels=["T1 A\nT2 A\nT3 A","T1 A\nT2 D\nT3 D",
                                                     "T1 A\nT2 D\nT3 A","T1 A\nT2 A\nT3 D"])
        colourbar(fig,axes,im,"Established boundaries (%)",p)
        f.save(fig,"trajectories_"+slug(s),"supplement",f"{s}: protective trajectories by method. "
               "All begin aligned. Columns correspond to A-A-A, A-D-D, A-D-A and A-A-D at Turns "
               "1/2/3; each row sums to 100%. Method cohorts here are descriptive, not the "
               "all-method matched contrast cohort.","dialogue_s01_trajectories")


def draw_ladder(f):
    d=f.read("dialogue_s08_ladder")
    fig,axes,p=f.layout(2,3,width=13.5,height=12,maps=True)
    for ax,label in zip(axes.flat,analysis.ORDER):
        b=subset(d,Measure="Strong Refusal",Model=label).set_index("Turn").reindex(TURNS)
        cols=[f"Age {a} (%)" for a in [7,9,11,13,15,17,18]]
        matrix=b[cols].T
        matrix.index=[7,9,11,13,15,17,18]
        im=heatmap(fig,ax,matrix,label,p,labels=TURN_LABELS)
        ax.set_ylabel("Age",fontsize=p*.8)
    colourbar(fig,axes,im,r"Strong Refusal, $S_t$ (%)",p)
    f.save(fig,"ladder","supplement","Strong Refusal across seven ages and three turns, "
           "on each model's balanced age cohort. This cohort is separate from the paired "
           "statutory-boundary figure.","dialogue_s08_ladder")


def draw_boundary(f):
    d=f.read("dialogue_s14_boundary")
    fig,axes,p=f.layout(1,3,width=13.5,height=6.8)
    for i,(ax,t) in enumerate(zip(axes.flat,TURNS)):
        forest(ax,subset(d,Measure="Strong Refusal",Turn=t),"17 minus 18 (pp)",
               "CI Lower","CI Upper",TURN_SHORT[t],p,show_labels=i==0)
        ax.set_xlabel("")
    fig.supxlabel(r"$S_{17,t}-S_{18,t}$ (pp)",fontsize=p)
    f.save(fig,"boundary","supplement","Strong-Refusal difference between ages 17 and 18 "
           "with paired scenario-bootstrap 95% intervals. Cohorts require these two ages only: "
           "24 Gemini scenarios and 25 for every other model.","dialogue_s14_boundary")


def draw_adjusted(f):
    d=f.read("dialogue_s12_adjusted")
    fig,axes,p=f.layout(1,3,height=6.8)
    for i,(ax,t) in enumerate(zip(axes.flat,TURNS)):
        frame=zero_frame() if t=="Turn 1" else subset(d,Measure="Strong Refusal",Turn=t).rename(
            columns={"Erosion Difference (pp)":"Estimate","CI Lower":"Lower","CI Upper":"Upper"})
        forest(ax,frame,"Estimate","Lower","Upper",TURN_SHORT[t],p,show_labels=i==0,
               xlabel="")
    fig.supxlabel(
        r"$\Delta_t G_{\mathrm{AR}}-\Delta_t G_{\mathrm{H}}$ (pp)",
        fontsize=p,
    )
    f.save(fig,"adjusted","supplement","Age-Restricted minus Harmful Strong-Refusal gap "
           "erosion from Turn 1 across all turns. Turn 1 is the zero reference. Later whiskers "
           "are scenario-bootstrap 95% intervals. Strata comprise different scenario sets, so "
           "this is a descriptive control comparison, not causal identification.",
           "dialogue_s12_adjusted")


def draw_memory_gap(f):
    d=f.read("dialogue_03_memory")
    check=f.read("dialogue_s13_memory_gap")
    age=model_frame(subset(d,**{"Scenario Type":"Age Restricted"}))
    harm=model_frame(subset(d,**{"Scenario Type":"Harmful"}))
    block=pd.DataFrame({
        "Turn 1":0.0,
        "Turn 2":age["Held Turn 2 (%)"]-harm["Held Turn 2 (%)"],
        "Turn 3":age["Held Both (%)"]-harm["Held Both (%)"],
    },index=age.index)
    held=model_frame(check)["Held Both, Age Restricted Minus Harmful (pp)"]
    if not np.allclose(block["Turn 3"],held,atol=.03):
        raise ValueError("Memory-gap figure does not reconcile with dialogue_s13_memory_gap")
    block.index=MODEL_LABELS
    fig,axes,p=f.layout(1,1,width=8.2,height=6.8,maps=True)
    im=heatmap(fig,axes[0,0],block,"Age Restricted vs Harmful",p,
               signed=True,limit=45,labels=TURNS)
    colourbar(fig,axes,im,r"$M_{\mathrm{AR},t}-M_{\mathrm{H},t}$ (pp)",p)
    f.save(fig,"memory_gap","supplement","Age-Restricted minus Harmful difference in "
           "Safety Memory. Turn 1 is 0 by cohort definition; Turn 3 requires "
           "the boundary to hold at both later turns. Negative values indicate weaker "
           "age-conditioned memory.","dialogue_03_memory; dialogue_s13_memory_gap")


def draw_directional(f):
    d=f.read("dialogue_s04_directional")
    fig,axes,p=f.layout(2,3,height=8.8)
    part=subset(d,**{"Scenario Type":"Age Restricted"})
    for ax,model in zip(axes.flat,analysis.ORDER):
        house.panel(ax,model,p)
        series={}
        for direction in DIRECTION_COLOUR:
            row=subset(part,Model=model,Direction=direction)
            series[direction]=[float(row[f"{t} (%)"].iloc[0]) for t in TURNS]
        for direction in DIRECTION_COLOUR:
            values=series[direction]
            ax.plot(range(3),values,color=DIRECTION_COLOUR[direction],lw=2,
                    marker="o" if direction=="Over-Permissive" else "s",
                    ls="-" if direction=="Over-Permissive" else "--")
            for x,value in enumerate(values):
                other=next(k for k in DIRECTION_COLOUR if k!=direction)
                above=value>=series[other][x]
                ax.annotate(f"{value:.1f}",(x,value),xytext=(0,9 if above else -13),
                            textcoords="offset points",ha="center",fontsize=p*.62,
                            color=DIRECTION_COLOUR[direction],fontweight="bold")
        ax.set_xticks(range(3),TURN_LABELS,rotation=0)
        ax.set_ylim(-5,105);ax.set_yticks([0,25,50,75,100]);ax.label_outer()
    fig.supylabel(r"Action Defect, $D_t$ (%)",fontsize=p*1.10,color="black")
    handles=[Line2D([],[],color=DIRECTION_COLOUR[k],lw=2,
                    marker="o" if k=="Over-Permissive" else "s",
                    ls="-" if k=="Over-Permissive" else "--") for k in DIRECTION_COLOUR]
    house.legend(fig,handles,list(DIRECTION_COLOUR),p,ncol=2)
    f.save(fig,"directional","supplement","Age Restricted: directional Action Defect rates "
           "at all three turns. Over-Permissive means failure to "
           "maintain Strong Refusal when refusal is expected; Over-Restrictive means failure to "
           "provide Total Compliance when compliance is expected.","dialogue_s04_directional")


def draw_failure(f):
    d=f.read("dialogue_s09_failure")
    fig,axes,p=f.layout(2,3,height=10.8,maps=True)
    for row,s in enumerate(STRATA):
        for col,t in enumerate(TURNS):
            columns=[c+" (%)" for c in CELLS]
            b=model_frame(subset(d,Turn=t,**{"Scenario Type":s}))[columns]
            if not np.allclose(b.sum(axis=1),100,atol=.03):
                raise ValueError(f"Failure composition does not sum to 100: {s}, {t}")
            b.index=MODEL_LABELS
            im=heatmap(fig,axes[row,col],b,f"{s} ({TURN_SHORT[t]})",p,
                       labels=["Strong","Weak","Minimal","Total"])
    colourbar(fig,axes,im,"Replies in cohort (%)",p)
    f.save(fig,"failure","supplement","Outcome composition within the initially protective "
           "cohort for Age Restricted and Harmful scenarios at T1, T2 and T3. Strong and Weak "
           "denote refusal; Minimal and Total denote compliance. Strong Refusal at T3 is "
           "endpoint memory and does not necessarily indicate an unbroken boundary.",
           "dialogue_s09_failure")


def draw_domains(f):
    d=f.read("dialogue_s10_domains").sort_values("Defect Turn 3 (%)",ascending=False)
    block=d.set_index("Domain")[[f"Defect {t} (%)" for t in TURNS]]
    block.columns=TURNS
    fig,axes,p=f.layout(1,1,width=8.8,height=8.2,maps=True)
    im=heatmap(fig,axes[0,0],block,"Harm category",p,labels=TURNS)
    axes[0,0].set_ylabel("Domain",fontsize=p)
    colourbar(fig,axes,im,"Action Defect (%)",p)
    f.save(fig,"domains","supplement","Macro-Average Action Defect by harm category and "
           "turn. Domains are ordered by Turn-3 defect rate. Scenario counts differ by domain, "
           "so this figure is descriptive.","dialogue_s10_domains")


def slug(s):
    return s.lower().replace(" ","_")


MAIN={"outcomes":draw_outcomes,"defects":draw_defects,"memory":draw_memory,
      "methods_radar":draw_methods_radar,"age":draw_age,"roleplay":draw_roleplay}
SUPPLEMENT={"adjusted":draw_adjusted,"ladder":draw_ladder,
            "boundary":draw_boundary,"directional":draw_directional,
            "failure":draw_failure}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables",type=Path,default=analysis.TABLES)
    parser.add_argument("--output",type=Path,default=analysis.FIGURES/"dialogue")
    parser.add_argument("--set",choices=["main","supplement","all"],default="all")
    parser.add_argument("--only",nargs="+",choices=list(MAIN)+list(SUPPLEMENT))
    parser.add_argument("--png",action="store_true",help="Also export 120-dpi PNGs")
    parser.add_argument("--no-preview",action="store_true")
    args=parser.parse_args()
    jobs=(MAIN if args.set=="main" else SUPPLEMENT if args.set=="supplement" else MAIN|SUPPLEMENT)
    if args.only:jobs={k:v for k,v in jobs.items() if k in args.only}
    if not jobs:parser.error("No figures match --set and --only")
    f=Figures(args.tables,args.output,args.png,not args.no_preview)
    f.clean_retired()
    try:
        for job in jobs.values():job(f)
    finally:
        f.close()


if __name__=="__main__":
    main()
