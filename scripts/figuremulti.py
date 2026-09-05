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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import analysis
import figuresafe as house

MODELS = list(analysis.ORDER) + [analysis.MACRO]
MODEL_LABELS = list(analysis.ORDER) + ["Macro-Average"]
STRATA = ["Age Restricted", "Harmful"]
TURNS = ["Turn 1", "Turn 2", "Turn 3"]
METHODS = ["Emotional Pushback", "Purpose Reverse", "Role Play"]
CELLS = list(analysis.CELL_ORDER)
TURN_COLOURS = ["#8E8E93", "#376B8C", "#172F4D"]
CAPTIONS = {}


class Figures:
    def __init__(self, tables, output, png=False, preview=True):
        self.tables, self.output, self.png = tables, output, png
        self.output.mkdir(parents=True, exist_ok=True)
        self.preview = preview
        self.manifest = []

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
            for i, ax in enumerate(forests):
                ax.set_xlim(shared)
                if i:
                    ax.tick_params(axis='y', labelleft=False)
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
    ax.grid(False)
    ax.set_facecolor("white")
    im = ax.imshow(a, aspect="auto", cmap="RdBu_r" if signed else "Blues",
                   vmin=-limit if signed else 0, vmax=limit)
    ax.set_xticks(range(a.shape[1]), labels or list(matrix.columns), rotation=0)
    ax.set_yticks(range(a.shape[0]), list(matrix.index))
    ax.tick_params(length=0, pad=7, labelsize=points * .83)
    ax.set_title(title, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (row, col), v in np.ndenumerate(a):
        colour = house.readable_on(im.cmap(im.norm(v)))
        text = f"{v:+.1f}" if signed else f"{v:.1f}"
        ax.text(col, row, text, ha="center", va="center", color=colour,
                fontsize=points*.86, fontweight="bold")
    if a.shape[0] == 7:
        ax.axhline(5.5, color="white", linewidth=2.5)
    return im


def colourbar(fig, axes, image, label, points):
    bar = fig.colorbar(image, ax=list(np.asarray(axes).flat), shrink=.85, pad=.025,
                       fraction=.025, aspect=28)
    bar.set_label(label, fontsize=points*.85)
    bar.ax.tick_params(labelsize=points*.8)
    bar.outline.set_visible(False)


def forest(ax, frame, estimate, low, high, title, points, percent=False):
    ax._multi_forest = True
    d = model_frame(frame)
    x, lo, hi = (d[c].to_numpy(float) for c in (estimate, low, high))
    if not np.isfinite(np.r_[x, lo, hi]).all() or (lo > hi).any():
        raise ValueError(f"Invalid interval in {title}")
    house.panel(ax, title, points)
    ax.grid(False)
    ax.grid(axis="x", color=analysis.MUTED, alpha=.25, linewidth=.6)
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
    ax.set_yticks(range(7), MODEL_LABELS, fontsize=points*.78)
    ax.set_ylim(6.65, -.65)
    span = max(hi.max()-lo.min(), 10)
    ax.set_xlim((0, 100) if percent else (min(lo.min(), 0)-span*.07,
                                        max(hi.max(), 0)+span*.07))
    ax.axhline(5.5, color=analysis.MUTED, alpha=.35, lw=.7)
    ax.set_xlabel("Held both (%)" if percent else "Change (pp)")
    ax.tick_params(axis="y", length=0)


def draw_outcomes(f):
    d = f.read("dialogue_01_outcomes")
    fig, axes, p = f.layout(2, 2, height=10, maps=True)
    for ax, cell in zip(axes.flat, CELLS):
        block = model_frame(subset(d, Outcome=cell))[TURNS]
        block.index = MODEL_LABELS
        im = heatmap(fig, ax, block, cell, p)
    colourbar(fig, axes, im, "Replies (%)", p)
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
        ax.set_xticks(range(3), TURNS, rotation=0)
        ax.set_xlim(-.25, 2.25); ax.set_ylim(-7, 107)
        ax.set_yticks([0,25,50,75,100]); ax.label_outer()
    fig.supylabel("Action Defect (%)", fontsize=p)
    handles=[Line2D([], [], color=analysis.SCENARIO_COLOUR[s], marker="s" if i==0 else "o",
                    ls="--" if i==0 else "-") for i,s in enumerate(STRATA)]
    fig.legend(handles, STRATA, loc="outside lower center", ncol=2, fontsize=p*.85)
    f.save(fig,"defects","main","Action Defect by turn and stratum. Undefined age-neutral "
           "expectations are excluded; lines describe turn-specific rates, not cumulative breaks.",
           "dialogue_02_defects")


def draw_memory(f):
    d=f.read("dialogue_03_memory")
    fig, axes,p=f.layout(height=6.8)
    for ax,s in zip(axes.flat,STRATA):
        forest(ax,subset(d,**{"Scenario Type":s}),"Held Both (%)","Held Both CI Lower",
               "Held Both CI Upper",s,p,percent=True)
    f.save(fig,"memory","main","Strong Refusal retained at both later turns, conditional on "
           "expected refusal and Strong Refusal at Turn 1. Bars: 95% scenario-bootstrap intervals. "
           "Strata and models have different eligible opening cohorts.","dialogue_03_memory")


def draw_first_break(f):
    d=f.read("dialogue_03_memory")
    fig,axes,p=f.layout(height=6.6,maps=True)
    columns=["First Break Turn 2 (%)","First Break Turn 3 (%)","Held Both (%)"]
    for ax,s in zip(axes.flat,STRATA):
        block=model_frame(subset(d,**{"Scenario Type":s}))[columns]
        if not np.allclose(block.sum(axis=1),100,atol=.03):raise ValueError("Break partition")
        block.index=MODEL_LABELS
        im=heatmap(fig,ax,block,s,p,labels=["Turn 2","Turn 3","No break"])
    colourbar(fig,axes,im,"Established boundaries (%)",p)
    f.save(fig,"first_break","main","First loss of Strong Refusal among established protective "
           "boundaries. Turn 2, Turn 3 and no break partition the cohort; a later return does not "
           "erase an earlier break.","dialogue_03_memory")


def draw_methods(f):
    d=f.read("dialogue_05_methods")
    for s in STRATA:
        fig,axes,p=f.layout(height=6.9,maps=True)
        for ax,t in zip(axes.flat,TURNS[1:]):
            block=subset(d,**{"Scenario Type":s}).pivot(index="Model",columns="Method",
                         values=f"Defect Change {t} (pp)").reindex(index=MODELS,columns=METHODS)
            block.index=MODEL_LABELS
            im=heatmap(fig,ax,block,t,p,signed=True,limit=65,
                       labels=["Emotional\nPushback","Purpose\nReverse","Role Play"])
        colourbar(fig,axes,im,"Action Defect change (pp)",p)
        f.save(fig,"methods_"+slug(s),"main",f"{s}: change in Action Defect from Turn 1, "
               "matched across all three methods on shared opening seeds. Positive values mean "
               "more defects. Purpose Reverse endpoint improvements do not establish recovery.",
               "dialogue_05_methods")


def draw_age_stratum(f, s):
    d=f.read("dialogue_04_age")
    fig,axes,p=f.layout(height=6.8)
    for ax,t in zip(axes.flat,TURNS[1:]):
        forest(ax,subset(d,Measure="Strong Refusal",Turn=t,**{"Scenario Type":s}),
               "Erosion (pp)","Erosion CI Lower","Erosion CI Upper",t,p)
        ax.set_xlabel("Age-gap change (pp)")
    f.save(fig,"age_"+slug(s),"main" if s==STRATA[0] else "supplement",
           f"{s}: change from Turn 1 in the minor-minus-age-18 Strong-Refusal gap. "
           "Negative values indicate gap erosion. Whiskers are paired scenario-bootstrap "
           "95% intervals; each model retains its available scenario cohort.","dialogue_04_age")


def draw_age(f):
    draw_age_stratum(f, "Age Restricted")


def draw_age_control(f):
    draw_age_stratum(f, "Harmful")


def draw_roleplay(f):
    d=f.read("dialogue_06_roleplay")
    fig,axes,p=f.layout(height=6.8)
    for ax,arm in zip(axes.flat,["Age 9","Age 17"]):
        forest(ax,subset(d,Measure="Strong Refusal",Condition=arm+" minus Control"),
               "Change Turn 3 (pp)","Change Turn 3 CI Lower","Change Turn 3 CI Upper",
               arm+" vs Neutral",p)
        ax.set_xlabel("Age-gap change at Turn 3 (pp)")
    f.save(fig,"roleplay","main","Role Play: Turn-3 minus Turn-1 change in the "
           "age-minus-neutral Strong-Refusal gap on common scenarios. 95% paired scenario-bootstrap "
           "intervals. This is descriptive gap erosion, not proof of a displaced internal age state.",
           "dialogue_06_roleplay")


def draw_changes(f):
    d=f.read("dialogue_02_defects")
    for s in STRATA:
        fig,axes,p=f.layout(height=6.8)
        for ax,t in zip(axes.flat,TURNS[1:]):
            forest(ax,subset(d,**{"Scenario Type":s}),f"Change {t} (pp)",
                   f"Change {t} CI Lower",f"Change {t} CI Upper",t,p)
        f.save(fig,"changes_"+slug(s),"supplement",f"{s}: Action Defect change from Turn 1 "
               "with paired scenario-bootstrap 95% intervals.","dialogue_02_defects")


def draw_trajectories(f):
    d=f.read("dialogue_s01_trajectories")
    cols=["Aligned Aligned Aligned (%)","Aligned Defect Defect (%)",
          "Aligned Defect Aligned (%)","Aligned Aligned Defect (%)"]
    for s in STRATA:
        fig,axes,p=f.layout(3,1,width=10.5,height=16,maps=True)
        for ax,m in zip(axes.flat,METHODS):
            b=model_frame(subset(d,Method=m,**{"Scenario Type":s}))[cols];b.index=MODEL_LABELS
            if not np.allclose(b.sum(axis=1),100,atol=.03):raise ValueError("Route partition")
            im=heatmap(fig,ax,b,m,p,labels=["Held both","Broke both","Broke,\nreturned","Late break"])
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
        im=heatmap(fig,ax,matrix,label,p,labels=TURNS)
        ax.set_ylabel("Age",fontsize=p*.8)
        ax.set_xlabel(f"n = {int(b['Ladder Scenarios'].iloc[0])} scenarios",fontsize=p*.75)
    colourbar(fig,axes,im,"Strong Refusal (%)",p)
    f.save(fig,"ladder","supplement","Strong Refusal across seven ages and three turns, "
           "on each model's balanced age cohort. Gemini uses 20 scenarios; the other models use "
           "25. This cohort is separate from the paired statutory-boundary figure.","dialogue_s08_ladder")


def draw_boundary(f):
    d=f.read("dialogue_s14_boundary")
    fig,axes,p=f.layout(1,3,width=13.5,height=6.8)
    for ax,t in zip(axes.flat,TURNS):
        forest(ax,subset(d,Measure="Strong Refusal",Turn=t),"17 minus 18 (pp)",
               "CI Lower","CI Upper",t,p)
        ax.set_xlabel("")
    fig.supxlabel("Age 17 minus Age 18 (pp)",fontsize=p)
    f.save(fig,"boundary","supplement","Strong-Refusal difference between ages 17 and 18 "
           "with paired scenario-bootstrap 95% intervals. Cohorts require these two ages only: "
           "24 Gemini scenarios and 25 for every other model.","dialogue_s14_boundary")


def draw_adjusted(f):
    d=f.read("dialogue_s12_adjusted")
    fig,axes,p=f.layout(height=6.8)
    for ax,t in zip(axes.flat,TURNS[1:]):
        forest(ax,subset(d,Measure="Strong Refusal",Turn=t),"Erosion Difference (pp)",
               "CI Lower","CI Upper",t,p)
        ax.set_xlabel("Erosion difference (pp)")
    f.save(fig,"adjusted","supplement","Age-Restricted minus Harmful Strong-Refusal gap "
           "erosion from Turn 1. Scenario-bootstrap 95% intervals. Strata comprise different "
           "scenario sets; this is a descriptive control comparison, not causal identification.",
           "dialogue_s12_adjusted")


def draw_memory_gap(f):
    d=f.read("dialogue_s13_memory_gap")
    fig,axes,p=f.layout(1,1,width=7.8,height=6.8)
    forest(axes[0,0],d,"Held Both, Age Restricted Minus Harmful (pp)",
           "CI Lower","CI Upper","Age Restricted vs Harmful",p)
    axes[0,0].set_xlabel("Held-both difference (pp)")
    f.save(fig,"memory_gap","supplement","Difference in unbroken Strong-Refusal memory "
           "between Age Restricted and Harmful protective cohorts, with scenario-bootstrap "
           "95% intervals. Negative values indicate weaker age-conditioned memory.",
           "dialogue_s13_memory_gap")


def draw_directional(f):
    d=f.read("dialogue_s04_directional")
    fig,axes,p=f.layout(height=6.8)
    for ax,direction in zip(axes.flat,["Over-Permissive","Over-Restrictive"]):
        forest(ax,subset(d,Direction=direction,**{"Scenario Type":"Age Restricted"}),
               "Change Turn 3 (pp)","Change Turn 3 CI Lower","Change Turn 3 CI Upper",direction,p)
    f.save(fig,"directional","supplement","Age Restricted: directional Action Defect change "
           "at Turn 3 relative to Turn 1, with 95% intervals. Over-Permissive means failure to "
           "maintain Strong Refusal when refusal is expected; Over-Restrictive means failure to "
           "provide Total Compliance when compliance is expected.","dialogue_s04_directional")


def draw_failure(f):
    d=f.read("dialogue_s09_failure")
    fig,axes,p=f.layout(2,2,height=11,maps=True)
    for row,s in enumerate(STRATA):
        for col,t in enumerate(TURNS[1:]):
            b=model_frame(subset(d,Turn=t,**{"Scenario Type":s}))[[c+" (%)" for c in CELLS]]
            b.index=MODEL_LABELS
            im=heatmap(fig,axes[row,col],b,s+" / "+t,p,
                       labels=["Strong\nRefusal","Weak\nRefusal","Minimal\nCompliance","Total\nCompliance"])
    colourbar(fig,axes,im,"Established boundaries (%)",p)
    f.save(fig,"failure","supplement","Later outcomes within the initially protective cohort. "
           "Rows partition that cohort at each turn; Strong Refusal matches endpoint memory, "
           "not necessarily unbroken memory.","dialogue_s09_failure")


def slug(s):
    return s.lower().replace(" ","_")


MAIN={"outcomes":draw_outcomes,"defects":draw_defects,"memory":draw_memory,
      "first_break":draw_first_break,"methods":draw_methods,"age":draw_age,"roleplay":draw_roleplay}
SUPPLEMENT={"changes":draw_changes,"trajectories":draw_trajectories,"ladder":draw_ladder,
            "boundary":draw_boundary,"age_control":draw_age_control,"adjusted":draw_adjusted,
            "memory_gap":draw_memory_gap,"directional":draw_directional,"failure":draw_failure}


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
    try:
        for job in jobs.values():job(f)
    finally:
        f.close()


if __name__=="__main__":
    main()
