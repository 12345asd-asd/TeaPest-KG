# -*- coding: utf-8 -*-
"""
Publication figures for paper/main.tex (MDPI Agriculture).
Numbers copied verbatim from manuscript tables; vector PDF + check PNG.
  fig2_crossover / fig3_gating / fig4_curves
Panel letters sit BELOW each panel; no in-figure footnotes.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

BLUE, ORANGE, GREEN, RED = "#0072B2", "#E69F00", "#009E73", "#D55E00"
PURPLE, GRAY, DARK, FAINT = "#CC79A7", "#9AA5AB", "#333333", "#BFC8CD"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)

def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def panel_title(ax, txt, y=-0.30):
    ax.text(0.5, y, txt, transform=ax.transAxes, ha="center", va="top",
            fontsize=9.5, color=DARK)

# ============================================================ Figure 2 ====
def fig_crossover():
    data14 = {
        "Zero-Shot": (8.2, 21.0), "Flat-BM25": (60.7, 67.2),
        "KG-Only": (100.0, 47.8), "KG-RAG 2-hop": (99.4, 55.3),
        "KG-RAG 1-hop": (99.3, 57.9), "Profile-RAG": (77.0, 91.8),
        "Parallel-Fusion": (88.7, 86.1), "Gated-RAG": (77.0, 73.0),
    }
    data7 = {
        "Zero-Shot": (10.4, 21.6), "Flat-BM25": (67.7, 65.1),
        "KG-Only": (100.0, 47.8), "KG-RAG 2-hop": (94.1, 54.2),
        "KG-RAG 1-hop": (95.4, 57.0), "Profile-RAG": (89.3, 88.7),
        "Parallel-Fusion": (92.5, 86.4), "Gated-RAG": (89.3, 72.2),
    }
    hi = {"KG-RAG 1-hop": RED, "Profile-RAG": BLUE, "Parallel-Fusion": GREEN}
    faint_order = ["Zero-Shot", "Flat-BM25", "KG-Only", "KG-RAG 2-hop", "Gated-RAG"]
    tag14 = {"KG-RAG 1-hop": (1.035, 50.5), "Profile-RAG": (1.035, 94.8),
             "Parallel-Fusion": (1.035, 87.0), "Flat-BM25": (1.035, 67.2),
             "Zero-Shot": (1.035, 21.0)}
    tag7 = {"KG-RAG 1-hop": (1.035, 50.0), "Profile-RAG": (1.035, 93.2),
            "Parallel-Fusion": (1.035, 84.5), "Flat-BM25": (1.035, 65.1),
            "Zero-Shot": (1.035, 21.6)}
    gaps = (("Lookup gap (KG$-$Profile): +22.3",
             "Diagnosis gap (Profile$-$KG): +33.9"),
            ("Lookup gap (KG$-$Profile): +6.1 (n.s.)",
             "Diagnosis gap (Profile$-$KG): +31.7"))

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
    fig.subplots_adjust(bottom=0.20, top=0.88, wspace=0.08)
    x = np.array([0.0, 1.0])
    for ax, data, title, glines, tag in zip(
            axes, (data14, data7),
            ("(a) Qwen2.5-14B-Instruct", "(b) Qwen2.5-7B-Instruct"),
            gaps, (tag14, tag7)):
        for m in faint_order:
            ax.plot(x, data[m], "-", color=FAINT, lw=1.0, zorder=2)
            ax.scatter(x, data[m], s=16, color=FAINT, zorder=2)
        for m, c in hi.items():
            ax.plot(x, data[m], "-", color=c, lw=2.4, zorder=4)
            ax.scatter(x, data[m], s=34, color=c, zorder=5,
                       edgecolor="white", linewidth=0.6)
        for m, (dx, ty) in tag.items():
            ax.text(dx, ty, m.replace("KG-RAG ", "KG "), fontsize=8.4,
                    ha="left", va="center", color=hi.get(m, "#7E8A91"))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Named lookup\n(n=70)", "Symptom diagnosis\n(n=58)"])
        ax.set_ylim(0, 108)
        ax.set_xlim(-0.08, 1.62)
        clean(ax)
        ax.grid(axis="y", lw=0.5, color="#E4E8EB", zorder=0)
        panel_title(ax, title, y=-0.27)
    axes[0].set_ylabel("Macro recall (%)")
    handles = [
        Line2D([0], [0], color=RED, lw=2.4, marker="o", ms=5, label="KG-RAG 1-hop"),
        Line2D([0], [0], color=BLUE, lw=2.4, marker="o", ms=5, label="Profile-RAG"),
        Line2D([0], [0], color=GREEN, lw=2.4, marker="o", ms=5, label="Parallel-Fusion"),
        Line2D([0], [0], color=FAINT, lw=1.2,
               label="Other methods (Zero/Few-Shot, Flat-BM25, KG-Only, 2-hop, Gated)"),
    ]
    fig.legend(handles=handles, loc="upper center", frameon=False, ncol=4,
               bbox_to_anchor=(0.5, 1.02), handlelength=1.6, columnspacing=1.3,
               fontsize=8.8)
    save(fig, "fig2_crossover")

# ============================================================ Figure 3 ====
def _op_panel(ax, pts, title, labels):
    cls = {
        "Zero-Shot": "base", "Few-Shot": "base", "Flat-BM25": "base",
        "KG-Only": "base", "KG-RAG 2-hop": "base", "KG-RAG 1-hop": "base",
        "Profile-RAG": "profile", "Parallel-Fusion": "fusion",
        "Gated-RAG": "gated", "DTU-KG-RAG 1-hop": "dtuother",
        "DTU-Fusion strict": "dtu", "DTU-Fusion advisory": "adv",
    }
    color = {"base": "#9AA5AB", "profile": PURPLE, "fusion": BLUE,
             "gated": ORANGE, "dtuother": "#7B4FA0", "dtu": RED, "adv": "#E8834A"}
    marker = {"base": "o", "profile": "s", "fusion": "o", "gated": "D",
              "dtuother": "v", "dtu": "*", "adv": "P"}
    size = {"base": 24, "profile": 32, "fusion": 38, "gated": 34,
            "dtuother": 34, "dtu": 100, "adv": 42}
    for name, (xx, yy) in pts.items():
        k = cls[name]
        ax.scatter(xx, yy, s=size[k], c=color[k], marker=marker[k], zorder=5,
                   edgecolor="white", linewidth=0.5)
    for name, (tx, ty, ha, va) in labels.items():
        ax.annotate(name.replace("DTU-Fusion ", "DTU "), pts[name],
                    xytext=(tx, ty), textcoords="data", fontsize=8.2, ha=ha, va=va,
                    color=DARK if name.startswith(("DTU", "Parallel", "Gated")) else "#555")
    ax.axhline(0, color=RED, lw=0.9, ls=":", zorder=1)
    ax.set_xlim(8, 99)
    ax.set_ylim(-23, 112)
    clean(ax)
    ax.grid(lw=0.5, color="#E4E8EB", zorder=0)
    panel_title(ax, title, y=-0.24)

def fig_gating():
    pts14 = {
        "Zero-Shot": (14.0, 73.3), "Few-Shot": (15.2, 83.3),
        "Flat-BM25": (63.6, 70.0), "KG-Only": (76.3, 100.0),
        "KG-RAG 2-hop": (79.4, 93.3), "KG-RAG 1-hop": (80.5, 86.7),
        "Profile-RAG": (83.7, 76.7), "Parallel-Fusion": (87.5, 53.3),
        "Gated-RAG": (75.1, 0.0), "DTU-KG-RAG 1-hop": (86.5, 0.0),
        "DTU-Fusion strict": (84.1, 0.0), "DTU-Fusion advisory": (86.4, 13.3),
    }
    pts7 = {
        "Zero-Shot": (15.5, 80.0), "Few-Shot": (16.9, 90.0),
        "Flat-BM25": (66.5, 76.7), "KG-Only": (76.3, 100.0),
        "KG-RAG 2-hop": (76.0, 100.0), "KG-RAG 1-hop": (78.0, 93.3),
        "Profile-RAG": (89.0, 76.7), "Parallel-Fusion": (89.8, 40.0),
        "Gated-RAG": (81.6, 0.0), "DTU-KG-RAG 1-hop": (86.3, 0.0),
        "DTU-Fusion strict": (88.0, 0.0), "DTU-Fusion advisory": (90.4, 6.7),
    }
    labels14 = {
        "Zero-Shot": (24.0, 66.0, "center", "top"),
        "Flat-BM25": (58.0, 63.0, "center", "top"),
        "KG-Only": (76.3, 106.0, "center", "bottom"),
        "Profile-RAG": (84.6, 82.0, "left", "bottom"),
        "Parallel-Fusion": (88.6, 50.0, "left", "top"),
        "Gated-RAG": (71.0, -15.0, "right", "top"),
        "DTU-Fusion strict": (84.1, -15.0, "center", "top"),
        "DTU-KG-RAG 1-hop": (85.3, 8.0, "right", "bottom"),
        "DTU-Fusion advisory": (89.4, 20.0, "left", "bottom"),
    }
    labels7 = {
        "Zero-Shot": (25.0, 72.0, "center", "top"),
        "Flat-BM25": (62.0, 68.0, "center", "top"),
        "KG-Only": (76.3, 106.0, "center", "bottom"),
        "Profile-RAG": (89.6, 82.0, "left", "bottom"),
        "Parallel-Fusion": (90.6, 36.0, "left", "top"),
        "Gated-RAG": (78.0, -15.0, "right", "top"),
        "DTU-Fusion strict": (88.0, -15.0, "center", "top"),
        "DTU-KG-RAG 1-hop": (84.6, 6.0, "right", "bottom"),
        "DTU-Fusion advisory": (91.0, 15.0, "left", "bottom"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 7.0))
    fig.subplots_adjust(hspace=0.70, wspace=0.30, top=0.97, bottom=0.06,
                        left=0.08, right=0.98)
    axa, axb, axc, axd = axes.flat
    _op_panel(axa, pts14, "(a) Operating points, 14B", labels14)
    _op_panel(axb, pts7, "(b) Operating points, 7B", labels7)
    axa.set_xlabel("In-distribution macro recall (%, n=128)")
    axb.set_xlabel("In-distribution macro recall (%, n=128)")
    axa.set_ylabel("OOD non-abstention (%, n=30), lower safer")

    # ---- (c) pollution
    configs = ["Ungated", "DTU strict", "w/o reg.",
               "w/o reg.\n& R-prune", "Soft order", "Hard R\n$\\tau$=0.45"]
    carry  = [100.0, 14.7, 20.8, 68.3, 41.8, 0.0]
    forged = [100.0, 0.0, 85.2, 89.7, 0.0, 0.0]
    collat = [0.0, 0.0, 0.0, 0.0, 0.0, 45.0]
    adopt14 = [47.0, 5.2, 9.1, np.nan, 27.4, np.nan]
    xc = np.arange(6)
    w = 0.26
    axc.bar(xc - w, carry, w, label="Carry (all units)",
            color="#5F7885", zorder=3)
    axc.bar(xc, forged, w, label="Forged ban carry", color=RED, zorder=3)
    axc.bar(xc + w, collat, w, label="Clean collateral", color=ORANGE, zorder=3)
    axc.plot(xc, adopt14, "-o", color=GREEN, ms=3.8, lw=1.7, zorder=5,
             label="LLM adoption (14B)")
    axc.set_xticks(xc)
    axc.set_xticklabels(configs, fontsize=7.8)
    for lbl in axc.get_xticklabels():
        lbl.set_rotation(28)
        lbl.set_horizontalalignment("right")
    axc.set_xlim(-0.55, 5.75)
    axc.set_ylim(0, 140)
    axc.set_ylabel("Rate (%)")
    clean(axc)
    axc.grid(axis="y", lw=0.5, color="#E4E8EB", zorder=0)
    axc.legend(frameon=True, framealpha=0.95, edgecolor="#D5DBE0",
               fontsize=7.8, loc="lower right", bbox_to_anchor=(1.0, 0.66),
               ncol=1, columnspacing=0.8, handletextpad=0.35,
               borderaxespad=0.2, labelspacing=0.4, borderpad=0.4)
    panel_title(axc, "(c) Pollution stress test, k=10%", y=-0.30)

    # ---- (d) dumbbells
    cats = ["Beyond-KG\nabstain (n=40)", "Numeric fabr.\n(n=40)",
            "ID recall\n(n=128)"]
    before14, after14 = [62.5, 15.0, 87.5], [87.5, 2.5, 84.1]
    before7, after7 = [72.5, 20.0, 89.8], [87.5, 10.0, 90.4]
    xd = np.arange(3)
    for i in range(3):
        axd.plot([xd[i] - 0.16]*2, [before14[i], after14[i]],
                 color=BLUE, lw=1.6, zorder=3)
        axd.plot([xd[i] + 0.16]*2, [before7[i], after7[i]],
                 color=BLUE, lw=1.6, ls="--", zorder=3)
    axd.scatter(xd - 0.16, before14, marker="o", s=46,
                facecolor="#BFC8CD", edgecolor=DARK, lw=0.6, zorder=5,
                label="Ungated, 14B")
    axd.scatter(xd - 0.16, after14, marker="o", s=50, facecolor=RED,
                edgecolor="white", lw=0.6, zorder=5, label="DTU strict, 14B")
    axd.scatter(xd + 0.16, before7, marker="s", s=42, facecolor="#BFC8CD",
                edgecolor=DARK, lw=0.6, zorder=5, label="Ungated, 7B")
    axd.scatter(xd + 0.16, after7, marker="s", s=46, facecolor=RED,
                edgecolor="white", lw=0.6, zorder=5, label="DTU strict, 7B")
    db = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
    # category 0 (positive changes): 14B label above its marker, 7B label
    # to the right in the empty inter-category band
    axd.text(xd[0] - 0.16, max(before14[0], after14[0]) + 5.5, "+25.0",
             fontsize=8.0, ha="center", va="center", color=BLUE, bbox=db)
    axd.text(xd[0] + 0.36, max(before7[0], after7[0]), "+15.0",
             fontsize=8.0, ha="left", va="center", color=BLUE, bbox=db)
    # category 1 (negative changes): labels beside the bottom markers
    axd.text(xd[1] - 0.34, min(before14[1], after14[1]), "-12.5",
             fontsize=8.0, ha="right", va="center", color=BLUE, bbox=db)
    axd.text(xd[1] + 0.36, min(before7[1], after7[1]), "-10.0",
             fontsize=8.0, ha="left", va="center", color=BLUE, bbox=db)
    # category 2 (ID recall): near-zero changes, labels centred above/below
    axd.text(xd[2] - 0.16, 79.6, "-3.5", fontsize=8.0, ha="center",
             va="center", color=BLUE, bbox=db)
    axd.text(xd[2] + 0.16, 94.2, "+1.8", fontsize=8.0, ha="center",
             va="center", color=BLUE, bbox=db)
    axd.set_xticks(xd)
    axd.set_xticklabels(cats, fontsize=8.2)
    axd.set_xlim(-0.6, 2.75)
    axd.set_ylim(0, 100)
    axd.set_ylabel("Rate (%)")
    clean(axd)
    axd.grid(axis="y", lw=0.5, color="#E4E8EB", zorder=0)
    axd.legend(frameon=True, framealpha=0.95, edgecolor="#D5DBE0",
               fontsize=7.8, loc="center", bbox_to_anchor=(0.50, 0.40),
               ncol=2, columnspacing=0.8, handletextpad=0.3,
               borderaxespad=0.2, labelspacing=0.35, borderpad=0.4)
    panel_title(axd, "(d) Independent manual vs. coverage cost", y=-0.30)

    save(fig, "fig3_gating")

# ============================================================ Figure 4 ====
def fig_curves():
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(7.9, 3.0))
    fig.subplots_adjust(bottom=0.26, top=0.93, wspace=0.42)

    # (a) mixing weight
    lam = [0.0, 0.25, 0.5, 0.75, 1.0]
    rec = [91.8, 92.6, 92.8, 87.0, 57.9]
    axa.plot(lam, rec, "-o", color=BLUE, lw=1.9, ms=5, zorder=4)
    axa.axhspan(87, 93, color=GREEN, alpha=0.08, zorder=0)
    axa.set_ylim(50, 102)
    axa.annotate("57.9", xy=(1.0, 57.9), xytext=(0.80, 54.5), fontsize=8,
                 ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                 arrowprops=dict(arrowstyle="->", lw=0.7, color=DARK))
    axa.set_xlabel("Mixing weight $\\lambda$ (triples vs. profiles)")
    axa.set_ylabel("Diagnosis recall (%, n=58)")
    axa.set_xticks(lam)
    clean(axa)
    axa.grid(lw=0.5, color="#E4E8EB", zorder=0)
    panel_title(axa, "(a) Mixing weight", y=-0.34)

    # (b) coverage boundary
    alpha = [0.10, 0.25, 0.50, 0.75, 1.0]
    ceil_ = [54.1, 56.2, 71.2, 90.9, 100.0]
    e2e = [np.nan, 38.7, 56.0, 73.5, 82.8]
    axb.plot(alpha, ceil_, "--s", color="#8A969D", lw=1.5, ms=4,
             label="Gold ceiling")
    axb.plot(alpha, e2e, "-o", color=BLUE, lw=1.9, ms=5,
             label="End-to-end")
    axb.axvspan(0.10, 0.75, color=RED, alpha=0.05, zorder=0)
    axb.annotate("failure boundary\n($\\approx$75% retained)", xy=(0.75, 74.0),
                 xytext=(0.14, 88), fontsize=8.2, color="#7A1F00",
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="#7A1F00"))
    axb.set_xlabel("Retained triple fraction $\\alpha$")
    axb.set_ylabel("Recall / gold ceiling (%)")
    axb.set_ylim(30, 104)
    axb.set_xticks(alpha)
    for lbl in axb.get_xticklabels():
        lbl.set_fontsize(7.6)
        lbl.set_rotation(28)
        lbl.set_horizontalalignment("right")
    clean(axb)
    axb.grid(lw=0.5, color="#E4E8EB", zorder=0)
    axb.legend(frameon=True, framealpha=0.95, edgecolor="#D5DBE0",
               fontsize=8.0, loc="lower right", bbox_to_anchor=(0.99, 0.04),
               borderaxespad=0.2, borderpad=0.35, labelspacing=0.3)
    panel_title(axb, "(b) Graph-coverage boundary", y=-0.34)

    # (c) review curve
    rev = [0, 10, 20, 30, 40]
    kept = [82.3, 83.1, 87.3, 91.9, 91.6]
    caught = [0, 14.3, 42.9, 67.9, 71.4]
    axc.plot(rev, kept, "-o", color=BLUE, lw=1.9, ms=5)
    axc.set_xlabel("Routed to human review (%)")
    axc.set_ylabel("Kept-question accuracy (%)", color=BLUE)
    axc.tick_params(axis="y", labelcolor=BLUE)
    axc.set_ylim(77, 106)
    axc.set_xlim(-2, 57)
    axc.set_xticks(rev)
    axc2 = axc.twinx()
    axc2.plot(rev, caught, "-s", color=ORANGE, lw=1.9, ms=5)
    axc2.set_ylabel("Errors caught (%)", color=ORANGE)
    axc2.tick_params(axis="y", labelcolor=ORANGE)
    axc2.set_ylim(0, 100)
    axc2.spines["top"].set_visible(False)
    clean(axc)
    axc.grid(lw=0.5, color="#E4E8EB", zorder=0)
    # compact legend identifying the two series (spelled out in the caption)
    axc.legend([Line2D([0], [0], color=BLUE, lw=1.9, marker="o", ms=4),
                Line2D([0], [0], color=ORANGE, lw=1.9, marker="s", ms=4)],
               ["Kept accuracy", "Errors caught"], frameon=True,
               framealpha=0.92, edgecolor="#D5DBE0", fontsize=7.6,
               loc="lower right", bbox_to_anchor=(0.99, 0.06),
               borderpad=0.35, labelspacing=0.3, handletextpad=0.4)
    panel_title(axc, "(c) Selective-review curve", y=-0.34)

    save(fig, "fig4_curves")

if __name__ == "__main__":
    fig_crossover()
    fig_gating()
    fig_curves()
    print("done ->", OUT)
