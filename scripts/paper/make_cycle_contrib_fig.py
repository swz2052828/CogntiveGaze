"""Figure 1: the data-scarcity cycle (left) and the four contributions mapped onto
the cycle steps they address (right). Contribution badges are labelled, so the
mapping does not rely on colour."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path("/springbrook/share/eng/esrpxk/results/paper/figs_core")
STEPS = [("Few usable\npublic corpora", []),
         ("Designs converge on\nthe same corpora", ["C1", "C2"]),
         ("New clinical data\nheld in silos", ["C1"]),
         ("Release blocked\nby privacy", ["C3"]),
         ("Standard exits fail", ["C2", "C3", "C4"]),
         ("Corpus stays\nunreleased", ["C4"])]
COL = {"C1": "#4c72b0", "C2": "#55a868", "C3": "#c44e52", "C4": "#8172b2"}
CARDS = [
 ("C1", "CognitiveGaze dataset (§3)",
  "18 participants, 7 clinical oculomotor tasks,\nsmartphone and EyeLink 1000 recorded together", "steps 2, 3"),
 ("C2", "Gaze utility benchmark (§4, §8)",
  "4 architectures, 4 train–test conditions, noise floor,\nno-information reference, pupil test, clinical error budget", "steps 2, 5"),
 ("C3", "Privacy measurement protocol (§6, §7)",
  "re-identification (A1) and linkability (A2) against\nmeasured floors, released as the privaudit tool", "steps 4, 5"),
 ("C4", "Privacy–utility analysis (§7–§10)",
  "3 generator families on one plane, two-part condition,\nseven design guidelines, controlled release", "steps 5, 6"),
]
fig = plt.figure(figsize=(13.0, 6.0))
ax = fig.add_axes([0.0, 0.0, 0.47, 1.0]); ax.axis("off"); ax.set_aspect("equal")
R, W, H = 3.0, 2.45, 1.0
ang = np.pi / 2 - np.arange(6) * 2 * np.pi / 6
pos = [(R * np.cos(a), R * np.sin(a)) for a in ang]
for i, ((t, cs), (x, y)) in enumerate(zip(STEPS, pos)):
    hot = i == 4
    ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.05",
                                fc="#fde9e7" if hot else "#eef3f8", ec="#c0392b" if hot else "#34495e",
                                lw=1.6 if hot else 1.0))
    ax.text(x, y + 0.05, f"{i + 1}. {t}", ha="center", va="center", fontsize=9, fontweight="bold")
    for k, c in enumerate(cs):                      # contribution badges under the box
        bx = x + (k - (len(cs) - 1) / 2) * 0.52
        ax.add_patch(Circle((bx, y - H / 2 - 0.27), 0.22, fc=COL[c], ec="black", lw=0.6, zorder=4))
        ax.text(bx, y - H / 2 - 0.27, c, ha="center", va="center", fontsize=7.5, color="white",
                fontweight="bold", zorder=5)
for i in range(6):
    a0, a1 = ang[i] - 0.36, ang[(i + 1) % 6] + 0.36
    ax.add_patch(FancyArrowPatch((R * np.cos(a0), R * np.sin(a0)), (R * np.cos(a1), R * np.sin(a1)),
                                 connectionstyle="arc3,rad=-0.28", arrowstyle="-|>", mutation_scale=14,
                                 lw=1.3, color="#34495e"))
ax.text(0, 0.35, "The data-scarcity cycle", ha="center", fontsize=11, fontweight="bold")
ax.text(0, -0.35, "It could break at step 5.\nThis paper measures\nwhether it does.", ha="center",
        va="center", fontsize=8.5, color="#c0392b")
ax.set_xlim(-4.4, 4.4); ax.set_ylim(-4.3, 4.0)

bx = fig.add_axes([0.49, 0.03, 0.50, 0.94]); bx.axis("off")
XW = 10 * (0.50 * 13.0) / (0.94 * 6.0)          # equal data units in x and y, so badges stay round
bx.set_xlim(0, XW); bx.set_ylim(0, 10); bx.set_aspect("equal")
bx.text(0.2, 9.55, "Contributions of this paper", fontsize=11, fontweight="bold", va="center")
for j, (c, title, body, steps) in enumerate(CARDS):
    y0 = 7.0 - j * 2.3
    bx.add_patch(FancyBboxPatch((0.2, y0), XW - 0.45, 1.95, boxstyle="round,pad=0.06", fc="white", ec=COL[c], lw=1.6))
    bx.add_patch(Circle((0.85, y0 + 0.98), 0.42, fc=COL[c], ec="black", lw=0.6))
    bx.text(0.85, y0 + 0.98, c, ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    bx.text(1.55, y0 + 1.5, title, fontsize=10, fontweight="bold", va="center")
    bx.text(1.55, y0 + 0.72, body, fontsize=8.6, va="center", linespacing=1.35)
    bx.text(XW - 0.4, y0 + 1.5, steps, fontsize=8.6, ha="right", va="center", style="italic", color="#333")
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_cycle_contrib.{ext}", dpi=220, bbox_inches="tight")
print("wrote", OUT / "fig_cycle_contrib.png")
