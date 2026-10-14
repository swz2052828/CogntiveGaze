"""Figure 1: the data-scarcity cycle in smartphone gaze research (conceptual)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("/springbrook/share/eng/esrpxk/results/paper/figs_core")
STAGES = [
    ("Few usable\npublic corpora", "faces recorded under\nconditions a study can use"),
    ("Designs converge on\nthe same corpora", "hand-held phone, small screen,\ntarget position as gaze label"),
    ("New clinical data\nrecorded", "large display, oculomotor tasks;\nheld in separate silos"),
    ("Release blocked\nby privacy", "face video = personal and\nhealth data (GDPR, HIPAA)"),
    ("Standard exits fail", "blur / mask: gaze geometry lost\nHE / FL: no shared benchmark\ncontrolled access: restricted"),
    ("Corpus stays\nunreleased or restricted", "the next group finds\nthe same few corpora"),
]
R, W, H = 3.0, 2.55, 1.2
fig, ax = plt.subplots(figsize=(8.6, 8.0))
ang = np.pi / 2 - np.arange(6) * 2 * np.pi / 6
pos = [(R * np.cos(a), R * np.sin(a)) for a in ang]
for i, ((t, d), (x, y)) in enumerate(zip(STAGES, pos)):
    hot = i == 4
    ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.06",
                                fc="#fde9e7" if hot else "#eef3f8",
                                ec="#c0392b" if hot else "#34495e", lw=1.6 if hot else 1.0))
    ax.text(x, y + 0.22, f"{i + 1}. {t}", ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(x, y - 0.33, d, ha="center", va="center", fontsize=7.6, color="#333")
for i in range(6):
    a0, a1 = ang[i] - 0.36, ang[(i + 1) % 6] + 0.36
    p0 = (R * 1.0 * np.cos(a0), R * 1.0 * np.sin(a0))
    p1 = (R * 1.0 * np.cos(a1), R * 1.0 * np.sin(a1))
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle="arc3,rad=-0.28",
                                 arrowstyle="-|>", mutation_scale=16, lw=1.4, color="#34495e"))
ax.text(0, 0.55, "The data-scarcity cycle", ha="center", fontsize=11.5, fontweight="bold")
ax.text(0, -0.15, "Possible break at step 5:\nde-identification that removes\nidentity and keeps gaze.\n\n"
        "This paper measures whether\nany current method does.", ha="center", va="center",
        fontsize=8.6, color="#c0392b")
ax.set_xlim(-4.6, 4.6); ax.set_ylim(-4.2, 4.2); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_cycle.{ext}", dpi=220, bbox_inches="tight")
print("wrote", OUT / "fig_cycle.pdf")
