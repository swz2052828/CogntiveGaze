"""Figure 2: from the research problem (the data-scarcity cycle of Figure 1) to the
research questions and the four contributions. Each row: cycle step(s) -> problem ->
question -> contribution. Labels carry the mapping, so it survives grayscale."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("/springbrook/share/eng/esrpxk/results/paper/figs_core")
ROWS = [
 ("Steps 1–3", "Corpora inherit the hand-held\nprotocol and target labels, and\nclinical recordings stay in silos",
  "Can smartphone gaze be recorded under\na clinical protocol with physiological\nground truth?",
  "C1", "CognitiveGaze dataset (§3)", "18 participants, 7 oculomotor tasks,\nsmartphone and EyeLink 1000 together"),
 ("Step 5", "Blurring and masking remove the\ngaze geometry a model needs",
  "How much gaze does a de-identified\nrelease keep for models trained on it?",
  "C2", "Gaze utility benchmark (§4, §8)", "4 architectures, 4 train–test conditions,\npupil test, clinical error budget"),
 ("Step 4", "Face video is personal and health\ndata, so it cannot simply be shared",
  "Is a de-identified release private,\nand under which test?",
  "C3", "Privacy measurement protocol (§6, §7)", "A1 and A2 against measured floors,\nreleased as the privaudit tool"),
 ("Steps 5–6", "No standard exit yields a usable\npublic corpus, so data stay unreleased",
  "Can any de-identification make a\ngaze corpus releasable, and what\nwould it need?",
  "C4", "Privacy–utility analysis (§7–§10)", "3 generator families on one plane,\ntwo-part condition, design guidelines"),
]
fig, ax = plt.subplots(figsize=(13.5, 7.6)); ax.axis("off"); ax.set_xlim(0, 135); ax.set_ylim(0, 76)
def box(x, y, w, h, fc, ec, lw=1.0, style="round,pad=0.4"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
def arrow(x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=14, lw=1.2, color="#34495e"))
# motivation banner
box(1, 68.5, 133, 6.0, "#f4f6f8", "#34495e")
ax.text(67.5, 71.5, "Motivation: smartphone eye tracking could move oculomotor assessment out of the clinic,\n"
        "but it records the face, so the data it produces cannot be shared (Figure 1)",
        ha="center", va="center", fontsize=10.2, fontweight="bold", linespacing=1.35)
# column headers
for x, t in ((17, "Problem in the data-scarcity cycle"), (59, "Research question"), (107, "Contribution of this paper")):
    ax.text(x, 64.5, t, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#34495e")
for k, (steps, prob, q, c, title, body) in enumerate(ROWS):
    y = 49.0 - k * 13.6
    box(1, y, 33, 11.0, "#fde9e7" if "5" in steps else "#eef3f8", "#c0392b" if "5" in steps else "#34495e")
    ax.text(2.6, y + 9.1, steps, fontsize=8.5, fontweight="bold", color="#c0392b" if "5" in steps else "#34495e")
    ax.text(2.6, y + 4.3, prob, fontsize=8.6, va="center", linespacing=1.3)
    arrow(34.8, 41.2, y + 5.5)
    box(42, y, 34, 11.0, "white", "#7f8c8d", style="round,pad=0.4")
    ax.text(43.6, y + 5.5, q, fontsize=8.8, va="center", style="italic", linespacing=1.3)
    arrow(76.8, 83.2, y + 5.5)
    box(84, y, 50, 11.0, "white", "#2c3e50", lw=1.6)
    ax.add_patch(FancyBboxPatch((85.5, y + 3.0), 6.0, 5.0, boxstyle="round,pad=0.3", fc="#2c3e50", ec="#2c3e50"))
    ax.text(88.5, y + 5.5, c, ha="center", va="center", fontsize=11, color="white", fontweight="bold")
    ax.text(93.6, y + 8.0, title, fontsize=9.6, fontweight="bold", va="center")
    ax.text(93.6, y + 3.6, body, fontsize=8.5, va="center", linespacing=1.3)
# outcome banner
box(1, 0.8, 133, 6.0, "#eef3f8", "#34495e")
ax.text(67.5, 3.8, "Outcome: no tested configuration is both private and useful. CognitiveGaze is released under\n"
        "controlled access, with requirements and an audit tool for the next gaze dataset",
        ha="center", va="center", fontsize=10.0, linespacing=1.35)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_contributions.{ext}", dpi=220, bbox_inches="tight")
print("wrote", OUT / "fig_contributions.png")
