"""The three core figures of the privacy-utility argument, from result files only.

  fig_core1_privacy.png   release-only linkage: the 2x2 on the face crop, and the
                          eye-ROI linkage of every arm with subject-bootstrap CIs
  fig_core2_utility.png   utility penalty under three training/testing conditions
  fig_core3_tradeoff.png  the privacy-utility plane: no point is both private and
                          useful

No number is typed in by hand except the ARI floor (0.089) and the mean-predictor
error (9.37 cm), both computed elsewhere and cited in the captions.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path("/springbrook/share/eng/esrpxk/results")
OUT = RES / "paper" / "figs_core"
OUT.mkdir(parents=True, exist_ok=True)
FLOOR_ARI = 0.089
MEAN_PRED = 9.37      # cm, predict-the-training-mean, averaged over the 5 folds
BBS = ["mobilenet_v3", "affnet", "mgazenet"]
BB_LABEL = {"mobilenet_v3": "MobileNet-V3", "affnet": "AFFNet", "mgazenet": "MGazeNet"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def j(p):
    return json.load(open(RES / p))


def t1(p):
    return j(p)["ari_k18"]


def boot(p):
    d = j(p)
    return d["ari_point"], d["ari_boot_ci95"]


# ------------------------------------------------------------------ data
face = {
    ("face", "fixed"): t1("anon_dp2/t1_dp2t1_per_subject.json"),
    ("face", "frame"): t1("anon_dp2/t1_dp2t1_per_frame.json"),
    ("body", "fixed"): t1("anon_dp2/fullbody/t1_dp2fb_per_subject.json"),
    ("body", "frame"): t1("anon_dp2/fullbody/t1_dp2fb_per_frame.json"),
}
face_simswap = [t1("anon_swap/t1_ProcessedSwap_selfalign.json"),
                t1("anon_swap2/t1_ProcessedSwap2_selfalign.json")]
face_orig = t1("anon_stage3b/t1sa_none.json")
face_fams = [t1("anon_fams/t1_fams_per_subject.json"), t1("anon_fams/t1_fams_per_frame.json")]

eye_arms = [
    ("Original", "anon_ari_ci/ari_original_{}.json"),
    ("SimSwap, identity A", "anon_ari_ci/ari_swap_t1_{}.json"),
    ("SimSwap, identity B", "anon_ari_ci/ari_swap_t2_{}.json"),
    ("DP2 face, fixed", "anon_ari_ci/ari_dp2_persubject_{}.json"),
    ("DP2 face, per frame", "anon_ari_ci/ari_dp2_perframe_{}.json"),
    ("DP2 full body, fixed", "anon_dp2/fullbody/ari_dp2fb_per_subject_{}.json"),
    ("DP2 full body, per frame", "anon_dp2/fullbody/ari_dp2fb_per_frame_{}.json"),
    ("FAMS (diffusion), fixed seed", "anon_fams/ari_fams_per_subject_{}.json"),
    ("FAMS (diffusion), per frame", "anon_fams/ari_fams_per_frame_{}.json"),
]
eye = {name: {e: boot(p.format(e)) for e in ("appleLeftEye", "appleRightEye")}
       for name, p in eye_arms}

util = j("anon_train/table3_three_conditions.json")
COND = [("ii", "(ii) train real, test de-id"),
        ("i_real", "(i) train de-id, test real"),
        ("i_anon", "(i) train de-id, test de-id")]
ARMS = [("swap1", "SimSwap A"), ("swap2", "SimSwap B"),
        ("dp2s", "DP2 face,\nfixed"), ("dp2f", "DP2 face,\nper frame"),
        ("dp2fbs", "DP2 full body,\nfixed"), ("dp2fbf", "DP2 full body,\nper frame")]


def pen(arm, cond):
    return np.array([util[bb][arm][cond]["penalty"] for bb in BBS])


no_info = np.array([MEAN_PRED - util[bb]["baseline"]["mean"] for bb in BBS])

# ============================================================ figure 1
fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.4),
                           gridspec_kw={"width_ratios": [1, 1.6]})
M = np.array([[face[("face", "fixed")], face[("face", "frame")]],
              [face[("body", "fixed")], face[("body", "frame")]]])
im = a.imshow(M, cmap="Reds", vmin=0, vmax=1)
for r in range(2):
    for c in range(2):
        a.text(c, r, f"{M[r, c]:.3f}", ha="center", va="center", fontsize=13,
               fontweight="bold", color="white" if M[r, c] > 0.6 else "black")
a.set_xticks([0, 1], ["fixed identity\nper participant", "fresh identity\nper frame"])
a.set_yticks([0, 1], ["DP2 face mode\n(edits face box)", "DP2 full body\n(edits whole person)"])
a.set_title(f"(a) Face-crop linkage, ARI @ k=18 (floor {FLOOR_ARI})\n"
            f"original {face_orig:.3f}, SimSwap A/B {face_simswap[0]:.3f}/{face_simswap[1]:.3f}, "
            f"FAMS {face_fams[0]:.3f}/{face_fams[1]:.3f}", fontsize=9)
for s in a.spines.values():
    s.set_visible(False)
plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)

names = [n for n, _ in eye_arms]
y = np.arange(len(names))[::-1]
for off, e, col in ((-0.15, "appleLeftEye", "#1f77b4"), (0.15, "appleRightEye", "#ff7f0e")):
    pts = np.array([eye[n][e][0] for n in names])
    lo = np.array([eye[n][e][1][0] for n in names])
    hi = np.array([eye[n][e][1][1] for n in names])
    b.errorbar(pts, y + off, xerr=[pts - lo, hi - pts], fmt="o", color=col, ms=5,
               capsize=2, label="left eye" if e == "appleLeftEye" else "right eye")
b.axvline(FLOOR_ARI, color="grey", ls="--", lw=1)
b.text(FLOOR_ARI + 0.01, y[-1] - 0.55, "floor", color="grey", fontsize=8)
b.set_yticks(y, names)
b.set_xlim(0, 1.02)
b.set_xlabel("ARI @ k=18 (95% subject-bootstrap CI)")
b.set_title("(b) Eye-ROI linkage, release-only attacker (A2)", fontsize=9)
b.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_core1_privacy.png", dpi=200)
fig.savefig(OUT / "fig_core1_privacy.pdf")
plt.close(fig)

# ============================================================ figure 2
fig, axes = plt.subplots(1, 2, figsize=(14, 4.3), sharey=False)
cols = ["#7f7f7f", "#2ca02c", "#9467bd"]
for ax, suffix, title in ((axes[0], "", "(a) Full Synthetic: face AND eyes de-identified"),
                          (axes[1], "_oldeye", "(b) Hybrid: de-identified face, ORIGINAL eyes")):
    x = np.arange(len(ARMS))
    w = 0.26
    for k, (cond, lab) in enumerate(COND):
        vals = [pen(a + suffix, cond) for a, _ in ARMS]
        m = [v.mean() for v in vals]
        ax.bar(x + (k - 1) * w, m, w, color=cols[k], label=lab, alpha=0.85)
        for i, v in enumerate(vals):
            ax.scatter(np.full(3, x[i] + (k - 1) * w), v, s=9, color="black", zorder=3)
    ax.axhline(0, color="black", lw=0.8)
    if suffix == "":
        ax.axhspan(no_info.min(), no_info.max(), color="red", alpha=0.12)
        ax.axhline(no_info.mean(), color="red", ls="--", lw=1)
        ax.text(len(ARMS) - 0.5, no_info.mean() + 0.3,
                "no gaze information\n(predict the mean)", color="red",
                ha="right", fontsize=8)
    else:
        ax.set_ylim(-0.6, 1.4)
        ax.axhspan(-0.2, 0.2, color="grey", alpha=0.15)
        ax.text(len(ARMS) - 0.5, 0.22, "init noise floor (2 x SD)", color="grey",
                ha="right", fontsize=8)
    ax.set_xticks(x, [l for _, l in ARMS])
    ax.set_ylabel("error penalty vs real-data model (cm)")
    ax.set_title(title, fontsize=9)
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
nd = sorted({util[bb][a][c]["n_draws"] for bb in BBS for a, _ in ARMS
             for c in ("i_real", "i_anon") if util[bb][a].get(c)})
nd_txt = (f"{nd[0]} initialisation draw{'s' if nd[0] > 1 else ''}" if len(nd) == 1
          else f"{nd[0]}-{nd[-1]} initialisation draws")
fig.text(0.5, -0.01, "bars: mean of MobileNet-V3, AFFNet, MGazeNet; dots: each backbone. "
         f"(ii) uses 5 initialisation draws, (i) {nd_txt}.", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig_core2_utility.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "fig_core2_utility.pdf", bbox_inches="tight")
plt.close(fig)

# ============================================================ figure 3
# Release linkage = the most linkable crop the release contains (face crop, left
# and right eye ROI). Utility = condition (i), train on the release, test on real
# faces, mean over the three backbones.
eye_mean = {n: np.mean([eye[n][e][0] for e in eye[n]]) for n in eye}
pts = [
    # label, x(penalty), face ARI, eye ARI, marker, colour
    ("Original data", 0.0, face_orig, eye_mean["Original"], "*", "black"),
    ("SimSwap A, Full Synthetic", pen("swap1", "i_real").mean(), face_simswap[0],
     eye_mean["SimSwap, identity A"], "o", "#1f77b4"),
    ("SimSwap A, Hybrid", pen("swap1_oldeye", "i_real").mean(), face_simswap[0],
     eye_mean["Original"], "s", "#1f77b4"),
    ("SimSwap B, Full Synthetic", pen("swap2", "i_real").mean(), face_simswap[1],
     eye_mean["SimSwap, identity B"], "o", "#17becf"),
    ("SimSwap B, Hybrid", pen("swap2_oldeye", "i_real").mean(), face_simswap[1],
     eye_mean["Original"], "s", "#17becf"),
    ("DP2 face fixed, Full Synthetic", pen("dp2s", "i_real").mean(), face[("face", "fixed")],
     eye_mean["DP2 face, fixed"], "o", "#ff7f0e"),
    ("DP2 face fixed, Hybrid", pen("dp2s_oldeye", "i_real").mean(), face[("face", "fixed")],
     eye_mean["Original"], "s", "#ff7f0e"),
    ("DP2 face per-frame, Full Synthetic", pen("dp2f", "i_real").mean(), face[("face", "frame")],
     eye_mean["DP2 face, per frame"], "o", "#d62728"),
    ("DP2 face per-frame, Hybrid", pen("dp2f_oldeye", "i_real").mean(), face[("face", "frame")],
     eye_mean["Original"], "s", "#d62728"),
    ("DP2 full body fixed, Full Synthetic", pen("dp2fbs", "i_real").mean(), face[("body", "fixed")],
     eye_mean["DP2 full body, fixed"], "o", "#8c564b"),
    ("DP2 full body fixed, Hybrid", pen("dp2fbs_oldeye", "i_real").mean(), face[("body", "fixed")],
     eye_mean["Original"], "s", "#8c564b"),
    ("DP2 full body per-frame, Full Synthetic", pen("dp2fbf", "i_real").mean(), face[("body", "frame")],
     eye_mean["DP2 full body, per frame"], "o", "purple"),
    ("DP2 full body per-frame, Hybrid", pen("dp2fbf_oldeye", "i_real").mean(), face[("body", "frame")],
     eye_mean["Original"], "s", "purple"),
]
fig, ax = plt.subplots(figsize=(8.2, 5.4))
ax.add_patch(plt.Rectangle((-0.6, -0.02), 1.1, 0.32, color="green", alpha=0.10))
ax.text(-0.55, 0.26, "wanted:\nprivate AND useful", color="green", fontsize=8, va="top")
# Points cluster at y = 1, so they are numbered and listed in a legend.
offsets = {}
for i, (lab, x, fa, ea, mk, c) in enumerate(pts, 1):
    yv = max(fa, ea)
    ax.scatter(x, yv, marker=mk, s=110 if mk == "*" else 55, color=c, zorder=3,
               edgecolor="black", lw=0.5,
               label=f"{i}. {lab}  ({x:+.2f} cm, ARI {yv:.3f})")
    k = (round(x * 2) / 2, round(yv, 1))      # stagger numbers of nearby points
    n = offsets.get(k, 0); offsets[k] = n + 1
    ax.annotate(str(i), (x, yv), xytext=(-4 + 9 * (n % 3), 8 + 9 * (n // 3)),
                textcoords="offset points", fontsize=7, fontweight="bold")
# The one point that leaves the top row: say why it is not a solution.
fbx = pen("dp2fbf", "i_real").mean()
fby = max(face[("body", "frame")], eye_mean["DP2 full body, per frame"])
ax.annotate("12: the only configuration that breaks\nlinkage — and a model trained on it\n"
            "is no better than predicting the mean", (fbx, fby), xytext=(-175, -30),
            textcoords="offset points", fontsize=7, color="purple",
            arrowprops=dict(arrowstyle="->", color="purple", lw=0.8))
ax.axvline(no_info.mean(), color="red", ls="--", lw=0.8)
ax.text(no_info.mean() + 0.05, 1.04, "no gaze\ninformation", color="red", fontsize=7, va="bottom")
ax.axhline(FLOOR_ARI, color="grey", ls="--", lw=0.8)
ax.text(5.9, FLOOR_ARI + 0.01, "linkage floor", color="grey", fontsize=7, ha="right")
ax.set_xlim(-0.6, 6.0)
ax.set_ylim(-0.02, 1.12)
ax.set_xlabel("utility cost: error penalty (cm), trained on the release, tested on real faces")
ax.set_ylabel("release linkage: ARI of the most linkable released crop")
ax.set_title("Privacy-utility plane: circles = Full Synthetic, squares = Hybrid", fontsize=9)
ax.legend(loc="center left", fontsize=6.5, frameon=True, framealpha=0.9,
          bbox_to_anchor=(0.10, 0.60), handletextpad=0.4, labelspacing=0.45)
fig.tight_layout()
fig.savefig(OUT / "fig_core3_tradeoff.png", dpi=200)
fig.savefig(OUT / "fig_core3_tradeoff.pdf")
plt.close(fig)

print("wrote", sorted(p.name for p in OUT.iterdir()))
print("no-information penalty per backbone:", dict(zip(BBS, np.round(no_info, 2))))
for lab, x, fa, ea, *_ in pts:
    print(f"  {lab:36s} penalty {x:+.2f}  face {fa:.3f}  eye {ea:.3f}")
