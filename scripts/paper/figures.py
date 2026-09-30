"""Paper figures, built only from result files -- never from numbers typed in.

Each figure is written as PDF (for the paper) and PNG (for review), and next to it
a CSV of exactly the values plotted, so every mark in a figure can be checked
against a table.

Design rules applied (see the dataviz method): one y-scale per plot, colour by
the job it does (identity, never rank), validated categorical slots on a white
print surface (#2a78d6 / #eb6834 / #1baf7a pass all-pairs CVD and normal-vision
floors; aqua is below 3:1 contrast so it is only ever used with a direct label),
hairline grid, thin marks, legends for >=2 series plus selective direct labels,
reference lines in muted ink rather than a series colour.
"""
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("/springbrook/share/eng/esrpxk/results")
OUT = R / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
GREY_SERIES = "#a8a79f"
TEXTW = 5.5   # inches, single-column ACM acmsmall text width

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0,
    "ytick.major.size": 0, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 7, "pdf.fonttype": 42,
    "savefig.dpi": 300, "figure.facecolor": "white", "axes.facecolor": "white",
})


def style(ax, grid_axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)


def save(fig, name, rows, header):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    with open(OUT / f"{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    print(f"wrote {name}.pdf/.png/.csv ({len(rows)} rows)")


# --------------------------------------------------------------- Figure 1
def fig_leakage():
    """TAR@FAR=1e-3 for original vs SimSwap crops, with the nuisance floor."""
    ci = {r["source"]: r for r in csv.DictReader(open(R / "paper" / "privacy_ci.csv"))}
    import json
    floor = json.load(open(R / "anon_stage3b" / "floor_background.json"))["TAR@FAR=0.001"]
    items = [  # label, source, group
        ("Face crop",       "anon_stage3b/t2_none.scores.npz",                      "Original"),
        ("Right eye crop",  "anon_swap/peri_ProcessedData_appleRightEye.scores.npz", "Original"),
        ("Left eye crop",   "anon_swap/peri_ProcessedData_appleLeftEye.scores.npz",  "Original"),
        ("Face crop",       "anon_swap/t2_ProcessedSwap.scores.npz",                 "SimSwap"),
        ("Right eye crop",  "anon_swap/peri_ProcessedSwap_appleRightEye.scores.npz", "SimSwap"),
        ("Left eye crop",   "anon_swap/peri_ProcessedSwap_appleLeftEye.scores.npz",  "SimSwap"),
    ]
    fig, ax = plt.subplots(figsize=(TEXTW, 2.4))
    rows, ys, labels = [], [], []
    y = 0
    for gi, grp in enumerate(("Original", "SimSwap")):
        col = BLUE if grp == "Original" else ORANGE
        for lab, src, g in items:
            if g != grp:
                continue
            r = ci[src]
            v, lo, hi = (100 * float(r[k]) for k in ("TAR@FAR=0.001", "TAR@FAR=0.001_lo", "TAR@FAR=0.001_hi"))
            ax.plot([lo, hi], [y, y], color=col, lw=1.4, solid_capstyle="round", zorder=2)
            ax.scatter([v], [y], s=34, color=col, edgecolor="white", linewidth=1.2, zorder=3,
                       label=grp if lab == "Face crop" else None)
            ax.text(hi + 1.5, y, f"{v:.1f}%", va="center", ha="left", fontsize=7, color=INK2)
            rows.append([grp, lab, round(v, 2), round(lo, 2), round(hi, 2), src])
            ys.append(y); labels.append(lab); y += 1
        y += 0.6
    ax.axvline(100 * floor, color=MUTED, lw=0.8, zorder=1)
    ax.text(100 * floor + 1.2, -0.75, f"session-nuisance floor {100*floor:.1f}%",
            color=MUTED, fontsize=7, va="center", ha="left")
    rows.append(["floor", "face-free background patch", round(100 * floor, 2), "", "", "anon_stage3b/floor_background.json"])
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    ax.set_ylim(y - 0.4, -1.2)
    ax.set_xlim(-3, 110); ax.set_xlabel("True-accept rate at FAR = 1e-3 (%), subject-level 95% CI")
    style(ax, "x")
    ax.legend(loc="lower right", handletextpad=0.3)
    save(fig, "fig1_periocular_leakage", rows, ["group", "input", "TAR@1e-3 %", "CI lo", "CI hi", "source"])


# --------------------------------------------------------------- Figure 2
def parse_planc(path):
    out, bb = {}, None
    for ln in open(path):
        m = re.match(r"^(\S+)\s+metric=", ln.strip())
        if m:
            bb = m.group(1); continue
        m = re.match(r"\s+(blur[\d.]+|blackbox)\s+([+-][\d.]+)\s+\[([+-][\d.]+),([+-][\d.]+)\]\s+\S+\s+(\d+)/17\s+\S+\s+\S+\s+(\S+)", ln)
        if m and bb:
            op, d, lo, hi, hurt, v = m.groups()
            out[(bb, op)] = (float(d), float(lo), float(hi), int(hurt), v)
    return out


def fig_noninferiority(K="9"):
    ii = parse_planc(R / "plan_c" / f"planc_condii_K{K}.txt")
    iii = parse_planc(R / "plan_c" / f"planc_condiii_K{K}.txt")
    bbs = [("mobile_vit", "MobileViT\nmultistream"), ("convnextv2", "ConvNeXt-V2\nmultistream"),
           ("itracker", "iTracker\nmultistream"), ("face_only_mobile_vit", "MobileViT\nface only")]
    ops = [("blur0.10", "blur 0.10·IOD"), ("blur0.20", "blur 0.20·IOD"),
           ("blur0.50", "blur 0.50·IOD"), ("blackbox", "face region removed")]
    fig, axes = plt.subplots(1, 4, figsize=(TEXTW, 2.3), sharey=True)
    rows = []
    for ax, (bb, title) in zip(axes, bbs):
        for j, (op, lab) in enumerate(ops):
            for cond, tab, col, off, name in (("ii", ii, GREY_SERIES, -0.14, "clean enrolment (ii)"),
                                              ("iii", iii, BLUE, 0.14, "matched enrolment (iii)")):
                d, lo, hi, hurt, v = tab[(bb, op)]
                yy = j + off
                ax.plot([lo, hi], [yy, yy], color=col, lw=1.3, solid_capstyle="round")
                ax.scatter([d], [yy], s=22, color=col, edgecolor="white", linewidth=1.0, zorder=3,
                           label=name if (bb == "mobile_vit" and j == 0) else None)
                rows.append([K, bb, op, cond, d, lo, hi, hurt, v])
        ax.axvline(0, color=AXIS, lw=0.6)
        ax.axvline(0.40, color=MUTED, lw=0.8)
        ax.set_title(title, fontsize=7.5, color=INK, loc="left")
        style(ax, "x")
    # one shared x range wide enough for EVERY interval: a clipped CI misreads
    xmin = min(r[5] for r in rows); xmax = max(r[6] for r in rows)
    for ax in axes:
        ax.set_xlim(min(-0.4, xmin - 0.15), xmax + 0.15)
        ax.set_xticks([t for t in (0, 1, 2, 3) if t <= xmax + 0.15])
    axes[0].set_yticks(range(len(ops))); axes[0].set_yticklabels([l for _, l in ops])
    axes[0].text(0.46, -0.55, "margin 0.40", color=MUTED, fontsize=6.5, va="center", ha="left")
    axes[0].set_ylim(len(ops) - 0.5, -0.8)
    fig.supxlabel("Gaze-error penalty vs no anonymisation (cm), subject-paired 95% CI", fontsize=8, color=INK2, y=-0.04)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.20))
    save(fig, f"fig2_noninferiority_K{K}", rows,
         ["K", "backbone", "operator", "condition", "mean diff cm", "CI lo", "CI hi", "subjects hurt /17", "verdict"])


# --------------------------------------------------------------- Figure 4
def fig_bland_altman(K="9", bb="mobile_vit", op="blackbox", cond="iii"):
    tab = {}
    for r in csv.DictReader(open(R / "anon_stage4_subject_all.csv")):
        if r["cond"] == cond and r["k"] == K and r["backbone"] == bb and r["fc_ft"] not in ("", "nan"):
            tab.setdefault(r["operator"], {})[int(r["rec"])] = float(r["fc_ft"])
    recs = sorted(set(tab["none"]) & set(tab[op]))
    a = np.array([tab["none"][s] for s in recs]); b = np.array([tab[op][s] for s in recs])
    mean, diff = (a + b) / 2, b - a
    bias, sd = diff.mean(), diff.std(ddof=1)
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    fig, ax = plt.subplots(figsize=(TEXTW * 0.62, 2.5))
    ax.axhline(0, color=AXIS, lw=0.6)
    ax.axhline(0.40, color=MUTED, lw=0.8)
    ax.axhline(bias, color=BLUE, lw=1.2)
    for v in (lo, hi):
        ax.axhline(v, color=BLUE, lw=0.8, alpha=0.55)
    ax.scatter(mean, diff, s=26, color=BLUE, edgecolor="white", linewidth=1.0, zorder=3)
    xr = ax.get_xlim()[1]
    fmt = lambda v, n: (f"{v:+.{n}f}").replace("-", "\u2212")
    for v, t, c in ((bias, f"bias {fmt(bias, 3)}", BLUE), (hi, f"+1.96 SD {fmt(hi, 2)}", INK2),
                    (lo, f"\u22121.96 SD {fmt(lo, 2)}", INK2), (0.40, "margin 0.40", MUTED)):
        ax.text(xr, v, " " + t, va="center", ha="left", fontsize=6.5, color=c)
    ax.set_xlabel("Mean of the two conditions (cm)")
    ax.set_ylabel("Face removed \u2212 none (cm)")
    style(ax, "y")
    rows = [[s, round(x, 4), round(y, 4), round(m, 4), round(d, 4)] for s, x, y, m, d in zip(recs, a, b, mean, diff)]
    save(fig, f"fig4_bland_altman_{bb}_{op}_K{K}", rows, ["recording", "none cm", f"{op} cm", "mean", "difference"])


# --------------------------------------------------------------- Figure 5
def fig_linkage():
    """Release-only linkage, on the face crop (a) and on the eye crop (b).

    Panel (a): two numbers matter per arm and only one can be a position. ARI
    carries the axis, frames-detected is annotated. Without the second, the
    conventional operators look protective when what they actually do is destroy
    the face detection that the attack -- and the gaze pipeline -- depends on.

    Panel (b) is the control panel (a) cannot supply. The session-nuisance floor
    is a face-free patch of the room, so it rules out room and session cues but
    NOT hair, ears, neck, clothing or face outline: those sit inside the face
    crop and outside the floor's patch, and the swap never edits them. The 120 px
    eye crop excludes all of them, and in the fully swapped arms the eye pixels
    are themselves synthetic. Linkage survives there, so it cannot be attributed
    to the channels panel (a) leaves open.
    """
    import json

    def rd(p):
        d = json.load(open(R / p))
        return d["ari_k18"], d["k_selected"], d.get("n_frames"), d.get("det_fail")

    def rd_peri(p, tag):
        for d in json.load(open(R / p)):
            if d["tag"] == tag:
                return d
        raise KeyError(f"{tag} not in {p}")

    face_arms = [
        ("No anonymisation", "anon_stage3b/t1sa_none.json", BLUE),
        ("blur 0.10·IOD", "anon_stage3b/t1sa_blur0.10.json", GREY_SERIES),
        ("blur 0.20·IOD", "anon_stage3b/t1sa_blur0.20.json", GREY_SERIES),
        ("blur 0.50·IOD", "anon_stage3b/t1sa_blur0.50.json", GREY_SERIES),
        ("pixelate 0.06·IOD", "anon_stage3b/t1sa_pixel0.06.json", GREY_SERIES),
        ("face region removed", "anon_stage3b/t1sa_blackbox.json", GREY_SERIES),
        ("SimSwap template 1", "anon_swap/t1_ProcessedSwap_selfalign.json", ORANGE),
        ("SimSwap template 2", "anon_swap2/t1_ProcessedSwap2_selfalign.json", ORANGE),
    ]
    # Pooled halves: the attacker of a public release holds the whole corpus, not
    # half of it. Per-half values are in the source JSONs and agree within 0.09.
    eye_arms = [
        ("Original, left", "anon_peri_linkage/peri_linkage_ProcessedData.json", "left_pooled", BLUE),
        ("Original, right", "anon_peri_linkage/peri_linkage_ProcessedData.json", "right_pooled", BLUE),
        ("SimSwap t1, left", "anon_peri_linkage/peri_linkage_ProcessedSwap.json", "left_pooled", ORANGE),
        ("SimSwap t1, right", "anon_peri_linkage/peri_linkage_ProcessedSwap.json", "right_pooled", ORANGE),
        ("SimSwap t2, left", "anon_peri_linkage/peri_linkage_ProcessedSwap2.json", "left_pooled", ORANGE),
        ("SimSwap t2, right", "anon_peri_linkage/peri_linkage_ProcessedSwap2.json", "right_pooled", ORANGE),
    ]
    floor_ari, floor_k, floor_n, _ = rd("anon_stage3b/t1bg_p120.json")

    fig, (axa, axb) = plt.subplots(
        2, 1, figsize=(TEXTW, 4.9), sharex=True,
        gridspec_kw={"height_ratios": [len(face_arms), len(eye_arms)], "hspace": 0.42})
    rows, seen = [], set()

    # --- (a) face crop -----------------------------------------------------
    for i, (lab, p, col) in enumerate(face_arms):
        ari, k, n, df = rd(p)
        name = {BLUE: "no anonymisation", GREY_SERIES: "conventional operator",
                ORANGE: "GAN face replacement"}[col]
        axa.scatter([ari], [i], s=38, color=col, edgecolor="white", linewidth=1.2, zorder=3,
                    label=name if name not in seen else None)
        seen.add(name)
        axa.text(1.06, i, f"{n}/1080 frames", va="center", ha="left", fontsize=6.8, color=INK2)
        axa.text(ari, i - 0.36, f"k={k}", va="center", ha="center", fontsize=6.5, color=MUTED)
        rows.append(["a: face crop", lab, round(ari, 4), k, n, df, str(R / p)])
    axa.axvline(floor_ari, color=MUTED, lw=0.8)
    axa.text(floor_ari + 0.015, -0.78, f"floor {floor_ari:.3f} (k={floor_k})",
             color=MUTED, fontsize=6.8, va="center")
    rows.append(["a: face crop", "session-nuisance floor", round(floor_ari, 4), floor_k,
                 floor_n, "", "anon_stage3b/t1bg_p120.json"])
    axa.set_yticks(range(len(face_arms))); axa.set_yticklabels([a[0] for a in face_arms])
    axa.set_ylim(len(face_arms) - 0.4, -1.15)
    axa.set_title("(a)  face crop — hair edges, spectacles and face outline in frame",
                  loc="left", color=INK, pad=6)
    style(axa, "x")
    axa.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02), handletextpad=0.3)

    # --- (b) eye crop ------------------------------------------------------
    for i, (lab, p, tag, col) in enumerate(eye_arms):
        d = rd_peri(p, tag)
        axb.scatter([d["ari_k18"]], [i], s=38, color=col, edgecolor="white", linewidth=1.2,
                    zorder=3)
        axb.text(1.06, i, f"purity {d['purity_k18']:.2f}", va="center", ha="left",
                 fontsize=6.8, color=INK2)
        axb.text(d["ari_k18"], i - 0.36, f"kₐᵤₜₒ={d['k_selected']}",
                 va="center", ha="center", fontsize=6.5, color=MUTED)
        rows.append(["b: eye crop", lab, round(d["ari_k18"], 4), d["k_selected"],
                     d["n_frames"], round(d["ari_kauto"], 4), str(R / p)])
    axb.axvline(floor_ari, color=MUTED, lw=0.8)
    axb.text(floor_ari + 0.015, -0.78, f"floor {floor_ari:.3f} (borrowed from panel a)",
             color=MUTED, fontsize=6.8, va="center")
    axb.set_yticks(range(len(eye_arms))); axb.set_yticklabels([a[0] for a in eye_arms])
    axb.set_ylim(len(eye_arms) - 0.4, -1.15)
    axb.set_xlim(-0.03, 1.05)
    axb.set_title("(b)  120 px eye crop — no hair, ears, neck, clothing or outline",
                  loc="left", color=INK, pad=6)
    axb.set_xlabel("Adjusted Rand index at known k = 18, attacker holding only the release")
    style(axb, "x")

    save(fig, "fig5_linkage_release_only", rows,
         ["panel", "arm", "ARI@k=18", "k selected", "frames", "detection failures / ARI at auto k",
          "source"])


# --------------------------------------------------------------- Figure 3
def fig_threat_models():
    """The same anonymised eye crops under two attackers.

    One axis, one metric; the attacker is the categorical variable. This is the
    figure that keeps a reader from quoting a privacy number without its threat
    model -- the two series differ by up to 60 points on identical data.
    """
    import json
    def cross(p):
        d = json.load(open(R / p)); return d["TAR@FAR=0.001"]
    def release(p):
        d = json.load(open(R / p)); return d["direct_TAR@FAR=0.001"]
    arms = [
        ("Original", "anon_swap/peri_ProcessedData_%s.json", "anon_adaptive/peri_ProcessedData_%s.json"),
        ("SimSwap t1", "anon_swap/peri_ProcessedSwap_%s.json", "anon_adaptive/peri_ProcessedSwap_%s.json"),
        ("SimSwap t2", "anon_swap2/periX_ProcessedSwap2_%s.json", "anon_swap2/peri_ProcessedSwap2_%s.json"),
    ]
    eyes = [("appleLeftEye", "left"), ("appleRightEye", "right")]
    floor = json.load(open(R / "anon_stage3b" / "floor_background.json"))["TAR@FAR=0.001"] * 100
    fig, ax = plt.subplots(figsize=(TEXTW, 2.5))
    rows, y, ticks, labels = [], 0, [], []
    for lab, pc, pr in arms:
        for eye, eshort in eyes:
            c, r = 100 * cross(pc % eye), 100 * release(pr % eye)
            ax.plot([c, r], [y, y], color=GRID, lw=1.0, zorder=1)
            ax.scatter([c], [y], s=34, color=BLUE, edgecolor="white", lw=1.1, zorder=3,
                       label="attacker holds enrolment photographs" if y == 0 else None)
            ax.scatter([r], [y], s=34, color=ORANGE, edgecolor="white", lw=1.1, zorder=3,
                       label="attacker holds only the release" if y == 0 else None)
            # one label per row, always to the RIGHT of the rightmost dot: a label
            # placed left of a near-zero value lands on top of the y tick label
            txt = f"{c:.1f}% \u2192 {r:.0f}%" if abs(r - c) > 6 else f"{c:.0f}% / {r:.0f}%"
            ax.text(max(c, r) + 2.5, y, txt, va="center", fontsize=6.8, color=INK2)
            ticks.append(y); labels.append(f"{lab}, {eshort}")
            rows.append([lab.replace("\n", " "), eshort, round(c, 2), round(r, 2)])
            y += 1
        y += 0.5
    ax.axvline(floor, color=MUTED, lw=0.8)
    ax.text(floor + 1.5, -0.9, f"floor {floor:.1f}% (enrolment-photograph attacker)",
            color=MUTED, fontsize=6.6, va="center")
    rows.append(["floor", "background patch", round(floor, 2), ""])
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    ax.set_ylim(y - 0.6, -1.3); ax.set_xlim(-4, 118)
    ax.set_xlabel("True-accept rate at FAR = 1e-3 (%)")
    style(ax, "x")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, handletextpad=0.3)
    save(fig, "fig3_threat_models", rows,
         ["arm", "eye", "TAR originals-in-hand %", "TAR release-only %"])


if __name__ == "__main__":
    fig_leakage()
    fig_noninferiority("9")
    fig_noninferiority("72")
    fig_bland_altman()
    fig_linkage()
    fig_threat_models()
