import re
L = open("/springbrook/share/eng/esrpxk/results/paper/MANUSCRIPT_long.md").read().split("\n")
def tables_of(num, all_blocks=False):
    i = next(k for k, l in enumerate(L) if l.startswith(f"**Table {num}.**"))
    out, k, blocks = [], i + 1, 0
    while k < len(L):
        l = L[k]
        if l.startswith("**Table") or l.startswith("#") or (blocks and not all_blocks and not l.startswith("|")):
            break
        if l.startswith("|") or (all_blocks and l.startswith("*(")):
            out.append(l)
            if l.startswith("|") and (k + 1 >= len(L) or not L[k + 1].startswith("|")):
                blocks += 1; out.append("")
        k += 1
    return "\n".join(out).strip()
abstract = ""
T = {}
T["TABLE2"] = ("**Table 2.** SimSwap under A1 (TAR at FAR 1e-3, ArcFace) and A2 (ARI at k = 18), with 95% "
               "participant-level intervals. In Hybrid the face crops are synthetic and the eye crops are unaltered.\n\n"
               + tables_of(4).replace("| Released stream | Real Data |", "| Released stream | Unprocessed |"))
T["TABLE3"] = ("**Table 3.** Face-crop A1 for every family: released face crops matched against the participant's "
               "original face crops. TAR at FAR 1e-3 with 95% participant-level intervals. The floors are 1.6% for "
               "ArcFace and 3.52% for FaceNet.\n\n" + tables_of(2))
T["TABLE4"] = ("**Table 4.** FAMS strength sweep on the privacy sample. Settings with more than 10% non-face outputs "
               "are not used for privacy conclusions. Pupil r: correlation of the synthetic pupil with horizontal "
               "EyeLink gaze (0.87 for unprocessed data and 0.79 for SimSwap A). n/a: not evaluated.\n\n" + tables_of(3).replace("| — | — | — | — | — |", "| n/a | n/a | n/a | n/a | n/a |"))
T["TABLE5"] = ("**Table 5.** Gaze error (cm) under R→D for SimSwap: mean over five folds, then mean ± SD over five "
               "initialisations. Values in parentheses give the change against R→R.\n\n" + tables_of(5))
T["TABLE6"] = ("**Table 6.** Hybrid minus Full Synthetic under R→D (cm), paired within initialisation, mean ± SD "
               "over five draws. Negative values mean that the genuine eye crops recover accuracy.\n\n" + tables_of(7))
t7 = tables_of(8)
for a, b in [(", Full Synthetic", ", FS"), ("DeepPrivacy2 ", "DP2 "), ("FAMS (diffusion)", "FAMS"),
             ("*No-information reference (predict the mean)*", "*No-information reference*")]:
    t7 = t7.replace(a, b)
T["TABLE7"] = ("**Table 7.** Penalty against R→R (cm): mean over MobileNet-V3 Large, AFFNet and MGazeNet, with "
               "[min, max] across them. D→R and D→D are averaged over five initialisations each. R→R errors are 4.98, "
               "5.67 and 4.96 cm. FS stands for Full Synthetic and DP2 for DeepPrivacy2. Negative Hybrid values "
               "indicate no detectable difference rather than gains.\n\n" + t7)
T["TABLE8"] = ("**Table 8.** Pupil-position preservation on 4,680 identical frames: within-participant correlation of "
               "the released pupil position with the original (horizontal, vertical) and with EyeLink gaze, "
               "mean over participants.\n\n" + tables_of(9))
T["TABLE9"] = ("**Table 9.** Stream isolation: eyes-only and face-only models retrained on each release (five "
               "initialisations), with the penalty against the same model under R→R (cm, mean ± SD).\n\n" + tables_of(10))
T["TABLE10"] = ("**Table 10.** Landmark release: gaze error of gradient-boosted regressors (uncalibrated) and "
                "privacy of the released features, with 95% participant-level intervals.\n\n" + tables_of(11))
T["TABLE11"] = ("**Table 11.** What each biomarker must resolve, compared with the calibrated error of 1.8–3.7°. "
                "Requirements are taken from published normative ranges.\n\n" + tables_of(13).replace("**usable** — measured bias −16.8 ms, under one 33 ms frame", "**usable** (bias −16.8 ms, less than one 33 ms frame)").replace("one frame; 100°/s detection threshold", "one frame, with a 100°/s detection threshold"))
T["TABLE12"] = """**Table 12.** Design guidelines for releasing gaze data. privaudit implements the checks for G1, G2, G4 and G5.

| # | Guideline | Evidence |
|---|---|---|
| G1 | Run A1 on every released stream, including the face crop, and test linkability (A2) as the anonymisation criterion. | Synthetic eye crops reach 0.2% under A1 but 43% within the release, and synthetic face crops 18–24% under A1. |
| G2 | Measure the floor on your own recordings, using a face-free patch of the same frames. | TAR 1.57%, Rank-1 50.5%, ARI 0.074–0.089. |
| G3 | Do not buy utility by releasing genuine eyes. | Hybrid recovers 82–111% of utility, and genuine eye crops verify at 62–80%. |
| G4 | Normalise crop size and per-crop photometry. | Crop size alone identifies 66.7% of participants, and photometry alone links at ARI 0.87–0.90. |
| G5 | When removing identifiers, redraw identity per frame and edit everything that is released. | Only this configuration breaks face-crop linkage (0.297), and verification still reaches 36%. |
| G6 | Report utility under D→R beside the no-information reference, and test the pupil directly. | FAMS transfers best, yet its pupils track gaze less closely than SimSwap's (17/17 participants). |
| G7 | If any check fails, release under controlled access and state which channels leak. | No tested protocol is both private and useful. |"""
T["TABLES1"] = ("**Table S1.** R→D for DeepPrivacy2 and FAMS, as mean ± SD over five initialisations. Values in "
                "parentheses (listed directly for FAMS) give the change against R→R (Table 5).\n\n" + tables_of(6, all_blocks=True))
T["TABLES2"] = ("**Table S2.** Penalty for iTracker against R→R (5.85 cm). D→R and D→D come from a single "
                "initialisation, and predicting the mean corresponds to +3.52 cm.\n\n" + tables_of(12))
s = open("MANUSCRIPT_condensed_prose.md").read().replace("@@ABSTRACT@@", abstract)
for k, v in T.items():
    assert f"@@{k}@@" in s, k
    s = s.replace(f"@@{k}@@", v)
assert "@@" not in s
s = re.sub(r"\n{3,}", "\n\n", s)
s = re.sub(r"(?m)^(\*\([a-z]\)[^\n]*\*)\n(?=\|)", r"\1\n\n", s)   # sub-table labels need their own paragraph
open("MANUSCRIPT_condensed.md", "w").write(s)
main = s.split("# SUPPLEMENTARY MATERIAL")[0]
print("main words", len(main.split()), "| supplement", len(s.split("# SUPPLEMENTARY MATERIAL")[1].split()), "| abstract", len(abstract.split()))
