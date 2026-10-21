import re
L = open("/springbrook/share/eng/esrpxk/results/paper/MANUSCRIPT.md").read().split("\n")
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
abstract = "\n".join(L[L.index("## ABSTRACT") + 1: next(k for k, l in enumerate(L) if l.startswith("**CCS"))]).strip()
T = {}
T["TABLE2"] = ("**Table 2.** SimSwap under A1 (TAR at FAR 1e-3, ArcFace) and A2 (ARI at k = 18), with 95% "
               "participant-level intervals. Hybrid face crops are synthetic; its eye crops are unaltered.\n\n"
               + tables_of(4).replace("| Released stream | Real Data |", "| Released stream | Unprocessed |"))
T["TABLE3"] = ("**Table 3.** Face-crop A1 for every family: released face crops matched against the participant's "
               "original face crops. TAR at FAR 1e-3 with 95% participant-level intervals; floors 1.6% (ArcFace) "
               "and 3.52% (FaceNet).\n\n" + tables_of(2))
T["TABLE4"] = ("**Table 4.** FAMS strength sweep on the privacy sample. Settings with more than 10% non-face outputs "
               "are not used for privacy conclusions. Pupil r: correlation of the synthetic pupil with horizontal "
               "EyeLink gaze (unprocessed 0.87; SimSwap A 0.79).\n\n" + tables_of(3))
T["TABLE5"] = ("**Table 5.** Gaze error (cm) under R→D for SimSwap: mean over five folds, then mean ± SD over five "
               "initialisations; in parentheses, the change against R→R.\n\n" + tables_of(5))
T["TABLE6"] = ("**Table 6.** Hybrid minus Full Synthetic under R→D (cm), paired within initialisation, mean ± SD "
               "over five draws; negative values mean the genuine eye crops recover accuracy.\n\n" + tables_of(7))
t7 = tables_of(8)
for a, b in [(", Full Synthetic", ", FS"), ("DeepPrivacy2 ", "DP2 "), ("FAMS (diffusion)", "FAMS"),
             ("*No-information reference (predict the mean)*", "*No-information reference*")]:
    t7 = t7.replace(a, b)
T["TABLE7"] = ("**Table 7.** Penalty against R→R (cm): mean over MobileNet-V3 Large, AFFNet and MGazeNet, with "
               "[min, max] across them; D→R and D→D over five initialisations each. R→R errors are 4.98, 5.67 and "
               "4.96 cm. FS: Full Synthetic; DP2: DeepPrivacy2. Negative Hybrid values denote no detectable "
               "difference, not gains.\n\n" + t7)
T["TABLE8"] = ("**Table 8.** Pupil-position preservation on 4,680 identical frames: within-participant correlation of "
               "the released pupil position with the original (horizontal, vertical) and with EyeLink gaze, "
               "mean over participants.\n\n" + tables_of(9))
T["TABLE9"] = ("**Table 9.** Stream isolation: eyes-only and face-only models retrained on each release (five "
               "initialisations); penalty against the same model under R→R (cm, mean ± SD).\n\n" + tables_of(10))
T["TABLE10"] = ("**Table 10.** Landmark release: gaze error of gradient-boosted regressors (uncalibrated) and "
                "privacy of the released features, with 95% participant-level intervals.\n\n" + tables_of(11))
T["TABLE11"] = ("**Table 11.** What each biomarker must resolve, against the calibrated error of 1.8–3.7°; "
                "requirements from published normative ranges.\n\n" + tables_of(13))
T["TABLE12"] = """**Table 12.** Design guidelines for releasing gaze data. privaudit implements the checks for G1, G2, G4 and G5.

| # | Guideline | Evidence |
|---|---|---|
| G1 | Run A1 on every released stream, face crop included, and test linkability (A2) as the anonymisation criterion. | Synthetic eye crops: 0.2% under A1, 43% within the release; synthetic face crops: 18–24% under A1. |
| G2 | Measure the floor on your own recordings, using a face-free patch of the same frames. | TAR 1.57%, Rank-1 50.5%, ARI 0.074–0.089. |
| G3 | Do not buy utility by releasing genuine eyes. | Hybrid recovers 85–106% of utility; genuine eye crops verify at 62–80%. |
| G4 | Normalise crop size and per-crop photometry. | Crop size alone identifies 66.7%; photometry alone links at ARI 0.87–0.90. |
| G5 | When stripping identifiers, resample identity and edit everything released. | Only that configuration breaks face-crop linkage (0.297); verification remains 36%. |
| G6 | Report utility under D→R beside the no-information reference, and test the pupil directly. | FAMS transfers best, yet its pupils track gaze less closely than SimSwap's (17/17 participants). |
| G7 | If any check fails, release under controlled access and state which channels leak. | No tested protocol is both private and useful. |"""
T["TABLES1"] = ("**Table S1.** R→D for DeepPrivacy2 and FAMS: mean ± SD over five initialisations; in parentheses "
                "(FAMS: as listed), the change against R→R (Table 5).\n\n" + tables_of(6, all_blocks=True))
T["TABLES2"] = ("**Table S2.** iTracker: penalty against R→R (5.85 cm); D→R and D→D from one initialisation; "
                "predicting the mean corresponds to +3.52 cm.\n\n" + tables_of(12))
s = open("MANUSCRIPT_condensed_prose.md").read().replace("@@ABSTRACT@@", abstract)
for k, v in T.items():
    assert f"@@{k}@@" in s, k
    s = s.replace(f"@@{k}@@", v)
assert "@@" not in s
s = re.sub(r"\n{3,}", "\n\n", s)
open("MANUSCRIPT_condensed.md", "w").write(s)
main = s.split("# SUPPLEMENTARY MATERIAL")[0]
print("main words", len(main.split()), "| supplement", len(s.split("# SUPPLEMENTARY MATERIAL")[1].split()), "| abstract", len(abstract.split()))
