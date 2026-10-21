import re, sys
sys.path.insert(0, ".")
from handwritten import DATASET, MODELS, DEID
P = "/springbrook/share/eng/esrpxk/results/paper/CCF_PASTE_ORDER.md"
L = open(P).read().splitlines()
def r(n): return open(n + ".md").read().strip()
def block(start, end_excl):
    """Lines between two anchors (exclusive), raw."""
    i = next(k for k, l in enumerate(L) if l.startswith(start))
    j = next(k for k in range(i + 1, len(L)) if L[k].startswith(end_excl))
    return L[i:j]
def table_after(anchor):
    i = next(k for k, l in enumerate(L) if anchor in l)
    k = i
    while not L[k].startswith("|"): k += 1
    t = []
    while k < len(L) and L[k].startswith("|"): t.append(L[k]); k += 1
    return "\n".join(t)

abstract = r("abstract")
intro = r("intro")
contrib = r("contrib")
related = r("related")
threat = r("threat")
leak = r("leak")
pre = r("pre"); c9c = r("c9c"); cter = r("c9cter"); cbis = r("c9cbis")
disc = r("disc"); limits = r("limits")
# xref is applied to reused draft text only (defined below, used after definition)

# split pieces
cter_main, cter_itr = cter.split("The fourth architecture agrees", 1)
cter_itr = "The fourth architecture agrees" + cter_itr
cbis_a, cbis_b = cbis.split("This is what makes", 1); cbis_b = "This is what makes" + cbis_b
lim_body, concl = limits.split("**Conclusion.**", 1)

# leak: split so the 2x2 figure can follow the DeepPrivacy2 paragraph; strip the in-text DP2 table into a numbered one later
tab3 = table_after("### 9a. Table 3")
tab_dp2f = table_after("### 9a-bis.")
tab_dp2fb = table_after("DeepPrivacy2 **full body** (FB_cse_mask_face")
tab_fams = table_after("FAMS (diffusion; §8), same five initialisations")
tab_pair = table_after("### 9b. Table 4")
tab5 = table_after("### 9c-ter.")
tab_itr = table_after("**iTracker, retrained (D→R, D→D), one initialisation**")
tab_bio = table_after("**Which number to anchor on.**")
tab_cal = table_after("Calibrated, per arm:")
tab_P = table_after("### Table P")
tab_guid = table_after("## 9.6 Design Guidelines")

# Contributions list lead-in
contrib = "This paper makes four contributions.\n\n" + contrib

# section / table cross-references in the reused text
def xref(s):
    s = s.replace("§Data Availability", "the Data Availability statement")
    s = s.replace("(§9.6)", "(§10)").replace("§9.6", "§10")
    s = re.sub(r"§8(?![.\d])", "§7", s)   # old results §8 -> final §7; new text uses §8.x (final) already
    s = s.replace("Table 4", "Table 5@").replace("Table 5,", "Table 6,").replace("Table 5:", "Table 6:")
    s = s.replace("Table 5@", "Table 5")
    s = s.replace("in our earlier analysis", "on these data")
    return s
for _n in ("abstract","intro","contrib","related","threat","leak","pre","c9c","cter_main","cter_itr","cbis_a","cbis_b","disc","lim_body","concl"):
    globals()[_n] = xref(globals()[_n])
pieces = []
A = pieces.append

A("# Keep Your Eyes On It: Measuring Gaze Utility and Re-identification Risk in De-identified Smartphone Gaze Data\n")
A("## ABSTRACT\n\n" + abstract.replace("\n>", "").replace("\n\n", "\n\n"))
A("**CCS Concepts:** • **Security and privacy** → **Pseudonymity, anonymity and untraceability**; • **Human-centered computing** → **Empirical studies in ubiquitous and mobile computing**; • **Computing methodologies** → *Biometrics*.\n\n"
  "**Additional Key Words and Phrases:** smartphone gaze estimation, face de-identification, re-identification, linkage attack, privacy–utility trade-off, ocular biometrics, dataset release, privacy auditing")
A("# 1 INTRODUCTION\n\n" + intro.replace("(Figure 1).", "(Figure 1).\n\n![Figure 1](figs_core/fig_cycle.pdf)\n\n@@FIG1@@\n\n", 1))
A(contrib)
A("# 2 RELATED WORK\n\n" + related)
A(DATASET.strip()); A(MODELS.strip()); A(DEID.strip())
threat = threat.replace("For a corpus intended for public release\nA2 is the operative attacker,", "The two can disagree by up to 60 points on identical crops (Figure S2). For a corpus intended for public release\nA2 is the operative attacker,")
assert "Figure S2" in threat
A("# 6 THREAT MODEL AND PRIVACY EVALUATION\n\n" + threat)
leak = re.sub(r"(gives the two-part condition as a factorial result, on the face crop,\n.*?floor 1\.6%\)):\n\n\| Edit domain.*?\n\n", 
    r"\1 (Figure 6a). Face-crop linkage is 1.000 with face-mode editing and a fixed identity, 0.982 when either the edit domain is widened to the whole person or the identity is resampled every frame, and 0.297 only when both are; verification falls from 89.7–100% in the first three cells to 35.7% [27, 45] in the fourth.\n\n", leak, flags=re.S)
assert "| Edit domain" not in leak
A("# 7 IDENTITY LEAKAGE UNDER DE-IDENTIFICATION\n\nTable 2 summarises the leakage of each SimSwap release under both attackers; Table 2a gives face-crop re-identification for every family, and Figure 6 linkability.\n\n" + leak)
P_note = ("**Table 2.** Identity leakage across release strategies. A1: the attacker holds enrolment photographs; "
          "A2: the attacker holds only the release. TAR at FAR = 1e-3 (ArcFace) and ARI at k = 18, with 95% "
          "participant-level bootstrap intervals.\n\n" + tab_P)
A(P_note)
A("![Figure 6](figs_core/fig_core1_privacy.pdf)\n\n@@FIG6@@")

A("# 8 GAZE UTILITY UNDER DE-IDENTIFICATION\n\n## 8.1 Utility protocol\n\n" + pre)
A("## 8.2 Models trained on real data (R→D)\n\n"
  "Table 3 reports SimSwap under R→D; Table 4 reports DeepPrivacy2 and FAMS under the same models, "
  "initialisations and harness. Before any DeepPrivacy2 or FAMS number was computed, the real-data arm was re-run "
  "through the same pipeline and reproduced Table 3 to within 0.0012 cm on all 100 folds.\n\n"
  "**Table 3.** Gaze error (cm) under R→D for SimSwap: trained on real data, deployed on de-identified "
  "crops. Mean over five folds, then mean ± SD over five independent initialisations; in parentheses, the change "
  "against the same model's real-data baseline. The SD is the noise floor of the comparison (0.04–0.20 cm on the "
  "baseline): no difference smaller than it can be supported in either direction.\n\n" + tab3)
A("**Table 4.** R→D for DeepPrivacy2 (face and full-body modes, fixed and per-frame identity) and FAMS "
  "(fresh noise per frame, all 210,924 task frames). Mean ± SD over five initialisations; in parentheses, or as "
  "listed for FAMS, the change against the real-data baseline in Table 3.\n\n"
  "*(a) DeepPrivacy2, face mode*\n\n" + tab_dp2f + "\n\n*(b) DeepPrivacy2, full-body mode*\n\n" + tab_dp2fb +
  "\n\n*(c) FAMS*\n\n" + tab_fams)
A("Deployed on DeepPrivacy2 crops, real-data models are worse than predicting the mean: 15.9–20.5 cm in face mode "
  "and 14.6–17.8 cm in full-body mode, against 9.37 cm. A model given repainted eyes is confidently wrong. Hybrid "
  "recovers 91–101% of the face-mode penalty and 93–99% of the full-body one, as it does for SimSwap. Every one of "
  "the 421,848 full-body frames had the participant detected and repainted. In 2,861 of them (1.4%) the detector "
  "also returned a second person region without a surface map, which the pipeline leaves unedited; the "
  "participant's released crops in those frames are changed as much as in all others, so no original pixels reach "
  "the release. Under deployment alone FAMS looks worse than SimSwap (+4.1 to +5.1 cm, against +2.4 to +3.7 cm for SimSwap A), "
  "and Hybrid recovers 93–100% of its penalty. Section 8.3 shows that most of the DeepPrivacy2 and FAMS penalty "
  "under R→D is domain shift.")
A("**Table 5.** Hybrid minus Full Synthetic (cm), paired within initialisation, mean ± SD over five draws; negative "
  "means the genuine ocular crops recover accuracy. All 40 draw-level values are negative, and the across-draw SD "
  "is an order of magnitude below the effect in every cell.\n\n" + tab_pair)
A(c9c)
A("Changing the source template from Identity A to Identity B changes the Full Synthetic error by +0.63 to +3.38 cm, "
  "but the Hybrid error by only −0.34 to +0.86 cm, and Hybrid recovers most of the penalty under both templates "
  "(Table 5). The synthesis penalty depends strongly on the template; the Hybrid error changes much less.")

A("## 8.3 Models retrained on the release (D→R and D→D)\n\n"
  "**Table 6.** Penalty (cm) against each model's own R→R error (MobileNet-V3 Large 4.98, AFFNet 5.67, MGazeNet "
  "4.96 cm, five-draw means), under R→D, D→R and D→D: mean over the three architectures, with [min, max] across "
  "them. D→R and D→D are means over five independent initialisations per cell (per-draw SDs 0.01–0.83 cm); "
  "R→D is recomputed over the same three architectures so the columns are comparable. Retraining uses "
  "the same recipe and participant partition as the real-data models. iTracker (about 9 h per fold) is reported "
  "separately in Table 8. Negative Hybrid values read as indistinguishable from baseline, not as gains.\n\n" + tab5)
A("FAMS was retrained on every task frame with the same five initialisations as every other arm. On a 10% frame "
  "subset, against a real-data model retrained on the same subset, the FAMS penalty on real faces was +0.32 cm, "
  "against +3.26 cm for SimSwap A and +3.46 cm for DeepPrivacy2 retrained on exactly those frames. The subset "
  "understated the FAMS penalty (+0.94 cm on all frames) but not its ranking: FAMS is better than SimSwap A and "
  "DeepPrivacy2 in all 15 paired draws (architecture × initialisation), on the subset and again on all frames.")
cter_main = cter_main.replace(
    "| Released per frame |", "**Table 7.** Releasing FaceMesh eye geometry instead of pixels: gaze error of gradient-boosted "
    "regressors (no per-participant calibration) and release-only privacy of the released features, with 95% "
    "participant-level intervals.\n\n| Released per frame |", 1)
A(cter_main.strip())
A("Figure 7 places every protocol on one plane, utility against release-only linkage; Figure S1 shows the per-architecture penalties under all three conditions.")
A("![Figure 7](figs_core/fig_core3_tradeoff.pdf)\n\n@@FIG7@@")
A("**Table 8.** iTracker under R→D, D→R and D→D, one initialisation for the retrained conditions. Penalty (cm) "
  "against its five-draw R→R error of 5.85 cm; predicting the mean corresponds to +3.52 cm.\n\n" + tab_itr)
A(cter_itr.strip())
A("Fold 2 is hard for every model, real data included (7.3–8.5 cm), which compresses all penalties in that fold. "
  "Tested on the release, the SimSwap Full Synthetic penalty is positive in 148 of 150 fold-level cells (two "
  "templates × three architectures × five folds × five initialisations).")

A("## 8.4 What the cost means for clinical measurement\n\n"
  "A centimetre figure on its own does not say whether a protocol is usable; it has to be set against what each "
  "biomarker needs to resolve. Tables 3–6 report the uncalibrated error, which is the right quantity for comparing "
  "arms, because holding calibration out isolates the effect of the imagery. It is the wrong quantity for asking "
  "whether the instrument is clinically usable, because a deployed screening tool calibrates each user. With "
  "per-participant calibration the same four architectures reach 3.70–5.11 cm on real data, which at the 80–120 cm "
  "viewing distance of our protocol is 1.8–3.7°.\n\n"
  "**Table 9.** What each oculomotor biomarker must resolve, against the calibrated error of 1.8–3.7°. "
  "Requirements are derived from published normative ranges.\n\n" + tab_bio)
A(cbis_a.strip())
A("**Table 10.** Calibrated error per release protocol, in centimetres and in degrees of visual angle at 120 and "
  "80 cm, as ranges over the four architectures.\n\n" + tab_cal)
A(cbis_b.strip())
A("Webcam-based eye tracking validated on clinically relevant saccade and free-viewing paradigms reports about "
  "1.42° mean error [CITE: webcamval]. That figure is obtained after careful per-user calibration, so the "
  "like-for-like comparison is with our calibrated 1.8–3.7°, not with the uncalibrated numbers in Table 3; on that "
  "basis the two sit in the same regime, with ours somewhat coarser.")

A("# 9 DISCUSSION\n\nEvery claim in this section concerns appearance-based gaze data — face and eye crops released "
  "for training gaze models. We make no claim about other sensing modalities; the evidence does not reach them.\n\n" + disc)
A("# 10 DESIGN GUIDELINES FOR RELEASING GAZE DATA\n\n"
  "Table 11 distils the results into seven guidelines for authors of appearance-based gaze datasets, each with the "
  "evidence behind it and how to check it. The full fifteen-item checklist and the audit tool are in the "
  "supplementary material [CITE: privaudit-repo].\n\n**Table 11.** Design guidelines for releasing gaze data.\n\n"
  + tab_guid.replace("(retraining protocol, §9)", "(retraining protocol, §8.3)"))
A("# 11 LIMITATIONS\n\n" + lim_body.strip())
A("# 12 CONCLUSION\n\n" + concl.strip())
A("# DATA AVAILABILITY\n\nCognitiveGaze contains facial video and is shared under controlled access: researchers "
  "apply to the data holder and sign a data-use agreement that prohibits re-identification and linkage. Derived "
  "per-frame gaze estimates and oculomotor measures form an open tier, whose residual linkage is reported in §7. "
  "The audit tool `privaudit`, the fifteen-item release checklist and the analysis code are available at "
  "[CITE: privaudit-repo].")

# captions
cap = open(P).read()
def caption(head):
    i = cap.index(head); j = cap.index("\n\n", cap.index("\n>", i))
    body = cap[cap.index("\n>", i):j]
    return re.sub(r"\n> ?", " ", body).strip()
FIG1 = "**Figure 1.** " + caption("**Caption, Figure 1 (data-scarcity cycle).**")
FIG6 = "**Figure 6.** **Release-only linkage.** " + caption("**Caption, release-only linkage.**")
FIG7 = "**Figure 7.** " + caption("**Caption, privacy–utility plane.**")
S1 = "**Figure S1.** **Utility under R→D, D→R and D→D.** " + caption("**Caption, utility under three conditions.**")
S2 = "**Figure S2.** " + caption("**Caption, two attackers (supplement).**")
A("# SUPPLEMENTARY FIGURES\n\n![Figure S1](figs_core/fig_core2_utility.pdf)\n\n" + S1 +
  "\n\n![Figure S2](figures/fig3_threat_models.pdf)\n\n" + S2)

doc = "\n\n".join(pieces)
doc = doc.replace("@@FIG1@@", FIG1).replace("@@FIG6@@", xref(FIG6)).replace("@@FIG7@@", xref(FIG7))
# Final table numbering by order of appearance of captions ("**Table X.**"),
# so tables added with temporary labels (2a, 6b, ...) are numbered sequentially.
labels = re.findall(r"\*\*Table (\d+[a-z]?)\.\*\*", doc)
assert len(labels) == len(set(labels)), labels
num = {lab: str(i + 1) for i, lab in enumerate(labels)}
doc = re.sub(r"Tables (\d+[a-z]?)–(\d+[a-z]?)", lambda m: f"Tables @{num[m.group(1)]}–@{num[m.group(2)]}", doc)
doc = re.sub(r"Table (\d+[a-z]?)(?![\d])", lambda m: f"Table @{num[m.group(1)]}" if m.group(1) in num else m.group(0), doc)
doc = doc.replace("Table @", "Table ").replace("Tables @", "Tables ").replace("–@", "–")
doc = re.sub(r"\[CITE: ([^\]]+)\]", r"[\1]", doc)
for x, y in [("floor 1.6%) (Figure 6a).", "floor 1.6%; Figure 6a)."),
             ("its full-body mode — the\npaper's main contribution,", "its full-body mode — the\nmain contribution of the DeepPrivacy2 paper,")]:
    assert x in doc, x
    doc = doc.replace(x, y)
doc = re.sub(r"\n{3,}", "\n\n", doc)
open("/springbrook/share/eng/esrpxk/results/paper/MANUSCRIPT_long.md", "w").write(doc + "\n")
print(len(doc.split()), "words")
