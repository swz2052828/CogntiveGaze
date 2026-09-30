"""Number registry for the paper (file named registry.py, never numbers.py: a module
 named `numbers` on sys.path shadows the standard library and breaks numpy): every figure in the text comes from here.

Reads result files only. Writes results/paper/numbers.json (machine-readable, for
the manuscript build) and results/paper/NUMBERS.md (for a human cross-check).
A number that is quoted in the draft but not in this registry should be treated
as unverified.

Anything whose source is known to be superseded or pending carries a `status`
field rather than being silently omitted.
"""
import csv
import json
import re
from pathlib import Path

R = Path("/springbrook/share/eng/esrpxk/results")
N = {}


def put(key, value, source, **extra):
    N[key] = {"value": value, "source": source, **extra}


def js(p):
    return json.load(open(R / p))


# ---- privacy: identification with subject-level CIs --------------------------
ci = {r["source"]: r for r in csv.DictReader(open(R / "paper" / "privacy_ci.csv"))}
for key, src in [
    ("t2.original.face", "anon_stage3b/t2_none.scores.npz"),
    ("t2.original.eye_right", "anon_swap/peri_ProcessedData_appleRightEye.scores.npz"),
    ("t2.original.eye_left", "anon_swap/peri_ProcessedData_appleLeftEye.scores.npz"),
    ("t2.simswap1.face", "anon_swap/t2_ProcessedSwap.scores.npz"),
    ("t2.simswap1.eye_right", "anon_swap/peri_ProcessedSwap_appleRightEye.scores.npz"),
    ("t2.simswap1.eye_left", "anon_swap/peri_ProcessedSwap_appleLeftEye.scores.npz"),
    ("t2.simswap2.face", "anon_swap2/t2_ProcessedSwap2.scores.npz"),
    ("t2.simswap2.eye_right", "anon_swap2/periX_ProcessedSwap2_appleRightEye.scores.npz"),
    ("t2.simswap2.eye_left", "anon_swap2/periX_ProcessedSwap2_appleLeftEye.scores.npz"),
    ("t2.blur0.20.face", "anon_stage3b/t2_blur0.20.scores.npz"),
    ("t2.blur0.50.face", "anon_stage3b/t2_blur0.50.scores.npz"),
    ("t2.pixel0.25.face", "anon_stage3b/t2_pixel0.25.scores.npz"),
    ("t2.blackbox.face", "anon_stage3b/t2_blackbox.scores.npz"),
    ("t2.blackbox_eyeflat.face", "anon_stage3b/t2_blackbox_eyeflat.scores.npz"),
    ("t2.blur0.50_unmasked.face", "anon_stage3b/t2_blur0.50_unmasked.scores.npz"),
]:
    if src not in ci:
        put(key, None, src, status="missing"); continue
    r = ci[src]
    for m in ("TAR@FAR=0.001", "dprime", "rank1"):
        put(f"{key}.{m}", float(r[m]), "paper/privacy_ci.csv <- " + src,
            ci95=[float(r[m + "_lo"]), float(r[m + "_hi"])], n_subjects=int(r["n_subjects"]))

fl = js("anon_stage3b/floor_background.json")
put("floor.t2.TAR@FAR=0.001", fl["TAR@FAR=0.001"], "anon_stage3b/floor_background.json")
put("floor.t2.dprime", fl["background_dprime"], "anon_stage3b/floor_background.json")
put("floor.t2.rank1", fl["background_direct_rank1"], "anon_stage3b/floor_background.json")
put("iod.px.mean", fl["iod_px_mean"], "anon_stage3b/floor_background.json",
    range=[fl["iod_px_min"], fl["iod_px_max"]])

# ---- privacy: the release-only eye attack (periocular_attack.py) --------------
# No CIs: this script stores no score matrix. Keys are prefixed `t5ro.` so a
# release-only number can never be confused with the cross-domain `t2.` one --
# they differ by up to 60 points on identical data.
for key, p in [("t5ro.original.eye_left", "anon_adaptive/peri_ProcessedData_appleLeftEye.json"),
               ("t5ro.original.eye_right", "anon_adaptive/peri_ProcessedData_appleRightEye.json"),
               ("t5ro.simswap1.eye_left", "anon_adaptive/peri_ProcessedSwap_appleLeftEye.json"),
               ("t5ro.simswap1.eye_right", "anon_adaptive/peri_ProcessedSwap_appleRightEye.json"),
               ("t5ro.simswap2.eye_left", "anon_swap2/peri_ProcessedSwap2_appleLeftEye.json"),
               ("t5ro.simswap2.eye_right", "anon_swap2/peri_ProcessedSwap2_appleRightEye.json")]:
    if not (R / p).exists():
        put(key, None, p, status="pending"); continue
    d = js(p)
    put(f"{key}.TAR@FAR=0.001", d["direct_TAR@FAR=0.001"], p,
        dprime=d["direct_dprime"], rank1=d["arcface_direct_rank1"],
        informed_probe_rank1=d["informed_probe_rank1"], note="no CI: script stores no score matrix")

# ---- privacy: linkage ---------------------------------------------------------
bg = js("anon_stage3b/t1bg_p120.json")
put("floor.t1.ari_k18", bg["ari_k18"], "anon_stage3b/t1bg_p120.json", k_selected=bg["k_selected"])
for key, p in [("t1.original", "anon_stage3b/t1_none.json"),
               ("t1.blackbox", "anon_stage3b/t1_blackbox.json"),
               ("t1.blur0.50", "anon_stage3b/t1_blur0.50.json"),
               ("t1.simswap1", "anon_swap/t1_ProcessedSwap.json"),
               ("t1.simswap1.selfalign", "anon_swap/t1_ProcessedSwap_selfalign.json"),
               ("t1.simswap2", "anon_swap2/t1_ProcessedSwap2.json"),
               ("t1.simswap2.selfalign", "anon_swap2/t1_ProcessedSwap2_selfalign.json")]:
    if not (R / p).exists():
        put(key, None, p, status="pending"); continue
    d = js(p)
    put(f"{key}.ari_k18", d["ari_k18"], p, k_selected=d["k_selected"], n_frames=d["n_frames"])

# ---- privacy: adaptive attacker ----------------------------------------------
for p in sorted((R / "anon_adaptive").glob("t4_*.json")) + sorted((R / "anon_swap2").glob("t4_*.json")):
    d = json.load(open(p))
    for mp in ("none", "shift", "procrustes", "ridge", "ridge_insample"):
        if mp in d["maps"]:
            put(f"t4.{d['tag']}.{mp}.TAR@FAR=0.001", d["maps"][mp]["TAR@FAR=0.001"],
                str(p.relative_to(R)), dprime=d["maps"][mp]["dprime"], rank1=d["maps"][mp]["rank1"])

# ---- utility: non-inferiority --------------------------------------------------
pat = re.compile(r"\s+(blur[\d.]+|pixel[\d.]+|blackbox)\s+([+-][\d.]+)\s+\[([+-][\d.]+),([+-][\d.]+)\]\s+(\S+)\s+(\d+)/17\s+(\S+)\s+(\S+)\s+(\S+)")
for cond in ("ii", "iii"):
    for K in ("72", "9"):
        f = R / "plan_c" / f"planc_cond{cond}_K{K}.txt"
        bb = None
        for ln in open(f):
            m = re.match(r"^(\S+)\s+metric=\S+\s+control 'none' = ([\d.]+)", ln.strip())
            if m:
                bb = m.group(1)
                put(f"util.cond{cond}.K{K}.{bb}.none_cm", float(m.group(2)), str(f.relative_to(R)))
                continue
            m = re.search(r"ICC ([\d.]+)", ln)
            if m and bb:
                put(f"util.cond{cond}.K{K}.{bb}.icc", float(m.group(1)), str(f.relative_to(R)))
            m = pat.match(ln)
            if m and bb:
                op, d, lo, hi, dz, hurt, p, ph, v = m.groups()
                put(f"util.cond{cond}.K{K}.{bb}.{op}.penalty_cm", float(d), str(f.relative_to(R)),
                    ci95=[float(lo), float(hi)], dz=float(dz), hurt=f"{hurt}/17", p_holm=float(ph), verdict=v)
            m = re.search(r"bias ([+-][\d.]+) cm\s+LoA \[([+-][\d.]+), ([+-][\d.]+)\]", ln)
            if m and bb:
                put(f"util.cond{cond}.K{K}.{bb}.bland_altman_blackbox", float(m.group(1)), str(f.relative_to(R)),
                    loa=[float(m.group(2)), float(m.group(3))])
put("util.margin_cm", 0.40, "PLAN_C_RESULT.md (pre-declared from init-noise floor)")

# ---- write ---------------------------------------------------------------------
out = R / "paper"
json.dump(N, open(out / "numbers.json", "w"), indent=1)
with open(out / "NUMBERS.md", "w") as fh:
    fh.write("# Paper number registry\n\nGenerated by `scripts/paper/numbers.py`. Do not edit by hand.\n\n")
    fh.write("| key | value | 95% CI / extra | status | source |\n|---|---|---|---|---|\n")
    for k, v in N.items():
        val = v["value"]
        sval = "—" if val is None else (f"{val:.4g}" if isinstance(val, float) else str(val))
        extra = ", ".join(f"{a}={b}" for a, b in v.items() if a not in ("value", "source", "status"))
        fh.write(f"| `{k}` | {sval} | {extra} | {v.get('status','')} | {v['source']} |\n")
print(f"{len(N)} numbers -> {out/'numbers.json'}, {out/'NUMBERS.md'}")
pending = [k for k, v in N.items() if v.get("status")]
print("pending/missing:", pending or "none")
