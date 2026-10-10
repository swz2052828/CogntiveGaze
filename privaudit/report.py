"""Map attack results onto the release checklist (CHECKLIST.md) and render.

Decision rule, everywhere: a channel LEAKS when the lower bound of its 95%
participant-bootstrap interval lies above the floor. The floor is the same attack
run on a face-free patch of the same recordings (session nuisance: lighting,
seating, camera); without one, verdicts fall back to a statistical null and are
marked as such, because on single-session data the null understates the floor.
"""
import json

PASS, FAIL, WARN, NA = "PASS", "FAIL", "WARN", "NOT RUN"


def _ci_lo(d, key):
    ci = d.get(key)
    return None if ci is None else ci[0]


def _tar_verdict(v, f_tar, have_floor, action):
    ref = f_tar if have_floor else 0.001
    ci, pt = v.get("TAR@FAR=0.001_ci95"), v["TAR@FAR=0.001"]
    tag = "floor" if have_floor else "FAR"
    if ci is None:   # interval refused: fall back to the point, and say so
        leak = pt > 2 * ref
        return (FAIL if leak else WARN,
                f"TAR@FAR=1e-3 {pt:.2%} (interval invalid; point vs 2x {tag} {2 * ref:.2%})",
                action if leak else "too few participants for an interval")
    leak = ci[0] > ref
    return (FAIL if leak else PASS,
            f"TAR@FAR=1e-3 {pt:.2%} [{ci[0]:.2%}, {ci[1]:.2%}] vs {tag} {ref:.2%}",
            action if leak else "")


def verdicts(r):
    floor = r.get("floor") or {}
    f_tar = (floor.get("verification") or {}).get("TAR@FAR=0.001")
    f_ari = (floor.get("linkage") or {}).get("ari_known_k")
    have_floor = f_tar is not None and f_ari is not None
    out = []

    def add(item, stream, status, evidence, action=""):
        out.append({"item": item, "stream": stream, "status": status,
                    "evidence": evidence, "action": action})

    add("R0 session floor", "-", PASS if have_floor else WARN,
        (f"TAR@FAR=1e-3 {f_tar:.2%}, ARI {f_ari:.3f}" if have_floor else
         "no face-free floor supplied (--floor-frames); verdicts compare against a "
         "statistical null and will overstate leakage on single-session data"),
        "" if have_floor else "supply full frames so a face-free floor can be measured")

    for s, d in r["streams"].items():
        lk, vf = d.get("linkage"), d.get("verification")
        if lk:
            lo = _ci_lo(lk, "ari_known_k_ci95")
            ref = f_ari if have_floor else lk["perm_null_ari_max"]
            if lo is None:
                add("R2 release-only linkage", s, WARN,
                    f"ARI {lk['ari_known_k']:.3f}; bootstrap interval invalid", "inspect")
            else:
                leak = lo > ref
                add("R2 release-only linkage", s, FAIL if leak else PASS,
                    f"ARI {lk['ari_known_k']:.3f} [{lo:.3f}, {lk['ari_known_k_ci95'][1]:.3f}] "
                    f"vs {'floor' if have_floor else 'perm. null'} {ref:.3f}",
                    "resample the synthetic identity per session AND release nothing the "
                    "transform does not edit" if leak else "")
            # discoverable = an attacker who must guess the cohort size still
            # recovers more partition than the floor (k itself may be over-split)
            # and selects a cluster count of the right order (k=2 for 18 people is
            # not discovery, whatever its ARI)
            disc = (lk.get("ari_auto_k") is not None
                    and lk.get("k_selected", 0) >= 0.5 * lk["n_participants"]
                    and lk["ari_auto_k"] > (f_ari if have_floor else 0.1))
            add("R7 cohort discoverable without k", s, WARN if disc else PASS,
                f"k selected {lk.get('k_selected')} of {lk['n_participants']}, "
                f"ARI {lk.get('ari_auto_k', float('nan')):.3f}",
                "the partition is found without being told the cohort size" if disc else "")
        if vf:
            add("R3 release-only verification", s, *_tar_verdict(vf, f_tar, have_floor,
                "this stream identifies participants within the release"))
        a1 = d.get("enrolment")
        if a1:
            add("R4 enrolment re-identification", s, *_tar_verdict(a1, f_tar, have_floor,
                "photographs of a participant identify their released frames"))
        else:
            add("R4 enrolment re-identification", s, NA, "no --enrolment root")
        ph = d.get("photometric")
        if ph:
            ref = f_ari if have_floor else 0.1
            leak = ph["ari_known_k"] > ref
            add("R5 photometric channel", s, FAIL if leak else PASS,
                f"21 summary statistics, no recogniser: ARI {ph['ari_known_k']:.3f}",
                "normalise per-crop photometry (exposure, white balance) before release"
                if leak else "")
        nc = d.get("null_control")
        if nc:
            ok = nc["mean_abs_diff"] is not None and nc["mean_abs_diff"] >= 1.0
            add("R8 transform applied", s, PASS if ok else FAIL,
                f"mean |delta| vs original {nc['mean_abs_diff'] or 0:.1f} over {nc['n_pairs']} frames",
                "" if ok else "the release is (near-)identical to the original")

    g = r.get("geometry")
    if g:
        leak = g["expected_rank1"] is not None and g["expected_rank1"] > 2 * g["chance_rank1"]
        add("R6 crop-geometry metadata", "+".join(r["streams"]), FAIL if leak else PASS,
            f"expected Rank-1 {g['expected_rank1']:.1%} (chance {g['chance_rank1']:.1%}), "
            f"{g['identifying_bits']:.2f} of {g['max_bits']:.2f} bits, without pixels",
            "resize every released crop to one fixed size" if leak else "")
    return out


def overall(vs):
    hard = [v for v in vs if v["status"] == FAIL]
    return ("NOT RELEASABLE WITHOUT ACCESS CONTROL" if hard
            else "NO MEASURED LEAK ABOVE FLOOR (see WARN items)")


def markdown(r, vs):
    L = [f"# Privacy audit — `{r['release']}`", "",
         f"privaudit {r['version']} · recogniser {r['recogniser']} · "
         f"{r['n_per_half']}+{r['n_per_half']} frames per participant per stream", "",
         f"**Overall: {overall(vs)}**", "",
         "| Item | Stream | Status | Evidence | Action |", "|---|---|---|---|---|"]
    for v in vs:
        L.append(f"| {v['item']} | {v['stream']} | **{v['status']}** | {v['evidence']} | {v['action']} |")
    L += ["", "Decision rule: a channel leaks when the lower bound of its 95% "
          "participant-level bootstrap interval exceeds the face-free floor. "
          "Items not measured by this tool are in CHECKLIST.md.", ""]
    return "\n".join(L)


def write(r, out_dir):
    vs = verdicts(r)
    r["verdicts"], r["overall"] = vs, overall(vs)
    (out_dir / "audit.json").write_text(json.dumps(r, indent=2))
    (out_dir / "audit.md").write_text(markdown(r, vs))
    return vs
