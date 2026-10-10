"""Does privaudit reproduce the CognitiveGaze study's numbers? Run after the four
audits of run_privaudit_validate.sbatch.

Eye-stream verification and linkage use exactly the study's sampling and
computation, so they must match to floating-point precision; anything else means
the package and the paper disagree and neither should be trusted until resolved.
The face stream is NOT compared: the study's face-crop linkage (T1) detects and
aligns on the released frame with a separate frame sample; privaudit's default
direct embedding is a different attack and is reported, not checked.

    python -m privaudit.validate_cognitivegaze <audit_root>
"""
import json
import sys
from pathlib import Path

R = Path("/springbrook/share/eng/esrpxk/results")
TOL = 1e-6

CASES = {
    # audit dir      peri ref (TAR)                         ARI ref                           photometric ref
    "ProcessedData": ("anon_adaptive/peri_ProcessedData_{e}.json",
                      "anon_ari_ci/ari_original_{e}.json",
                      "anon_session_confound/illum_ProcessedData_{e}.json"),
    "ProcessedSwap": ("anon_adaptive/peri_ProcessedSwap_{e}.json",
                      "anon_ari_ci/ari_swap_t1_{e}.json",
                      "anon_session_confound/illum_ProcessedSwap_{e}.json"),
    "ProcessedDP2t1_per_subject": ("anon_dp2/peri_dp2t1per_subject_{e}.json",
                                   "anon_ari_ci/ari_dp2_persubject_{e}.json", None),
    "ProcessedDP2fb_per_frame": ("anon_dp2/fullbody/peri_dp2fbper_frame_{e}.json",
                                 "anon_dp2/fullbody/ari_dp2fb_per_frame_{e}.json", None),
}


# Embeddings the study saved, for the computation check (same frames, same order).
EMB = {"ProcessedData": "anon_adaptive/emb_ProcessedData_{e}.npz",
       "ProcessedSwap": "anon_adaptive/emb_ProcessedSwap_{e}.npz",
       "ProcessedDP2t1_per_subject": "anon_dp2/emb_dp2t1per_subject_{e}.npz",
       "ProcessedDP2fb_per_frame": "anon_dp2/fullbody/emb_dp2fbper_frame_{e}.npz"}


def computation_check():
    """Layer 1: privaudit's statistics on the study's OWN saved embeddings. This
    isolates the computation from recogniser inference, so it must be exact."""
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from privaudit import attacks
    rows, bad = [], 0
    for case, (peri, ari, _) in CASES.items():
        for e in ("appleLeftEye", "appleRightEye"):
            z = np.load(R / EMB[case].format(e=e), allow_pickle=True)
            X1, y1, X2, y2 = (z[k] for k in ("X1", "y1", "X2", "y2"))
            v = attacks.verification(X1, y1, X2, y2, n_boot=10)
            lk = attacks.linkage(X1, y1, X2, y2, n_boot=1000, n_perm=1)
            p = json.load(open(R / peri.format(e=e)))
            b = json.load(open(R / ari.format(e=e)))
            for what, got, ref in (("TAR@1e-3", v["TAR@FAR=0.001"], p["direct_TAR@FAR=0.001"]),
                                   ("ARI point", lk["ari_known_k"], b["ari_point"]),
                                   ("ARI CI lo", lk["ari_known_k_ci95"][0], b["ari_boot_ci95"][0]),
                                   ("ARI CI hi", lk["ari_known_k_ci95"][1], b["ari_boot_ci95"][1])):
                ok = abs(got - ref) <= TOL
                bad += 0 if ok else 1
                rows.append((case, f"{e} {what}", got, ref, "ok" if ok else "MISMATCH"))
    return rows, bad


def main(audit_root):
    audit_root = Path(audit_root)
    rows, bad = [], 0

    # Layer 2 (end to end): privaudit re-embeds the crops. GPU inference is not
    # bit-deterministic across runs (embeddings differ by ~1e-4), which can move a
    # boundary frame across a cluster or threshold. Allowed: ARI within 0.015, TAR
    # within 0.0015 (one probe in 2160 is 0.00046), photometric exact (no model).
    E2E = {"ARI": 0.015, "TAR": 0.0015, "rank-1": 0.0015}

    def check(case, what, got, ref, exact=True):
        nonlocal bad
        tol = 0.02 if not exact else next((t for k, t in E2E.items() if k in what), TOL)
        ok = got is not None and ref is not None and abs(got - ref) <= tol
        bad += 0 if ok else 1
        rows.append((case, what, got, ref, "ok" if ok else "MISMATCH"))

    for case, (peri, ari, ph) in CASES.items():
        a = json.load(open(audit_root / case / "audit.json"))
        for e in ("appleLeftEye", "appleRightEye"):
            s = a["streams"][e]
            p = json.load(open(R / peri.format(e=e)))
            check(case, f"{e} A2 TAR@1e-3", s["verification"]["TAR@FAR=0.001"],
                  p["direct_TAR@FAR=0.001"])
            check(case, f"{e} A2 rank-1", s["verification"]["rank1"], p["arcface_direct_rank1"])
            b = json.load(open(R / ari.format(e=e)))
            check(case, f"{e} ARI point", s["linkage"]["ari_known_k"], b["ari_point"])
            check(case, f"{e} ARI CI lo", s["linkage"]["ari_known_k_ci95"][0], b["ari_boot_ci95"][0])
            check(case, f"{e} ARI CI hi", s["linkage"]["ari_known_k_ci95"][1], b["ari_boot_ci95"][1])
            if ph:
                q = json.load(open(R / ph.format(e=e)))
                check(case, f"{e} photometric ARI", s["photometric"]["ari_known_k"],
                      q["ari_known_k"])
        if case == "ProcessedData":
            f = json.load(open(R / "anon_peri_controls/peri_floor_arcface_appleLeftEye.json"))
            check(case, "floor TAR@1e-3", a["floor"]["verification"]["TAR@FAR=0.001"],
                  f["direct_TAR@FAR=0.001"])
            # The study's crop-geometry table (IMWUT_ANON_PROTOCOL.md) gives 66.7%
            # for face + eye crop size; privaudit scores frame-level, so close, not exact.
            check(case, "geometry expected rank-1 (face+eyes)", a["geometry"]["expected_rank1"],
                  0.667, exact=False)

    print("== Layer 2: end to end (privaudit re-embeds the crops) ==")
    w = max(len(r[1]) for r in rows)
    for c, what, got, ref, st in rows:
        g = "None" if got is None else f"{got:.6f}"
        print(f"{c:28s} {what:{w}s}  got {g}  ref {ref:.6f}  {st}")
    print(f"\n{len(rows) - bad}/{len(rows)} reproduced"
          + ("" if not bad else f"  -- {bad} MISMATCH(ES)"))
    print("\n== Layer 1: computation on the study's saved embeddings (must be exact) ==")
    rows1, bad1 = computation_check()
    for c, what, got, ref, st in rows1:
        print(f"{c:28s} {what:{w}s}  got {got:.6f}  ref {ref:.6f}  {st}")
    print(f"\n{len(rows1) - bad1}/{len(rows1)} exact")
    return 1 if (bad or bad1) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
