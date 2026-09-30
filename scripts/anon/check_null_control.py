"""Null-control gate for a condition (iii) run.

On the `none` arm the operator is a no-op, so enrolment frames anonymised by
`none` must equal the original enrolment frames and condition (iii) must
reproduce condition (ii) to within the JPEG re-encode floor. Any real gap there
is a fault in the ENROLMENT path, not a result -- and it will otherwise be read
as "matched enrolment changes the number", which is exactly how the first
condition (iii) run (job 1272012) was misreported before the crop-convention
bug in build_anon_calib_support.py was found.

`base` is the sharper probe of the two: it never touches the calibration support
at all, so it must agree EXACTLY. `fc_ft` and `meta` do touch it, so they are
allowed the re-encode floor but no more. face_only_mobile_vit is the most
sensitive backbone to a face-crop convention error and is reported first.

Run this BEFORE reading any condition (iii) contrast.

  python scripts/anon/check_null_control.py
"""
import argparse
import csv
from pathlib import Path

import numpy as np

R = Path("/springbrook/share/eng/esrpxk/runs")
# face_only first: a face-crop convention error hits it hardest, which is what
# located the last bug.
BBS = ["face_only_mobile_vit", "mobile_vit", "convnextv2", "itracker"]


def fold_map(path, col):
    if not Path(path).is_file():
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        v = r.get(col, "")
        if v not in ("", "nan"):
            out[int(r["fold"])] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond2", default=str(R / "anon_stage4"))
    ap.add_argument("--cond3", default=str(R / "anon_stage4_cond3"))
    ap.add_argument("--tol-base", type=float, default=1e-3,
                    help="base never uses the support, so it must match exactly")
    ap.add_argument("--tol-calib", type=float, default=0.05,
                    help="calibrated columns may differ by the re-encode floor only")
    args = ap.parse_args()

    verdict = True
    print(f"{'backbone':<22}{'column':<11}{'cond(ii)':>10}{'cond(iii)':>11}"
          f"{'diff':>9}{'tol':>8}  status")
    print("-" * 78)
    for bb in BBS:
        a2 = Path(args.cond2) / f"s4_{bb}_none.csv"
        a3 = Path(args.cond3) / f"s4c3_{bb}_none.csv"
        for col in ("base", "fc_ft", "meta"):
            m2, m3 = fold_map(a2, col), fold_map(a3, col)
            common = sorted(set(m2) & set(m3))
            if not common:
                print(f"{bb:<22}{col:<11}{'--':>10}{'--':>11}{'--':>9}{'--':>8}  NO DATA")
                continue
            v2 = np.mean([m2[f] for f in common])
            v3 = np.mean([m3[f] for f in common])
            d = abs(v3 - v2)
            tol = args.tol_base if col == "base" else args.tol_calib
            ok = d <= tol
            verdict &= ok
            print(f"{bb:<22}{col:<11}{v2:>10.4f}{v3:>11.4f}{d:>9.4f}{tol:>8.3f}"
                  f"  {'ok' if ok else 'FAIL'}   (n={len(common)})")

    print()
    if verdict:
        print("NULL CONTROL PASSED -- condition (iii) may be read.")
    else:
        print("NULL CONTROL FAILED -- the enrolment path differs from condition (ii)")
        print("on an arm where the operator does nothing. Do NOT report any")
        print("condition (iii) contrast until this is explained.")
    raise SystemExit(0 if verdict else 1)


if __name__ == "__main__":
    main()
