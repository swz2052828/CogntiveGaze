"""How much of the released face crop is byte-exact original pixels?

This is the number that explains the Stage 3b result. Every operator in the bank
acts on the face region but must leave the two eye-crop boxes untouched (§4, the
utility constraint). Those boxes sit *inside* the released face crop, so a fixed
fraction of the released artefact is original pixels no matter how aggressive the
operator is -- and it is the most identity-dense fraction. An operator's privacy
is therefore bounded by this share before it does anything at all.

Geometry is subject-constant (§2.2), so the fraction is a per-subject constant;
the per-frame spread is reported anyway as a check on that claim.
"""
import argparse
import json
from pathlib import Path

import numpy as np

DATA = Path("/springbrook/share/eng/esrpxk/datasets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom-root", default=str(DATA / "anon_geometry"))
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    rows = []
    for f in sorted(Path(args.geom_root).glob("*.npz")):
        g = np.load(f)
        fa, le, ri = g["appleFace"], g["appleLeftEye"], g["appleRightEye"]
        fw, fh = fa[:, 2], fa[:, 3]
        # Only the part of an eye box that falls inside the face box is actually
        # present in the released face crop, so intersect rather than assume.
        prot = np.zeros(len(fa))
        for eye in (le, ri):
            ix = np.clip(np.minimum(fa[:, 0] + fw, eye[:, 0] + eye[:, 2]) -
                         np.maximum(fa[:, 0], eye[:, 0]), 0, None)
            iy = np.clip(np.minimum(fa[:, 1] + fh, eye[:, 1] + eye[:, 3]) -
                         np.maximum(fa[:, 1], eye[:, 1]), 0, None)
            prot += ix * iy
        pct = 100.0 * prot / (fw * fh)
        iod = np.abs((le[:, 0] + le[:, 2] / 2.0) - (ri[:, 0] + ri[:, 2] / 2.0))
        rows.append({"rec": f.stem, "n_frames": int(len(fa)),
                     "face_w": float(fw[0]), "eye_w": float(le[0, 2]),
                     "iod_px": float(iod.mean()),
                     "protected_pct_mean": float(pct.mean()),
                     "protected_pct_sd": float(pct.std())})

    pcts = np.array([r["protected_pct_mean"] for r in rows])
    summary = {"n_subjects": len(rows),
               "protected_pct_mean": float(pcts.mean()),
               "protected_pct_min": float(pcts.min()),
               "protected_pct_max": float(pcts.max()),
               "max_within_subject_sd": float(max(r["protected_pct_sd"] for r in rows)),
               "per_subject": rows}

    print(f"{'rec':8s} {'faceW':>6s} {'eyeW':>5s} {'IOD':>6s} {'prot%':>7s} {'sd':>6s}")
    for r in rows:
        print(f"{r['rec']:8s} {r['face_w']:6.0f} {r['eye_w']:5.0f} {r['iod_px']:6.1f} "
              f"{r['protected_pct_mean']:7.1f} {r['protected_pct_sd']:6.3f}")
    print(f"\nbyte-exact share of the released face crop: "
          f"mean {summary['protected_pct_mean']:.1f}%, "
          f"range {summary['protected_pct_min']:.1f}-{summary['protected_pct_max']:.1f}%")
    print(f"max within-subject sd: {summary['max_within_subject_sd']:.3f} pp "
          f"(0 confirms the geometry is subject-constant, §2.2)")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(out, "w"), indent=2)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
