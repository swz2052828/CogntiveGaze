"""Diagnostic root: hold appleFace at the P0 (blackbox) arm, vary only the eye crops.

Why this exists. The P-axis operators grey the periocular surround wherever it
appears. The eye boxes lie INSIDE the face box (verified: face [317,163,320,320],
left eye [477,183,120,120]), so re-cutting appleFace from the masked frame stamps
the same grey into the released face crop. That is correct for privacy -- you
cannot leave the periocular pixels in appleFace and claim to have removed them --
but it means the measured P1 utility cost is a SUM of two channels:

  (a) the eye stream loses periocular context, and
  (b) the face stream loses the eye windows that were its only content under
      blackbox.

It also means face_only_mobile_vit is NOT a valid falsification arm for this
operator: it reads appleFace, which the operator legitimately modifies, and it
duly degraded +4.56 cm.

This root isolates (a). appleFace is hard-linked from the blackbox arm, so it is
byte-identical across P0/P1/P2 and the face stream contributes nothing to the
difference; the eye folders are hard-linked from the P-axis arm. Hard links, so
the cost is directory entries, not data blocks -- we are inode-constrained.

It is a DIAGNOSTIC, not a deployable configuration: its appleFace still carries
the original eye pixels and therefore still leaks the periocular identity that
the P-axis exists to remove. Never report it as a privacy-utility operating point.
"""
import argparse, os
from pathlib import Path

D = Path("/springbrook/share/eng/esrpxk/datasets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True, help="P1 / P2")
    ap.add_argument("--face-arm", default=str(D / "anon" / "blackbox"))
    ap.add_argument("--rec", required=True)
    args = ap.parse_args()

    face_src = Path(args.face_arm) / args.rec
    eye_src = D / "anon_paxis" / args.level / args.rec
    dst = D / "anon_paxis" / f"{args.level}_facefixed" / args.rec

    n = {}
    for fold, src in (("appleFace", face_src), ("appleLeftEye", eye_src),
                      ("appleRightEye", eye_src)):
        (dst / fold).mkdir(parents=True, exist_ok=True)
        c = 0
        # The eye arm is the subsampled one (1200/rec); it defines the frame set,
        # so link the face crops for exactly those frames and no others. Linking
        # all 11,718 face crops would make the root inconsistent across folders.
        names = sorted(p.name for p in (eye_src / "appleFace").glob("*.jpg"))
        for name in names:
            s, t = src / fold / name, dst / fold / name
            if not s.exists():
                continue
            if t.exists():
                t.unlink()
            os.link(s, t)
            c += 1
        n[fold] = c
    print(f"[{args.rec}] {args.level}_facefixed {n}")


if __name__ == "__main__":
    main()
