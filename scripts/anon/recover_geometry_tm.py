"""Recover exact crop geometry by template-matching stored crops in the frame.

ProcessedData crops are axis-aligned squares cut at NATIVE resolution (face
300-350 px, eyes 110-120 px, varying per subject), not detector-generated fixed
size images. So the geometry is recoverable exactly rather than approximately:
locate each stored crop inside its OriginalData frame by normalised
cross-correlation. Stage-0 measurement: corr 0.9993, MAE ~1.0 (JPEG re-encode).

Temporal locality is exploited -- the search window is centred on the previous
frame's hit and only widens to a full-frame search when correlation drops.

  python scripts/anon/recover_geometry_tm.py --rec 00006 --limit 300
  python scripts/anon/recover_geometry_tm.py --rec 00006 --out geom/00006.npz
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

DATA = Path("/springbrook/share/eng/esrpxk/datasets")
FOLDERS = ("appleFace", "appleLeftEye", "appleRightEye")


def match_scaled(frame, tmpl, scales, prev_scale=None):
    """Return (x, y, w, h, scale, corr) for a template stored at a DIFFERENT size
    than it occupies in the frame.

    Task crops are stored at native size, so plain matchTemplate is exact. The
    CALIBRATION support crops are stored resized to 224x224 while occupying
    ~300-400 px (face) and ~110-130 px (eyes) in the frame, so the scale has to be
    searched. `prev_scale` narrows the search once a recording's scale is known --
    it is near-constant within a subject, as it is for the task crops.
    """
    order = scales
    if prev_scale is not None:
        order = sorted(scales, key=lambda s: abs(s - prev_scale))
    best = (None, None, None, None, None, -1.0)
    for sc in order:
        h = int(round(tmpl.shape[0] * sc)); w = int(round(tmpl.shape[1] * sc))
        if h < 8 or w < 8 or h > frame.shape[0] or w > frame.shape[1]:
            continue
        tt = cv2.resize(tmpl, (w, h), interpolation=cv2.INTER_AREA)
        r = cv2.matchTemplate(frame, tt, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        if mx > best[5]:
            best = (loc[0], loc[1], w, h, sc, float(mx))
        if prev_scale is not None and mx >= 0.985:
            break            # scale is stable within a subject; stop early
    return best


def match(frame, tmpl, prev, pad):
    """Return (x, y, corr). Windowed search around `prev`, else full frame."""
    th, tw = tmpl.shape[:2]
    fh, fw = frame.shape[:2]
    if prev is not None:
        x0 = max(0, prev[0] - pad); y0 = max(0, prev[1] - pad)
        x1 = min(fw, prev[0] + tw + pad); y1 = min(fh, prev[1] + th + pad)
        sub = frame[y0:y1, x0:x1]
        if sub.shape[0] >= th and sub.shape[1] >= tw:
            r = cv2.matchTemplate(sub, tmpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx >= 0.95:
                return x0 + loc[0], y0 + loc[1], float(mx)
    r = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(r)
    return loc[0], loc[1], float(mx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pad", type=int, default=48, help="windowed search margin (px)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    odir = DATA / "OriginalData" / args.rec
    pdir = DATA / "ProcessedData" / args.rec
    names = sorted(p.name for p in (pdir / "appleFace").glob("*.jpg"))
    if args.limit:
        names = names[: args.limit]

    prev = {f: None for f in FOLDERS}
    rows, corrs, full_searches = [], {f: [] for f in FOLDERS}, 0
    t0 = time.time()
    for i, name in enumerate(names):
        frame = cv2.imread(str(odir / name))
        if frame is None:
            continue
        row = {"frame": name}
        ok = True
        for f in FOLDERS:
            t = cv2.imread(str(pdir / f / name))
            if t is None:
                ok = False
                break
            was = prev[f]
            x, y, c = match(frame, t, prev[f], args.pad)
            if was is None or c < 0.95:
                full_searches += 1
            prev[f] = (x, y)
            corrs[f].append(c)
            row[f] = (x, y, t.shape[1], t.shape[0], round(c, 5))
        if ok:
            rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(names)}  {(time.time()-t0)/(i+1)*1000:.1f} ms/frame", flush=True)

    dt = time.time() - t0
    print(f"[{args.rec}] {len(rows)} frames in {dt:.1f}s "
          f"({dt/max(1,len(rows))*1000:.1f} ms/frame), full searches {full_searches}")
    for f in FOLDERS:
        c = np.asarray(corrs[f])
        if len(c):
            print(f"  {f:14s} corr min {c.min():.4f} mean {c.mean():.4f} "
                  f"| <0.99: {(c < 0.99).sum()} frames")

    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            frames=np.array([r["frame"] for r in rows]),
            **{f: np.array([r[f] for r in rows], dtype=np.float64) for f in FOLDERS})
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
