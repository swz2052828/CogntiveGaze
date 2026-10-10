"""python -m privaudit --release ROOT --out DIR [options]

See README.md. Runs on CPU (--gpu -1) or GPU (--gpu 0); a full audit of an
18-participant release takes minutes on a GPU, most of it the bootstrap.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__, attacks, data, recognise, report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="privaudit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, help="candidate release, GazeCapture layout")
    ap.add_argument("--streams", nargs="+",
                    default=["appleFace", "appleLeftEye", "appleRightEye"])
    ap.add_argument("--align-detect", nargs="*", default=[],
                    help="streams to embed after detecting a face ON THE RELEASED "
                         "crop (release-only attacker; detection failures are "
                         "counted). Others are embedded directly at 112x112, which "
                         "is what a face detector cannot do on a 120 px eye crop.")
    ap.add_argument("--enrolment", help="root with enrolment imagery of the same "
                                         "participants (enables A1)")
    ap.add_argument("--original", help="unprocessed root, same layout (enables the "
                                       "null control R8)")
    ap.add_argument("--floor-frames", help="root of FULL frames, <pid>/<frame> with "
                                           "the same names as the crops (enables the "
                                           "session-nuisance floor R0)")
    ap.add_argument("--floor-patch", type=int, default=120,
                    help="top-left square of each full frame used for the floor; it "
                         "must contain no face")
    ap.add_argument("--floor-stream", help="whose frame sample the floor reuses "
                                           "(default: first stream not in --align-detect)")
    ap.add_argument("--recogniser", default="arcface", choices=["arcface", "facenet"])
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--n-per-half", type=int, default=60)
    ap.add_argument("--n-photometric", type=int, default=120)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["verification", "linkage", "photometric", "geometry",
                             "floor"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rec = recognise.build(a.recogniser, a.gpu)
    r = {"version": __version__, "release": a.release, "recogniser": a.recogniser,
         "n_per_half": a.n_per_half, "streams": {}, "argv": sys.argv[1:]}
    samples = {}

    for s in a.streams:
        print(f"[privaudit] stream {s}", flush=True)
        sample, skipped = data.half_sample(a.release, s, a.n_per_half)
        samples[s] = sample
        d = {"n_participants": len(sample), "skipped_participants": skipped}
        X1, y1, X2, y2, fail = attacks.embed_halves(a.release, s, rec, sample,
                                                    align_detect=s in a.align_detect)
        d["embedding"] = {"align_detect": s in a.align_detect, "failed_frames": fail}
        np.savez_compressed(out / f"emb_{s}.npz", X1=X1.astype(np.float32), y1=y1,
                            X2=X2.astype(np.float32), y2=y2)
        if "verification" not in a.skip:
            d["verification"] = attacks.verification(X1, y1, X2, y2, n_boot=a.n_boot)
        if "linkage" not in a.skip:
            d["linkage"] = attacks.linkage(X1, y1, X2, y2, n_boot=a.n_boot)
        if "photometric" not in a.skip:
            d["photometric"] = attacks.photometric(a.release, s, a.n_photometric)
        if a.enrolment:
            g, _ = data.half_sample(a.enrolment, s, a.n_per_half)
            G1, g1, _, _, _ = attacks.embed_halves(a.enrolment, s, rec, g,
                                                   align_detect=s in a.align_detect)
            d["enrolment"] = attacks.verification(G1, g1, X2, y2, n_boot=a.n_boot)
        if a.original:
            d["null_control"] = attacks.null_control(a.release, a.original, s, sample)
        r["streams"][s] = d
        (out / "audit.partial.json").write_text(json.dumps(r, indent=2, default=str))

    if "geometry" not in a.skip:
        r["geometry"] = attacks.geometry(a.release, a.streams, samples[a.streams[0]])

    if a.floor_frames and "floor" not in a.skip:
        fs = a.floor_stream or next((s for s in a.streams if s not in a.align_detect),
                                    a.streams[0])
        print(f"[privaudit] floor on {fs}'s frame sample, {a.floor_patch}px patch", flush=True)
        F1, f1, F2, f2, fail = attacks.embed_halves(
            None, fs, rec, samples[fs],
            loader=lambda pid, n: data.load_floor_patch(a.floor_frames, pid, n, a.floor_patch))
        r["floor"] = {"stream_sample": fs, "patch": a.floor_patch, "failed_frames": fail,
                      "verification": attacks.verification(F1, f1, F2, f2, n_boot=a.n_boot),
                      "linkage": attacks.linkage(F1, f1, F2, f2, n_boot=a.n_boot)}

    vs = report.write(r, out)
    print(report.markdown(r, vs))
    (out / "audit.partial.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
