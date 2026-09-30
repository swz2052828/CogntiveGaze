"""Subject-level bootstrap confidence intervals for the identification metrics.

Every privacy number in the protocol so far is a point estimate. The probes are
~60 frames per subject, and frames of one subject are strongly correlated (same
session, same lighting, same face), so treating the ~1,080 probes as independent
would make any interval absurdly narrow. The resampling unit is therefore the
PROBE SUBJECT: each replicate draws 18 subjects with replacement and takes all of
their probe rows. The gallery is held fixed -- the attacker's enrolment set is not
a random quantity in the threat model -- which also avoids duplicate gallery
columns that would break Rank-1.

Metric definitions are copied from id_attack.py, and the script refuses to
report a CI whose point estimate does not reproduce the stored JSON: an interval
around a differently-computed number is worse than no interval.

  python scripts/paper/privacy_ci.py --out results/paper/privacy_ci.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

R = Path("/springbrook/share/eng/esrpxk/results")


def metrics(S, probe, gallery):
    gallery = np.asarray(gallery)
    labels = (gallery[None, :] == probe[:, None]).astype(np.int32)
    rank1 = float((gallery[S.argmax(1)] == probe).mean())
    sc, lb = S.ravel().astype(np.float64), labels.ravel()
    gen, imp = sc[lb == 1], sc[lb == 0]
    dp = float((gen.mean() - imp.mean()) / np.sqrt(0.5 * (gen.var() + imp.var()) + 1e-12))
    thr = np.quantile(imp, 1.0 - 1e-3)
    tar3 = float((gen >= thr).mean())
    return {"dprime": dp, "rank1": rank1, "TAR@FAR=0.001": tar3}


def boot(S, probe, gallery, B, rng):
    subs = np.unique(probe)
    rows = {s: np.where(probe == s)[0] for s in subs}
    out = {k: [] for k in ("dprime", "rank1", "TAR@FAR=0.001")}
    for _ in range(B):
        pick = rng.choice(subs, size=len(subs), replace=True)
        idx = np.concatenate([rows[s] for s in pick])
        m = metrics(S[idx], probe[idx], gallery)
        for k in out:
            out[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    files = sorted(list((R / "anon_stage3b").glob("t2_*.scores.npz"))
                   + list((R / "anon_swap").glob("*.scores.npz"))
                   + list((R / "anon_swap2").glob("*.scores.npz")))
    rows, bad = [], []
    for f in files:
        d = np.load(f)
        S, probe, gallery = d["S"], d["probe_subject"], d["gallery_subject"]
        m = metrics(S, probe, gallery)
        js = Path(str(f).replace(".scores.npz", ".json"))
        ref = json.load(open(js)) if js.exists() else {}
        for k, rk in (("dprime", "dprime"), ("rank1", "rank1"), ("TAR@FAR=0.001", "TAR@FAR=0.001")):
            if rk in ref and abs(ref[rk] - m[k]) > 1e-6:
                bad.append(f"{f.name}: {k} recomputed {m[k]:.6f} vs json {ref[rk]:.6f}")
        ci = boot(S, probe, gallery, args.boot, rng)
        row = {"source": str(f.relative_to(R)), "n_probe": len(probe),
               "n_subjects": len(np.unique(probe))}
        for k in ("dprime", "rank1", "TAR@FAR=0.001"):
            row[k] = round(m[k], 4)
            row[k + "_lo"] = round(ci[k][0], 4)
            row[k + "_hi"] = round(ci[k][1], 4)
        rows.append(row)
        print(f"{f.name:48} d'={m['dprime']:.2f} [{ci['dprime'][0]:.2f},{ci['dprime'][1]:.2f}]  "
              f"TAR@1e-3={m['TAR@FAR=0.001']*100:5.1f}% [{ci['TAR@FAR=0.001'][0]*100:.1f},{ci['TAR@FAR=0.001'][1]*100:.1f}]",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")
    if bad:
        print("\nREPRODUCTION FAILURES -- these intervals are NOT around the published numbers:")
        for b in bad:
            print("  ", b)
        raise SystemExit(1)
    print("all point estimates reproduce the stored JSON")


if __name__ == "__main__":
    main()
