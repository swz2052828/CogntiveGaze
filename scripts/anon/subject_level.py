"""Plan C, subject level: per-recording errors instead of per-fold means.

The Stage 4 CSVs record one row per fold, i.e. the mean over the 3-4 recordings
held out in that fold. That throws away the unit the design is actually paired
on. Recording-level 5-fold CV gives every recording exactly one held-out
measurement per (backbone, operator), so the natural analysis is paired WITHIN
SUBJECT across operators: n=17 subjects rather than n=5 folds, and the
between-subject variance -- which is large, subjects differ by several cm --
cancels out of the difference instead of inflating its interval.

The per-recording numbers were already being written to the run logs
("Fold 0 rec=13 K=72(calib) base=... fc_ft=..."); nothing needs recomputing.

  python scripts/anon/subject_level.py --out results/anon_stage4_subject.csv
"""
import argparse
import csv
import re
from pathlib import Path

RUNS = Path("/springbrook/share/eng/esrpxk/runs/anon_stage4/logs")
# "Fold 0 rec=13 K=72(calib) base=5.1124 svr=6.0092 svr_embed=4.4528 fc_ft=3.3050 meta=2.6151"
LINE = re.compile(
    r"Fold (?P<fold>\d+) rec=(?P<rec>\d+) K=(?P<k>\d+)\((?P<mode>\w+)\) (?P<rest>.*)")
KV = re.compile(r"(\w+)=([0-9.]+|nan)")
NAME = re.compile(
    r"s4(?P<cond>c3)?_(?P<bb>.+?)_(?P<op>none|blur[\d.]+|pixel[\d.]+|blackbox)_f(?P<fold>\d)\.log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(RUNS), nargs="+",
                    help="one or more log directories; cond ii/iii inferred from filename")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows, seen = [], set()
    logdirs = args.logs if isinstance(args.logs, list) else [args.logs]
    for f in sorted(q for d in logdirs for q in Path(d).glob("*.log")):
        m = NAME.match(f.name)
        if not m:
            print(f"  (skipped, unparsed name: {f.name})")
            continue
        bb, op = m.group("bb"), m.group("op")
        cond = "iii" if m.group("cond") else "ii"
        for line in f.read_text(errors="replace").splitlines():
            lm = LINE.search(line)
            if not lm:
                continue
            rec, fold = int(lm.group("rec")), int(lm.group("fold"))
            # A log can be appended to across reruns; keep the LAST value for a
            # (backbone, operator, recording) and count how often that happens,
            # so a silently duplicated run cannot double-weight a subject.
            # The key MUST include K. Without it, feeding a K=9 log directory
            # alongside a K=72 one silently overwrites every K=72 row with its
            # K=9 counterpart -- same cond, backbone, operator and recording --
            # and the resulting file looks complete.
            kk = int(lm.group("k"))
            key = (cond, kk, bb, op, rec)
            vals = dict(KV.findall(lm.group("rest")))
            row = {"cond": cond, "backbone": bb, "operator": op, "rec": rec,
                   "fold": fold, "k": kk, "mode": lm.group("mode")}
            row.update({k: (float(v) if v != "nan" else "") for k, v in vals.items()})
            if key in seen:
                rows = [r for r in rows
                        if not (r["cond"] == cond and r["k"] == kk
                                and r["backbone"] == bb
                                and r["operator"] == op and r["rec"] == rec)]
            seen.add(key)
            rows.append(row)

    cols = ["cond", "backbone", "operator", "rec", "fold", "k", "mode",
            "base", "svr", "svr_embed", "fc_ft", "meta"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["cond"], r["backbone"],
                                            r["operator"], r["rec"])):
            w.writerow(r)
    conds = sorted({r["cond"] for r in rows})
    print(f"  conditions: {conds}")
    bbs = sorted({r["backbone"] for r in rows})
    ops = sorted({r["operator"] for r in rows})
    recs = sorted({r["rec"] for r in rows})
    print(f"{len(rows)} rows -> {out}")
    print(f"  backbones {len(bbs)}: {bbs}")
    print(f"  operators {len(ops)}")
    print(f"  recordings {len(recs)}: {recs}")
    print(f"  expected if complete: {len(bbs)*len(ops)*len(recs)}")


if __name__ == "__main__":
    main()
