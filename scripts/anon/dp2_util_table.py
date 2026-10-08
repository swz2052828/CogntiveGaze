"""DP2 rows for Table 3, with a harness gate first.

Gate: the `none` arm re-run inside the DP2 eval job must reproduce, fold by
fold, the `base` of the `none` CSV that Table 3 was built from for the same
draw. If it does not, the checkpoints or the harness differ and no DP2 number
is written.

Then, per arm, the `base` column (no per-participant calibration, as Table 3):
mean over the 5 folds within a draw, then mean +- SD over the 5 draws, and the
penalty against that draw's own `none`. SimSwap arms are read from the existing
CSVs so both families sit in one table on one harness.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path("/springbrook/share/eng/esrpxk/runs")
BBS = ["itracker", "mobilenet_v3", "affnet", "mgazenet"]
DP2_ARMS = ["dp2s", "dp2s_oldeye", "dp2f", "dp2f_oldeye"]
SWAP_ARMS = ["swap1", "swap1_oldeye", "swap2", "swap2_oldeye"]


def old_csv(draw, bb, arm):
    if draw == 0:
        return RUNS / "anon_swap_utility" / f"su_{bb}_{arm}.csv"
    return RUNS / "initvar_utility" / f"iv_rep{draw}_{bb}_{arm}.csv"


def base_by_fold(path):
    d = pd.read_csv(path).drop_duplicates("fold", keep="last").set_index("fold")
    if sorted(d.index) != [0, 1, 2, 3, 4]:
        raise SystemExit(f"{path}: folds {sorted(d.index)}, expected 0-4")
    return d["base"].sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.01, help="cm, per fold")
    ap.add_argument("--json-out", default="/springbrook/share/eng/esrpxk/results/anon_dp2/util_table.json")
    args = ap.parse_args()

    # ---------------------------------------------------------------- gate
    worst = 0.0
    for draw in range(5):
        for bb in BBS:
            new = base_by_fold(RUNS / "dp2_utility" / f"dp2_d{draw}_{bb}_none.csv")
            old = base_by_fold(old_csv(draw, bb, "none"))
            gap = float((new - old).abs().max())
            worst = max(worst, gap)
            if gap > args.tol:
                raise SystemExit(f"HARNESS GATE FAILED draw{draw} {bb}: none differs from "
                                 f"Table 3's by up to {gap:.4f} cm. Not writing DP2 rows.")
    print(f"harness gate PASS: none reproduces Table 3 to within {worst:.4f} cm on all 100 folds")

    # ---------------------------------------------------------------- table
    out = {}
    for bb in BBS:
        per_draw = {a: [] for a in ["none"] + SWAP_ARMS + DP2_ARMS}
        for draw in range(5):
            per_draw["none"].append(base_by_fold(old_csv(draw, bb, "none")).mean())
            for a in SWAP_ARMS:
                per_draw[a].append(base_by_fold(old_csv(draw, bb, a)).mean())
            for a in DP2_ARMS:
                per_draw[a].append(base_by_fold(RUNS / "dp2_utility" / f"dp2_d{draw}_{bb}_{a}.csv").mean())
        none = np.array(per_draw["none"])
        row = {}
        for a, v in per_draw.items():
            v = np.array(v)
            pen = v - none
            row[a] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                          penalty=float(pen.mean()), penalty_sd=float(pen.std(ddof=1)),
                          draws=[float(x) for x in v])
        # share of the full-replacement penalty that restoring original eyes recovers
        for full, hyb in (("swap1", "swap1_oldeye"), ("swap2", "swap2_oldeye"),
                          ("dp2s", "dp2s_oldeye"), ("dp2f", "dp2f_oldeye")):
            row[hyb]["recovered_frac"] = 1 - row[hyb]["penalty"] / row[full]["penalty"]
        out[bb] = row

    cols = ["none", "swap1", "swap1_oldeye", "dp2s", "dp2s_oldeye", "dp2f", "dp2f_oldeye"]
    print("\nbase error, cm (mean over 5 draws; penalty vs none in brackets)")
    print(f"{'model':14s}" + "".join(f"{c:>18s}" for c in cols))
    for bb, row in out.items():
        cells = [f"{row['none']['mean']:.2f}±{row['none']['sd']:.2f}".rjust(18)]
        for c in cols[1:]:
            cells.append(f"{row[c]['mean']:.2f} ({row[c]['penalty']:+.2f})".rjust(18))
        print(f"{bb:14s}" + "".join(cells))
    print("\nrecovered fraction by restoring original eyes:")
    for bb, row in out.items():
        print(f"  {bb:14s} " + "  ".join(
            f"{h}={row[h]['recovered_frac']:.2f}" for h in
            ("swap1_oldeye", "swap2_oldeye", "dp2s_oldeye", "dp2f_oldeye")))

    p = Path(args.json_out); p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(gate_max_abs_cm=worst, table=out), open(p, "w"), indent=2)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
