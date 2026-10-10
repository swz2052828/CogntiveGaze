"""Table 3 with three columns per de-identified arm (phase 1: three fast backbones).

  (ii)        clean-trained, de-identified test        existing 5-draw CSVs
  (i)-real    trained on de-identified, real test      runs/anon_train/eval
  (i)-anon    trained and tested on de-identified      runs/anon_train/eval

Baseline for every column is the 5-draw clean model on real data, whose SD is
the init noise floor. (i) is a single initialisation draw, so a difference
between (i) and baseline smaller than ~2x that SD is not a finding.
Refuses to write anything if a CSV lacks any of folds 0-4.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path("/springbrook/share/eng/esrpxk/runs")
BBS = ["mobilenet_v3", "affnet", "mgazenet"]
PAIRS = [("swap1", "swap1_oldeye"), ("swap2", "swap2_oldeye"),
         ("dp2s", "dp2s_oldeye"), ("dp2f", "dp2f_oldeye"),
         ("dp2fbs", "dp2fbs_oldeye"), ("dp2fbf", "dp2fbf_oldeye")]


def fold_mean(path):
    d = pd.read_csv(path).drop_duplicates("fold", keep="last")
    if sorted(d["fold"]) != [0, 1, 2, 3, 4]:
        raise SystemExit(f"{path}: folds {sorted(d['fold'])}, expected 0-4")
    return float(d["base"].mean())


def cond2(bb, arm, draw):
    if arm.startswith("dp2"):
        return RUNS / "dp2_utility" / f"dp2_d{draw}_{bb}_{arm}.csv"
    if draw == 0:
        return RUNS / "anon_swap_utility" / f"su_{bb}_{arm}.csv"
    return RUNS / "initvar_utility" / f"iv_rep{draw}_{bb}_{arm}.csv"


def base0(bb, draw):
    return cond2(bb, "none", draw) if draw else RUNS / "anon_swap_utility" / f"su_{bb}_none.csv"


def main():
    out = {}
    for bb in BBS:
        clean = np.array([fold_mean(base0(bb, d)) for d in range(5)])
        row = {"baseline": dict(mean=float(clean.mean()), sd=float(clean.std(ddof=1)))}
        for full, hyb in PAIRS:
            for arm in (full, hyb):
                c = {}
                try:
                    c["ii"] = float(np.mean([fold_mean(cond2(bb, arm, d)) for d in range(5)]))
                except FileNotFoundError:
                    c["ii"] = None
                draws = {}
                for t in ("real", "anon"):
                    # draw 1 = original run; draws 2-5 carry a _rep<N> suffix.
                    # Only complete draws (all five folds) are used.
                    vals = []
                    for rep in range(1, 6):
                        sfx = "" if rep == 1 else f"_rep{rep}"
                        p = RUNS / "anon_train" / "eval" / f"at_{arm}_{bb}_test{t}{sfx}.csv"
                        if p.is_file():
                            try:
                                vals.append(fold_mean(p))
                            except SystemExit:
                                pass
                    c[f"i_{t}"] = float(np.mean(vals)) if vals else None
                    draws[f"i_{t}"] = vals
                row[arm] = {k: (None if v is None else dict(
                                err=v, penalty=v - clean.mean(),
                                n_draws=len(draws.get(k, [])) or 5,
                                sd=(float(np.std(draws[k], ddof=1)) if len(draws.get(k, [])) > 1
                                    else None),
                                draw_penalties=[d - clean.mean() for d in draws.get(k, [])]))
                            for k, v in c.items()}
            for col in ("ii", "i_real", "i_anon"):
                f, h = row[full][col], row[hyb][col]
                if f and h and f["penalty"] > 0:
                    h["recovered_frac"] = 1 - h["penalty"] / f["penalty"]
        out[bb] = row

    for bb, row in out.items():
        b = row["baseline"]
        print(f"\n{bb}: baseline {b['mean']:.2f} ± {b['sd']:.2f} cm (5 clean draws)")
        print(f"  {'arm':14s}{'(ii) clean-train':>22s}{'(i) test real':>22s}{'(i) test anon':>22s}")
        for full, hyb in PAIRS:
            for arm in (full, hyb):
                cells = []
                for col in ("ii", "i_real", "i_anon"):
                    c = row[arm][col]
                    if c is None:
                        cells.append("—".rjust(22)); continue
                    s = f"{c['err']:.2f} ({c['penalty']:+.2f})"
                    if c.get("sd") is not None:
                        s += f"±{c['sd']:.2f}/{c['n_draws']}"
                    if "recovered_frac" in c:
                        s += f" {100 * c['recovered_frac']:.0f}%"
                    cells.append(s.rjust(22))
                print(f"  {arm:14s}" + "".join(cells))

    p = Path("/springbrook/share/eng/esrpxk/results/anon_train/table3_three_conditions.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
