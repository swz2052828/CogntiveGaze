"""FAMS utility on the every-10th-frame subsample, against a MATCHED baseline.

Unlike every other arm, FAMS was generated for 10% of the frames (diffusion is
~6 s/frame), so its baseline is the real-data model on the same subset:
  (ii)  clean-trained checkpoints (5 init draws), evaluated on the subset manifest
        for both arms -- `none_sub10` vs `famsf` / `famsf_oldeye`
  (i)   models RETRAINED on the 10% subset -- real-data subset (`none_sub10`) vs
        FAMS subset -- tested on the full real test set and on the subset release;
        penalties are paired within initialisation draw, then mean +- SD.
Writes results/anon_fams/util_table.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path("/springbrook/share/eng/esrpxk/runs")
OUT = Path("/springbrook/share/eng/esrpxk/results/anon_fams/util_table.json")
ARMS = ["famsf", "famsf_oldeye"]


def fold_mean(p):
    if not p.is_file():
        return None
    d = pd.read_csv(p).drop_duplicates("fold", keep="last")
    return float(d["base"].mean()) if sorted(d["fold"]) == [0, 1, 2, 3, 4] else None


def main():
    out = {"ii": {}, "i": {}}
    for bb in ["itracker", "mobilenet_v3", "affnet", "mgazenet"]:
        row = {}
        base = [fold_mean(RUNS / "dp2_utility" / f"dp2_d{d}_{bb}_none_sub10.csv") for d in range(5)]
        for arm in ARMS:
            pen = [fold_mean(RUNS / "dp2_utility" / f"dp2_d{d}_{bb}_{arm}.csv") - b
                   for d, b in enumerate(base) if b is not None
                   and fold_mean(RUNS / "dp2_utility" / f"dp2_d{d}_{bb}_{arm}.csv") is not None]
            row[arm] = dict(penalty=float(np.mean(pen)), sd=float(np.std(pen, ddof=1)) if len(pen) > 1 else None,
                            n_draws=len(pen))
        row["baseline"] = float(np.mean([b for b in base if b is not None]))
        out["ii"][bb] = row
    E = RUNS / "anon_train" / "eval"
    for bb in ["mobilenet_v3", "affnet", "mgazenet"]:
        row = {}
        for test in ("real", "anon"):
            for arm in ARMS:
                pen = []
                for rep in range(1, 6):
                    sfx = "" if rep == 1 else f"_rep{rep}"
                    b = fold_mean(E / f"at_none_sub10_{bb}_test{test}{sfx}.csv")
                    a = fold_mean(E / f"at_{arm}_{bb}_test{test}{sfx}.csv")
                    if a is not None and b is not None:
                        pen.append(a - b)
                row[f"{arm}_{test}"] = dict(penalty=float(np.mean(pen)) if pen else None,
                                            sd=float(np.std(pen, ddof=1)) if len(pen) > 1 else None,
                                            n_draws=len(pen), draws=pen)
        out["i"][bb] = row
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    for cond in ("ii", "i"):
        print(f"== condition ({cond})")
        for bb, row in out[cond].items():
            print(f"  {bb:13s}", {k: (round(v["penalty"], 2) if v["penalty"] is not None else None,
                                     v["n_draws"]) for k, v in row.items() if isinstance(v, dict)})
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
