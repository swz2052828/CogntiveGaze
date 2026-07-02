"""Compare base vs subject-adversarial (adv) models on the clean held-out val.
Both are the EXISTING (blink-contaminated-trained) checkpoints, scored on the
non-blink val subset (Step-1 style). adv_gain = base_clean - adv_clean
(>0 means the adversarial subject-invariance training helped accuracy).
Also shows each variant's blink inflation (err_all - err_clean).
"""
import csv
import glob
from pathlib import Path

import numpy as np

BASE = Path("/springbrook/share/eng/esrpxk/results/clean_eval")
ADV = Path("/springbrook/share/eng/esrpxk/results/clean_eval_adv")
OUT = Path("/springbrook/share/eng/esrpxk/results/base_vs_adv_summary.md")


def fold_mean(fp, col):
    rr = list(csv.DictReader(open(fp)))
    return np.mean([float(r[col]) for r in rr]) if rr else None


def main():
    rows = []
    for fp in sorted(glob.glob(str(BASE / "*.csv"))):
        bb = Path(fp).stem
        b_clean = fold_mean(fp, "err_clean")
        b_all = fold_mean(fp, "err_all")
        afp = ADV / f"{bb}.csv"
        a_clean = fold_mean(afp, "err_clean") if afp.is_file() else None
        a_all = fold_mean(afp, "err_all") if afp.is_file() else None
        rows.append(dict(bb=bb, b_clean=b_clean, b_infl=b_all - b_clean,
                         a_clean=a_clean,
                         a_infl=(a_all - a_clean) if a_clean is not None else None,
                         adv_gain=(b_clean - a_clean) if a_clean is not None else None))
    rows.sort(key=lambda r: (r["a_clean"] is None, r["a_clean"] if r["a_clean"] is not None else 0))

    hdr = f"{'backbone':32s} {'base_clean':>10} {'adv_clean':>10} {'adv_gain':>9} {'base_infl':>10} {'adv_infl':>9}"
    L = ["# base vs subject-adversarial (adv) on clean held-out val (cm)",
         "", "base_clean / adv_clean = existing base / adv model on non-blink val.",
         "adv_gain = base_clean - adv_clean  (>0 = adversarial training helped).",
         "*_infl = blink inflation (err_all - err_clean) for that variant.",
         "", "```", hdr, "-" * len(hdr)]
    done = [r for r in rows if r["a_clean"] is not None]
    for r in rows:
        ac = f"{r['a_clean']:.4f}" if r['a_clean'] is not None else "  pending"
        ag = f"{r['adv_gain']:+.4f}" if r['adv_gain'] is not None else "   --"
        ai = f"{r['a_infl']:+.4f}" if r['a_infl'] is not None else "   --"
        L.append(f"{r['bb']:32s} {r['b_clean']:>10.4f} {ac:>10} {ag:>9} {r['b_infl']:>+10.4f} {ai:>9}")
    if done:
        mb = np.mean([r["b_clean"] for r in done]); ma = np.mean([r["a_clean"] for r in done])
        nbetter = sum(1 for r in done if r["adv_gain"] > 0)
        L += ["-" * len(hdr),
              f"{'COHORT MEAN':32s} {mb:>10.4f} {ma:>10.4f} {mb-ma:>+9.4f}",
              f"# adv better on {nbetter}/{len(done)} backbones"]
    L.append("```")
    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
