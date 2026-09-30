"""Plan D: three ways to put a gaze model on anonymised imagery, compared per subject.

zeroshot  clean-trained model deployed on anonymised data (condition (iii) runs)
retrain   trained from scratch on anonymised data
finetune  clean model fine-tuned on anonymised data

All three are evaluated with the identical metacompare call and matched
anonymised enrolment, so the comparison is within one harness. Pairing is by
RECORDING, as everywhere else in this programme: between-subject spread is
several cm and would swamp the effect if the unit were the fold.
"""
import csv, re
from collections import defaultdict
from pathlib import Path
import numpy as np

RUNS = Path("/springbrook/share/eng/esrpxk/runs")
RES = Path("/springbrook/share/eng/esrpxk/results")
LINE = re.compile(r"Fold (?P<fold>\d+) rec=(?P<rec>\d+) K=(?P<k>\d+)\(calib\) .*?fc_ft=(?P<v>[0-9.]+)")

# --- protocols trained on anonymised data ------------------------------------
tab = defaultdict(dict)          # (proto, seed, bb, op, K) -> {rec: fc_ft}
for f in (RUNS / "anon_retrain_eval" / "logs").glob("*.log"):
    m = re.match(r"(\w+?)_s(\d+)_(.+?)_(blackbox|blur[\d.]+)_K(\d+)_f\d\.log", f.name)
    if not m:
        continue
    proto, seed, bb, op, K = m.groups()
    for ln in f.read_text(errors="replace").splitlines():
        g = LINE.search(ln)
        if g and g.group("k") == K:
            tab[(proto, seed, bb, op, K)][int(g.group("rec"))] = float(g.group("v"))

# --- zero-shot condition (iii), from the existing subject-level table ---------
zs = defaultdict(dict)
for r in csv.DictReader(open(RES / "anon_stage4_subject_all.csv")):
    if r["cond"] == "iii" and r["fc_ft"] not in ("", "nan"):
        zs[(r["backbone"], r["operator"], r["k"])][int(r["rec"])] = float(r["fc_ft"])

rng = np.random.default_rng(0)
def ci(d):
    d = np.asarray(d); idx = rng.integers(0, len(d), (20000, len(d)))
    return float(np.percentile(d[idx].mean(1), 2.5)), float(np.percentile(d[idx].mean(1), 97.5))

rows = []
print(f"{'K':>3} {'backbone':22}{'operator':10}{'protocol':16}{'cm':>7}{'vs zero-shot':>14}{'95% CI':>18}{'n':>4}")
for K in ("72", "9"):
    for bb in ("mobile_vit", "convnextv2"):
        for op in ("blur0.20", "blackbox"):
            base = zs[(bb, op, K)]
            print(f"{K:>3} {bb:22}{op:10}{'zeroshot':16}{np.mean(list(base.values())):7.3f}{'':>14}{'':>18}{len(base):4}")
            rows.append([K, bb, op, "zeroshot", "-", round(float(np.mean(list(base.values()))), 4), "", "", len(base)])
            for (proto, seed) in (("retrain", "42"), ("finetune", "42"), ("retrain", "1"), ("retrain", "7")):
                cur = tab.get((proto, seed, bb, op, K))
                if not cur:
                    continue
                recs = sorted(set(cur) & set(base))
                d = np.array([cur[s] - base[s] for s in recs])
                lo, hi = ci(d)
                rows.append([K, bb, op, proto, seed, round(float(np.mean([cur[s] for s in recs])), 4),
                             round(float(d.mean()), 4), f"[{lo:+.3f},{hi:+.3f}]", len(recs)])
                print(f"{'':>3} {'':22}{'':10}{proto+' s'+seed:16}{np.mean([cur[s] for s in recs]):7.3f}"
                      f"{d.mean():+14.3f}{f'[{lo:+.2f},{hi:+.2f}]':>18}{len(recs):4}")
    print()

out = RES / "paper" / "retrain_protocols.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["K", "backbone", "operator", "protocol", "seed", "fc_ft cm", "diff vs zeroshot cm", "95% CI", "n_subjects"])
    w.writerows(rows)
print("wrote", out)
