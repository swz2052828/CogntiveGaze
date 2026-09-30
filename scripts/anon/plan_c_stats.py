"""Plan C: subject-level statistics on the Stage 4 privacy-utility comparison.

Unit of analysis is the SUBJECT, not the fold. Recording-level 5-fold CV gives
each recording exactly one held-out measurement per (backbone, operator), so the
design is balanced and every operator is paired within subject against `none`.
The subject random intercept then cancels out of the difference rather than
inflating its interval, which matters here: between-subject spread is several cm
and dwarfs the effects being tested.

Independence: the 17 recordings analysed are 17 distinct individuals. Three
participants recorded twice in the wider study (subid 3/12, 4/7, 2/6) but only
one session of each pair survives into this set (7 is dropped by the meanno7
split, 3 and 2 are outside the 6-23 range), so no person contributes two rows.

Reported per (backbone, operator):
  * paired mean difference vs `none` with a subject-resampled bootstrap CI
  * one-sided non-inferiority test against a PRE-DECLARED margin, alpha .025
  * Cohen's dz, and the number of subjects hurt
Plus, per backbone: a random-intercept variance decomposition, and
Bland-Altman agreement for the headline `none` vs `blackbox` contrast.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

OPS = ["blur0.03", "blur0.10", "blur0.20", "blur0.35", "blur0.50",
       "pixel0.03", "pixel0.06", "pixel0.12", "pixel0.25", "pixel0.40", "blackbox"]


def student_t_sf(t, df):
    """Upper-tail P(T > t). Regularised incomplete beta, no scipy dependency."""
    x = df / (df + t * t)
    ib = _betainc(df / 2.0, 0.5, x)
    p = 0.5 * ib
    return p if t > 0 else 1.0 - p


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = _lgamma(a) + _lgamma(b) - _lgamma(a + b)
    front = np.exp(np.log(x) * a + np.log(1 - x) * b - lbeta) / a
    if x < (a + 1) / (a + b + 2):
        return front * _cf(a, b, x)
    return 1.0 - _betainc(b, a, 1 - x)


def _cf(a, b, x, it=300):
    f, c, d = 1.0, 1.0, 0.0
    for i in range(it + 1):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        delta = c * d
        f *= delta
        if abs(1 - delta) < 1e-12:
            break
    return f - 1.0


def _lgamma(z):
    from math import lgamma
    return lgamma(z)


def load(path, metric, cond=None, k=None):
    """tab[backbone][operator][rec] -> value, restricted to one condition and K.

    Filtering on K is not optional once the input holds more than one enrolment
    size: the table is keyed by recording, so K=9 rows would silently overwrite
    their K=72 counterparts and the result would look complete.
    """
    tab = defaultdict(lambda: defaultdict(dict))
    n = 0
    for r in csv.DictReader(open(path)):
        v = r.get(metric, "")
        if v in ("", "nan"):
            continue
        if cond is not None and r.get("cond", "ii") != cond:
            continue
        if k is not None and str(r.get("k", "")) != str(k):
            continue
        tab[r["backbone"]][r["operator"]][int(r["rec"])] = float(v)
        n += 1
    ks = {r.get("k") for r in csv.DictReader(open(path))}
    if k is None and len(ks) > 1:
        raise SystemExit(
            f"{path} holds several enrolment sizes ({sorted(ks)}). Pass --k, or "
            f"rows will overwrite each other by recording.")
    return tab


def compare_conditions(path, metric, margin, boot, rng, k=None):
    """Enrolment protocol as a within-subject factor.

    Condition (ii) enrols on ORIGINAL calibration frames, (iii) on frames
    anonymised by the same operator. The release-only deployment can only offer
    (iii), so the question is whether the (ii) numbers -- which is what Stage 4a
    reported -- flatter the result. Paired within subject, so this is the
    operator x protocol interaction the protocol asks for, restricted to the
    contrast that matters rather than fitted as a full factorial.
    """
    ii = load(path, metric, "ii", k)
    iii = load(path, metric, "iii", k)
    bbs = sorted(set(ii) & set(iii))
    if not bbs:
        print("\n(no condition-iii rows yet; skipping protocol comparison)")
        return
    print(f"\n{'#'*94}\nENROLMENT PROTOCOL  (iii) matched-anonymised  vs  (ii) clean   metric={metric}")
    print("positive = condition (iii) is WORSE, i.e. (ii) was flattering the result")
    for bb in bbs:
        print(f"\n  === {bb}")
        print(f"  {'operator':<11}{'(ii)':>8}{'(iii)':>8}{'diff':>9}{'95% CI':>18}"
              f"{'vs none: (ii)':>15}{'(iii)':>9}")
        n_ii = ii[bb].get("none", {})
        n_iii = iii[bb].get("none", {})
        for op in ["none"] + OPS:
            a, b = ii[bb].get(op, {}), iii[bb].get(op, {})
            common = sorted(set(a) & set(b))
            if len(common) < 3:
                continue
            d = np.array([b[r] - a[r] for r in common])
            bs = np.array([rng.choice(d, len(d), replace=True).mean()
                           for _ in range(boot)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            # the operator penalty within each protocol, which is the quantity
            # the paper actually claims
            def pen(tabop, ctrl):
                k = sorted(set(tabop) & set(ctrl))
                return np.mean([tabop[r] - ctrl[r] for r in k]) if k else float("nan")
            p_ii = pen(a, n_ii) if n_ii else float("nan")
            p_iii = pen(b, n_iii) if n_iii else float("nan")
            print(f"  {op:<11}{np.mean([a[r] for r in common]):>8.3f}"
                  f"{np.mean([b[r] for r in common]):>8.3f}{d.mean():>+9.3f}"
                  f"  [{lo:+.2f},{hi:+.2f}]{p_ii:>+15.3f}{p_iii:>+9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/springbrook/share/eng/esrpxk/results/anon_stage4_subject.csv")
    ap.add_argument("--metric", default="fc_ft")
    ap.add_argument("--margin", type=float, default=0.40)
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--cond", default="ii",
                    help="enrolment condition to analyse: ii (clean enrolment) or "
                         "iii (enrolment anonymised by the same operator)")
    ap.add_argument("--k", default=None,
                    help="enrolment size to analyse (9 or 72). Required when the "
                         "input CSV holds more than one.")
    ap.add_argument("--compare-conditions", action="store_true",
                    help="also test (iii) against (ii) paired within subject")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    M = args.margin
    tab = load(args.csv, args.metric, args.cond, args.k)
    if not tab:
        raise SystemExit(f"no rows for cond={args.cond!r} in {args.csv}")

    for bb in sorted(tab):
        ctrl = tab[bb].get("none", {})
        if not ctrl:
            continue
        print(f"\n{'='*94}\n{bb}   metric={args.metric}   control 'none' = "
              f"{np.mean(list(ctrl.values())):.3f} cm over {len(ctrl)} subjects")

        # --- variance decomposition (why the pairing matters) ----------------
        recs = sorted(set.intersection(*[set(tab[bb][o]) for o in tab[bb]]))
        ops_all = sorted(tab[bb])
        Y = np.array([[tab[bb][o][r] for o in ops_all] for r in recs])   # S x O
        S, O = Y.shape
        gm = Y.mean()
        ms_s = O * ((Y.mean(1) - gm) ** 2).sum() / (S - 1)
        ms_o = S * ((Y.mean(0) - gm) ** 2).sum() / (O - 1)
        resid = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm
        ms_e = (resid ** 2).sum() / ((S - 1) * (O - 1))
        var_u = max(0.0, (ms_s - ms_e) / O)
        icc = var_u / (var_u + ms_e) if (var_u + ms_e) > 0 else float("nan")
        print(f"  random-intercept model  y[s,o] = mu + beta_o + u_s + e")
        print(f"    between-subject SD  {np.sqrt(var_u):6.3f} cm     "
              f"residual SD {np.sqrt(ms_e):6.3f} cm     ICC {icc:.3f}")
        print(f"    operator MS {ms_o:8.3f}  vs residual MS {ms_e:8.3f}  "
              f"-> F({O-1},{(S-1)*(O-1)}) = {ms_o/ms_e:6.2f}")
        print(f"  (ICC {icc:.2f} => {icc*100:.0f}% of variance is WHO the subject is; "
              f"pairing removes it)")

        # Collect first, then Holm-correct across operators within a backbone --
        # the protocol pre-registers the correction, and 11 operators tested at
        # .025 each would otherwise carry a familywise error near 25%.
        stats = []
        for op in OPS:
            cur = tab[bb].get(op, {})
            common = sorted(set(ctrl) & set(cur))
            if len(common) < 3:
                continue
            d = np.array([cur[r] - ctrl[r] for r in common])
            n = len(d)
            bs = np.array([rng.choice(d, n, replace=True).mean()
                           for _ in range(args.boot)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            sd = d.std(ddof=1)
            dz = d.mean() / sd if sd > 0 else float("inf")
            # one-sided non-inferiority: H0 mu >= M  vs  H1 mu < M
            t = (d.mean() - M) / (sd / np.sqrt(n))
            p = 1.0 - student_t_sf(t, n - 1)     # P(T < t)
            stats.append([op, d, n, lo, hi, dz, p, None])

        # Holm step-down on the non-inferiority p-values.
        order = sorted(range(len(stats)), key=lambda i: stats[i][6])
        k = len(stats)
        running = 0.0
        for rank, i in enumerate(order):
            adj = min(1.0, stats[i][6] * (k - rank))
            running = max(running, adj)          # enforce monotonicity
            stats[i][7] = running

        print(f"\n  {'operator':<11}{'mean d':>9}{'95% CI (subj)':>19}{'dz':>7}"
              f"{'hurt':>7}{'p(NI)':>9}{'p_holm':>9}{'verdict':>16}")
        for op, d, n, lo, hi, dz, p, ph in stats:
            verdict = ("NON-INFERIOR" if (hi < M and ph < 0.025)
                       else "INFERIOR" if lo > M else "inconclusive")
            print(f"  {op:<11}{d.mean():>+9.3f}  [{lo:+.2f},{hi:+.2f}]{dz:>9.2f}"
                  f"{f'{(d>0).sum()}/{n}':>7}{p:>9.4f}{ph:>9.4f}{verdict:>16}")

        # --- Bland-Altman on the headline contrast --------------------------
        bx = tab[bb].get("blackbox", {})
        common = sorted(set(ctrl) & set(bx))
        if common:
            a = np.array([ctrl[r] for r in common])
            b = np.array([bx[r] for r in common])
            diff, mean = b - a, (a + b) / 2
            bias, sdd = diff.mean(), diff.std(ddof=1)
            loa = (bias - 1.96 * sdd, bias + 1.96 * sdd)
            r = np.corrcoef(mean, diff)[0, 1]
            print(f"\n  Bland-Altman  none vs blackbox   n={len(common)}")
            print(f"    bias {bias:+.3f} cm   LoA [{loa[0]:+.3f}, {loa[1]:+.3f}] cm"
                  f"   SD of differences {sdd:.3f}")
            print(f"    proportional-bias check: corr(mean, diff) = {r:+.3f}"
                  f"   ({'no trend' if abs(r) < 0.5 else 'TREND -- bias depends on error level'})")

    d_lo = np.degrees(np.arctan(M / 120.0))
    d_hi = np.degrees(np.arctan(M / 80.0))
    print(f"\nmargin Delta = {M} cm  =  {d_lo:.3f}-{d_hi:.3f} deg of visual angle "
          f"at the measured 80-120 cm viewing distance")
    print(f"one-sided alpha .025, Holm-corrected across operators within backbone, "
          f"{args.boot} subject-resampled draws")
    print(f"enrolment condition: {args.cond}")

    if args.compare_conditions:
        compare_conditions(args.csv, args.metric, M, args.boot, rng, args.k)


if __name__ == "__main__":
    main()
