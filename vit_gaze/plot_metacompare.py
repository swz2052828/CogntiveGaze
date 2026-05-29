"""Plot the calibration-points-vs-error curve from one or more metacompare CSVs.

Reads the CSV(s) written by ``metacompare --csv-out`` (one row per fold per K),
groups by K, and plots mean +/- std-across-folds error for each method
(base / svr / meta / meta_adv) on a single axes. This is the headline figure:
how each calibration method's error falls as the number of calibration frames
K grows, and where meta overtakes (or undercuts) SVR.

Usage:
    python -m vit_gaze.plot_metacompare --csv metacompare.csv --out kcurve.png
    python -m vit_gaze.plot_metacompare --csv a.csv b.csv --out kcurve.png
"""

import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

_METHODS = ("base", "svr", "svr_embed", "fc_ft", "meta", "meta_adv")
_LABELS = {
    "base": "Base (no calibration)",
    "svr": "Per-subject SVR (correction)",
    "svr_embed": "Per-subject SVR (embeddings, Zhu et al.)",
    "fc_ft": "Head-only fine-tune (Zhu et al.)",
    "meta": "Meta-adapter",
    "meta_adv": "Meta-adapter (subject-adv features)",
}
_STYLE = {
    "base": dict(color="0.5", marker="o", linestyle="--"),
    "svr": dict(color="tab:orange", marker="s", linestyle="-"),
    "svr_embed": dict(color="tab:brown", marker="P", linestyle="-"),
    "fc_ft": dict(color="tab:red", marker="v", linestyle="-"),
    "meta": dict(color="tab:blue", marker="^", linestyle="-"),
    "meta_adv": dict(color="tab:green", marker="D", linestyle="-"),
}


def _load(csv_paths):
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    if "K" not in df.columns:
        raise ValueError("CSV has no 'K' column; is this a metacompare CSV?")
    return df


def _wilcoxon_paired(df, m1, m2):
    """Per-K paired Wilcoxon signed-rank test of (m1 - m2).

    Each (fold, seed) row gives one paired observation. Returns
    ``{K: (p_value, n_pairs, sign)}`` where ``sign`` is -1 if m1<m2 on average
    (m1 wins), +1 if m1>m2. Returns ``None`` if scipy is unavailable.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    out = {}
    for k in sorted(df["K"].unique()):
        sub = df[df["K"] == k]
        a = sub[m1].to_numpy()
        b = sub[m2].to_numpy()
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        if len(a) < 6:
            out[int(k)] = (float("nan"), len(a), 0)
            continue
        try:
            _, p = wilcoxon(a, b)
        except ValueError:
            p = float("nan")
        sign = -1 if np.mean(a) < np.mean(b) else 1
        out[int(k)] = (float(p), len(a), sign)
    return out


def _stars(p):
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot(csv_paths, out_path, title=None, sig_pairs=()):
    df = _load(csv_paths)
    methods = [m for m in _METHODS if m in df.columns and df[m].notna().any()]
    ks = sorted(df["K"].unique())
    # Aggregate over all (fold, seed) rows at a given K. If only one seed was run
    # std-bands reflect across-fold spread; with a seed sweep they also include
    # seed-to-seed noise -- the honest error bars for the headline figure.
    seed_aware = "seed" in df.columns and df["seed"].nunique() > 1

    fig, ax = plt.subplots(figsize=(7, 5))
    means_by_method, stds_by_method = {}, {}
    for m in methods:
        means, stds = [], []
        for k in ks:
            vals = df.loc[df["K"] == k, m].dropna().to_numpy()
            means.append(np.mean(vals) if len(vals) else np.nan)
            stds.append(np.std(vals) if len(vals) > 1 else 0.0)
        means, stds = np.array(means), np.array(stds)
        means_by_method[m] = means
        stds_by_method[m] = stds
        ax.plot(ks, means, label=_LABELS[m], **_STYLE[m])
        ax.fill_between(ks, means - stds, means + stds, alpha=0.15,
                        color=_STYLE[m]["color"])

    # Paired-Wilcoxon significance markers: stars annotated above the winning
    # method's curve at each K where p < 0.05. The full p-value table is also
    # printed to stdout so the user can quote exact numbers.
    if sig_pairs:
        sig_summary = []
        for m1, m2 in sig_pairs:
            if m1 not in methods or m2 not in methods:
                continue
            result = _wilcoxon_paired(df, m1, m2)
            if result is None:
                print("scipy not available; skipping significance test.")
                break
            for k_idx, k in enumerate(ks):
                p, n, sign = result[int(k)]
                stars = _stars(p)
                if not stars:
                    continue
                winner = m1 if sign < 0 else m2
                # offset the annotation just above the winning curve's lower std band
                y = means_by_method[winner][k_idx] - stds_by_method[winner][k_idx]
                ax.annotate(stars, xy=(k, y), xytext=(0, -8),
                            textcoords="offset points",
                            ha="center", va="top",
                            color=_STYLE[winner]["color"], fontsize=10,
                            fontweight="bold")
            sig_summary.append((m1, m2, result))
        if sig_summary:
            print("\nPaired Wilcoxon signed-rank tests:")
            for m1, m2, result in sig_summary:
                print(f"  {_LABELS[m1]} vs {_LABELS[m2]}:")
                for k in ks:
                    p, n, sign = result[int(k)]
                    win = _LABELS[m1] if sign < 0 else _LABELS[m2]
                    if np.isnan(p):
                        print(f"    K={k}: insufficient pairs (n={n})")
                    else:
                        print(f"    K={k}: p={p:.4f} n={n} winner={win} {_stars(p)}")

    ax.set_xlabel("Calibration frames K")
    ax.set_ylabel("Mean gaze error (cm)")
    suffix = " (mean +/- std across folds x seeds)" if seed_aware else " (mean +/- std across folds)"
    ax.set_title((title or "Calibration error vs. number of calibration frames") + suffix)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _parse_sig_pair(s):
    if ":" not in s:
        raise argparse.ArgumentTypeError(f"--sig-pair must be 'method1:method2', got {s!r}")
    a, b = s.split(":", 1)
    if a not in _METHODS or b not in _METHODS:
        raise argparse.ArgumentTypeError(
            f"--sig-pair: methods must be one of {_METHODS}, got {s!r}")
    return (a, b)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", nargs="+", required=True,
                   help="One or more metacompare CSV files (concatenated).")
    p.add_argument("--out", default="metacompare_kcurve.png")
    p.add_argument("--title", default=None)
    p.add_argument(
        "--sig-pair", action="append", type=_parse_sig_pair, default=None,
        metavar="m1:m2",
        help="Per-K paired Wilcoxon signed-rank test between m1 and m2. Stars "
             "are drawn above the winning curve where p<0.05; full p-value "
             "table is printed to stdout. Repeatable. Default: meta:svr when "
             "--significance is set.",
    )
    p.add_argument(
        "--significance", action="store_true",
        help="Shortcut: enable a default meta:svr test if no --sig-pair given.",
    )
    args = p.parse_args()
    sig_pairs = args.sig_pair or []
    if args.significance and not sig_pairs:
        sig_pairs = [("meta", "svr")]
    path = plot(args.csv, args.out, title=args.title, sig_pairs=sig_pairs)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
