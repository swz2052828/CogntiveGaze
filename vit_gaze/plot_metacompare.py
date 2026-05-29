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

_METHODS = ("base", "svr", "meta", "meta_adv")
_LABELS = {
    "base": "Base (no calibration)",
    "svr": "Per-subject SVR",
    "meta": "Meta-adapter",
    "meta_adv": "Meta-adapter (subject-adv features)",
}
_STYLE = {
    "base": dict(color="0.5", marker="o", linestyle="--"),
    "svr": dict(color="tab:orange", marker="s", linestyle="-"),
    "meta": dict(color="tab:blue", marker="^", linestyle="-"),
    "meta_adv": dict(color="tab:green", marker="D", linestyle="-"),
}


def _load(csv_paths):
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    if "K" not in df.columns:
        raise ValueError("CSV has no 'K' column; is this a metacompare CSV?")
    return df


def plot(csv_paths, out_path, title=None):
    df = _load(csv_paths)
    methods = [m for m in _METHODS if m in df.columns and df[m].notna().any()]
    ks = sorted(df["K"].unique())

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in methods:
        means, stds = [], []
        for k in ks:
            vals = df.loc[df["K"] == k, m].dropna().to_numpy()
            means.append(np.mean(vals) if len(vals) else np.nan)
            stds.append(np.std(vals) if len(vals) > 1 else 0.0)
        means, stds = np.array(means), np.array(stds)
        ax.plot(ks, means, label=_LABELS[m], **_STYLE[m])
        ax.fill_between(ks, means - stds, means + stds, alpha=0.15,
                        color=_STYLE[m]["color"])

    ax.set_xlabel("Calibration frames K")
    ax.set_ylabel("Mean gaze error (cm)")
    ax.set_title(title or "Calibration error vs. number of calibration frames")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", nargs="+", required=True,
                   help="One or more metacompare CSV files (concatenated).")
    p.add_argument("--out", default="metacompare_kcurve.png")
    p.add_argument("--title", default=None)
    args = p.parse_args()
    path = plot(args.csv, args.out, title=args.title)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
