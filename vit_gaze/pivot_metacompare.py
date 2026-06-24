"""Pivot metacompare CSV(s) into the per-fold / per-K tables that go next to
the K-sweep figure in a paper.

The long-form ``metacompare.csv`` has one row per (fold, seed, K, ...). For a
writeup you usually want two condensed views:

1. **summary** (default): the headline table. Rows = K, columns = method,
   cells = "mean +/- std" aggregated across all (fold, seed) rows for that K.
   Matches the plotted K-sweep curve numerically.

2. **per-fold**: at a *single* K (``--k``), rows = fold, columns = method,
   cells = "mean +/- std" aggregated across seeds for that (fold, method).
   This is the per-fold breakdown that surfaces fold 2 as the hard one.

3. **per-method**: transpose of summary -- rows = method, columns = K.

Three output formats: ``csv`` (machine-readable), ``markdown`` (paste into a
draft), ``latex`` (booktabs-style tabular).

Usage:
    python -m vit_gaze.pivot_metacompare --csv metacompare.csv --view summary \\
                                          --format markdown --out summary.md
    python -m vit_gaze.pivot_metacompare --csv metacompare.csv --view per-fold \\
                                          --k 16 --format markdown
"""

import argparse

import numpy as np
import pandas as pd

_METHODS = ("base", "svr", "svr_embed", "fc_ft", "meta", "meta_adv")
_LABELS = {
    "base": "Base",
    "svr": "SVR (correction)",
    "svr_embed": "SVR (embeddings)",
    "fc_ft": "FC fine-tune",
    "meta": "Meta-adapter",
    "meta_adv": "Meta-adapter (adv)",
}


def _load(csv_paths):
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    if "K" not in df.columns:
        raise ValueError("CSV has no 'K' column; is this a metacompare CSV?")
    return df


def _active_methods(df):
    return [m for m in _METHODS if m in df.columns and df[m].notna().any()]


def _cell(mean, std):
    if np.isnan(mean):
        return "--"
    if np.isnan(std) or std == 0.0:
        return f"{mean:.3f}"
    return f"{mean:.3f} +/- {std:.3f}"


def _build_summary(df, group_col, methods):
    """Group by `group_col`, return DataFrame with one row per group and one
    column per method, cells as 'mean +/- std' over all rows in that group."""
    rows = []
    for g, sub in df.groupby(group_col):
        row = {group_col: g}
        for m in methods:
            vals = sub[m].dropna().to_numpy()
            mean = float(np.mean(vals)) if len(vals) else float("nan")
            std = float(np.std(vals)) if len(vals) > 1 else 0.0
            row[_LABELS[m]] = _cell(mean, std)
        rows.append(row)
    return pd.DataFrame(rows).set_index(group_col)


def view_summary(df):
    return _build_summary(df, "K", _active_methods(df))


def view_per_fold(df, k):
    sub = df[df["K"] == k]
    if sub.empty:
        raise ValueError(f"No rows for K={k}. Available K: {sorted(df['K'].unique())}")
    return _build_summary(sub, "fold", _active_methods(df))


def view_per_method(df):
    """Rows = method, columns = K. Transpose of summary, easier to read at-a-glance
    when comparing methods rather than K values."""
    methods = _active_methods(df)
    ks = sorted(df["K"].unique())
    rows = []
    for m in methods:
        row = {"method": _LABELS[m]}
        for k in ks:
            vals = df.loc[df["K"] == k, m].dropna().to_numpy()
            mean = float(np.mean(vals)) if len(vals) else float("nan")
            std = float(np.std(vals)) if len(vals) > 1 else 0.0
            row[f"K={k}"] = _cell(mean, std)
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def _to_markdown(df, title=None):
    out = []
    if title:
        out.append(f"### {title}\n")
    try:
        out.append(df.to_markdown())  # requires tabulate
    except ImportError:
        # Hand-rolled pipe table so the script works in minimal envs.
        cols = [df.index.name or ""] + list(df.columns)
        rows = [cols, ["---"] * len(cols)]
        for idx, vals in df.iterrows():
            rows.append([str(idx)] + [str(v) for v in vals])
        widths = [max(len(r[i]) for r in rows) for i in range(len(cols))]
        out.extend("| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(cols))) + " |"
                   for r in rows)
    return "\n".join(out)


def _to_latex(df, title=None):
    body = df.to_latex(escape=False, column_format="l" + "r" * len(df.columns))
    if title:
        return f"% {title}\n{body}"
    return body


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", nargs="+", required=True,
                   help="One or more metacompare CSVs (concatenated).")
    p.add_argument("--view", choices=("summary", "per-fold", "per-method"),
                   default="summary")
    p.add_argument("--k", type=int, default=None,
                   help="K to slice on (required for --view per-fold).")
    p.add_argument("--format", choices=("csv", "markdown", "latex"),
                   default="markdown")
    p.add_argument("--out", default=None,
                   help="Write to this file (default: stdout).")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    df = _load(args.csv)
    if args.view == "summary":
        table = view_summary(df)
        title = args.title or "Calibration error vs K (mean +/- std)"
    elif args.view == "per-fold":
        if args.k is None:
            raise SystemExit("--view per-fold requires --k.")
        table = view_per_fold(df, args.k)
        title = args.title or f"Per-fold error at K={args.k} (mean +/- std across seeds)"
    else:
        table = view_per_method(df)
        title = args.title or "Per-method error across K (mean +/- std)"

    if args.format == "csv":
        text = table.to_csv()
    elif args.format == "latex":
        text = _to_latex(table, title)
    else:
        text = _to_markdown(table, title)

    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
