"""Aggregate Stage 3b (threat model T2, external reference) into the paper table.

Privacy axis is d' and TAR@FAR=1e-3 (§3): Rank-1 carries a ~50% session-nuisance
floor and Rank-1/AUC/TAR@FAR=1e-2 saturate, so they are reported but do not carry
the argument. The background-nuisance control is printed as the floor row.

d' is recomputed from the persisted score matrix where a run predates the metric
being added, so no GPU rerun is needed to complete the column.
"""
import argparse
import json
from pathlib import Path

import numpy as np

RES = Path("/springbrook/share/eng/esrpxk/results/anon_stage3b")
# sweep order, not alphabetical: the table is a monotone strength curve
ORDER = ["none", "blur0.03", "blur0.10", "blur0.20", "blur0.35", "blur0.50",
         "pixel0.03", "pixel0.06", "pixel0.12", "pixel0.25", "pixel0.40", "blackbox"]
LABEL = {"none": "none (control)", "blackbox": "black box (face)"}


def dprime_from_scores(path: Path):
    """Recover d' for runs written before id_attack.py emitted it."""
    npz = path.with_suffix(".scores.npz")
    if not npz.exists():
        return None
    z = np.load(npz, allow_pickle=True)
    S, py, subs = z["S"], z["probe_subject"], z["gallery_subject"]
    labels = (subs[None, :] == py[:, None]).astype(bool).ravel()
    s = S.ravel()
    gen, imp = s[labels], s[~labels]
    return float((gen.mean() - imp.mean()) /
                 np.sqrt(0.5 * (gen.var() + imp.var()) + 1e-12))


def fmt(v, spec, dash="--"):
    return dash if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", default=str(RES))
    ap.add_argument("--out", default=None, help="markdown file to write")
    args = ap.parse_args()
    res = Path(args.res_dir)

    rows, missing = [], []
    for op in ORDER:
        f = res / f"t2_{op}.json"
        if not f.exists():
            missing.append(op)
            continue
        d = json.load(open(f))
        if d.get("dprime") is None:
            d["dprime"] = dprime_from_scores(f)
        rows.append((op, d))

    floor = None
    ff = res / "floor_background.json"
    if ff.exists():
        floor = json.load(open(ff))

    L = []
    L.append("| Operator | d' | TAR@FAR=1e-3 | TAR@FAR=1e-2 | Rank-1 | ROC-AUC | "
             "gender agr. | age MAE (y) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for op, d in rows:
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            LABEL.get(op, op),
            fmt(d.get("dprime"), ".2f"),
            fmt(d.get("TAR@FAR=0.001"), ".1%"),
            fmt(d.get("TAR@FAR=0.01"), ".1%"),
            fmt(d.get("rank1"), ".1%"),
            fmt(d.get("roc_auc"), ".4f"),
            fmt(d.get("gender_agreement"), ".3f"),
            fmt(d.get("age_mae_years"), ".1f")))
    if floor:
        L.append("| **background floor** (no face) | {} | {} | {} | {} | {} | -- | -- |".format(
            fmt(floor.get("background_dprime"), ".2f"),
            fmt(floor.get("TAR@FAR=0.001"), ".1%"),
            fmt(floor.get("TAR@FAR=0.01"), ".1%"),
            fmt(floor.get("background_direct_rank1"), ".1%"),
            fmt(floor.get("background_direct_auc"), ".4f")))
    if rows:
        n = {d["n_probe"] for _, d in rows}
        L.append("")
        L.append(f"n_subjects = {rows[0][1]['n_subjects']}, chance Rank-1 = "
                 f"{rows[0][1]['chance_rank1']:.1%}, n_probe = "
                 f"{'/'.join(str(x) for x in sorted(n))}.")
    if missing:
        L.append("")
        L.append(f"MISSING (not yet run): {', '.join(missing)}")

    # --- T1: linkage within the release ------------------------------------
    t1_rows = []
    for op in ORDER:
        entry = {}
        for key, suffix in (("orig", "t1"), ("self", "t1sa")):
            f = res / f"{suffix}_{op}.json"
            if f.exists():
                entry[key] = json.load(open(f))
        if entry:
            t1_rows.append((op, entry))

    if t1_rows:
        L.append("")
        L.append("**T1 — linkage within the release** (no reference of any kind). "
                 "`orig-align` follows the conservative §3 rule (landmarks from the "
                 "original); `self-align` is the strict release-only attacker. The gap "
                 "between them is protection that comes from breaking face detection "
                 "rather than from removing identity.")
        L.append("")
        L.append("| Operator | ARI k=18 | purity k=18 | k chosen | ARI k-auto | "
                 "pair d' | pair AUC | ARI k=18 (self) | pair d' (self) | det-fail (self) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for op, e in t1_rows:
            a, b = e.get("orig", {}), e.get("self", {})
            L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                LABEL.get(op, op),
                fmt(a.get("ari_k18"), ".3f"), fmt(a.get("purity_k18"), ".1%"),
                fmt(a.get("k_selected"), "d"), fmt(a.get("ari_kauto"), ".3f"),
                fmt(a.get("dprime_same_diff"), ".2f"), fmt(a.get("pair_auc"), ".4f"),
                fmt(b.get("ari_k18"), ".3f"), fmt(b.get("dprime_same_diff"), ".2f"),
                fmt(b.get("det_fail"), "d")))
        L.append("")
        L.append("Chance ARI = 0. Perfect recovery of the 18 individuals = 1.0. "
                 "Background floor (face-free patch through the identical pipeline): "
                 "ARI 0.089, purity 0.243, k_selected 2, pair d' 0.361.")
        L.append("")
        L.append("**Read the last three columns with care** "
                 "(`results/anon_stage3b/T1_BACKGROUND_FLOOR.md`): "
                 "purity@18 reads 1.000 for every operator in every regime — including "
                 "ones whose ARI collapses to 0.215 — because when detection fails on "
                 "most frames the survivors are trivially separable. It is a valid floor "
                 "check and useless as an operator ranking. For the same reason "
                 "**pair d' is not comparable across self-align rows with heavy "
                 "detection failure**: a row computed on a handful of surviving frames "
                 "can exceed the originals'. `k_selected` is the sharper discriminator — "
                 "the floor cannot find the right k, and every alignment-intact operator "
                 "lands on exactly 18.")

    md = "\n".join(L)
    print(md)
    if args.out:
        Path(args.out).write_text(md + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
