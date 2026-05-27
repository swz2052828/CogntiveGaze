"""Pure numeric gaze metrics (no plotting).

Ported from ``GazePreprocessor_swz`` with the algorithms unchanged. Plotting
helpers that happened to live next to these (e.g. dwell-time ribbons) now live
in ``plotting.py``; this module stays import-light so it can be used headless.
"""

import numpy as np
from scipy.stats import pearsonr, ttest_rel
from scipy.spatial.distance import euclidean

try:
    from fastdtw import fastdtw
    HAS_DTW = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_DTW = False
    print("Warning: 'fastdtw' not found. DTW will be skipped.")


def calculate_velocity(sample_rate, xy_data):
    """2D speed profile from ``[2, N]`` coordinates, padded to length N."""
    if xy_data is None or xy_data.shape[1] < 2:
        return np.zeros(0)
    dx = np.diff(xy_data[0])
    dy = np.diff(xy_data[1])
    dist = np.sqrt(dx ** 2 + dy ** 2)
    return np.pad(dist * sample_rate, (0, 1), 'edge')


def calculate_signal_metrics(xy_pred, xy_gt):
    """RMSE and (optional) length-normalized DTW between two ``[2, N]`` signals."""
    if not xy_pred.shape[1] == xy_gt.shape[1]:
        print('Error: Output and EyeLink do not have the same shape!')
        return {'RMSE': np.nan, 'DTW': np.nan}
    err = np.sqrt(np.mean(np.sum((xy_pred - xy_gt) ** 2, axis=0)))
    dtw_dist = -1
    if HAS_DTW:
        dist, _ = fastdtw(xy_pred.T, xy_gt.T, dist=euclidean)
        dtw_dist = dist
    return {'RMSE': err, 'DTW': dtw_dist / xy_pred.shape[1]}


def calculate_entropy(heatmap, epsilon=1e-10):
    """Shannon entropy of a heatmap (higher = more dispersed scanning)."""
    heatmap_sum = np.sum(heatmap)
    if heatmap_sum == 0:
        return 0.0
    prob_flat = (heatmap / heatmap_sum).flatten()
    prob_flat = prob_flat[prob_flat > 0]
    return -np.sum(prob_flat * np.log2(prob_flat + epsilon))


def calculate_heatmap_correlation(map1, map2):
    """Pearson correlation between two flattened heatmaps."""
    h1, h2 = map1.flatten(), map2.flatten()
    if np.std(h1) == 0 or np.std(h2) == 0:
        return 0.0
    r, _ = pearsonr(h1, h2)
    return r


def print_summary_statistics(title, data1, data2):
    """Print Mean +/- SD, Pearson r, MAE and paired t-test for Output vs EyeLink."""
    print(f"\n=== {len(data1)} Cohort Summary Statistics (Mean +/- SD) ===")
    print(f"{title} (Output) : {data1.mean():.3f} +/- {data1.std():.3f}")
    print(f"{title} (EyeLink): {data2.mean():.3f} +/- {data2.std():.3f}")
    if not len(data1) == len(data2):
        print('Warning! Data does not have the same length!')
        return
    r, p = pearsonr(data1, data2)
    mae = np.mean(np.abs(data1 - data2))
    print(f"  -> Pearson r: {r:.3f} (p={p:.4f}), MAE: {mae:.3f}")
    t_stat, p_val = ttest_rel(data1, data2)
    print(f"  -> T-test t: {t_stat:.3f} (p={p_val})\n")


def calculate_all_biomarker_iccs(biomarker_data_dict):
    """ICC2 (two-way mixed, absolute agreement) per biomarker -> paper-ready table.

    ``biomarker_data_dict``: {name: (smartphone_array, eyelink_array)}.
    """
    import pandas as pd
    import pingouin as pg

    results = []
    for name, (smart_data, gt_data) in biomarker_data_dict.items():
        smart_data = np.array(smart_data, dtype=float)
        gt_data = np.array(gt_data, dtype=float)
        assert len(smart_data) == len(gt_data), f"Length mismatch for {name}"

        df_wide = pd.DataFrame({'Smart': smart_data, 'GT': gt_data}).dropna()
        n_valid = len(df_wide)
        if n_valid < 3:
            print(f"Skipping {name}: Not enough valid paired data.")
            continue

        df_clean = pd.DataFrame({
            'Subject': np.tile(np.arange(1, n_valid + 1), 2),
            'System': ['Smartphone'] * n_valid + ['EyeLink'] * n_valid,
            'Score': np.concatenate([df_wide['Smart'].values, df_wide['GT'].values]),
        })
        icc = pg.intraclass_corr(data=df_clean, targets='Subject',
                                 raters='System', ratings='Score')
        target = icc[icc['Type'] == 'ICC2'].iloc[0]
        icc_val = target['ICC']
        ci_lower, ci_upper = target['CI95%'][0], target['CI95%'][1]
        p_val = target['pval']
        if icc_val < 0.50:
            interp = "Poor"
        elif icc_val < 0.75:
            interp = "Moderate"
        elif icc_val < 0.90:
            interp = "Good"
        else:
            interp = "Excellent"
        results.append({
            'Biomarker': name, 'N Valid': n_valid, 'ICC': round(icc_val, 3),
            '95% CI': f"[{ci_lower:.2f}, {ci_upper:.2f}]",
            'P-value': f"{p_val:.3e}" if p_val < 0.001 else f"{p_val:.3f}",
            'Reliability': interp,
        })
    return pd.DataFrame(results)


def run_mixed_anova_for_metric(smart_fv, smart_vs, eyelink_fv, eyelink_vs,
                               metric_name="Shannon Entropy"):
    """2x2 mixed ANOVA: Task (between) x System (within) for one metric."""
    import pandas as pd
    import pingouin as pg

    n_fv, n_vs = len(smart_fv), len(smart_vs)
    print(n_fv, n_vs)
    fv_ids = np.arange(1, n_fv + 1)
    vs_ids = np.arange(n_fv + 1, n_fv + n_vs + 1)

    df = pd.concat([
        pd.DataFrame({'Trial_ID': fv_ids, 'Task': 'Free View', 'System': 'Smartphone', 'Score': smart_fv}),
        pd.DataFrame({'Trial_ID': fv_ids, 'Task': 'Free View', 'System': 'EyeLink', 'Score': eyelink_fv}),
        pd.DataFrame({'Trial_ID': vs_ids, 'Task': 'Search', 'System': 'Smartphone', 'Score': smart_vs}),
        pd.DataFrame({'Trial_ID': vs_ids, 'Task': 'Search', 'System': 'EyeLink', 'Score': eyelink_vs}),
    ], ignore_index=True).dropna()

    print(f"\n{'=' * 56}\n  TWO-WAY MIXED ANOVA: {metric_name.upper()}\n{'=' * 56}")
    anova = pg.mixed_anova(data=df, dv='Score', within='System',
                           between='Task', subject='Trial_ID')
    print(anova.round(4).to_string(index=False))
    p_int = anova[anova['Source'] == 'Interaction']['p-unc'].values[0]
    if p_int > 0.05:
        print(f"CONCLUSION: Interaction NOT significant (p = {p_int:.3f}). "
              "Systems track the task change the same way.")
    else:
        print(f"CONCLUSION: Interaction IS significant (p = {p_int:.3f}). "
              "Systems measure the task change differently.")
    return anova
