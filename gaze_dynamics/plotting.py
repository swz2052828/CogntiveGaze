"""All matplotlib/seaborn rendering for the pipeline, consolidated.

Behavior: figures are always saved (headless Agg backend by default) and only
shown interactively when ``configure(show=True)`` has been called (wired to the
CLI ``--show`` flag). Calibration-specific plotters from the original
``GazePlot_swz`` are intentionally omitted (calibration is out of scope).
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless-safe default; configure(show=True) switches it
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import pearsonr, linregress  # noqa: E402
from scipy.signal import correlate  # noqa: E402

from .preprocess import pad_frames_with_rules, shift_array  # noqa: E402
from .metrics import calculate_velocity, print_summary_statistics  # noqa: E402
from . import config  # noqa: E402

_SHOW = False


def configure(show=False):
    """Enable/disable interactive display. Save-to-file always happens."""
    global _SHOW
    _SHOW = show
    if show:
        try:
            matplotlib.use("TkAgg", force=True)
        except Exception:  # pragma: no cover - depends on local GUI backend
            print("No interactive backend available; figures will only be saved.")


def _render(save_path=None, fig=None, dpi=300):
    """Save (if path given) and either show or close, per the configured mode."""
    fig = fig or plt.gcf()
    if save_path:
        out_dir = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if _SHOW:
        plt.show()
    else:
        plt.close(fig)


# --- Blink figures -----------------------------------------------------------

def plot_temporal_trends(blink_df, window_min=1, save_path='temporal_trends.png'):
    """Blink rate and mean duration over experiment time, with linear trends."""
    if blink_df.empty:
        return
    df = blink_df.copy()
    df['min_bin'] = (df['timestamp'] / window_min).astype(int)
    rates = df.groupby('min_bin').size().reset_index(name='total')
    durs = (df.groupby(['subject_id', 'min_bin'])['duration'].mean()
            .groupby('min_bin').mean().reset_index(name='avg_dur'))
    durs['avg_dur'] *= 100 / 3

    slope_r, _, r_v, p_v, _ = linregress(rates['min_bin'], rates['total'])
    slope_d, _, r_d, p_d, _ = linregress(durs['min_bin'], durs['avg_dur'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    sns.regplot(data=rates, x='min_bin', y='total', ax=ax1, color='darkred',
                scatter_kws={'color': 'coral'})
    ax1.set_title(f'A. Blink Rate Trend (Slope: {slope_r:.3f}, Pearson r: {r_v:.3f} p: {p_v:.5f})',
                  loc='left', fontweight='bold')
    ax1.set_ylabel('Total Rate (counts/min)')
    sns.regplot(data=durs, x='min_bin', y='avg_dur', ax=ax2, color='black',
                scatter_kws={'color': 'teal'})
    ax2.set_title(f'B. Blink Duration Trend (Slope: {slope_d:.3f}, Pearson r: {r_d:.3f} p: {p_d:.5f})',
                  loc='left', fontweight='bold')
    ax2.set(ylabel='Mean Duration (ms/count)', xlabel='Experiment Time (Minutes)')
    plt.tight_layout()
    _render(save_path, fig)


def plot_validation(subject_df, save_path='validation.png'):
    """Predicted-vs-GT blink count correlation + Bland-Altman (No-Glasses only)."""
    if subject_df.empty:
        return
    df = subject_df[subject_df['eyewear'] == 'No Glasses'].copy()
    if df.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.regplot(data=df, x='our_count', y='gt_count', ax=ax1, color='black',
                scatter_kws={'s': 100, 'color': 'royalblue'})
    ax1.set_title(f"Correlation (r={pearsonr(df['gt_count'], df['our_count'])[0]:.3f})",
                  fontweight='bold')
    ax1.set(xlabel='Predicted Count', ylabel='GT Count')

    df['avg'] = (df['gt_count'] + df['our_count']) / 2
    df['diff'] = df['gt_count'] - df['our_count']
    md, sd = df['diff'].mean(), df['diff'].std()
    sns.scatterplot(data=df, x='avg', y='diff', s=100, color='royalblue', ax=ax2)
    ax2.axhline(md, color='black', label=f'Bias: {md:.1f}')
    ax2.axhline(md + 1.96 * sd, color='gray', ls='--')
    ax2.axhline(md - 1.96 * sd, color='gray', ls='--')
    ax2.set_title('Bland-Altman Agreement', fontweight='bold')
    ax2.legend()
    plt.tight_layout()
    _render(save_path, fig)


def plot_variance(df, save_path='subject_variance.png'):
    """Inter-subject blink-duration variability."""
    fig = plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='subject_id', y='duration', palette='Spectral')
    plt.xticks(rotation=45)
    plt.ylabel('Blink Duration (ms)')
    plt.title('Inter-Subject Duration Variability')
    plt.tight_layout()
    _render(save_path, fig)


def plot_performance_metrics(subject_df, save_path='performance.png'):
    """Recall/Precision/Accuracy/F1 distribution, split by eyewear."""
    if subject_df.empty:
        return
    melted = subject_df.melt(id_vars=['subject_id', 'eyewear'],
                             value_vars=['Recall', 'Precision', 'Accuracy', 'F1'],
                             var_name='Metric', value_name='Score')
    fig = plt.figure(figsize=(10, 5))
    sns.boxplot(data=melted, x='Metric', y='Score', hue='eyewear',
                palette={'No Glasses': 'royalblue', 'Glasses': 'crimson'})
    plt.title('Algorithm Performance Distribution: Impact of Glasses', fontweight='bold')
    plt.ylim(0, 1.05)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    _render(save_path, fig)


def plot_split_half(paired_df, p_val, save_path='split_half_time.png'):
    """First-50% vs last-50% blink-duration spaghetti + boxplot with significance."""
    fig = plt.figure(figsize=(7, 6))
    for _, row in paired_df.iterrows():
        color = 'crimson' if row['Value_late'] > row['Value_early'] else 'royalblue'
        plt.plot([0, 1], [row['Value_early'], row['Value_late']], color=color, alpha=0.4, linewidth=1)
    plot_data = pd.concat([
        paired_df[['subject_id', 'Value_early']].rename(columns={'Value_early': 'Value'}).assign(Condition='First 50%'),
        paired_df[['subject_id', 'Value_late']].rename(columns={'Value_late': 'Value'}).assign(Condition='Last 50%'),
    ])
    sns.boxplot(data=plot_data, x='Condition', y='Value', palette='Pastel1', width=0.3)
    y_max = plot_data['Value'].max()
    h = y_max * 0.05
    plt.plot([0, 0, 1, 1], [y_max + h, y_max + 2 * h, y_max + 2 * h, y_max + h], lw=1.5, c='k')
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    plt.text(0.5, y_max + 2.5 * h, f"{sig}\n(p={p_val:.4f})", ha='center', va='bottom', fontweight='bold')
    plt.title('Fatigue Analysis: First vs. Last 50% Time', fontweight='bold')
    plt.ylabel('Mean Blink Duration (ms)')
    plt.ylim(top=y_max + 5 * h)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    _render(save_path, fig)


# --- Saccade / tracking figures ---------------------------------------------

def plot_task_performance(xy_pred, xy_gt, fs, task_name, saccade_vel_thresh,
                          target_radius, target_x=None, target_y=None, save_path=None):
    """3-panel X / Y / velocity tracking comparison (Output vs EyeLink)."""
    t1 = np.arange(xy_pred.shape[1])
    vel_p = calculate_velocity(fs, xy_pred)
    vel_g = calculate_velocity(fs, xy_gt)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    ax1.plot(t1, xy_pred[0], label='Smartphone Gaze', color='blue', alpha=0.8)
    ax1.plot(t1, xy_gt[0], color='red', linestyle='--', label='Ground Truth', alpha=0.6)
    ax1.set_ylabel('X Position (px)')
    ax1.set_title(f'{task_name} Task - Horizontal Tracking', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':')

    ax2.plot(t1, xy_pred[1], label='Smartphone Gaze', color='blue', alpha=0.8)
    ax2.plot(t1, xy_gt[1], color='red', linestyle='--', alpha=0.6)
    ax2.set_ylabel('Y Position (px)')
    ax2.set_xlabel('Time (Frames)')
    ax2.grid(True, linestyle=':')

    if target_x is not None and target_y is not None:
        t2 = np.arange(len(target_x))
        ax1.plot(t2, target_x, label='Correct Area', color='black', linestyle='--', linewidth=2)
        ax2.plot(t2, target_y, label='Correct Area', color='black', linestyle='--', linewidth=2)
        ax1.fill_between(t2, target_x - target_radius, target_x + target_radius,
                         color='green', alpha=0.1, label='Target Zone')
        ax2.fill_between(t2, target_y - target_radius, target_y + target_radius,
                         color='green', alpha=0.1)

    ax3.plot(t1, vel_g, color='red', linestyle='--', alpha=0.4, label='Ground Truth Speed')
    ax3.plot(t1, vel_p, color='blue', label='Smartphone Speed', linewidth=1)
    ax3.axhline(saccade_vel_thresh, color='gold', label='Saccade Threshold', linewidth=1)
    ax3.set_ylabel("Velocity (px/s)")
    ax3.set_xlabel("Time (Frames)")
    ax3.set_title(f"{task_name} Task: Velocity Profile", fontweight='bold', fontsize=10)
    ax3.legend(loc='upper right', fontsize='small')
    ax3.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    _render(save_path, fig)


def plot_aggregate_tracking(fs, subject_pred_list, subject_gt_list, frames_norm_list,
                            output_dir, task_name="Tracking Task", target_trace=None,
                            max_shift_ms=1000):
    """Align all subjects to the stimulus/correlation and plot mean +/- SD traces."""
    n_subs = len(subject_pred_list)
    max_len = max(s.shape[0] for s in subject_pred_list)
    aligned_preds = np.full((n_subs, max_len), np.nan)
    aligned_gts = np.full((n_subs, max_len), np.nan)
    aligned_fs, gts, preds, corrs = [], [], [], []
    align_to = 'Stimulus Onset'

    print(f"Aligning {n_subs} subjects to the Stimulus Trace...")
    for i in range(n_subs):
        gt, pred, frames = subject_gt_list[i], subject_pred_list[i], frames_norm_list[i]
        if 'Smooth Pursuit' in task_name:
            align_to, corrs = 'Stimulus Onset', 0
            g, p, f, lag_frames = gt, pred, frames, 0
            aligned_gts[i, frames] = gt
            aligned_preds[i, frames] = pred
        elif 'Optokinetic Nystagmus' in task_name:
            align_to = 'Maximum Average Correlate'
            f, p = pad_frames_with_rules(frames, pred, max_len)
            f, g = pad_frames_with_rules(frames, gt, max_len)
            n_points = min(len(g), len(target_trace))
            g_centered = g[:n_points] - np.mean(g[:n_points])
            t_centered = target_trace[:n_points] - np.mean(target_trace[:n_points])
            corr = correlate(t_centered, g_centered, mode='full')
            corrs.append(np.max(corr))
            lags = np.arange(-(n_points - 1), n_points)
            lag_frames = lags[np.argmax(corr)]
            if abs(lag_frames) > (max_shift_ms / (1000 / fs)):
                lag_frames = 0
            aligned_gts[i, :] = shift_array(g, lag_frames, max_len)
            aligned_preds[i, :] = shift_array(p, lag_frames, max_len)
        gts.append(g)
        preds.append(p)
        aligned_fs.append((np.array(f) + lag_frames) * (1000.0 / fs))

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_gt, std_gt = np.nanmean(aligned_gts, axis=0), np.nanstd(aligned_gts, axis=0)
        mean_pred, std_pred = np.nanmean(aligned_preds, axis=0), np.nanstd(aligned_preds, axis=0)
    time = np.arange(max_len) * (1000.0 / fs)

    fig = plt.figure(figsize=(15, 6))
    for i in range(n_subs):
        plt.plot(aligned_fs[i], preds[i], color='blue', alpha=0.5, lw=0.5)
        plt.plot(aligned_fs[i], gts[i], color='red', alpha=0.5, lw=0.5)
    plt.plot(time, mean_gt, '#800000', lw=2.5, label='Mean EyeLink')
    plt.fill_between(time, mean_gt - std_gt, mean_gt + std_gt, color='red', alpha=0.15, label='EyeLink SD')
    plt.plot(time, mean_pred, '#000080', lw=2.5, label='Mean Smartphone')
    plt.fill_between(time, mean_pred - std_pred, mean_pred + std_pred, color='blue', alpha=0.15, label='Smartphone SD')
    mean_corr = np.mean(corrs) if np.ndim(corrs) else corrs
    plt.title(f"{task_name} (Aligned to {align_to} {mean_corr:.0f})", fontsize=14, fontweight='bold')
    plt.xlabel("Time (ms)")
    plt.ylabel("Horizontal Position (px)")
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    _render(os.path.join(output_dir, f'{task_name} {mean_corr:.0f}.jpg'), fig)


# --- Heatmap / scanpath helpers (reused by HeatmapAnalyzer) ------------------

def visualize_target(ax, x, y, radius):
    """Draw a dashed target circle + center cross on an axis."""
    import matplotlib.patches as patches
    ax.add_patch(patches.Circle((x, y), radius, linewidth=2, edgecolor='red',
                                facecolor='none', linestyle='--'))
    ax.plot(x, y, 'rx', markersize=5)


def plot_scanpath(ax, xy_data, height, color='red', label='Gaze', alpha=0.6, step=100):
    """Chronological scanpath: faint saccade lines + sampled fixation dots."""
    if xy_data is None or xy_data.shape[1] == 0:
        return
    x = xy_data[0]
    y = height - xy_data[1]
    ax.plot(x, y, color=color, linewidth=1, alpha=alpha / 2, zorder=1)
    ax.scatter(x[::step], y[::step], s=40, color=color, edgecolor='white',
               alpha=alpha, label=label, zorder=2)
    for i, (xi, yi) in enumerate(zip(x[::step][:10], y[::step][:10])):
        ax.text(xi + 5, yi + 5, str(i + 1), color=color, fontsize=9, fontweight='bold')


def plot_error_vectors(ax, xy_out, xy_gt, height, step=200):
    """Connect Output to GT points at sampled times, colored by error magnitude."""
    if xy_out is None or xy_gt is None:
        return
    n_points = min(xy_out.shape[1], xy_gt.shape[1])
    for idx in np.arange(0, n_points, step):
        x_out, y_out = xy_out[0, idx], height - xy_out[1, idx]
        x_gt, y_gt = xy_gt[0, idx], height - xy_gt[1, idx]
        error = np.sqrt((x_out - x_gt) ** 2 + (y_out - y_gt) ** 2)
        color = 'green' if error < 100 else 'orange' if error < 200 else 'red'
        ax.plot([x_gt, x_out], [y_gt, y_out], color=color, alpha=0.6, linewidth=1)
        ax.scatter(x_out, y_out, color=color, s=10, alpha=0.5)


def plot_bivariate_contours(ax, xy_out, xy_gt, width, height, bg_image=None):
    """KDE contour overlay comparing Output vs GT gaze density."""
    if xy_out is None or xy_gt is None:
        return
    df_out = pd.DataFrame({'x': xy_out[0], 'y': height - xy_out[1]})
    df_gt = pd.DataFrame({'x': xy_gt[0], 'y': height - xy_gt[1]})
    if bg_image is not None:
        ax.imshow(np.flipud(bg_image), origin='lower', alpha=0.6)
    gt_colors = ["#ff9999", "#ff0000", "#800000"]
    pred_colors = ["#80b3ff", "#0044ff", "#000080"]
    try:
        sns.kdeplot(data=df_gt, x='x', y='y', ax=ax, levels=3, thresh=0.1,
                    colors=gt_colors, linewidths=2.5, linestyles='--', alpha=0.9)
    except Exception as e:
        print(f"Skipping GT KDE: {e}")
    try:
        sns.kdeplot(data=df_out, x='x', y='y', ax=ax, levels=3, thresh=0.1,
                    colors=pred_colors, linewidths=2.5, linestyles='-', alpha=0.9)
    except Exception as e:
        print(f"Skipping Pred KDE: {e}")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color='#800000', lw=2.5, linestyle='--', label='EyeLink (GT)'),
        Line2D([0], [0], color='#000080', lw=2.5, linestyle='-', label='Smartphone'),
    ], loc='upper right', fontsize='small', framealpha=0.9)


def calculate_dwell_time(ax, xy_out, xy_gt, target_x, target_y, radius, sample_rate=30):
    """Time inside a circular AOI for Output and GT, with a presence-ribbon plot.

    Returns ``(dt_out_ms, pct_out, dt_gt_ms, pct_gt)``.
    """
    if xy_out is None or xy_out.shape[1] == 0 or xy_gt is None or xy_gt.shape[1] == 0:
        return 0, 0, 0.0, 0.0
    radius_sq = radius ** 2
    ms_per_sample = 1000 / sample_rate
    presence_out = (xy_out[0] - target_x) ** 2 + (xy_out[1] - target_y) ** 2 <= radius_sq
    presence_gt = (xy_gt[0] - target_x) ** 2 + (xy_gt[1] - target_y) ** 2 <= radius_sq
    dt_out = np.sum(presence_out) * ms_per_sample
    dt_gt = np.sum(presence_gt) * ms_per_sample
    p_out = (np.sum(presence_out) / len(presence_out)) * 100 if len(presence_out) > 0 else 0
    p_gt = (np.sum(presence_gt) / len(presence_gt)) * 100 if len(presence_gt) > 0 else 0

    if ax is not None:
        time_axis = np.arange(len(presence_out)) * ms_per_sample
        ax.fill_between(time_axis, 0.6, 0.6 + (presence_out * 0.4), color='royalblue',
                        alpha=0.7, label=f'Out: {dt_out:.0f}ms')
        ax.fill_between(time_axis, 0.1, 0.1 + (presence_gt * 0.4), color='crimson',
                        alpha=0.7, label=f'GT: {dt_gt:.0f}ms')
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0.3, 0.8])
        ax.set_yticklabels(['EyeLink', 'Smartphone'])
        ax.set_xlabel("Time (ms)")
        ax.set_title("AOI Gaze Presence")
        ax.legend(loc='upper right', fontsize='x-small', frameon=True)
        ax.grid(axis='x', linestyle='--', alpha=0.4)
    return dt_out, p_out, dt_gt, p_gt


# --- Generic comparison figures ----------------------------------------------

def plot_spatial_variance_heatmap(xy_data, fixs, width=None, height=None,
                                  title="Fixation Spatial Distribution", save_path=None):
    """Hexbin density of gaze with fixation-centroid crosses."""
    width = width or config.SCREEN_RES_PX[0]
    height = height or config.SCREEN_RES_PX[1]
    fig = plt.figure(figsize=(16, 9))
    hb = plt.hexbin(xy_data[0], height - xy_data[1], gridsize=30, cmap='inferno', mincnt=1)
    plt.colorbar(hb, label='Fixation Density')
    for s, e in fixs:
        plt.plot(np.mean(xy_data[0, s:e]), height - np.mean(xy_data[1, s:e]),
                 'w+', markersize=10, alpha=0.7)
    plt.title(title + "\n(White crosses = Fixation Centroids)", fontweight='bold')
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.tight_layout()
    _render(save_path, fig)


def plot_bland_altman(metric_name, smart_data, gt_data, save_path=None):
    """Bland-Altman agreement plot (Smartphone vs EyeLink); returns (bias, lo, hi)."""
    smart = np.array(smart_data, dtype=float)
    gt = np.array(gt_data, dtype=float)
    valid = (~np.isnan(smart)) & (~np.isnan(gt))
    smart, gt = smart[valid], gt[valid]
    means = np.mean([smart, gt], axis=0)
    diffs = smart - gt
    bias, sd_diff = np.mean(diffs), np.std(diffs, ddof=1)
    upper_loa, lower_loa = bias + 1.96 * sd_diff, bias - 1.96 * sd_diff

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(means, diffs, alpha=0.7, s=60, color='blue', edgecolor='white')
    ax.axhline(bias, color='red', linestyle='-', linewidth=2, label='Mean Bias')
    ax.axhline(upper_loa, color='black', linestyle='--', linewidth=1.5, label='95% LoA')
    ax.axhline(lower_loa, color='black', linestyle='--', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    x_pos = np.max(means) * 0.98
    ax.text(x_pos, bias, f'Bias: {bias:.1f}', color='red', va='center')
    ax.text(x_pos, upper_loa, f'+1.96 SD: {upper_loa:.1f}', color='black', va='bottom')
    ax.text(x_pos, lower_loa, f'-1.96 SD: {lower_loa:.1f}', color='black', va='top')
    ax.set_title(f"Bland-Altman Plot: {metric_name}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Mean of Smartphone and EyeLink")
    ax.set_ylabel("Difference (Smartphone - EyeLink)")
    ax.set_xlim(np.min(means) * 0.95, np.max(means) * 1.15)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    _render(save_path, fig)
    print(f"--- Bland-Altman: {metric_name} --- Bias {bias:.2f}, "
          f"95% LoA [{lower_loa:.2f}, {upper_loa:.2f}], SD {sd_diff:.2f}")
    return bias, lower_loa, upper_loa


def plot_task_pair(task_name, metric_name, arr1, arr2=None, save_path=None):
    """Line comparison of an Output/EyeLink metric across one or two tasks."""
    fig = plt.figure(figsize=(15, 4))
    if arr2 is None:
        print_summary_statistics(f"{metric_name} {task_name}", arr1[0], arr1[1])
        plt.plot(np.arange(len(arr1[0])), arr1[0], label=f"{task_name} (Output)")
        plt.plot(np.arange(len(arr1[0])), arr1[1], label=f"{task_name} (EyeLink)")
    else:
        print_summary_statistics(f"{metric_name} All", np.concatenate((arr1[0], arr1[1])),
                                 np.concatenate((arr2[0], arr2[1])))
        n0 = len(arr1[0])
        plt.plot(np.arange(n0), arr1[0], label=f"{task_name[0]} (Output)")
        plt.plot(np.arange(n0), arr2[0], label=f"{task_name[0]} (EyeLink)")
        plt.plot(np.arange(n0, n0 + len(arr1[1])), arr1[1], label=f"{task_name[1]} (Output)")
        plt.plot(np.arange(n0, n0 + len(arr1[1])), arr2[1], label=f"{task_name[1]} (EyeLink)")
    plt.legend(loc='upper right', fontsize='small')
    plt.title(f"{metric_name}", fontweight='bold', fontsize=10)
    plt.tight_layout()
    _render(save_path, fig)
