"""Spatial gaze analysis: heatmaps, entropy, dwell, scanpaths, KDE contours.

Ported from ``HeatmapAnalyzer_swz``. Low-level drawing helpers now live in
``plotting``; this class keeps the panel-orchestration loops. Two crashers from
the original ``plot_heatmaps`` / ``plot_new_heatmaps`` (they called
``load_background_image`` with one argument) are fixed by threading a
``bg_img_path_base`` through (flagged FIX:).
"""

import os

import numpy as np
from scipy.ndimage import gaussian_filter

from .. import config, plotting
from ..geometry import concat_xy
from ..io import load_gaze_data, load_background_image
from ..metrics import calculate_entropy


class HeatmapAnalyzer:
    def __init__(self, output_dir, W=None, H=None, sigma=40):
        self.output_dir = output_dir
        self.W = W or config.SCREEN_RES_PX[0]
        self.H = H or config.SCREEN_RES_PX[1]
        self.sigma = sigma
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_heatmap(self, xy_data):
        """Gaussian-blurred fixation density map from ``[2, N]`` int coords."""
        heatmap = np.zeros((self.H, self.W), dtype=np.float32)
        x_coords = np.clip(xy_data[0], 0, self.W - 1)
        y_coords = np.clip(xy_data[1], 0, self.H - 1)
        np.add.at(heatmap, (y_coords, x_coords), 1)
        return gaussian_filter(heatmap, sigma=self.sigma)

    def load_and_process(self, sub_list, heatmap_dir, img_indices, grid_bins=(64, 36)):
        """Build per-subject heatmaps + entropy for one image index."""
        xy, hm, ents = [], [], []
        for sub_id in sub_list:
            num = img_indices - 1
            xy_out, xy_gaze, _ = load_gaze_data(heatmap_dir, num, sub_id)
            if xy_out is None:
                continue
            if not xy_out:  # FIX: was `if xy_out == []` (clarity; lists only)
                ents.append([0, 0])
                continue
            copy_out = concat_xy(xy_out)
            copy_gt = concat_xy(xy_gaze)
            hm_out = self.generate_heatmap(copy_out)
            hm_gaze = self.generate_heatmap(copy_gt)
            ent1, ent2 = calculate_entropy(hm_out), calculate_entropy(hm_gaze)
            print(f"  Entropy: {ent1:.4f} (Output) {ent2:.4f} (EyeLink) bits")
            xy.append([xy_out, xy_gaze])
            hm.append([hm_out, hm_gaze])
            ents.append([ent1, ent2])
        return xy, np.array(hm), np.array(ents)

    def plot_figures(self, bg_img_path_base, xy, img_ids, targets=None):
        """Scanpath + error-vector + AOI-dwell panels per image with a target."""
        for i in range(len(img_ids)):
            xy_out, xy_gaze = xy[i]
            copy_out = concat_xy(xy_out)
            copy_gt = concat_xy(xy_gaze)
            if not (targets and i in targets):
                continue
            tx, ty, tr = targets[i]
            bg_image = load_background_image(bg_img_path_base, self.W, self.H, img_ids[i])
            fig, axes = plotting.plt.subplots(2, 2, figsize=(20, 12))

            axes[0, 0].imshow(np.flipud(bg_image), origin='lower')
            plotting.plot_scanpath(axes[0, 0], copy_out, self.H, color='blue', label='Smartphone')
            plotting.plot_scanpath(axes[0, 0], copy_gt, self.H, color='red', label='EyeLink')
            axes[0, 0].legend()

            axes[1, 0].imshow(np.flipud(bg_image), origin='lower')
            plotting.plot_error_vectors(axes[1, 0], copy_out, copy_gt, self.H)
            axes[1, 0].set_title("Spatial Deviation (Error Vectors)")

            axes[0, 1].imshow(np.flipud(bg_image), origin="lower")
            axes[0, 1].set_title(f"No. {i} ID: {img_ids[i]}")
            plotting.visualize_target(axes[0, 1], tx, ty, tr)
            axes[0, 1].text(10, self.H - 50, f"Target: ({tx},{ty})", color='red',
                            fontsize=10, backgroundcolor='white')

            dt_out, p_out, dt_gt, p_gt = plotting.calculate_dwell_time(
                axes[1, 1], copy_out, copy_gt, tx, self.H - ty, tr)
            print(f"  Dwell (Output): {dt_out:.0f}ms ({p_out:.1f}%)")
            print(f"  Dwell (EyeLink): {dt_gt:.0f}ms ({p_gt:.1f}%)")
            axes[0, 0].axis("off")
            axes[1, 0].axis("off")
            plotting.plt.tight_layout()
            plotting._render(os.path.join(self.output_dir, f'scanpath_{i}.jpg'), fig)

    def plot_heatmaps(self, bg_img_path_base, img_ids, hm_v, hm_s, targets=None):
        """5-panel heatmap overlays (original + FV/VS x Output/GT) per image.

        FIX: ``bg_img_path_base`` added; original called load_background_image
        with a single argument.
        """
        for i in range(len(img_ids)):
            hm_out1, hm_gaze1 = hm_v[i]
            hm_out2, hm_gaze2 = hm_s[i]
            bg_image = load_background_image(bg_img_path_base, self.W, self.H, img_ids[i])
            fig, axes = plotting.plt.subplots(1, 5, figsize=(20, 4))
            ax_orig, ax_fv_out, ax_fv_gaze, ax_vs_out, ax_vs_gaze = axes
            vmax1 = max(hm_out1.max(), hm_gaze1.max())
            vmax2 = max(hm_out2.max(), hm_gaze2.max())

            ax_orig.imshow(np.flipud(bg_image), origin="lower")
            for ax, hmap, vmax in [(ax_fv_out, hm_out1, vmax1), (ax_fv_gaze, hm_gaze1, vmax1),
                                   (ax_vs_out, hm_out2, vmax2), (ax_vs_gaze, hm_gaze2, vmax2)]:
                ax.imshow(np.flipud(bg_image), origin="lower")
                ax.imshow(np.flipud(hmap), cmap="jet", alpha=0.6, origin="lower", vmin=0, vmax=vmax)
            if targets and i in targets:
                tx, ty, tr = targets[i]
                plotting.visualize_target(ax_vs_out, tx, ty, tr)
                plotting.visualize_target(ax_vs_gaze, tx, ty, tr)
            for ax in axes:
                ax.axis("off")
            plotting.plt.tight_layout()
            plotting._render(os.path.join(self.output_dir, f'heatmap_analysis_{i}.jpg'), fig)

    def plot_new_heatmaps(self, bg_img_path_base, img_ids, xy_v, xy_s, targets=None):
        """3-panel KDE-contour comparison (original + FV + VS) per image.

        FIX: ``bg_img_path_base`` added (same one-argument bug as plot_heatmaps).
        """
        for i in range(len(img_ids)):
            xy_out1, xy_gaze1 = xy_v[i]
            xy_out2, xy_gaze2 = xy_s[i]
            copy_out1, copy_gt1 = concat_xy(xy_out1), concat_xy(xy_gaze1)
            copy_out2, copy_gt2 = concat_xy(xy_out2), concat_xy(xy_gaze2)
            bg_image = load_background_image(bg_img_path_base, self.W, self.H, img_ids[i])
            fig, (ax_orig, ax_fv, ax_vs) = plotting.plt.subplots(1, 3, figsize=(18, 5))

            ax_orig.imshow(np.flipud(bg_image), origin="lower")
            ax_orig.axis("off")
            if targets and i in targets:
                tx, ty, tr = targets[i]
                plotting.visualize_target(ax_orig, tx, ty, tr)
                ax_orig.text(10, self.H - 50, "Target Area", color='red', backgroundcolor='white')
            plotting.plot_bivariate_contours(ax_fv, copy_out1, copy_gt1, self.W, self.H, bg_image)
            plotting.plot_bivariate_contours(ax_vs, copy_out2, copy_gt2, self.W, self.H, bg_image)
            plotting.plt.tight_layout()
            plotting._render(os.path.join(self.output_dir, f'contour_analysis_{i}.jpg'), fig)
