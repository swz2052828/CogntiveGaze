"""Iris detection + template-match tracking (upstream of the gaze metrics).

Ported from ``IrisTracker_swz``. The original ``run_tracking`` referenced
several names that are undefined in its own scope (``tx``/``ty`` start point,
the template width/height ``t_w``/``t_h``, and it used the *list* returned by
``save_template`` as a single template). Those are the kind of "clear crasher"
fixes requested -- each is marked FIX: -- but note this module is the least
complete of the uploads and its tracking logic should be re-validated on real
video before trusting its output.
"""

import os

import numpy as np
import cv2 as cv
from skimage.filters import gaussian
from skimage.segmentation import active_contour

from .. import plotting

plt = plotting.plt


def run_acm(img, alpha=0.1, beta=1.0, gamma=0.03):
    """Active-contour fit of the eye region."""
    s = np.linspace(0, 2 * np.pi, 400)
    r = img.shape[0] * 0.5 + img.shape[0] * 0.45 * np.sin(s)
    c = img.shape[1] * 0.5 + img.shape[1] * 0.45 * np.cos(s)
    init = np.array([r, c]).T
    return active_contour(gaussian(img, 3, preserve_range=False), init,
                          alpha=alpha, beta=beta, gamma=gamma)


def get_roi_mask(img, snake):
    """Bounding-box ROI mask from a snake contour (background whited out)."""
    mask = np.zeros(img.shape, dtype=np.uint8)
    y_min, x_min = np.min(snake, axis=0).astype(int)
    y_max, x_max = np.max(snake, axis=0).astype(int)
    y_min, x_min = max(0, y_min), max(0, x_min)
    mask[y_min:y_max, x_min:x_max] = 255
    masked_img = img.copy()
    masked_img[mask == 0] = 255
    return masked_img


def detect_hough_circles(img, params=[350, 10, 13, 20]):
    """Hough-transform circle (iris) detection."""
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 40,
                              param1=params[0], param2=params[1],
                              minRadius=params[2], maxRadius=params[3])
    if circles is not None:
        print(f"Found {len(circles)} circles in template frame.")
        return circles[0, :]
    print("No circles found in template frame.")
    return None


def select_best_circle(img, circles, ratio_thresh=0.6):
    """Pick the circle with the darkest interior (most iris-like)."""
    bw = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv.THRESH_BINARY, 11, 2)
    for c in circles:
        cx, cy, r = c
        mask = np.zeros_like(bw)
        cv.circle(mask, (cx, cy), r, 255, -1)
        total = np.sum(mask == 255)
        if total == 0:
            continue
        if np.sum((bw == 0) & (mask == 255)) / total > ratio_thresh:
            return c
    return None


def show_template(frame, template, circle_info=None, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(frame, cmap='gray')
    if circle_info is not None:
        cx, cy, cr = circle_info
        ax[0].add_patch(plt.Circle((cx, cy), cr, color='r', fill=False, linewidth=2))
        ax[0].set_title(f"Initial Detection (x, y): ({cx:.1f}, {cy:.1f}) r: {cr:.1f}")
    ax[1].imshow(template, cmap='gray')
    ax[1].set_title(f"Extracted Template Shape: {template.shape}")
    plotting._render(save_path, fig)


def show_tracking_step(crop, loc, w, h, idx, score, save_path=None):
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(crop, cmap='gray')
    ax.add_patch(Rectangle(loc, w, h, edgecolor='r', facecolor='none'))
    ax.set_title(f"Frame {idx} | Score: {score:.2f}")
    plotting._render(save_path, fig)


class IrisTracker:
    def __init__(self, loader, directory_path, visualization=True):
        self.loader = loader
        self.directory_path = directory_path
        self.sub_id = loader.sub_id
        self.visualize = visualization
        self.roi_margins = {'top': 0.0, 'bot': 1.0, 'left': 0.1, 'right': 0.6}

    def run_tracking(self, frames, suffix, offset=(0, 0), params=[180, 10, 5, 15],
                     frame_list=None, template_id=None, box_size=(400, 250),
                     start_xy=(0, 0), match_thresh=0.48):
        """Template-match the iris across ``frames``; 0,0 marks a lost/blink frame."""
        if len(frames) == 0:
            print("No frames loaded.")
            return

        templates = self.save_template(suffix, offset, params, frame_list, template_id)
        if not templates:
            print("No template produced; aborting tracking.")
            return
        template = templates[0]                  # FIX: original used the list as a template
        t_h, t_w = template.shape[:2]            # FIX: original referenced undefined t_w/t_h

        pts_e0, pts_e1, match_scores = [], [], []
        tx, ty = start_xy                        # FIX: original referenced undefined tx/ty
        for i, frame in enumerate(frames):
            if i == 0:
                curr_x, curr_y = tx, ty
            else:
                curr_x, curr_y = pts_e0[-1], pts_e1[-1]
                if curr_x == 0:
                    curr_x, curr_y = tx, ty
            search_h, search_w = box_size[1], box_size[0]
            y_min = max(0, curr_y - search_h // 2)
            x_min = max(0, curr_x - search_w // 2)
            crop = frame[y_min:y_min + search_h, x_min:x_min + search_w]

            res = cv.matchTemplate(crop, template, cv.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv.minMaxLoc(res)
            if max_val >= match_thresh:
                pts_e0.append(x_min + max_loc[0])
                pts_e1.append(y_min + max_loc[1])
            else:
                pts_e0.append(0)
                pts_e1.append(0)
            match_scores.append(max_val)

            if self.visualize and i % 5 == 0:
                show_tracking_step(crop, (max_loc[0], max_loc[1]), t_w, t_h, i, max_val,
                                   save_path=f"{self.directory_path}/track_{self.sub_id}_{i}.png")
        return np.array(pts_e0), np.array(pts_e1), np.array(match_scores)

    def save_template(self, suffix, offset=(0, 0), params=[180, 10, 5, 15],
                      frame_list=None, save_id=None):
        """Extract iris-crop templates from reference frames; optionally save one."""
        if frame_list is None:
            frame_list = np.arange(self.loader.t_init, self.loader.t_init + 300, 30)
        print(f"Generating Template {frame_list}...")
        template_list, frame_indice = [], []
        tr = 0
        for i, frame_id in enumerate(frame_list):
            img, frame_idx = self.loader.load_frames(self.directory_path, frame_idx=frame_id)
            circles = detect_hough_circles(img, params)
            if circles is None:
                return None
            tx, ty, tr = circles[0]
            tx += offset[0]
            ty += offset[1]
            left, right, top, bot = np.uint16(np.around([tx - tr, tx + tr, ty - tr, ty + tr]))
            template = img[top:bot, left:right]
            if self.visualize:
                show_template(img, template, (tx, ty, tr),
                              save_path=f"{self.directory_path}/template_{self.sub_id}_{i}.png")
            template_list.append(template)
            frame_indice.append(frame_idx)

        if save_id is not None and save_id > 0:
            template_name = f"{self.loader.output_prefix}_{frame_indice[save_id]}_{suffix}_{tr:.0f}.png"
            cv.imwrite(os.path.join(self.loader.dir_path, template_name), template_list[save_id])
            print(f"Saved {suffix} to {template_name}")
        return template_list

    def save_results(self, frame_start, pts_e0, pts_e1):
        """Persist tracked points, preserving any existing head data."""
        filename = f"{self.loader.output_prefix}_{frame_start}"
        try:
            with open(filename, "rb") as f:
                head_0 = np.load(f)
                head_1 = np.load(f)
                _ = np.load(f)
                _ = np.load(f)
        except FileNotFoundError:
            print("Warning: Previous file not found. Creating new with dummy head data.")
            head_0 = np.zeros_like(pts_e0)
            head_1 = np.zeros_like(pts_e1)
        with open(filename, "wb") as f:
            np.save(f, head_0)
            np.save(f, head_1)
            np.save(f, pts_e0)
            np.save(f, pts_e1)
        print(f"Saved results to {filename}")
