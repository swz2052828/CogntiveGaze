"""Data access: subject loader + gaze/background file readers.

Ported from ``GazeDataLoader_swz``. The only structural change is that the
per-subject acquisition tables and the global task timeline now come from
``config`` instead of being inlined in ``__init__``.
"""

import os
import glob
import re

import numpy as np
import cv2 as cv
from PIL import Image
from tqdm import tqdm

from . import config


class GazeDataLoader:
    def __init__(self, data_path, idx):
        self.data_path = data_path

        self.task_begin, self.task_len, self.sacc = config.build_task_timeline()

        self.sub_id = config.SUBJECT_IDS[idx]
        self.calib = config.CALIB_BEGINS[idx]
        self.s2f = config.S2FS[idx]
        self.t_init = config.T_INITS[idx]
        self.r_lim = config.R_LIMS[idx]

        self.dir_path = f"{data_path}/subid_{self.sub_id}"
        self.output_prefix = f"subid_{self.sub_id}"

    def load_ascii_data(self, rmblink=20):
        """Read the EyeLink ASCII file: blinks, validation points, gaze (xyt)."""
        filepath = f'{self.data_path}/ASCII/EXPFILE_{self.sub_id}.asc'
        blink_stamp, validate_stamp, pts, xyt = [], [], [], []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    cols = line.split()
                    if len(cols) < 5:
                        continue
                    if cols[0].isdigit() and float(cols[3]) != 0:
                        xyt.append([float(cols[1]), float(cols[2]), int(cols[0])])
                    elif cols[0] == 'EBLINK':
                        blink_stamp.append([int(cols[2]) - 2 * rmblink, int(cols[3]) + 2 * rmblink])
                    elif cols[0] == 'MSG' and cols[2] == 'VALIDATE':
                        validate_stamp.append(int(cols[1]))
                        ox, oy = cols[8].split(',')
                        pts.append([int(ox), int(oy)])
        except FileNotFoundError:
            print(f"Error: File {filepath} not found.")
            return None, None, None, None
        return (np.array(blink_stamp), np.array(validate_stamp),
                np.array(pts), np.array(xyt).T)

    def load_task_binary(self, task_id):
        """Load one task's binary (head_0, head_1, pts_0, pts_1)."""
        prefix = f"{self.data_path}/subid_{self.sub_id}/subid_{self.sub_id}"
        if task_id == self.calib:
            filename = f"{prefix}_calib_{self.calib}"
        else:
            filename = f"{prefix}_{self.task_begin[task_id]}"
        try:
            print('Reading task file:', filename)
            with open(filename, "rb") as f:
                head_0 = np.load(f)
                head_1 = np.load(f)
                pts_0 = np.load(f)
                pts_1 = np.load(f)
            if task_id == self.calib:
                return head_0, head_1, pts_0, pts_1
            if pts_0.shape[0] - self.task_len[task_id] != 21:
                print('!WARNING!The length of this file is incorrect!!!')
            return np.stack((pts_0, pts_1, head_0, head_1))
        except FileNotFoundError:
            print(f"Binary file not found: {filename}")
            return np.zeros((4, self.task_len[task_id]))

    def load_frames(self, directory_path, frame_idx=None, suffix=None):
        """Read grayscale frames from ``directory_path`` (single frame or glob)."""
        if frame_idx is None:
            if suffix is None:
                search_path = os.path.join(directory_path, "*.jpg")
            else:
                file_pattern = f"{self.output_prefix}_*_{suffix}_*.png"
                search_path = os.path.join(directory_path, file_pattern)
            file_paths = glob.glob(search_path)
        else:
            if not isinstance(frame_idx, (int, np.integer)):
                print(f"Invalid frame_idx: {frame_idx}. It must be a single integer.")
                return None, None
            search_path = os.path.join(directory_path, f"{frame_idx:05d}.jpg")
            file_paths = [search_path]

        if not file_paths:
            print(f"No images found in {directory_path}")
            return None, None

        try:
            file_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f)))))
        except ValueError:
            file_paths.sort()
        print(f"Found {len(file_paths)} frames in {directory_path}")

        if frame_idx is not None:
            img = cv.imread(search_path, cv.IMREAD_GRAYSCALE)
            print(f'Loading Completed! {search_path}')
            return img, frame_idx

        frames, frame_indices = [], []
        for fp in tqdm(file_paths, desc="Loading Frames", unit="frame"):
            img = cv.imread(fp, cv.IMREAD_GRAYSCALE)
            if img is None:
                tqdm.write(f"Warning: Could not read {fp}")
                continue
            try:
                name_body = os.path.splitext(os.path.basename(fp))[0]
                parts = name_body.split('_')
                if suffix is None:
                    parsed_idx = int(parts[-1])
                elif f"{suffix}" in parts:
                    parsed_idx = int(parts[parts.index(f"{suffix}") - 1])
                else:
                    print(f"Warning: Filename format invalid (missing '{suffix}' keyword).")
                    continue
            except Exception as e:
                print(f"Error parsing filename {fp}: {e}")
                continue
            frames.append(img)
            frame_indices.append(parsed_idx)
        print('Loading Completed!')
        return frames, np.array(frame_indices)

    def extract_frames(self, start_frame=0, duration_sec=40, fps=30):
        """Extract and 180-rotate grayscale frames from the subject video."""
        video_path = os.path.join(self.dir_path, f"{self.output_prefix}.mp4")
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        frames = []
        end_frame = start_frame + (duration_sec * fps)
        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Extracting frames {start_frame} to {end_frame}...")
        for _ in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frames.append(np.rot90(np.rot90(gray)))
        cap.release()
        return np.array(frames)


def load_gaze_data(heatmap_path, img_number_str, sub_id=None):
    """Load per-item gaze files: each holds [xy_smooth, xy_compare, frames].

    Returns ``(xy_output_list, xy_gaze_list, frame_list)`` or ``(None, None, None)``.
    """
    if sub_id is not None:
        files = [f"{sub_id}_{idx}" for idx in img_number_str]
    else:
        files = [f for f in os.listdir(heatmap_path)
                 if re.search(fr"_({img_number_str})$", os.path.splitext(f)[0])]
    print(f"Matched files for {img_number_str}:", files)
    if not files:
        return None, None, None

    xy_output_list, xy_gaze_list, frame_list = [], [], []
    for file in files:
        try:
            with open(os.path.join(heatmap_path, file), "rb") as f:
                xy_output_list.append(np.load(f))
                xy_gaze_list.append(np.load(f))
                frame_list.append(np.load(f))
        except FileNotFoundError:
            print(f"Gaze file not found: {file}")
            continue
    return xy_output_list, xy_gaze_list, frame_list


def load_background_image(bg_img_path_base, W, H, img_id):
    """Load a stimulus image, resize to height H preserving aspect, pad/crop to W."""
    bg_path = os.path.join(bg_img_path_base, f"{img_id}.jpg")
    if not os.path.exists(bg_path):
        print(f"Warning: Background image {bg_path} not found.")
        return np.zeros((H, W, 3), dtype=np.uint8)

    bg_image = Image.open(bg_path)
    aspect_ratio = bg_image.width / bg_image.height
    new_width = int(H * aspect_ratio)
    bg_image = np.array(bg_image.resize((new_width, H)))

    if new_width < W:
        pad_left = (W - new_width) // 2
        pad_right = W - new_width - pad_left
        if bg_image.ndim == 3:
            bg_image = np.pad(bg_image, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant')
        else:
            bg_image = np.pad(bg_image, ((0, 0), (pad_left, pad_right)), mode='constant')
    elif new_width > W:
        start = (new_width - W) // 2
        bg_image = bg_image[:, start:start + W]
    return bg_image


def identify_events(data):
    """Group consecutive integers into ``[start, duration]`` events."""
    if len(data) == 0:
        return []
    data = np.sort(data)
    events = []
    start_val = current_val = data[0]
    count = 1
    for i in range(1, len(data)):
        if data[i] == current_val + 1:
            current_val = data[i]
            count += 1
        else:
            events.append([start_val, count])
            start_val = current_val = data[i]
            count = 1
    events.append([start_val, count])
    return np.array(events)
