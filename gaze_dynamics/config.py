"""Central configuration for the gaze-dynamics pipeline.

Every magic number that used to be scattered across the analysis scripts lives
here: screen geometry, sampling rate, the per-subject acquisition tables, the
experiment task timeline, and the saccade task target layouts. Keeping them in
one place is the main point of the refactor -- the analyzers and loaders read
from here instead of re-declaring constants.
"""

from dataclasses import dataclass, field

import numpy as np


# --- Screen / sampling -------------------------------------------------------

SCREEN_RES_PX = (1920, 1080)      # pixels (W, H)
SCREEN_SIZE_CM = (54.4, 30.4)     # physical size (W, H) in cm
SAMPLE_RATE = 30                  # gaze fps
SACCADE_VEL_THRESH = 70           # I-VT velocity threshold (px/frame * fps)


# --- Cohort ------------------------------------------------------------------

SUBJECT_IDS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
GLASSES_IDS = [7, 10, 12]

# Per-frame synchronization offsets used by the blink analysis (subject -> frame).
OFFSETS = {
    6: 5462, 7: 14998, 8: 12333, 9: 4851, 10: 8560, 11: 7988, 12: 3185, 13: 3926,
    14: 5823, 15: 4665, 16: 3352, 17: 5224, 18: 12839, 19: 4358, 20: 7889, 21: 4042,
    22: 3980, 23: 4074,
}


# --- Per-subject acquisition tables (indexed by position in SUBJECT_IDS) -----

CALIB_BEGINS = [3750, 13170, 9900, 3200, 4950, 6300, 1500, 2400, 3600, 2850,
                1800, 2250, 11250, 2550, 4500, 600, 2400, 2250]
S2FS = [11078.88, 49125.57, 44948.15, 7440.77, 27077.76, 110421.82,
        30904.21, 29722.62, 3227.32, 3629.49, 38569.85, 22729.22,
        24844.96, 16167.6, 19825.7, 26542.83, 26700.3, 30231.27]
T_INITS = [5462, 14998, 12333, 4851, 8560, 7988, 3185, 3926, 5823, 4665,
           3352, 5224, 12839, 4358, 7889, 4042, 3980, 4074]
R_LIMS = [80, 70, 180, 150, 80, 150, 90, 110, 170, 80, 130, 100, 120,
          90, 80, 90, 140, 80]


def build_task_timeline():
    """Reconstruct the global task timeline shared by every subject.

    Returns
    -------
    task_begin : np.ndarray  (47,)  frame index each task block starts at
    task_len   : np.ndarray  (47,)  block length in frames
    sacc       : np.ndarray  (39,)  saccade stimulus onset frames
    """
    a = np.array([0, 1050, 2490, 4230, 5280, 6540])
    start_b = 6540 + 7 * 30
    b = np.arange(start_b, start_b + 29 * 180, 180, dtype=int)
    start_c = 12570 + 6 * 30 + 450
    c = np.r_[12570 + 5 * 30, np.arange(start_c, start_c + 5 * 450, 450, dtype=int)]
    start_d = 12570 + 94 * 30
    d = np.arange(start_d, start_d + 6 * 450, 450, dtype=int)
    task_begin = np.concatenate((a, b, c, d))

    e = np.array([11, 22, 37, 14, 11, 6])
    f = np.full(29, 5)
    g = np.r_[9, np.full(11, 8)]
    task_len = (np.concatenate((e, f, g)) * 30 + 10).astype(int)

    sacc = np.arange(9, 9 + 39 * 30, 30, dtype=int)
    return task_begin, task_len, sacc


# --- Saccade task target layouts ---------------------------------------------

TASK_NAMES = ['Fixation', 'Pro-Saccade', 'Anti-Saccade', 'Smooth Pursuit',
              'Optokinetic Nystagmus']


def build_saccade_tasks(screen_px=SCREEN_RES_PX):
    """Build (stim_onsets, target_layouts, task_names) for the saccade tasks.

    target_layouts[task] is a list of [x, y, duration_frames] target stops.
    """
    stim_onsets = [
        np.array([0]),
        np.arange(90, 90 + 9 * 60, 60, dtype=int),
        np.arange(90, 90 + 16 * 60, 60, dtype=int),
        np.array([0, 90, 240]),
        np.array([0]),
    ]
    ref = np.array(screen_px)
    cent, mid = ref * 0.5
    right, bot = ref * 0.9
    left, top = ref * 0.1
    target_layouts = [
        [[cent, mid, 300]],
        [[cent, mid, 90], [right, top, 60], [right, bot, 60], [left, top, 60], [left, bot, 60]],
        [[cent, mid, 90], [left, bot, 60], [cent, mid, 60], [right, mid, 60], [cent, mid, 60],
         [cent, top, 60], [cent, mid, 60], [right, top, 60], [cent, mid, 60], [cent, bot, 60],
         [cent, mid, 60], [right, bot, 60], [cent, mid, 60], [left, mid, 60], [cent, mid, 60],
         [left, top, 60], [cent, mid, 90]],
        [[cent, mid, 90]],
        [[cent, mid, 10]],
    ]
    return stim_onsets, target_layouts, list(TASK_NAMES)


# --- Bundled config object ---------------------------------------------------

@dataclass
class GazeConfig:
    """One object the pipeline threads through every analyzer."""
    data_path: str = "data"
    screen_res_px: tuple = SCREEN_RES_PX
    screen_size_cm: tuple = SCREEN_SIZE_CM
    sample_rate: int = SAMPLE_RATE
    saccade_vel_thresh: int = SACCADE_VEL_THRESH
    subject_ids: list = field(default_factory=lambda: list(SUBJECT_IDS))
    glasses_ids: list = field(default_factory=lambda: list(GLASSES_IDS))

    @property
    def width(self):
        return self.screen_res_px[0]

    @property
    def height(self):
        return self.screen_res_px[1]
