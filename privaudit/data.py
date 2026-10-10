"""Release layout and frame sampling.

Layout: <root>/<participant>/<stream>/<frame>.{jpg,png}. Frame names must sort
in temporal order (GazeCapture and CognitiveGaze use zero-padded frame indices).

Sampling is deliberately identical to the CognitiveGaze attacks so that this
package reproduces their numbers: ONE generator seeded once, consumed in sorted
participant order; per participant, n frames from the first temporal half and n
from the second. Changing the order or the seed changes which frames are drawn.
"""
from pathlib import Path

import cv2
import numpy as np

EXTS = (".jpg", ".jpeg", ".png")


def participants(root, stream):
    root = Path(root)
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and (d / stream).is_dir())


def frames(root, pid, stream):
    d = Path(root) / pid / stream
    return sorted(p.name for p in d.iterdir() if p.suffix.lower() in EXTS)


def half_sample(root, stream, n_per_half, seed=0, pids=None):
    """{pid: (first_half_names, second_half_names)}; participants with fewer than
    4*n frames are skipped (and reported by the caller), never down-sampled."""
    rng = np.random.default_rng(seed)
    out, skipped = {}, []
    for pid in (pids or participants(root, stream)):
        names = frames(root, pid, stream)
        if len(names) < 4 * n_per_half:
            skipped.append((pid, len(names)))
            continue
        half = len(names) // 2
        a = rng.choice(names[:half], size=n_per_half, replace=False)
        b = rng.choice(names[half:], size=n_per_half, replace=False)
        out[pid] = (list(a), list(b))
    return out, skipped


def flat_sample(root, stream, n, seed=0, pids=None):
    """{pid: names}, n frames uniformly from the whole recording."""
    rng = np.random.default_rng(seed)
    out = {}
    for pid in (pids or participants(root, stream)):
        names = frames(root, pid, stream)
        if len(names) < n:
            continue
        out[pid] = list(rng.choice(names, size=n, replace=False))
    return out


def load(root, pid, stream, name):
    return cv2.imread(str(Path(root) / pid / stream / name))


def load_floor_patch(frames_root, pid, name, patch):
    """Top-left `patch`-square of the full frame with the same name: a face-free
    region of the same recording, for the session-nuisance floor. The caller is
    responsible for checking it contains no face (see --face-boxes)."""
    f = cv2.imread(str(Path(frames_root) / pid / name))
    return None if f is None else f[0:patch, 0:patch]
