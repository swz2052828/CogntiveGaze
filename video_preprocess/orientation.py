"""Open a source video with orientation handling that does not depend on the
environment.

The 18 source recordings carry THREE different container rotation tags:

    90 deg   subid_6..11, 13..18, 21..23   (15 recordings)
     0 deg   subid_12, subid_19
   180 deg   subid_20

OpenCV auto-applies that tag when CAP_PROP_ORIENTATION_AUTO is on, and whether
it is on depends on the build:

    envs/gaze     OpenCV 4.13.0   ORIENTATION_AUTO = 1   subid_6 reads 1080x1920
    envs/facedet  OpenCV 4.11.0   ORIENTATION_AUTO = 0   subid_6 reads 1920x1080

Every extraction in this project decodes raw (1920x1080) and then applies a
hard-coded cv2.ROTATE_90_CLOCKWISE, which is correct for all three tag classes --
verified with FaceMesh on the stored crops: ProcessedData and calib_support_K72
are upright for 00012 (tag 0), 00019 (tag 0), 00020 (tag 180) and the tag-90
recordings alike, tilt -8..0 deg and eye-line at 0.30-0.32 of face height for all.

That correctness holds only because those jobs ran under `facedet`, where AUTO is
off. The same code under `gaze` would decode the 15 tag-90 recordings
pre-rotated and then rotate them AGAIN, and would take subid_20 pre-rotated by
180 and then add 90. Nothing would raise; the crops would simply be sideways.

`open_upright` removes the dependency: it turns AUTO off explicitly, and checks
the decoded frame really is landscape before the caller rotates it.
"""
import cv2

RAW_SIZE = (1920, 1080)


def open_upright(path, expect_raw=RAW_SIZE):
    """VideoCapture with container auto-rotation disabled, size-checked.

    Returns the capture. The caller still applies its own rotation; this only
    guarantees the caller starts from raw, unrotated pixels in every env.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if expect_raw is not None and (w, h) != tuple(expect_raw):
        cap.release()
        raise RuntimeError(
            f"{path}: decoded {w}x{h}, expected {expect_raw[0]}x{expect_raw[1]}. "
            f"The container rotation tag is being applied by this OpenCV build "
            f"({cv2.__version__}) despite CAP_PROP_ORIENTATION_AUTO=0, so a "
            f"hard-coded rotation downstream would double-rotate. Decode under "
            f"envs/facedet, or derive the rotation from CAP_PROP_ORIENTATION_META "
            f"instead of hard-coding it.")
    return cap
