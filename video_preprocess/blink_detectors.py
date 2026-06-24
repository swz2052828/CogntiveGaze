"""Three swappable blink detectors.

  ear              6-point Eye Aspect Ratio (current default). Score range
                   roughly 0.0 (closed) to 0.4 (wide open). Default
                   threshold 0.2.
  contour_ratio    Uses all 16 eye contour points: height / width of the
                   axis-aligned eye bbox. Similar polarity to EAR but uses
                   more landmarks so it is a touch more stable.
  iris_visibility  When the eye closes the iris is occluded, so the iris
                   landmark spread (radius proxy) shrinks. Score is the iris
                   radius normalised by inter-canthi distance; threshold
                   default 0.04. Independent of the eye-contour landmarks,
                   so good as a sanity check against the two contour-based
                   methods.

A blink is "either eye is closed", consistent across detectors.
"""

from typing import Optional, Tuple

import numpy as np

from .bbox import bbox_from_points
from .blink import _ear
from .detection_base import BBox, BlinkDetector
from .detector import LandmarkSet


class EARBlinkDetector(BlinkDetector):
    name = "ear"
    needs_mesh = True

    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold

    def detect(
        self,
        frame_rgb: np.ndarray,
        left_eye_bbox: Optional[BBox] = None,
        right_eye_bbox: Optional[BBox] = None,
        mesh_landmarks: Optional[LandmarkSet] = None,
    ) -> Tuple[bool, float, float]:
        if mesh_landmarks is None:
            return (False, float("nan"), float("nan"))
        ear_l = _ear(mesh_landmarks.left_eye_ear)
        ear_r = _ear(mesh_landmarks.right_eye_ear)
        return (ear_l < self.threshold or ear_r < self.threshold, ear_l, ear_r)


class ContourRatioBlinkDetector(BlinkDetector):
    """Height / width of the eye contour bbox. Closed eye -> small ratio."""

    name = "contour_ratio"
    needs_mesh = True

    def __init__(self, threshold: float = 0.18):
        self.threshold = threshold

    @staticmethod
    def _ratio(contour: np.ndarray) -> float:
        x0, y0, x1, y1 = bbox_from_points(contour)
        w = x1 - x0
        h = y1 - y0
        if w < 1e-6:
            return 0.0
        return h / w

    def detect(
        self,
        frame_rgb: np.ndarray,
        left_eye_bbox=None,
        right_eye_bbox=None,
        mesh_landmarks: Optional[LandmarkSet] = None,
    ) -> Tuple[bool, float, float]:
        if mesh_landmarks is None:
            return (False, float("nan"), float("nan"))
        r_l = self._ratio(mesh_landmarks.left_eye)
        r_r = self._ratio(mesh_landmarks.right_eye)
        return (r_l < self.threshold or r_r < self.threshold, r_l, r_r)


class IrisVisibilityBlinkDetector(BlinkDetector):
    """Vertical iris extent / inter-canthi span. Closed eye -> small score.

    When the eye closes the eyelids occlude the top and bottom of the iris, so
    the *vertical* spread of the iris landmarks collapses. (The max-radius /
    fitted-circle measure does NOT collapse, because refine_landmarks keeps
    fitting a full circle even when occluded -- that earlier metric detected
    essentially no blinks. Validated on 6000 frames / 5 subjects: vertical
    extent reaches F1=0.70 vs 0.44 for max-radius, treating EAR as truth.)

    Normalising by the inter-canthi (eye-corner to eye-corner) distance makes
    the score scale-invariant across subjects. Independent of the eye-contour
    EAR/contour-ratio landmarks, so it remains a useful cross-check.
    """

    name = "iris_visibility"
    needs_mesh = True

    def __init__(self, threshold: float = 0.113):
        self.threshold = threshold

    @staticmethod
    def _norm_vertical_extent(iris_pts: np.ndarray, scale: float) -> float:
        vert = float(iris_pts[:, 1].max() - iris_pts[:, 1].min())
        if scale < 1e-6:
            return 0.0
        return vert / scale

    def detect(
        self,
        frame_rgb: np.ndarray,
        left_eye_bbox=None,
        right_eye_bbox=None,
        mesh_landmarks: Optional[LandmarkSet] = None,
    ) -> Tuple[bool, float, float]:
        if mesh_landmarks is None:
            return (False, float("nan"), float("nan"))
        # Inter-canthi distance from the two outer corners of left and right eye
        # (landmark indices 33 and 263 in FaceMesh, which sit at positions 0 and
        # 8 of our LEFT_EYE_INDICES and RIGHT_EYE_INDICES contours).
        left_outer = mesh_landmarks.left_eye[0]
        right_outer = mesh_landmarks.right_eye[8]
        inter_canthi = float(np.linalg.norm(right_outer - left_outer))
        score_l = self._norm_vertical_extent(mesh_landmarks.left_iris, inter_canthi)
        score_r = self._norm_vertical_extent(mesh_landmarks.right_iris, inter_canthi)
        return (
            score_l < self.threshold or score_r < self.threshold,
            score_l,
            score_r,
        )


class TemplateMatchingBlinkDetector(BlinkDetector):
    """Pupil-template matching (the Experiment 2 mechanism).

    A per-subject pupil template (a small grayscale crop of the open-eye pupil,
    from extracted_data/subid_<id>/subid_<id>_<frame>_template_<id>.png) is slid
    over the eye region with cv2.matchTemplate(TM_CCOEFF_NORMED). When the eye is
    open the pupil is visible and the peak correlation is high; when the eye
    closes the pupil is occluded, no location exceeds the match threshold
    ("No match" in the original notebook), so a low peak => blink.

    Score per eye = max peak correlation over all of the subject's templates.
    A blink is "either eye lost its pupil" (min over eyes < threshold),
    consistent with the EAR / contour-ratio convention.

    Needs the eye bboxes (from the eye detector), not the mesh; the eye region
    is padded so the pupil has room to be found and the template fits.
    """

    name = "template_match"
    needs_mesh = False

    def __init__(self, threshold: float = 0.5, templates_dir=None,
                 subject_id=None, pad_w: float = 0.5, pad_h: float = 0.6):
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("opencv-python is required.") from exc
        import glob
        self._cv2 = cv2
        self.threshold = threshold if threshold is not None else 0.5
        self.pad_w = pad_w
        self.pad_h = pad_h
        self.templates = []
        if templates_dir is not None and subject_id is not None:
            paths = sorted(glob.glob(
                f"{templates_dir}/subid_{subject_id}/*template*.png"))
            for p in paths:
                t = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if t is not None:
                    self.templates.append(t)
        if not self.templates:
            raise ValueError(
                f"No templates loaded for subject {subject_id} in {templates_dir}")

    def _score_eye(self, gray, bbox):
        if bbox is None:
            return float("nan")
        h_img, w_img = gray.shape[:2]
        x0, y0, x1, y1 = bbox
        bw, bh = x1 - x0, y1 - y0
        px, py = bw * self.pad_w, bh * self.pad_h
        rx0 = max(0, int(round(x0 - px)))
        ry0 = max(0, int(round(y0 - py)))
        rx1 = min(w_img, int(round(x1 + px)))
        ry1 = min(h_img, int(round(y1 + py)))
        crop = gray[ry0:ry1, rx0:rx1]
        if crop.size == 0:
            return float("nan")
        best = 0.0
        for t in self.templates:
            th, tw = t.shape[:2]
            ch, cw = crop.shape[:2]
            search = crop
            # matchTemplate needs the template no larger than the search image;
            # pad the crop with edge replication if a template is bigger.
            if th > ch or tw > cw:
                pad_b = max(0, th - ch)
                pad_r = max(0, tw - cw)
                search = self._cv2.copyMakeBorder(
                    crop, 0, pad_b, 0, pad_r, self._cv2.BORDER_REPLICATE)
            res = self._cv2.matchTemplate(search, t, self._cv2.TM_CCOEFF_NORMED)
            best = max(best, float(res.max()))
        return best

    def detect(
        self,
        frame_rgb: np.ndarray,
        left_eye_bbox: Optional[BBox] = None,
        right_eye_bbox: Optional[BBox] = None,
        mesh_landmarks: Optional[LandmarkSet] = None,
    ) -> Tuple[bool, float, float]:
        gray = self._cv2.cvtColor(frame_rgb, self._cv2.COLOR_RGB2GRAY)
        sl = self._score_eye(gray, left_eye_bbox)
        sr = self._score_eye(gray, right_eye_bbox)
        scores = [s for s in (sl, sr) if s == s]  # drop NaN
        if not scores:
            return (False, sl, sr)
        is_blink = min(scores) < self.threshold
        return (is_blink, sl, sr)


BLINK_DETECTORS = {
    EARBlinkDetector.name: EARBlinkDetector,
    ContourRatioBlinkDetector.name: ContourRatioBlinkDetector,
    IrisVisibilityBlinkDetector.name: IrisVisibilityBlinkDetector,
    TemplateMatchingBlinkDetector.name: TemplateMatchingBlinkDetector,
}


def build_blink_detector(name: str, threshold: Optional[float] = None,
                         **kwargs) -> BlinkDetector:
    if name not in BLINK_DETECTORS:
        raise ValueError(
            f"Unknown blink detector '{name}'. Available: {list(BLINK_DETECTORS)}"
        )
    cls = BLINK_DETECTORS[name]
    if name == "template_match":
        return cls(threshold=threshold, **kwargs)
    return cls(threshold=threshold) if threshold is not None else cls()
