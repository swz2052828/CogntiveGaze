"""Per-subject SVR calibration baseline.

The standard published approach: fit two RBF-SVRs (one per output coordinate)
on K calibration pairs ``(predicted_xy, true_xy)``, then apply at inference to
transform model predictions to subject-calibrated coordinates. Used here as the
apples-to-apples comparison point against the meta-learned adapter at matched K.

Hyperparameters follow the common defaults reported in the gaze-calibration
literature (RBF kernel, ``C=1.0``, ``gamma='scale'``, ``epsilon=0.1``); expose
the ``C`` knob since it is the one most worth tuning at small K.
"""

import numpy as np


class SVRCalibrator:
    def __init__(self, C=1.0, epsilon=0.1, gamma="scale"):
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.models_ = None

    def fit(self, pred_xy, true_xy):
        """``pred_xy``, ``true_xy``: ``[K, 2]`` arrays in the same coordinate frame."""
        from sklearn.svm import SVR
        pred = np.asarray(pred_xy, dtype=np.float64)
        true = np.asarray(true_xy, dtype=np.float64)
        if pred.shape != true.shape or pred.ndim != 2 or pred.shape[1] != 2:
            raise ValueError(f"pred and true must be [K, 2]; got {pred.shape} vs {true.shape}")
        self.models_ = [
            SVR(kernel="rbf", C=self.C, epsilon=self.epsilon, gamma=self.gamma).fit(pred, true[:, 0]),
            SVR(kernel="rbf", C=self.C, epsilon=self.epsilon, gamma=self.gamma).fit(pred, true[:, 1]),
        ]
        return self

    def transform(self, pred_xy):
        if self.models_ is None:
            raise RuntimeError("SVRCalibrator not fit yet.")
        pred = np.asarray(pred_xy, dtype=np.float64)
        return np.stack([self.models_[0].predict(pred), self.models_[1].predict(pred)], axis=1)
