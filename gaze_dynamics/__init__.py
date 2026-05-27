"""Unified gaze-dynamics analysis pipeline.

Merges the previously separate SaccadeAnalyzer / BlinkAnalyzer / HeatmapAnalyzer
scripts and their Iris/BlinkSync upstream into one package that consumes gaze
coordinates (model "Output" vs EyeLink ground truth) and produces dynamics
metrics and figures.

Submodules are imported explicitly (e.g. ``from gaze_dynamics import metrics``)
so that importing the package does not pull in heavy optional dependencies
(opencv, pingouin, fastdtw, ...).
"""

__all__ = [
    "config", "geometry", "metrics", "preprocess", "events", "io",
]
