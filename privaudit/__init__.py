"""privaudit -- release-time privacy audit for appearance-based gaze datasets.

Given a candidate release in GazeCapture layout (<release>/<participant>/<stream>/
<frame>.jpg), run the attacks that decide whether it can be shared, against the
floors that make their numbers interpretable, and map each result onto the
release checklist (CHECKLIST.md).

    python -m privaudit --release /data/release --streams appleFace appleLeftEye \
        appleRightEye --floor-frames /data/frames --out audit/

The attacks and their sampling are those of the CognitiveGaze study; the
validation script reproduces that study's numbers from this package.
"""
__version__ = "0.1.0"
