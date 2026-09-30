"""A second recogniser family, so a privacy number is not one model's opinion.

Every identity figure in this study comes from InsightFace ArcFace (w600k_r50).
A reviewer will reasonably ask whether "the eye crop verifies at 62-67%" is a
property of the data or of ArcFace. This module supplies a drop-in with the same
`embed_aligned` contract so the attacks can be re-run unchanged on a second,
architecturally unrelated recogniser trained on a different corpus.

ArcFace: ResNet-50, trained on WebFace600K, margin-based softmax, 112x112 BGR.
FaceNet: Inception-ResNet-v1, trained on VGGFace2, triplet loss, 160x160 RGB.

Different backbone, different training set, different objective. Agreement
between them is therefore evidence about the imagery rather than about one
model's inductive bias.

Detection and alignment stay with ArcFace in both cases: the attacker is assumed
to use the best available detector and to swap only the recogniser, which is the
conservative choice and keeps the two arms comparable frame for frame.
"""

import numpy as np


class FaceNetRecogniser:
    """facenet_pytorch InceptionResnetV1(vggface2), ArcFace's embed interface."""

    def __init__(self, ctx_id: int = -1):
        import torch
        from facenet_pytorch import InceptionResnetV1

        self._torch = torch
        self.device = f"cuda:{ctx_id}" if ctx_id >= 0 else "cpu"
        self.net = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def embed_aligned(self, aimg):
        """aimg: BGR uint8, any size (the attacks pass 112x112). Returns L2-normed 512-d."""
        import cv2

        rgb = cv2.cvtColor(cv2.resize(aimg, (160, 160)), cv2.COLOR_BGR2RGB)
        t = self._torch.from_numpy(rgb).permute(2, 0, 1).float()
        t = (t - 127.5) / 128.0                      # fixed_image_standardization
        with self._torch.no_grad():
            e = self.net(t.unsqueeze(0).to(self.device)).cpu().numpy().ravel()
        return e / (np.linalg.norm(e) + 1e-9)


def build(name: str, ctx_id: int = -1, **kw):
    """`arcface` (default, unchanged behaviour) or `facenet`."""
    if name == "arcface":
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from anon.id_attack import ArcFace

        return ArcFace(ctx_id=ctx_id, **kw)
    if name == "facenet":
        return FaceNetRecogniser(ctx_id=ctx_id)
    raise ValueError(f"unknown recogniser '{name}'; choices: arcface, facenet")
