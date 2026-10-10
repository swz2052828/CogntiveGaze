"""Pretrained recognisers. None is trained on the audited data.

ArcFace (InsightFace buffalo_l, w600k_r50) is the default. FaceNet
(Inception-ResNet-v1, VGGFace2) is a second, unrelated family; quote each against
its own floor, never mix them.
"""
import numpy as np


class ArcFace:
    def __init__(self, gpu=-1, det_size=320, threads=4):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        ort.set_default_logger_severity(3)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads          # default pool = one per core
        so.inter_op_num_threads = 1
        prov = (["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu >= 0
                else ["CPUExecutionProvider"])
        self.app = FaceAnalysis(name="buffalo_l", providers=prov,
                                allowed_modules=["detection", "recognition"],
                                session_options=so)
        self.app.prepare(ctx_id=gpu, det_size=(det_size, det_size))
        self.rec = self.app.models["recognition"]

    def embed(self, img112):
        e = self.rec.get_feat(img112).flatten()
        return e / (np.linalg.norm(e) + 1e-9)

    def detect(self, bgr):
        faces = self.app.get(bgr)
        if not faces:
            return None
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return faces[-1].kps

    def embed_detected(self, bgr):
        """Detect on the RELEASED image (release-only attacker); None if no face."""
        from insightface.utils import face_align
        kps = self.detect(bgr)
        if kps is None:
            return None
        return self.embed(face_align.norm_crop(bgr, landmark=kps, image_size=112))


class FaceNet:
    def __init__(self, gpu=-1):
        import torch
        from facenet_pytorch import InceptionResnetV1
        self.torch = torch
        self.device = f"cuda:{gpu}" if gpu >= 0 else "cpu"
        self.net = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self._arc = None

    def embed(self, img112):
        import cv2
        rgb = cv2.cvtColor(cv2.resize(img112, (160, 160)), cv2.COLOR_BGR2RGB)
        t = (self.torch.from_numpy(rgb).permute(2, 0, 1).float() - 127.5) / 128.0
        with self.torch.no_grad():
            e = self.net(t.unsqueeze(0).to(self.device)).cpu().numpy().ravel()
        return e / (np.linalg.norm(e) + 1e-9)

    def embed_detected(self, bgr):
        raise NotImplementedError("detection-aligned FaceNet: use --recogniser arcface "
                                  "for face streams with --align-detect")


def build(name, gpu=-1):
    if name == "arcface":
        return ArcFace(gpu=gpu)
    if name == "facenet":
        return FaceNet(gpu=gpu)
    raise ValueError(f"unknown recogniser {name!r}")
