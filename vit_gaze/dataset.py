from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data

# OpenCV decode + bilinear resize is much faster than PIL + BICUBIC and matches
# the CNN baseline's loader (ITrackerData). Force single-threaded so DataLoader
# workers do not oversubscribe CPU cores. These run once at import and are
# inherited by forked workers.
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _load_image_cv2(path, size):
    """Decode -> resize (bilinear) -> CHW float tensor, ImageNet-normalized.

    A missing/corrupt file falls back to a black frame instead of crashing,
    matching the CNN baseline. With the init-time existence check removed, this
    fallback is what guards against the occasional missing crop.
    """
    img = cv2.imread(str(path))
    if img is None:
        img = np.zeros((size, size, 3), dtype=np.uint8)
    else:
        if img.shape[0] != size or img.shape[1] != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


class PairedFaceGazeDataset(data.Dataset):
    def __init__(
        self,
        data_path,
        mean_path,
        metadata_path=None,
        raw_root=None,
        synthetic_root=None,
        raw_folder="appleFace",
        synthetic_folder="appleFaceFake",
        image_size=224,
        use_synthetic=True,
        require_synthetic=True,
    ):
        self.data_path = Path(data_path) if data_path is not None else None
        self.raw_root = Path(raw_root) if raw_root is not None else None
        self.synthetic_root = Path(synthetic_root) if synthetic_root is not None else None
        self.raw_folder = raw_folder
        self.synthetic_folder = synthetic_folder
        self.image_size = image_size
        self.use_synthetic = use_synthetic

        metadata_path = self._resolve_metadata_path(metadata_path, mean_path)
        if not metadata_path.is_file():
            raise FileNotFoundError(f"metadata.mat not found: {metadata_path}")

        metadata = sio.loadmat(metadata_path, squeeze_me=True, struct_as_record=False)
        rec_nums = np.asarray(metadata["labelRecNum"]).astype(np.int32)
        frame_indices = np.asarray(metadata["frameIndex"]).astype(np.int32)
        gazes = np.stack(
            [
                np.asarray(metadata["labelDotXCam"]).astype(np.float32),
                np.asarray(metadata["labelDotYCam"]).astype(np.float32),
            ],
            axis=1,
        )

        self.samples = []
        missing_raw = 0
        missing_synthetic = 0
        for rec, frame, gaze in zip(rec_nums, frame_indices, gazes):
            raw_path = self._image_path(rec, frame, synthetic=False)
            synthetic_path = self._image_path(rec, frame, synthetic=True)
            if not raw_path.is_file():
                missing_raw += 1
                continue
            if use_synthetic and not synthetic_path.is_file():
                missing_synthetic += 1
                if require_synthetic:
                    continue
            self.samples.append((raw_path, synthetic_path, gaze, int(rec), int(frame)))

        if not self.samples:
            raise RuntimeError(
                "No usable face samples found. Check image roots and metadata, "
                "or pass --allow-missing-synthetic for debugging synthetic inputs."
            )

        if missing_raw:
            print(f"Skipped {missing_raw} rows without raw face images.")
        if missing_synthetic:
            if require_synthetic:
                print(f"Skipped {missing_synthetic} rows without synthetic face images.")
            else:
                print(f"Found {missing_synthetic} rows without synthetic face images; using raw images as fallback.")

        self.gazes = np.stack([sample[2] for sample in self.samples]).astype(np.float32)
        self.recordings = np.asarray([sample[3] for sample in self.samples], dtype=np.int32)
        print(
            f"Loaded {len(self.samples)} metadata-aligned samples. "
            f"Raw root: {self.raw_root or self.data_path}. "
            f"Synthetic root: {self.synthetic_root or self.data_path}."
        )

    def _resolve_metadata_path(self, metadata_path, mean_path):
        if metadata_path is not None:
            return Path(metadata_path)
        if self.data_path is None:
            raise ValueError("Provide --data-path or --metadata-path.")
        return self.data_path / mean_path / "metadata.mat"

    def _image_path(self, rec, frame, synthetic):
        filename = f"{frame:05d}.jpg"
        if synthetic:
            if self.synthetic_root is not None:
                return self.synthetic_root / f"{rec:05d}" / filename
            return self.data_path / f"{rec:05d}" / self.synthetic_folder / filename

        if self.raw_root is not None:
            return self.raw_root / f"{rec:05d}" / filename
        return self.data_path / f"{rec:05d}" / self.raw_folder / filename

    def unique_recordings(self):
        return np.unique(self.recordings)

    def indices_for_recordings(self, recording_ids):
        return np.where(np.isin(self.recordings, np.asarray(recording_ids, dtype=np.int32)))[0].tolist()

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        return _load_image_cv2(path, self.image_size)

    def __getitem__(self, idx):
        raw_path, synthetic_path, gaze, rec, frame = self.samples[idx]
        raw = self._load_image(raw_path)
        if self.use_synthetic and synthetic_path.is_file():
            synthetic = self._load_image(synthetic_path)
        else:
            synthetic = raw.clone()

        return {
            "raw": raw,
            "synthetic": synthetic,
            "gaze": torch.from_numpy(gaze.copy()),
            "index": torch.tensor(idx, dtype=torch.long),
            "rec": torch.tensor(rec, dtype=torch.long),
            "frame": torch.tensor(frame, dtype=torch.long),
        }


def build_dataset(args, use_synthetic, require_synthetic):
    return PairedFaceGazeDataset(
        data_path=args.data_path,
        mean_path=args.mean_path,
        metadata_path=args.metadata_path,
        raw_root=args.raw_root,
        synthetic_root=args.synthetic_root,
        raw_folder=args.raw_folder,
        synthetic_folder=args.synthetic_folder,
        image_size=args.image_size,
        use_synthetic=use_synthetic,
        require_synthetic=require_synthetic,
    )


def tensor_to_image(tensor):
    image = tensor.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return image


class MultiStreamGazeDataset(data.Dataset):
    """Face + left eye + right eye + optional face-grid for multi-stream ViT.

    Layout matches the iTracker-style preprocessing already used by the
    project's CNN baselines (see ITrackerData.py reference):
      <data_path>/<rec:05d>/<face_folder>/<frame:05d>.jpg
      <eye_path>/<rec:05d>/<left_eye_folder>/<frame:05d>.jpg
      <eye_path>/<rec:05d>/<right_eye_folder>/<frame:05d>.jpg

    Metadata is loaded from <data_path>/<mean_path>/metadata.mat (or an explicit
    metadata_path). Expected fields: labelRecNum, frameIndex, labelDotXCam,
    labelDotYCam, and labelFaceGrid ([x0, y0, w, h] per row, only required when
    use_grid=True).

    Normalisation uses ImageNet stats (matches the existing vit_gaze pipeline
    and the ImageNet-pretrained ViT backbone), not the per-channel mean
    subtraction the CNN baselines use.
    """

    def __init__(
        self,
        data_path,
        mean_path,
        eye_path=None,
        metadata_path=None,
        face_folder="appleFace",
        left_eye_folder="appleLeftEye",
        right_eye_folder="appleRightEye",
        image_size=224,
        eye_size=224,
        grid_size=25,
        use_grid=False,
    ):
        self.data_path = Path(data_path)
        self.eye_path = Path(eye_path) if eye_path is not None else self.data_path
        self.face_folder = face_folder
        self.left_eye_folder = left_eye_folder
        self.right_eye_folder = right_eye_folder
        self.image_size = image_size
        self.eye_size = eye_size
        self.use_grid = use_grid
        self.grid_size = grid_size
        self.grid_len = grid_size * grid_size

        if metadata_path is None:
            metadata_path = self.data_path / mean_path / "metadata.mat"
        else:
            metadata_path = Path(metadata_path)
        if not metadata_path.is_file():
            raise FileNotFoundError(f"metadata.mat not found: {metadata_path}")

        metadata = sio.loadmat(metadata_path, squeeze_me=True, struct_as_record=False)
        rec_nums = np.asarray(metadata["labelRecNum"]).astype(np.int32)
        frame_indices = np.asarray(metadata["frameIndex"]).astype(np.int32)
        gazes = np.stack(
            [
                np.asarray(metadata["labelDotXCam"]).astype(np.float32),
                np.asarray(metadata["labelDotYCam"]).astype(np.float32),
            ],
            axis=1,
        )
        face_grids_meta = None
        if use_grid:
            if "labelFaceGrid" not in metadata:
                raise KeyError(
                    "labelFaceGrid not found in metadata.mat; cannot --use-grid."
                )
            face_grids_meta = np.asarray(metadata["labelFaceGrid"])

        # Build sample paths for every metadata row. We intentionally do NOT
        # stat each file here: the per-row triple os.stat() (face/eyeL/eyeR) is
        # slow on networked HPC filesystems and dominated dataset startup.
        # Missing/corrupt crops are handled at load time by the black-frame
        # fallback in _load_image_cv2, matching the CNN baseline.
        self.samples = []
        for rec, frame, gaze in zip(rec_nums, frame_indices, gazes):
            face_p = self.data_path / f"{rec:05d}" / face_folder / f"{frame:05d}.jpg"
            left_p = self.eye_path / f"{rec:05d}" / left_eye_folder / f"{frame:05d}.jpg"
            right_p = self.eye_path / f"{rec:05d}" / right_eye_folder / f"{frame:05d}.jpg"
            self.samples.append((face_p, left_p, right_p, gaze, int(rec), int(frame)))
        self.grid_params = list(face_grids_meta) if use_grid else []

        if not self.samples:
            raise RuntimeError(
                "No rows in metadata.mat. Check --data-path / --metadata-path."
            )

        self.gazes = gazes.astype(np.float32)
        self.recordings = rec_nums.astype(np.int32)

        self._grid_xs = np.arange(self.grid_len) % grid_size
        self._grid_ys = np.arange(self.grid_len) // grid_size

        print(
            f"Loaded {len(self.samples)} multistream samples. "
            f"Face root: {self.data_path}. Eye root: {self.eye_path}. "
            f"Grid: {'on' if use_grid else 'off'}."
        )

    def unique_recordings(self):
        return np.unique(self.recordings)

    def indices_for_recordings(self, recording_ids):
        return np.where(
            np.isin(self.recordings, np.asarray(recording_ids, dtype=np.int32))
        )[0].tolist()

    def _load_image(self, path, size):
        return _load_image_cv2(path, size)

    def _make_grid(self, params):
        grid = torch.zeros(self.grid_len, dtype=torch.float32)
        x0, y0, w, h = (int(v) for v in params)
        mask = (
            (self._grid_xs >= x0)
            & (self._grid_xs < x0 + w)
            & (self._grid_ys >= y0)
            & (self._grid_ys < y0 + h)
        )
        grid[mask] = 1.0
        return grid

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        face_p, left_p, right_p, gaze, rec, frame = self.samples[idx]
        item = {
            "face": self._load_image(face_p, self.image_size),
            "eye_left": self._load_image(left_p, self.eye_size),
            "eye_right": self._load_image(right_p, self.eye_size),
            "gaze": torch.from_numpy(gaze.copy()),
            "index": torch.tensor(idx, dtype=torch.long),
            "rec": torch.tensor(rec, dtype=torch.long),
            "frame": torch.tensor(frame, dtype=torch.long),
        }
        if self.use_grid:
            item["grid"] = self._make_grid(self.grid_params[idx])
        return item


def make_augment_transform(level, image_size):
    """Return a torchvision Compose transform for the given augmentation level, or None.

    Designed for normalized CHW float tensors from _load_image_cv2.
    NO horizontal flip: gaze labels are screen-relative; mirroring the image
    without mirroring the gaze target would corrupt training signal.
    """
    import torchvision.transforms as T
    if not level or level == "none":
        return None
    if level == "light":
        return T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomResizedCrop(image_size, scale=(0.90, 1.0), ratio=(1.0, 1.0)),
        ])
    if level == "medium":
        return T.Compose([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            T.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(1.0, 1.0)),
            T.RandomGrayscale(p=0.05),
        ])
    raise ValueError(f"Unknown augmentation level: {level!r}. Choose none/light/medium.")


class AugmentedSubset(data.Dataset):
    """Wraps a Subset and applies per-image augmentation to selected keys.

    Only wraps training subsets; validation subsets are left unwrapped so
    augmentation never affects reported metrics.
    """

    def __init__(self, subset, transform, image_keys=("face", "eye_left", "eye_right")):
        self.subset = subset
        self.transform = transform
        self.image_keys = image_keys

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        item = self.subset[idx]
        if self.transform is None:
            return item
        item = dict(item)
        for key in self.image_keys:
            if key in item:
                item[key] = self.transform(item[key])
        return item


def build_multistream_dataset(args):
    return MultiStreamGazeDataset(
        data_path=args.data_path,
        mean_path=args.mean_path,
        eye_path=getattr(args, "eye_path", None),
        metadata_path=args.metadata_path,
        face_folder=getattr(args, "face_folder", "appleFace"),
        left_eye_folder=getattr(args, "left_eye_folder", "appleLeftEye"),
        right_eye_folder=getattr(args, "right_eye_folder", "appleRightEye"),
        image_size=args.image_size,
        eye_size=getattr(args, "eye_size", 224),
        grid_size=getattr(args, "grid_size", 25),
        use_grid=getattr(args, "use_grid", False),
    )
