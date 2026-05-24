from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


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
        image = Image.open(path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        arr = np.asarray(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD

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
