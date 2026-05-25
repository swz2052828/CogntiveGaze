import random
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data
from PIL import Image


def resolve_metadata_path(data_path, mean_path, metadata_path):
    if metadata_path:
        return Path(metadata_path)
    if not data_path:
        raise ValueError("Provide --data-path or --metadata-path.")
    return Path(data_path) / mean_path / "metadata.mat"


def image_to_tensor(path, image_size):
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def tensor_to_pil(tensor):
    arr = (tensor.detach().cpu().clamp(-1, 1) + 1.0) * 127.5
    arr = arr.permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(arr)


def collect_images(root):
    root = Path(root)
    paths = []
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(root.glob(suffix))
        paths.extend(root.glob(f"*/*{suffix[1:]}"))
    return sorted(set(path for path in paths if path.is_file()))


class GazeSwapDataset(data.Dataset):
    def __init__(
        self,
        data_path=None,
        mean_path="mean7",
        metadata_path=None,
        source_root=None,
        target_root=None,
        image_size=224,
        limit=None,
        random_targets=False,
    ):
        if source_root is None:
            raise ValueError("Provide --source-root with <root>/<recording>/<frame>.jpg images.")

        self.source_root = Path(source_root)
        self.target_root = Path(target_root) if target_root else None
        self.image_size = image_size
        self.random_targets = random_targets

        metadata_path = resolve_metadata_path(data_path, mean_path, metadata_path)
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
        missing = 0
        for rec, frame, gaze in zip(rec_nums, frame_indices, gazes):
            source_path = self.source_root / f"{rec:05d}" / f"{frame:05d}.jpg"
            if not source_path.is_file():
                missing += 1
                continue
            self.samples.append((source_path, gaze, int(rec), int(frame)))
            if limit is not None and len(self.samples) >= limit:
                break

        if not self.samples:
            raise RuntimeError("No metadata-aligned source images were found.")

        self.target_paths = collect_images(self.target_root) if self.target_root else []
        if not self.target_paths:
            print("No --target-root images found. Target defaults to source image; identity change will be weak.")

        if missing:
            print(f"Skipped {missing} metadata rows without source images.")
        print(f"Loaded {len(self.samples)} gaze-swap samples.")

    def __len__(self):
        return len(self.samples)

    def _target_for_index(self, index):
        if not self.target_paths:
            return self.samples[index][0]
        if self.random_targets:
            return random.choice(self.target_paths)
        return self.target_paths[(index * 9973) % len(self.target_paths)]

    def __getitem__(self, index):
        source_path, gaze, rec, frame = self.samples[index]
        target_path = self._target_for_index(index)
        return {
            "source": image_to_tensor(source_path, self.image_size),
            "target": image_to_tensor(target_path, self.image_size),
            "gaze": torch.from_numpy(gaze.copy()),
            "rec": torch.tensor(rec, dtype=torch.long),
            "frame": torch.tensor(frame, dtype=torch.long),
            "source_path": str(source_path),
            "target_path": str(target_path),
        }
