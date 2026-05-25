from pathlib import Path

import torch
import torch.utils.data as data

from .dataset import GazeSwapDataset, tensor_to_pil
from .masks import blend_protected_regions, make_gaze_protection_mask
from .models import GazePreservingGenerator


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved_args = checkpoint.get("args", {})
    base_channels = int(saved_args.get("base_channels", args.base_channels))
    residual_blocks = int(saved_args.get("residual_blocks", args.residual_blocks))

    generator = GazePreservingGenerator(base_channels=base_channels, residual_blocks=residual_blocks).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    dataset = GazeSwapDataset(
        data_path=args.data_path,
        mean_path=args.mean_path,
        metadata_path=args.metadata_path,
        source_root=args.source_root,
        target_root=args.target_root,
        image_size=args.image_size,
        limit=args.limit,
        random_targets=args.random_targets,
    )
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    output_root = Path(args.output_root)

    print(f"Device: {device}")
    print(f"Writing swapped faces to {output_root}")
    generated = 0

    with torch.no_grad():
        for batch in loader:
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            protection = make_gaze_protection_mask(source.size(0), source.size(2), source.size(3), device, source.dtype)
            base_fake = generator(source, target)
            fake = blend_protected_regions(base_fake, source, protection) if args.copy_protected else base_fake

            for i in range(fake.size(0)):
                rec = int(batch["rec"][i])
                frame = int(batch["frame"][i])
                dst = output_root / f"{rec:05d}" / f"{frame:05d}.jpg"
                dst.parent.mkdir(parents=True, exist_ok=True)
                tensor_to_pil(fake[i]).save(dst, quality=95)
                generated += 1
                print(f"{batch['source_path'][i]} -> {dst}")

    print(f"Done. Generated {generated} gaze-preserving swapped images.")
