import time
from pathlib import Path

import torch
import torch.utils.data as data

from vit_gaze import accel

from .dataset import GazeSwapDataset
from .losses import (
    FrozenGazeCriterion,
    lsgan_discriminator_loss,
    lsgan_generator_loss,
    source_difference_penalty,
    total_variation_loss,
)
from .masks import blend_protected_regions, make_gaze_protection_mask, masked_l1
from .models import GazePreservingGenerator, PatchDiscriminator, init_weights


def make_loader(args):
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
    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return data.DataLoader(dataset, **loader_kwargs)


def save_checkpoint(args, generator, discriminator, optimizer_g, optimizer_d, epoch, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        },
        path,
    )


def save_sample_grid(source, target, fake, path):
    try:
        from torchvision.utils import save_image
    except ImportError:
        return

    count = min(4, source.size(0))
    grid = torch.cat([source[:count], target[:count], fake[:count]], dim=0)
    save_image((grid + 1.0) * 0.5, path, nrow=count)


def train(args):
    start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    accel.configure_backends(enable_tf32=not getattr(args, "no_tf32", False))
    if args.gaze_weight > 0 and not args.gaze_checkpoint:
        raise ValueError("--gaze-checkpoint is required when --gaze-weight > 0.")

    loader = make_loader(args)
    generator = GazePreservingGenerator(args.base_channels, args.residual_blocks).to(device)
    discriminator = PatchDiscriminator(args.base_channels).to(device)
    generator.apply(init_weights)
    discriminator.apply(init_weights)

    optimizer_g = torch.optim.AdamW(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999), weight_decay=args.weight_decay)
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999), weight_decay=args.weight_decay)
    gaze_criterion = (
        FrozenGazeCriterion(args.gaze_checkpoint, device=device, image_size=args.image_size)
        if args.gaze_weight > 0
        else None
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"GAN batches per epoch: {len(loader)}")
    print("Protected pixels are copied from the source image before gaze loss and saving.")

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        running_g = 0.0
        running_d = 0.0
        running_gaze_error = 0.0
        seen = 0

        for batch_idx, batch in enumerate(loader, start=1):
            batch_start = time.perf_counter()
            source = batch["source"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            gaze = batch["gaze"].to(device, non_blocking=True)
            protection = make_gaze_protection_mask(source.size(0), source.size(2), source.size(3), device, source.dtype)
            editable = 1.0 - protection

            base_fake = generator(source, target)
            fake = blend_protected_regions(base_fake, source, protection) if args.copy_protected else base_fake

            optimizer_d.zero_grad(set_to_none=True)
            real_logits = discriminator(target)
            fake_logits = discriminator(fake.detach())
            d_loss = lsgan_discriminator_loss(real_logits, fake_logits) * args.adv_weight
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            fake_logits_for_g = discriminator(fake)
            adv_loss = lsgan_generator_loss(fake_logits_for_g)
            gaze_loss = source.new_tensor(0.0)
            gaze_error = source.new_tensor(0.0)
            if gaze_criterion is not None:
                gaze_loss, gaze_error = gaze_criterion(fake, gaze)
            eye_loss = masked_l1(fake, source, protection)
            target_loss = masked_l1(fake, target, editable)
            diff_penalty = source_difference_penalty(fake, source, editable, args.source_difference_temperature)
            tv_loss = total_variation_loss(fake)

            g_loss = (
                args.adv_weight * adv_loss
                + args.gaze_weight * gaze_loss
                + args.eye_weight * eye_loss
                + args.target_weight * target_loss
                + args.source_difference_weight * diff_penalty
                + args.tv_weight * tv_loss
            )
            g_loss.backward()
            optimizer_g.step()

            batch_size = source.size(0)
            running_g += g_loss.item() * batch_size
            running_d += d_loss.item() * batch_size
            running_gaze_error += gaze_error.item() * batch_size
            seen += batch_size
            print(
                f"GAN epoch {epoch + 1}/{args.epochs} batch {batch_idx}/{len(loader)} "
                f"g_loss={g_loss.item():.6f} d_loss={d_loss.item():.6f} "
                f"gaze_loss={gaze_loss.item():.6f} gaze_error={gaze_error.item():.6f} "
                f"eye_loss={eye_loss.item():.6f} target_loss={target_loss.item():.6f} "
                f"source_diff_penalty={diff_penalty.item():.6f} "
                f"running_g_loss={running_g / max(1, seen):.6f} "
                f"running_d_loss={running_d / max(1, seen):.6f} "
                f"running_gaze_error={running_gaze_error / max(1, seen):.6f} "
                f"batch_time_sec={time.perf_counter() - batch_start:.2f}"
            )

            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

        save_checkpoint(
            args,
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            epoch + 1,
            out_dir / "last_gaze_preserving_swap_gan.pth",
        )
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                args,
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                epoch + 1,
                out_dir / f"epoch{epoch + 1:03d}_gaze_preserving_swap_gan.pth",
            )
            save_sample_grid(source.detach(), target.detach(), fake.detach(), sample_dir / f"epoch{epoch + 1:03d}.png")

        print(
            f"GAN epoch {epoch + 1}/{args.epochs} done "
            f"mean_g_loss={running_g / max(1, seen):.6f} "
            f"mean_d_loss={running_d / max(1, seen):.6f} "
            f"mean_gaze_error={running_gaze_error / max(1, seen):.6f} "
            f"epoch_time_sec={time.perf_counter() - epoch_start:.2f}"
        )

    print(f"GAN training done total_time_sec={time.perf_counter() - start:.2f}")
