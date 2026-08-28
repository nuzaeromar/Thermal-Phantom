"""Train the class-conditioned object-crop DDPM used by ThermalMirage."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.flir_dataset import FLIRCropDataset, FLIR_CLASS_NAMES, NUM_CLASSES
from models.diffusion_editor import ThermalUNet, DDPMScheduler


def csv_ints(value):
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_args():
    parser = argparse.ArgumentParser(description="Train class-conditioned thermal object diffusion")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", default="checkpoints/diffusion")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--min_box_area", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--channel_mult", type=csv_ints, default=(1, 2, 4, 8))
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--attn_resolutions", type=csv_ints, default=(16,))
    parser.add_argument("--cond_dim", type=int, default=64)
    parser.add_argument("--t_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def save_checkpoint(path, model, class_embedding, optimizer, scaler, epoch,
                    global_step, best_val, scheduler_config, args):
    payload = {
        "format_version": 1,
        "task": "class_conditioned_object_ddpm",
        "unet": model.state_dict(),
        "class_embedding": class_embedding.state_dict(),
        "model_config": model.model_config,
        "class_embedding_config": {
            "num_classes": NUM_CLASSES,
            "cond_dim": args.cond_dim,
            "class_names": list(FLIR_CLASS_NAMES),
        },
        "scheduler_config": scheduler_config,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val,
        "training_args": vars(args),
    }
    torch.save(payload, path)


@torch.no_grad()
def validate(model, class_embedding, scheduler, loader, device):
    model.eval()
    class_embedding.eval()
    total, count = 0.0, 0
    for crops, labels in loader:
        x0 = crops.to(device, non_blocking=True) * 2 - 1
        labels = labels.to(device, non_blocking=True).long()
        t = torch.randint(0, scheduler.T, (x0.shape[0],), device=device)
        x_t, noise = scheduler.q_sample(x0, t)
        loss = F.mse_loss(model(x_t, t, class_embedding(labels)), noise)
        total += float(loss) * x0.shape[0]
        count += x0.shape[0]
    return total / max(count, 1)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    requested = torch.device(args.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable; training on CPU")
        requested = torch.device("cpu")
    device = requested
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "training_config.json").write_text(
        json.dumps(vars(args), indent=2, default=list), encoding="utf-8"
    )

    train_set = FLIRCropDataset(
        args.data_root, split=args.train_split, crop_size=args.image_size,
        min_box_area=args.min_box_area,
    )
    val_set = FLIRCropDataset(
        args.data_root, split=args.val_split, crop_size=args.image_size,
        min_box_area=args.min_box_area,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = ThermalUNet(
        img_channels=3, base_ch=args.base_ch, channel_mult=args.channel_mult,
        num_res_blocks=args.num_res_blocks, attn_resolutions=args.attn_resolutions,
        cond_dim=args.cond_dim, t_dim=args.t_dim, dropout=args.dropout,
        img_size=args.image_size,
    ).to(device)
    class_embedding = torch.nn.Embedding(NUM_CLASSES, args.cond_dim).to(device)
    scheduler_config = {
        "T": args.timesteps,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
    }
    scheduler = DDPMScheduler(**scheduler_config)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(class_embedding.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch, global_step, best_val = 0, 0, float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["unet"])
        class_embedding.load_state_dict(ckpt["class_embedding"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val = float(ckpt.get("best_val_loss", best_val))
        print(f"[Resume] epoch={start_epoch}, step={global_step}, best_val={best_val:.6f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        class_embedding.train()
        running, seen = 0.0, 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for crops, labels in progress:
            x0 = crops.to(device, non_blocking=True) * 2 - 1
            labels = labels.to(device, non_blocking=True).long()
            t = torch.randint(0, scheduler.T, (x0.shape[0],), device=device)
            x_t, noise = scheduler.q_sample(x0, t)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                predicted_noise = model(x_t, t, class_embedding(labels))
                loss = F.mse_loss(predicted_noise, noise)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(class_embedding.parameters()), 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            running += float(loss.detach()) * x0.shape[0]
            seen += x0.shape[0]
            progress.set_postfix(train_loss=f"{running / seen:.5f}")

        val_loss = validate(model, class_embedding, scheduler, val_loader, device)
        train_loss = running / max(seen, 1)
        print(f"[Epoch {epoch + 1}] train_noise_mse={train_loss:.6f} val_noise_mse={val_loss:.6f}")
        save_checkpoint(
            out_dir / "latest.pt", model, class_embedding, optimizer, scaler,
            epoch, global_step, min(best_val, val_loss), scheduler_config, args,
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                out_dir / "best.pt", model, class_embedding, optimizer, scaler,
                epoch, global_step, best_val, scheduler_config, args,
            )
            print(f"[Checkpoint] New best model: {out_dir / 'best.pt'}")
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                out_dir / f"epoch_{epoch + 1:04d}.pt", model, class_embedding,
                optimizer, scaler, epoch, global_step, best_val,
                scheduler_config, args,
            )


if __name__ == "__main__":
    main()
