"""
ThermalMirage — Main Runner
===========================

Usage examples:

  # 1. Train GAN on FLIR crops
  python run.py --mode train_gan --data_root /path/to/FLIR --epochs 100

  # 2. Run attack on validation set (FGSM fallback, no trained diffusion needed)
  python run.py --mode attack --data_root /path/to/FLIR --attack_mode hide --src_class 0

  # 3. Run full attack with diffusion + GAN prior
  python run.py --mode attack --data_root /path/to/FLIR \\
      --gan_ckpt checkpoints/thermal_gan.pt \\
      --attack_mode hide --src_class 0 --tgt_class 3

  # 4. Run ablation study
  python run.py --mode ablation --data_root /path/to/FLIR

  # 5. Visualize results
  python run.py --mode visualize --result_dir results/

Arguments:
  --data_root      Path to FLIR ADAS dataset root
  --mode           train_gan | attack | ablation | visualize
  --attack_mode    hide | misclassify | background
  --src_class      Source class index (0=person,1=bicycle,2=car,3=dog)
  --tgt_class      Target class index (-1 = background/None)
  --scene_cond     night | low_contrast | clutter | blur
  --batch_size     Batch size (default 4)
  --epochs         GAN training epochs (default 100)
  --device         cuda | cpu
  --gan_ckpt       Path to pretrained GAN checkpoint
  --out_dir        Output directory
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
import os
import shutil

from data.flir_dataset    import (FLIRDataset, FLIRCropDataset, build_dataloader,
                                   FLIR_CLASS_NAMES, NUM_CLASSES,
                                   FLIR_NATIVE_SIZE,      # (512, 640) — H×W
                                   FLIR_TRAIN_SIZE, FLIR_VAL_SIZE)
from models.thermal_gan   import ThermalGANTrainer, ThermalGenerator, ThermalClassPrior
from models.diffusion_editor import (
    ThermalUNet, DDPMScheduler,
    ThermalObjectEditor, ThermalContextEditor,
)
from attack.detector_wrapper import (
    FasterRCNNWrapper, AdversarialGuide, ThermalPlausibilityLoss
)
from attack.joint_attack  import ThermalMirageAttack, AttackConfig, BatchAttackRunner
from eval.evaluator       import AttackEvaluator, AblationRunner
from utils.visualize      import (
    side_by_side_plot, class_prior_grid, attack_progress_plot, ablation_bar_chart
)

# Use constants imported from flir_dataset — single source of truth
# FLIR_CLASS_NAMES = ["person", "bicycle", "car", "dog"]
# NUM_CLASSES      = 4
# JSON cat IDs:  person=1, bicycle=2, car=3, dog=17
FLIR_CLASSES = FLIR_CLASS_NAMES   # alias for legacy references inside this file


# ── FLIR Directory Structure Handler ─────────────────────────────────────────

def setup_flir_structure(data_root: str) -> str:
    """
    Automatically detect and handle different FLIR dataset directory structures.
    
    Expected structure:
        FLIR_ADAS_1_3/
        ├── images/
        │   ├── train/
        │   └── val/
        └── annotations/
            ├── instances_train.json
            └── instances_val.json
    
    Common actual structure:
        FLIR_ADAS_1_3/
        ├── train/
        │   ├── thermal_8_bit/
        │   └── thermal_annotations.json (or .json, or without extension)
        └── val/
            ├── thermal_8_bit/
            └── thermal_annotations.json
    
    This function creates the expected structure by copying files if needed.
    Returns the path to use as data_root.
    """
    import shutil
    
    root = Path(data_root).resolve()
    
    # Check if already in expected format
    if (root / "images" / "train").exists() and (root / "annotations").exists():
        ann_train = root / "annotations" / "instances_train.json"
        ann_val = root / "annotations" / "instances_val.json"
        if ann_train.exists() and ann_val.exists():
            print(f"[Setup] FLIR structure already correct at {root}")
            return str(root)
    
    print(f"[Setup] Setting up FLIR directory structure...")
    
    # Create images and annotations directories
    images_dir = root / "images"
    annotations_dir = root / "annotations"
    images_dir.mkdir(exist_ok=True)
    annotations_dir.mkdir(exist_ok=True)
    
    setup_successful = False
    
    # Process train and val splits
    for split in ["train", "val"]:
        split_dir = root / split
        
        if not split_dir.exists():
            print(f"  ⚠ Warning: {split_dir} does not exist")
            continue
        
        # Handle images - look for thermal_8_bit folder
        source_img_dir = split_dir / "thermal_8_bit"
        target_img_dir = images_dir / split
        
        if source_img_dir.exists() and source_img_dir.is_dir():
            if not target_img_dir.exists():
                try:
                    # Try symlink first (faster)
                    os.symlink(source_img_dir, target_img_dir, target_is_directory=True)
                    print(f"  ✓ Linked {split} images: {source_img_dir} → {target_img_dir}")
                    setup_successful = True
                except (OSError, NotImplementedError):
                    # If symlink fails, create directory and note it
                    print(f"  ⚠ Could not create symlink, you may need to copy files manually")
                    target_img_dir.mkdir(exist_ok=True)
            else:
                print(f"  ✓ {split} images already set up")
                setup_successful = True
        else:
            print(f"  ⚠ Could not find images at {source_img_dir}")
        
        # Handle annotations - try multiple possible filenames
        annotation_candidates = [
            split_dir / "thermal_annotations.json",
            split_dir / "thermal_annotations",
            split_dir / f"{split}_thermal_annotations.json",
            split_dir / "annotations.json",
        ]
        
        source_ann_file = None
        for candidate in annotation_candidates:
            if candidate.exists() and candidate.is_file():
                source_ann_file = candidate
                break
        
        target_ann_file = annotations_dir / f"instances_{split}.json"
        
        if source_ann_file:
            if not target_ann_file.exists():
                try:
                    # Try symlink first
                    os.symlink(source_ann_file, target_ann_file)
                    print(f"  ✓ Linked {split} annotations: {source_ann_file.name} → {target_ann_file}")
                    setup_successful = True
                except (OSError, NotImplementedError):
                    # Fall back to copying the file
                    try:
                        shutil.copy2(source_ann_file, target_ann_file)
                        print(f"  ✓ Copied {split} annotations: {source_ann_file.name} → {target_ann_file}")
                        setup_successful = True
                    except Exception as e:
                        print(f"  ✗ Failed to copy {split} annotations: {e}")
            else:
                print(f"  ✓ {split} annotations already set up")
                setup_successful = True
        else:
            print(f"  ✗ Could not find annotation file for {split}")
            print(f"     Tried: {[c.name for c in annotation_candidates]}")
    
    if setup_successful:
        print(f"[Setup] FLIR structure setup complete")
    else:
        print(f"[Setup] ERROR: Could not set up FLIR structure")
        print(f"\nPlease check your directory structure:")
        print(f"  Current root: {root}")
        print(f"\nExpected to find:")
        print(f"  - {root}/train/thermal_8_bit/ (images)")
        print(f"  - {root}/train/thermal_annotations.json (or similar)")
        print(f"  - {root}/val/thermal_8_bit/ (images)")
        print(f"  - {root}/val/thermal_annotations.json (or similar)")
        print(f"\nRun 'ls -la {root}/train/' to see what's actually there.")
    
    return str(root)


# ── Build components ──────────────────────────────────────────────────────────

def build_detector(device: str, num_classes: int = NUM_CLASSES):
    """
    Build Faster R-CNN fine-tuned for FLIR classes.
    Internally uses num_classes+1 outputs (background + 4 FLIR classes).
    Labels returned by forward() are always internal 0-3.
    """
    print(f"[Setup] Loading Faster R-CNN  "
          f"(background + {num_classes} FLIR classes = {num_classes+1} total)")
    detector = FasterRCNNWrapper(
        num_flir_classes = num_classes,   # 4  (person, bicycle, car, dog)
        pretrained       = False,
        device           = device,
    )
    # Load YOUR fine-tuned weights
    ckpt_path = Path("checkpoints/fasterrcnn_flir.pt")
    if ckpt_path.exists():
        print(f"[Setup] Loading fine-tuned FLIR weights from {ckpt_path}")
        detector.model.load_state_dict(
            torch.load(str(ckpt_path), map_location=device)
        )
        print("[Setup] Fine-tuned detector loaded successfully")
    else:
        print(f"[Setup] WARNING: {ckpt_path} not found!")
        print("[Setup] Using randomly initialized weights - attack will fail!")
        print(f"[Setup] Make sure you ran: python fasterR-CNN.py first")
    
    detector.model.eval()  # Set to evaluation mode for inference
    return detector


def build_diffusion(device: str, img_size: int = 256):
    """Build lightweight diffusion UNet + scheduler."""
    print("[Setup] Building diffusion model...")
    unet = ThermalUNet(
        img_channels=3,
        base_ch=64,
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(8,),
        cond_dim=64,
        t_dim=256,
        img_size=img_size,
    )
    scheduler = DDPMScheduler(T=1000)
    return unet, scheduler


def build_attack(
    args,
    detector,
    gan_trainer=None,
    unet=None,
    scheduler=None,
    device: str = "cuda",
) -> ThermalMirageAttack:
    """Assemble the full ThermalMirageAttack pipeline."""

    tgt_class = args.tgt_class if args.tgt_class >= 0 else None

    # GAN prior
    gan_prior = None
    if gan_trainer is not None and unet is not None:
        import torchvision.models as tvm
        feat_extractor = tvm.resnet18(pretrained=True)
        feat_extractor.fc = torch.nn.Identity()
        feat_extractor.eval()
        gan_prior = ThermalClassPrior(
            generator       = gan_trainer.G,
            feature_extractor = feat_extractor,
            num_classes     = NUM_CLASSES,
            device          = device,
        )
        print("[Setup] Building GAN priors...")
        gan_prior.build_priors(n_samples=128)

    # Object editor (diffusion)
    obj_editor = None
    if unet is not None and scheduler is not None and tgt_class is not None:
        class_emb = torch.nn.Embedding(NUM_CLASSES, 64)
        obj_editor = ThermalObjectEditor(
            unet            = unet,
            scheduler       = scheduler,
            class_embeddings= class_emb,
            device          = device,
            inpaint_start_t = args.obj_start_t,
        )

    # Context editor
    scene_emb = torch.nn.Embedding(5, 64)   # 5 scene conditions
    ctx_editor = ThermalContextEditor(
        unet          = unet if unet is not None else ThermalUNet(img_size=256),
        scheduler     = scheduler if scheduler is not None else DDPMScheduler(),
        scene_embeddings = scene_emb,
        device        = device,
        ctx_start_t   = args.ctx_start_t,
    )

    # Adversarial guide
    guide = AdversarialGuide(detector, loss_scale=1.0)

    # Plausibility loss
    plaus_loss = ThermalPlausibilityLoss(
        lambda_smooth=1.0, lambda_boundary=0.5,
        lambda_hist=0.5,   lambda_range=1.0,
    )

    # Attack config
    cfg = AttackConfig(
        mode              = args.attack_mode,
        wrong_class       = tgt_class,
        src_class         = args.src_class,
        tgt_class         = tgt_class,
        scene_condition   = args.scene_cond,
        use_analytic_ctx  = (unet is None),   # analytic fallback if no trained diffusion
        context_strength  = 0.35,
        guidance_scale    = 0.08,
        n_refinement_steps= 3,
        max_perturbation  = 0.15,
        use_gan_prior     = (gan_prior is not None),
    )

    attack = ThermalMirageAttack(
        gan_prior  = gan_prior,
        obj_editor = obj_editor,
        ctx_editor = ctx_editor,
        guide      = guide,
        plaus_loss = plaus_loss,
        config     = cfg,
        device     = device,
    )
    return attack


# ── Mode: Train GAN ───────────────────────────────────────────────────────────

def run_train_gan(args):
    from torch.utils.data import DataLoader
    print("[Mode] Training Thermal GAN")

    crop_ds = FLIRCropDataset(
        root         = args.data_root,
        split        = "train",
        classes      = FLIR_CLASS_NAMES,
        crop_size    = 64,
        min_box_area = 400,
    )
    # Note: FLIRCropDataset internally uses FLIR_NATIVE_SIZE for the base dataset
    loader = DataLoader(crop_ds, batch_size=32, shuffle=True, num_workers=2)

    trainer = ThermalGANTrainer(
        num_classes = NUM_CLASSES,
        z_dim       = 128,
        device      = args.device,
    )
    trainer.train(loader, epochs=args.epochs, log_every=10)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    trainer.save(str(Path(args.out_dir) / "thermal_gan.pt"))

    print("[Done] GAN training complete.")


# ── Mode: Attack ──────────────────────────────────────────────────────────────

def run_attack(args):
    print("[Mode] Running ThermalMirage Attack")
    device = args.device

    # Dataloader — use native FLIR resolution (512, 640), NOT (640, 640)
    val_loader = build_dataloader(
        root        = args.data_root,
        split       = "val",
        batch_size  = args.batch_size,
        num_workers = 2,
        img_size    = FLIR_NATIVE_SIZE,   # (512, 640)  ← matches sensor output
        normalize   = False,              # we want [0,1] raw tensors for the attack
        return_masks= True,
    )

    # Detector
    detector = build_detector(device)

    # Optional GAN
    gan_trainer = None
    if args.gan_ckpt and Path(args.gan_ckpt).exists():
        print(f"[Setup] Loading GAN from {args.gan_ckpt}")
        from models.thermal_gan import ThermalGANTrainer
        gan_trainer = ThermalGANTrainer(num_classes=NUM_CLASSES, device=device)
        gan_trainer.load(args.gan_ckpt)

    # Optional diffusion
    unet, scheduler = None, None
    # Uncomment if you have a trained diffusion model:
    # unet, scheduler = build_diffusion(device)
    # unet.load_state_dict(torch.load(args.diffusion_ckpt)['unet'])

    # Build attack
    attack   = build_attack(args, detector, gan_trainer, unet, scheduler, device)
    runner   = BatchAttackRunner(
        attack           = attack,
        detector_fn      = lambda x: detector.forward(x),
        iou_threshold    = 0.5,
        conf_threshold   = 0.25,
    )
    evaluator = AttackEvaluator(num_classes=NUM_CLASSES)

    # Run
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    img_save_dir = out_dir / "images"
    img_save_dir.mkdir(exist_ok=True)

    for batch_idx, batch in enumerate(val_loader):
        print(f"\n[Attack] Batch {batch_idx + 1}")
        results = runner.run_batch(batch, verbose=True)

        for ri, res in enumerate(results):
            all_results.append({
                k: v for k, v in res.items()
                if not isinstance(v, torch.Tensor)
            })

            # Save comparison plots
            if res.get("adv_image") is not None and "images_raw" in batch:
                fig = side_by_side_plot(
                    clean      = batch["images_raw"][ri:ri+1],
                    adv        = res["adv_image"].unsqueeze(0) if res["adv_image"].dim()==3 else res["adv_image"],
                    gt_boxes   = batch["boxes"][ri],
                    gt_labels  = batch["labels"][ri],
                    title      = f"Image {batch['image_ids'][ri]}",
                    save_path  = str(img_save_dir / f"result_{batch_idx}_{ri}.png"),
                )
                import matplotlib.pyplot as plt
                plt.close(fig)

        if batch_idx >= 9:   # limit for quick runs; remove for full eval
            break

    # Aggregate and print
    summary = evaluator.aggregate(all_results)
    evaluator.print_summary(summary)
    evaluator.save_json(summary, str(out_dir / "eval_summary.json"))

    with open(out_dir / "per_image_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n[Done] Results saved to {out_dir}")


# ── Mode: Ablation ────────────────────────────────────────────────────────────

def run_ablation(args):
    print("[Mode] Ablation Study")
    from dataclasses import replace, fields

    device = args.device
    detector = build_detector(device)
    evaluator = AttackEvaluator(num_classes=NUM_CLASSES)

    val_loader = build_dataloader(
        root        = args.data_root,
        split       = "val",
        batch_size  = args.batch_size,
        num_workers = 2,
        img_size    = FLIR_NATIVE_SIZE,   # (512, 640)
        normalize   = False,
        return_masks= True,
    )

    # Collect a fixed set of batches
    batches = []
    for b in val_loader:
        batches.append(b)
        if len(batches) >= 5:
            break

    base_cfg  = AttackConfig(
        mode          = args.attack_mode,
        src_class     = args.src_class,
        tgt_class     = args.tgt_class if args.tgt_class >= 0 else None,
        scene_condition= args.scene_cond,
    )

    def attack_factory(cfg):
        return build_attack(
            argparse.Namespace(
                tgt_class=cfg.tgt_class if cfg.tgt_class is not None else -1,
                src_class=cfg.src_class,
                attack_mode=cfg.mode,
                scene_cond=cfg.scene_condition,
                obj_start_t=500,
                ctx_start_t=300,
            ),
            detector,
            device=device,
        )

    def runner_factory(attack):
        return BatchAttackRunner(
            attack       = attack,
            detector_fn  = lambda x: detector.forward(x),
        )

    ablation = AblationRunner(base_cfg, attack_factory, runner_factory, evaluator)
    summaries = ablation.run_all(batches, verbose=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(summaries, f, indent=2, default=str)

    fig = ablation_bar_chart(
        summaries,
        metrics    = ["attack_success_rate", "false_negative_rate", "mean_conf_drop"],
        save_path  = str(out_dir / "ablation_chart.png"),
    )
    import matplotlib.pyplot as plt
    plt.close(fig)

    print(f"[Done] Ablation results saved to {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def resolve_device(requested: str) -> str:
    """
    Validate the requested device and fall back to CPU if unavailable.

    Cases handled:
      - User passes --device cuda  but no GPU is present  → fall back to cpu
      - User passes --device cuda  and GPU is available    → use cuda
      - User passes --device cpu                           → always use cpu
      - CUDA driver not installed (RuntimeError on init)   → fall back to cpu

    Returns a validated device string ('cuda' or 'cpu').
    """
    if requested == "cpu":
        print("[Device] Using CPU as requested.")
        return "cpu"

    # Requested cuda — check it is actually usable
    if not torch.cuda.is_available():
        print("[Device] WARNING: --device cuda requested but torch.cuda.is_available() "
              "returned False.")
        print("[Device] Falling back to CPU. Training will be slower.")
        print("[Device] If you expected a GPU, check: nvidia-smi  and  "
              "that your SLURM --gres=gpu: line is correct.")
        return "cpu"

    # Try to actually initialise CUDA — catches missing driver errors
    try:
        torch.zeros(1).cuda()
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[Device] Using GPU: {name}  ({mem:.1f} GB VRAM)")
        return "cuda"
    except RuntimeError as e:
        print(f"[Device] WARNING: CUDA initialisation failed: {e}")
        print("[Device] Falling back to CPU.")
        return "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="ThermalMirage — FLIR Attack Pipeline")
    p.add_argument("--mode",        type=str, default="attack",
                   choices=["train_gan", "attack", "ablation", "visualize"])
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--out_dir",     type=str, default="results")
    p.add_argument("--device",      type=str, default="auto",
                   help="'cuda', 'cpu', or 'auto' (default: auto-detect)")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--epochs",      type=int, default=200)
    p.add_argument("--gan_ckpt",    type=str, default="")
    p.add_argument("--diffusion_ckpt", type=str, default="")

    # Attack options
    p.add_argument("--attack_mode", type=str, default="hide",
                   choices=["hide", "misclassify", "background"])
    p.add_argument("--src_class",   type=int, default=0,
                   help="0=person, 1=bicycle, 2=car, 3=dog")
    p.add_argument("--tgt_class",   type=int, default=-1,
                   help="Target class (-1 = background)")
    p.add_argument("--scene_cond",  type=str, default="night",
                   choices=["night", "low_contrast", "clutter", "blur"])
    p.add_argument("--obj_start_t", type=int, default=500)
    p.add_argument("--ctx_start_t", type=int, default=300)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Setup FLIR directory structure before doing anything else
    args.data_root = setup_flir_structure(args.data_root)

    # Resolve device BEFORE any model is created
    # Handles: --device auto, --device cuda (with no GPU), --device cpu
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = resolve_device(args.device)   # validates + falls back safely

    print(f"\n[ThermalMirage] Mode={args.mode}  Device={args.device}")
    print(f"  src_class={args.src_class}  tgt_class={args.tgt_class}")
    print(f"  attack_mode={args.attack_mode}  scene={args.scene_cond}")

    if args.mode == "train_gan":
        run_train_gan(args)
    elif args.mode == "attack":
        run_attack(args)
    elif args.mode == "ablation":
        run_ablation(args)
    else:
        print("[Visualize mode] Pass --result_dir to load saved results.")
