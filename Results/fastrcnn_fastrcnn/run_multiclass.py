"""
run_multiclass.py
Multi-class adaptive attack mode for ThermalMirage.

Usage:
    # Adaptive mode - test all strategies per object
    python run_multiclass.py \\
        --mode attack_adaptive \\
        --data_root /path/to/FLIR_ADAS_1_3 \\
        --gan_ckpt checkpoints/thermal_gan.pt \\
        --detector_ckpt checkpoints/fasterrcnn_flir.pt \\
        --batch_size 1 \\
        --device cuda \\
        --out_dir results/adaptive_attack/
    
    # Profiled mode - pre-compute best strategies, then use them
    python run_multiclass.py \\
        --mode profile \\
        --data_root /path/to/FLIR_ADAS_1_3 \\
        --gan_ckpt checkpoints/thermal_gan.pt \\
        --detector_ckpt checkpoints/fasterrcnn_flir.pt \\
        --profile_samples 200 \\
        --device cuda \\
        --out_dir checkpoints/
    
    # Attack using profiled strategies (faster)
    python run_multiclass.py \\
        --mode attack_profiled \\
        --data_root /path/to/FLIR_ADAS_1_3 \\
        --gan_ckpt checkpoints/thermal_gan.pt \\
        --detector_ckpt checkpoints/fasterrcnn_flir.pt \\
        --profile_path checkpoints/strategy_profile.json \\
        --batch_size 2 \\
        --device cuda \\
        --out_dir results/profiled_attack/
"""

import sys
import os
# CRITICAL: Add current directory to path FIRST before any other imports
# This prevents conflicts with installed packages that have the same name
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Import from your actual file structure
# Based on your directory layout: data/, attack/, models/, utils/, eval/
try:
    from data.flir_dataset import build_dataloader, FLIR_NATIVE_SIZE, NUM_CLASSES, FLIR_CLASS_NAMES
except ImportError:
    try:
        from flir_dataset import build_dataloader, FLIR_NATIVE_SIZE, NUM_CLASSES, FLIR_CLASS_NAMES
    except ImportError:
        print("[Error] Cannot import from flir_dataset. Make sure data/flir_dataset.py exists.")
        sys.exit(1)

try:
    from attack.detector_wrapper import FasterRCNNWrapper
except ImportError:
    try:
        from detector_wrapper import FasterRCNNWrapper
    except ImportError:
        print("[Error] Cannot import FasterRCNNWrapper. Make sure attack/detector_wrapper.py exists.")
        sys.exit(1)

try:
    from models.thermal_gan import ThermalGANTrainer, ThermalClassPrior
except ImportError:
    try:
        from thermal_gan import ThermalGANTrainer, ThermalClassPrior
    except ImportError:
        print("[Error] Cannot import ThermalGANTrainer. Make sure models/thermal_gan.py exists.")
        sys.exit(1)

# Import the attack selector (should be in same directory as this script)
from attack_selector import AttackStrategySelector

# Optional imports - will gracefully handle if not present
try:
    from eval.evaluator import AttackEvaluator
except ImportError:
    try:
        from evaluator import AttackEvaluator
    except ImportError:
        AttackEvaluator = None
        print("[Warning] AttackEvaluator not found, evaluation features disabled")

# Visualize import - check utils folder first
side_by_side_plot = None
try:
    # Try importing from utils folder (most likely location based on user's structure)
    from utils.visualize import side_by_side_plot
    print("[Info] Using utils.visualize for visualization")
except ImportError:
    try:
        # Fallback: check if visualize.py exists in current directory
        local_visualize_path = os.path.join(current_dir, 'visualize.py')
        if os.path.exists(local_visualize_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("local_visualize", local_visualize_path)
            local_visualize = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(local_visualize)
            side_by_side_plot = local_visualize.side_by_side_plot
            print("[Info] Using local visualize.py for visualization")
        else:
            # Try viz.visualize as final fallback
            try:
                from viz.visualize import side_by_side_plot
                print("[Info] Using viz.visualize for visualization")
            except ImportError:
                pass
    except Exception as e:
        print(f"[Warning] Could not load visualization: {e}")

if side_by_side_plot is None:
    print("[Warning] Visualization features will be disabled")

# Prefer the full ThermalMirage attack. SimpleAttackWrapper is only an emergency fallback.
FULL_ATTACK_AVAILABLE = False
SIMPLE_FALLBACK_AVAILABLE = False

try:
    from attack.joint_attack import ThermalMirageAttack, AttackConfig
    from attack.detector_wrapper import AdversarialGuide, ThermalPlausibilityLoss
    from models.diffusion_editor import (
        ThermalUNet, DDPMScheduler, ThermalObjectEditor, ThermalContextEditor
    )
    FULL_ATTACK_AVAILABLE = True
    print("[Info] Full ThermalMirageAttack is available")
except ImportError as e:
    print(f"[Warning] Full ThermalMirageAttack import failed: {e}")
    try:
        from joint_attack import ThermalMirageAttack, AttackConfig
        from detector_wrapper import AdversarialGuide, ThermalPlausibilityLoss
        from diffusion_editor import ThermalUNet, DDPMScheduler, ThermalObjectEditor, ThermalContextEditor
        FULL_ATTACK_AVAILABLE = True
        print("[Info] Full ThermalMirageAttack is available from local files")
    except ImportError as e2:
        print(f"[Warning] Local full attack import failed: {e2}")

try:
    from simple_attack_wrapper import SimpleAttackWrapper
    SIMPLE_FALLBACK_AVAILABLE = True
except ImportError:
    SimpleAttackWrapper = None

def parse_args():
    parser = argparse.ArgumentParser(description='ThermalMirage Multi-Class Attack')
    
    # Mode
    parser.add_argument('--mode', type=str, required=True,
                        choices=['profile', 'attack_adaptive', 'attack_profiled'],
                        help='profile: build strategy profile | '
                             'attack_adaptive: test all strategies per object | '
                             'attack_profiled: use pre-computed strategies')
    
    # Data
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to FLIR dataset root')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val'],
                        help='Dataset split to use')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (1 recommended for adaptive attack)')
    parser.add_argument('--num_images', type=int, default=None,
                        help='Number of images to process (None = all)')
    
    # Model checkpoints
    parser.add_argument('--gan_ckpt', type=str, required=True,
                        help='Path to GAN checkpoint')
    parser.add_argument('--detector_ckpt', type=str, default=None,
                        help='Path to detector checkpoint (optional)')
    parser.add_argument('--diffusion_ckpt', type=str, default=None,
                        help='Path to diffusion model checkpoint (optional)')
    
    # Profile mode specific
    parser.add_argument('--profile_samples', type=int, default=100,
                        help='Number of images to use for profiling')
    parser.add_argument('--profile_path', type=str,
                        default='checkpoints/strategy_profile.json',
                        help='Path to save/load strategy profile')
    
    # Attack parameters
    parser.add_argument('--test_all_strategies', action='store_true',
                        help='Test all possible strategies per object (slower but more thorough)')
    parser.add_argument('--top_k_strategies', type=int, default=3,
                        help='Number of top strategies to test when using profiled mode')
    
    # Scene/context attack settings
    parser.add_argument('--scene_cond', type=str, default=None,
                        choices=['day', 'night', 'low_contrast', 'high_contrast'],
                        help='Optional scene label/filter kept for backward compatibility')
    parser.add_argument('--context_attack', type=str, default='sequential',
                        choices=['sequential', 'contrast', 'smooth', 'cool', 'adaptive_cool',
                                 'night', 'low_contrast', 'blur', 'clutter'],
                        help='Context attack mode. sequential = contrast -> smoothing -> adaptive cooling')
    parser.add_argument('--context_strength', type=float, default=0.08,
                        help='Context L_inf epsilon / strength for physics-informed context perturbation')
    parser.add_argument('--context_tau', type=float, default=20.0,
                        help='Decay rate tau for distance-aware M_soft mask')
    parser.add_argument('--epsilon_tv', type=float, default=0.02,
                        help='TV smoothing threshold for masked context perturbation')
    parser.add_argument('--context_alpha', type=float, default=0.45,
                        help='Alpha for contrast reduction')
    parser.add_argument('--context_beta', type=float, default=0.65,
                        help='Beta for thermal smoothing')
    parser.add_argument('--context_gamma', type=float, default=0.45,
                        help='Gamma for adaptive context cooling')
    parser.add_argument('--max_perturbation', type=float, default=0.15,
                        help='Final L_inf perturbation budget for full attack')
    parser.add_argument('--n_refinement_steps', type=int, default=3,
                        help='Detector-in-the-loop refinement steps after object/context fusion')
    
    # Output
    parser.add_argument('--out_dir', type=str, default='results/multiclass',
                        help='Output directory for results')
    parser.add_argument('--save_images', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    
    return parser.parse_args()


def build_attack_engine(args):
    """Build the attack engine with full object+context attack preferred."""

    device = args.device

    # Load GAN
    print(f"[Setup] Loading GAN from {args.gan_ckpt}")
    gan_trainer = ThermalGANTrainer(num_classes=NUM_CLASSES, device=device)
    gan_trainer.load(args.gan_ckpt)
    gan_trainer.G.eval()
    print(f"[ThermalGANTrainer] Running on {device}")
    print(f"[GAN] Loaded checkpoint <- {args.gan_ckpt}")

    # Load detector
    print(f"[Setup] Loading detector...")
    detector = FasterRCNNWrapper(
        num_flir_classes=NUM_CLASSES,
        pretrained=True,
        device=device,
    )
    if args.detector_ckpt:
        detector.model.load_state_dict(
            torch.load(args.detector_ckpt, map_location=device)
        )
        print(f"[Detector] Loaded checkpoint <- {args.detector_ckpt}")
    detector.model.eval()

    # Build class prior extractor
    print(f"[Setup] Building class priors...")
    class_prior = ThermalClassPrior(
        generator=gan_trainer.G,
        feature_extractor=detector.get_feature_extractor(),
        num_classes=NUM_CLASSES,
        device=device,
    )
    class_prior.build_priors(n_samples=256)

    # Prefer full ThermalMirageAttack. This uses object branch + physics-informed context branch.
    if FULL_ATTACK_AVAILABLE:
        try:
            print("[Setup] Initializing full ThermalMirageAttack engine...")

            unet, scheduler = None, None
            obj_editor, ctx_editor = None, None

            # Optional diffusion branch. If unavailable, object branch falls back internally to FGSM,
            # while context branch uses the physics-informed analytic attack.
            if args.diffusion_ckpt and Path(args.diffusion_ckpt).exists():
                print(f"[Setup] Loading diffusion checkpoint <- {args.diffusion_ckpt}")
                unet = ThermalUNet(
                    img_channels=3,
                    base_ch=64,
                    channel_mult=(1, 2, 4),
                    num_res_blocks=2,
                    attn_resolutions=(8,),
                    cond_dim=64,
                    t_dim=256,
                    img_size=256,
                )
                scheduler = DDPMScheduler(T=1000)
                ckpt = torch.load(args.diffusion_ckpt, map_location=device)
                if isinstance(ckpt, dict) and "unet" in ckpt:
                    unet.load_state_dict(ckpt["unet"])
                else:
                    unet.load_state_dict(ckpt)
                unet.to(device).eval()

                class_emb = torch.nn.Embedding(NUM_CLASSES, 64).to(device)
                scene_emb = torch.nn.Embedding(5, 64).to(device)
                obj_editor = ThermalObjectEditor(
                    unet=unet,
                    scheduler=scheduler,
                    class_embeddings=class_emb,
                    device=device,
                    inpaint_start_t=500,
                )
                ctx_editor = ThermalContextEditor(
                    unet=unet,
                    scheduler=scheduler,
                    scene_embeddings=scene_emb,
                    device=device,
                    ctx_start_t=300,
                )
            else:
                print("[Setup] No diffusion checkpoint supplied; object branch will use FGSM fallback when needed")
                print("[Setup] Context branch will use physics-informed contrast/smoothing/cooling attack")

            guide = AdversarialGuide(detector, loss_scale=1.0)
            plaus_loss = ThermalPlausibilityLoss(
                lambda_smooth=1.0,
                lambda_boundary=0.5,
                lambda_hist=0.5,
                lambda_range=1.0,
            )

            cfg = AttackConfig(
                mode="hide",
                wrong_class=None,
                src_class=0,
                tgt_class=None,
                scene_condition=args.context_attack,
                use_analytic_ctx=True,  # physics-informed context attack is analytic and mask-based
                context_strength=args.context_strength,
                context_tau=args.context_tau,
                context_epsilon_tv=args.epsilon_tv,
                context_alpha=args.context_alpha,
                context_beta=args.context_beta,
                context_gamma=args.context_gamma,
                guidance_scale=0.08,
                n_refinement_steps=args.n_refinement_steps,
                max_perturbation=args.max_perturbation,
                use_gan_prior=True,
            )

            attack_engine = ThermalMirageAttack(
                gan_prior=class_prior,
                obj_editor=obj_editor,
                ctx_editor=ctx_editor,
                guide=guide,
                plaus_loss=plaus_loss,
                config=cfg,
                device=device,
            )
            print(f"[Setup] Full attack engine ready | context_attack={args.context_attack}")
            return attack_engine, detector
        except Exception as e:
            print(f"[Warning] Full attack engine initialization failed: {e}")
            print("[Warning] Falling back to SimpleAttackWrapper")

    if SIMPLE_FALLBACK_AVAILABLE:
        attack_engine = SimpleAttackWrapper(
            detector=detector,
            gan_generator=gan_trainer.G,
            class_prior=class_prior,
            device=device,
        )
        return attack_engine, detector

    print("[Error] No attack engine available. Check joint_attack.py or simple_attack_wrapper.py.")
    sys.exit(1)


def mode_profile(args):
    """Profile mode: determine best attack strategies per class."""
    
    print("\n" + "="*60)
    print("  MODE: PROFILE - Building Strategy Profile")
    print("="*60)
    
    # Build attack engine
    attack_engine, detector = build_attack_engine(args)
    
    # Build dataloader
    val_loader = build_dataloader(
        root=args.data_root,
        split='val',  # Always use validation for profiling
        batch_size=1,
        num_workers=4,
        img_size=FLIR_NATIVE_SIZE,
        normalize=False,
        return_masks=False
    )
    
    # Initialize strategy selector
    selector = AttackStrategySelector(
        num_classes=NUM_CLASSES,
        class_names=FLIR_CLASS_NAMES,
        device=args.device,
        mode='profiled'
    )
    
    # Run profiling
    selector.profile_strategies(
        attack_engine=attack_engine,
        val_loader=val_loader,
        save_path=args.profile_path,
        max_images=args.profile_samples
    )
    
    print(f"\n✓ Profiling complete. Results saved to {args.profile_path}")


def mode_attack(args, use_profile=False):
    """Attack mode: attack all objects with adaptive strategy selection."""
    
    mode_str = "PROFILED" if use_profile else "ADAPTIVE"
    print("\n" + "="*60)
    print(f"  MODE: ATTACK ({mode_str}) - Multi-Class Attack")
    print("="*60)
    
    # Build attack engine
    attack_engine, detector = build_attack_engine(args)
    
    # Build dataloader
    test_loader = build_dataloader(
        root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=4,
        img_size=FLIR_NATIVE_SIZE,
        normalize=False,
        return_masks=False
    )
    
    # Initialize strategy selector
    selector = AttackStrategySelector(
        num_classes=NUM_CLASSES,
        class_names=FLIR_CLASS_NAMES,
        device=args.device,
        mode='profiled' if use_profile else 'adaptive'
    )
    
    # Load profile if using profiled mode
    if use_profile:
        print(f"[Setup] Loading strategy profile from {args.profile_path}")
        selector.load_profile(args.profile_path)
    
    # Prepare output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = AttackEvaluator(num_classes=NUM_CLASSES)
    
    # Attack loop
    all_results = []
    num_processed = 0
    
    iterator = tqdm(test_loader, desc="Attacking images") if not args.verbose else test_loader
    
    for batch_idx, batch in enumerate(iterator):
        if args.num_images and num_processed >= args.num_images:
            break
        
        images = batch['images']
        boxes_list = batch['boxes']
        labels_list = batch['labels']
        image_ids = batch.get('image_ids', [batch_idx] * len(images))
        
        for img_idx, (image, boxes, labels, img_id) in enumerate(
            zip(images, boxes_list, labels_list, image_ids)
        ):
            if len(boxes) == 0:
                continue  # Skip images with no objects
            
            if args.verbose:
                print(f"\n{'='*60}")
                print(f"Image {num_processed + 1}: ID={img_id} | {len(boxes)} objects")
                print(f"{'='*60}")
            
            # Run multi-class attack on this image
            attack_result = selector.attack_multi_class_image(
                attack_engine=attack_engine,
                image=image.unsqueeze(0).to(args.device),
                boxes=boxes.to(args.device),
                labels=labels.to(args.device),
                test_all=args.test_all_strategies if not use_profile else False,
                verbose=args.verbose
            )
            
            adv_image = attack_result['adv_image']
            
            # Get detector predictions on clean and adversarial
            with torch.no_grad():
                clean_dets = detector.detect(image.unsqueeze(0).to(args.device))
                adv_dets = detector.detect(adv_image)
            
            object_successes = [
                bool(result.success)
                for _, _, result in attack_result['strategies_used']
            ]
            num_successful_objects = sum(object_successes)
            image_attack_success = num_successful_objects > 0

            # Compute metrics
            result_metrics = {
                'image_id': img_id,
                'num_objects': len(boxes),
                'num_successful_objects': num_successful_objects,
                'object_attack_success_rate': (
                    num_successful_objects / len(boxes) if len(boxes) > 0 else 0.0
                ),
                'image_attack_success': image_attack_success,
                'overall_success_rate': attack_result['overall_success_rate'],
                'per_class_success': attack_result['per_class_success'],
                'strategies_used': [
                    str(strat) for _, strat, _ in attack_result['strategies_used']
                ],
                'n_det_clean': len(clean_dets[0]['boxes']) if clean_dets else 0,
                'n_det_adv': len(adv_dets[0]['boxes']) if adv_dets else 0,
            }
            
            all_results.append(result_metrics)
            
            # Save visualization
            if args.save_images:
                img_out_path = out_dir / f"attack_{num_processed:04d}.png"
                side_by_side_plot(
                    clean=image.unsqueeze(0),
                    adv=adv_image.cpu(),
                    gt_boxes=boxes,
                    gt_labels=labels,
                    clean_dets=clean_dets[0] if clean_dets else None,
                    adv_dets=adv_dets[0] if adv_dets else None,
                    title=f"Image {num_processed} | Success: {attack_result['overall_success_rate']:.1%}",
                    save_path=str(img_out_path)
                )
                plt.close('all')
            
            num_processed += 1
    
    # Aggregate and print results
    print("\n" + "="*60)
    print("  ATTACK RESULTS SUMMARY")
    print("="*60)
    
    total_objects = sum(r['num_objects'] for r in all_results)
    total_successful_objects = sum(r['num_successful_objects'] for r in all_results)
    object_level_asr = (
        total_successful_objects / total_objects if total_objects > 0 else 0.0
    )
    image_level_asr = (
        np.mean([r['image_attack_success'] for r in all_results])
        if all_results else 0.0
    )
    overall_asr = np.mean([r['overall_success_rate'] for r in all_results]) if all_results else 0.0
    print(f"Images processed: {num_processed}")
    print(f"Object-level Attack Success Rate: {object_level_asr:.1%} "
          f"({total_successful_objects}/{total_objects} objects)")
    print(f"Image-level Attack Success Rate: {image_level_asr:.1%} "
          f"({sum(r['image_attack_success'] for r in all_results)}/{len(all_results)} images with >=1 success)")
    print(f"Mean per-image object success rate: {overall_asr:.1%}")
    
    # Per-class breakdown
    per_class_asr = {}
    for cls in range(NUM_CLASSES):
        class_successes = []
        for r in all_results:
            if cls in r['per_class_success']:
                class_successes.append(r['per_class_success'][cls])
        if class_successes:
            per_class_asr[cls] = np.mean(class_successes)
            print(f"  {FLIR_CLASS_NAMES[cls]:10s}: {per_class_asr[cls]:.1%}")
    
    # Strategy usage statistics
    print(f"\nStrategy Usage:")
    strategy_counts = {}
    for r in all_results:
        for strat_str in r['strategies_used']:
            strategy_counts[strat_str] = strategy_counts.get(strat_str, 0) + 1
    
    for strat, count in sorted(strategy_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {strat}: {count} times")
    
    # Save results to JSON
    results_path = out_dir / "attack_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'summary': {
                'num_images': num_processed,
                'num_objects': total_objects,
                'num_successful_objects': total_successful_objects,
                'object_level_attack_success_rate': object_level_asr,
                'image_level_attack_success_rate': image_level_asr,
                'overall_success_rate': overall_asr,
                'overall_success_rate_definition': 'mean per-image object success rate',
                'per_class_success_rate': per_class_asr,
                'strategy_usage': strategy_counts
            },
            'per_image_results': all_results,
            'strategy_statistics': dict(selector.strategy_stats)
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    print("="*60)


if __name__ == "__main__":
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Route to appropriate mode
    if args.mode == 'profile':
        mode_profile(args)
    elif args.mode == 'attack_adaptive':
        mode_attack(args, use_profile=False)
    elif args.mode == 'attack_profiled':
        mode_attack(args, use_profile=True)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
