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

# Import attack module
# Using SimpleAttackWrapper for now - replace with full ThermalMirageAttack when ready
try:
    from simple_attack_wrapper import SimpleAttackWrapper as AttackEngine
    print("[Info] Using SimpleAttackWrapper for testing")
    print("       Replace with full ThermalMirageAttack when attack_single_object() is implemented")
except ImportError:
    try:
        from attack.joint_attack import ThermalMirageAttack as AttackEngine
        print("[Info] Using ThermalMirageAttack from attack/")
    except ImportError:
        try:
            from joint_attack import ThermalMirageAttack as AttackEngine
            print("[Info] Using ThermalMirageAttack")
        except ImportError:
            AttackEngine = None
            print("[Error] No attack engine found!")
            print("        Please create simple_attack_wrapper.py in your project root")

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
    
    # Scene conditions (optional filtering)
    parser.add_argument('--scene_cond', type=str, default=None,
                        choices=['day', 'night', 'low_contrast', 'high_contrast'],
                        help='Filter images by scene condition')
    
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
    """Build the core attack engine with all components."""
    
    device = args.device
    
    # Load GAN
    print(f"[Setup] Loading GAN from {args.gan_ckpt}")
    gan_trainer = ThermalGANTrainer(num_classes=NUM_CLASSES, device=device)
    gan_trainer.load(args.gan_ckpt)
    gan_trainer.G.eval()
    print(f"[ThermalGANTrainer] Running on {device}")
    print(f"[GAN] Loaded checkpoint <- {args.gan_ckpt}")
    
    # Load Detector
    print(f"[Setup] Loading detector...")
    detector = FasterRCNNWrapper(
        num_flir_classes=NUM_CLASSES,
        pretrained=True,
        device=device
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
        device=device
    )
    class_prior.build_priors(n_samples=256)
    
    # Build attack engine using SimpleAttackWrapper
    print(f"[Setup] Initializing attack engine...")
    if AttackEngine is None:
        print("[Error] No attack engine available. Make sure simple_attack_wrapper.py exists.")
        sys.exit(1)
    
    attack_engine = AttackEngine(
        detector=detector,
        gan_generator=gan_trainer.G,
        class_prior=class_prior,
        device=device
    )
    
    return attack_engine, detector


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
        normalize=True,
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
        normalize=True,
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
            
            # Compute metrics
            result_metrics = {
                'image_id': img_id,
                'num_objects': len(boxes),
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
    
    overall_asr = np.mean([r['overall_success_rate'] for r in all_results])
    print(f"Images processed: {num_processed}")
    print(f"Overall Attack Success Rate: {overall_asr:.1%}")
    
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
                'overall_success_rate': overall_asr,
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
