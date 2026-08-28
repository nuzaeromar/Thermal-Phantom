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
import torch.nn.functional as F
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
    from attack.detector_wrapper import FasterRCNNWrapper, YOLOWrapper, RTDETRWrapper
except ImportError:
    try:
        from detector_wrapper import FasterRCNNWrapper, YOLOWrapper, RTDETRWrapper
    except ImportError:
        print("[Error] Cannot import detector wrappers. Make sure attack/detector_wrapper.py exists.")
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
        ThermalObjectEditor, ThermalContextEditor,
        load_object_diffusion_checkpoint,
    )
    FULL_ATTACK_AVAILABLE = True
    print("[Info] Full ThermalMirageAttack is available")
except ImportError as e:
    print(f"[Warning] Full ThermalMirageAttack import failed: {e}")
    try:
        from joint_attack import ThermalMirageAttack, AttackConfig
        from detector_wrapper import AdversarialGuide, ThermalPlausibilityLoss
        from diffusion_editor import (
            ThermalObjectEditor, ThermalContextEditor,
            load_object_diffusion_checkpoint,
        )
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
    parser.add_argument('--attack_classes', nargs='+', default=None,
                        choices=FLIR_CLASS_NAMES,
                        help='Restrict attack/evaluation to these FLIR classes, e.g. --attack_classes person')
    
    # Model checkpoints
    parser.add_argument('--gan_ckpt', type=str, required=True,
                        help='Path to GAN checkpoint')
    parser.add_argument('--detector_ckpt', type=str, default=None,
                        help='Path to detector checkpoint (optional)')
    parser.add_argument('--diffusion_ckpt', type=str, default=None,
                        help='Complete checkpoint from train_diffusion.py; enables object diffusion')
    parser.add_argument('--target_detector', type=str, default='fasterrcnn',
                        choices=['fasterrcnn', 'yolo', 'rtdetr'],
                        help='Detector used to measure attack success')
    parser.add_argument('--yolo_ckpt', type=str, default='yolo11n.pt',
                        help='YOLOv11/Ultralytics checkpoint used when --target_detector yolo')
    parser.add_argument('--yolo_conf', type=float, default=0.25,
                        help='YOLO confidence threshold used when --target_detector yolo')
    parser.add_argument('--rtdetr_ckpt', type=str, default='rtdetr-l.pt',
                        help='Ultralytics RT-DETR checkpoint used when --target_detector rtdetr')
    parser.add_argument('--rtdetr_conf', type=float, default=0.25,
                        help='RT-DETR confidence threshold used when --target_detector rtdetr')
    parser.add_argument('--loop_detector', type=str, default='fasterrcnn',
                        choices=['fasterrcnn', 'yolo'],
                        help='Detector used inside the attack loop for gradient guidance')
    parser.add_argument('--loop_yolo_ckpt', type=str, default=None,
                        help='YOLO checkpoint used when --loop_detector yolo; defaults to --yolo_ckpt')
    parser.add_argument('--loop_yolo_conf', type=float, default=None,
                        help='YOLO confidence threshold used when --loop_detector yolo; defaults to --yolo_conf')
    parser.add_argument('--eval_conf', type=float, default=None,
                        help='Confidence threshold for final target success/evaluation; defaults to the target detector threshold')
    parser.add_argument('--eval_iou', type=float, default=0.5,
                        help='IoU threshold for final target success, precision/recall, and mean-IoU matching')
    
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
    parser.add_argument('--compute_lpips', action='store_true',
                        help='Compute LPIPS perceptual distance if the lpips package is installed')
    
    # Output
    parser.add_argument('--out_dir', type=str, default='results/multiclass',
                        help='Output directory for results')
    parser.add_argument('--save_images', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')
    parser.add_argument('--verbose_refinement', '--verbose-refinement',
                        dest='verbose_refinement', action='store_true',
                        help='Print detector-in-the-loop refinement steps inside each tested strategy')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    
    return parser.parse_args()


def _box_iou_vectorized(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)


def _matched_detection(detections, query_box, H, W, iou_threshold=0.5):
    """Return confidence and class for the best detection matching a normalized GT box."""
    if detections is None or len(detections.get('boxes', [])) == 0:
        return 0.0, -1

    det_boxes = detections['boxes'].detach().float().cpu().clone()
    det_boxes[:, [0, 2]] /= W
    det_boxes[:, [1, 3]] /= H
    query = query_box.detach().float().cpu().unsqueeze(0)

    ious = _box_iou_vectorized(query, det_boxes)
    best_iou, best_idx = ious.max(dim=1)
    if best_iou.item() <= iou_threshold:
        return 0.0, -1

    score = detections['scores'][best_idx.item()].item()
    label = detections['labels'][best_idx.item()].item()
    return float(score), int(label)


def evaluate_target_detector_success(
    attack_result,
    clean_dets,
    adv_dets,
    boxes,
    H,
    W,
    conf_threshold=0.3,
    iou_threshold=0.5,
):
    """
    Re-score object attack success against the configured target detector.
    The attack itself may be guided by a different detector.
    """
    per_object = []
    for box_idx, strategy, _ in attack_result['strategies_used']:
        box = boxes[box_idx]
        clean_conf, clean_cls = _matched_detection(
            clean_dets,
            box,
            H,
            W,
            iou_threshold=iou_threshold,
        )
        adv_conf, adv_cls = _matched_detection(
            adv_dets,
            box,
            H,
            W,
            iou_threshold=iou_threshold,
        )
        conf_drop = clean_conf - adv_conf

        if strategy.mode == 'hide':
            success = clean_conf >= conf_threshold and adv_conf < conf_threshold
        else:
            success = clean_conf >= conf_threshold and adv_cls == strategy.tgt_class

        per_object.append({
            'box_idx': int(box_idx),
            'strategy': str(strategy),
            'success': bool(success),
            'clean_confidence': float(clean_conf),
            'adversarial_confidence': float(adv_conf),
            'confidence_drop': float(conf_drop),
            'clean_class': int(clean_cls),
            'adversarial_class': int(adv_cls),
        })

    return per_object


def summarize_target_success(per_object_target_results, labels):
    successes = [r['success'] for r in per_object_target_results]
    per_class_lists = {}
    for item in per_object_target_results:
        cls = int(labels[item['box_idx']].item())
        per_class_lists.setdefault(cls, []).append(item['success'])

    return {
        'num_successful_objects': sum(successes),
        'object_attack_success_rate': (sum(successes) / len(successes)) if successes else 0.0,
        'image_attack_success': any(successes),
        'per_class_success': {
            cls: float(np.mean(cls_successes))
            for cls, cls_successes in per_class_lists.items()
        },
    }


def _to_python_number(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _format_optional_metric(value):
    return "N/A" if value is None else f"{value:.4f}"


def _resolve_eval_thresholds(args):
    if args.eval_conf is not None:
        eval_conf = args.eval_conf
    elif args.target_detector == 'yolo':
        eval_conf = args.yolo_conf
    elif args.target_detector == 'rtdetr':
        eval_conf = args.rtdetr_conf
    else:
        eval_conf = 0.3

    return float(eval_conf), float(args.eval_iou)


def _prepare_predictions_for_metrics(detections):
    """Return CPU tensors in pixel xyxy format with internal FLIR labels 0-3."""
    if detections is None:
        return {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'scores': torch.zeros((0,), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.long),
        }

    return {
        'boxes': detections.get('boxes', torch.zeros((0, 4))).detach().float().cpu(),
        'scores': detections.get('scores', torch.zeros((0,))).detach().float().cpu(),
        'labels': detections.get('labels', torch.zeros((0,), dtype=torch.long)).detach().long().cpu(),
    }


def _prepare_gt_for_metrics(boxes, labels, H, W):
    """Convert normalized FLIR GT boxes to pixel xyxy format."""
    gt_boxes = boxes.detach().float().cpu().clone()
    if gt_boxes.numel() > 0:
        gt_boxes[:, [0, 2]] *= W
        gt_boxes[:, [1, 3]] *= H

    return {
        'boxes': gt_boxes,
        'labels': labels.detach().long().cpu(),
    }


def _precision_recall_and_mean_iou(pred, gt, iou_threshold=0.5, conf_threshold=0.3):
    pred_boxes = pred['boxes']
    pred_scores = pred['scores']
    pred_labels = pred['labels']
    gt_boxes = gt['boxes']
    gt_labels = gt['labels']

    keep = pred_scores >= conf_threshold
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    pred_labels = pred_labels[keep]

    matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)
    true_positives = 0
    matched_ious = []

    if len(pred_boxes) > 0:
        order = pred_scores.argsort(descending=True)
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]

        for pr_box, pr_label in zip(pred_boxes, pred_labels):
            candidates = (gt_labels == pr_label) & (~matched_gt)
            if candidates.sum().item() == 0:
                continue

            candidate_indices = torch.where(candidates)[0]
            ious = _box_iou_vectorized(pr_box.unsqueeze(0), gt_boxes[candidate_indices]).squeeze(0)
            best_iou, best_local_idx = ious.max(dim=0)

            if best_iou.item() >= iou_threshold:
                gt_idx = candidate_indices[best_local_idx]
                matched_gt[gt_idx] = True
                true_positives += 1
                matched_ious.append(float(best_iou.item()))

    false_positives = len(pred_boxes) - true_positives
    false_negatives = len(gt_boxes) - true_positives

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return {
        'precision': float(precision),
        'recall': float(recall),
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives),
        'mean_matched_iou': mean_iou,
        'matched_iou_sum': float(sum(matched_ious)),
        'matched_iou_count': int(len(matched_ious)),
    }


def _compute_ap_from_pr(precisions, recalls):
    if len(precisions) == 0:
        return 0.0

    precisions = np.asarray(precisions)
    recalls = np.asarray(recalls)
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    mrec = np.concatenate(([0.0], recalls, [1.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    changed = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))


def _compute_map_at_iou(all_preds, all_gts, iou_threshold, num_classes=NUM_CLASSES):
    per_class_ap = {}

    for cls in range(num_classes):
        class_predictions = []
        n_gt = 0

        for image_idx, (pred, gt) in enumerate(zip(all_preds, all_gts)):
            gt_mask = gt['labels'] == cls
            n_gt += int(gt_mask.sum().item())

            pred_mask = pred['labels'] == cls
            for box, score in zip(pred['boxes'][pred_mask], pred['scores'][pred_mask]):
                class_predictions.append({
                    'image_idx': image_idx,
                    'box': box,
                    'score': float(score.item()),
                })

        if n_gt == 0:
            per_class_ap[cls] = 0.0
            continue

        class_predictions.sort(key=lambda item: item['score'], reverse=True)
        matched_by_image = {
            image_idx: torch.zeros(int((gt['labels'] == cls).sum().item()), dtype=torch.bool)
            for image_idx, gt in enumerate(all_gts)
        }

        tps = []
        fps = []

        for pred_item in class_predictions:
            image_idx = pred_item['image_idx']
            gt = all_gts[image_idx]
            gt_boxes = gt['boxes'][gt['labels'] == cls]
            matched = matched_by_image[image_idx]

            if len(gt_boxes) == 0:
                tps.append(0)
                fps.append(1)
                continue

            ious = _box_iou_vectorized(pred_item['box'].unsqueeze(0), gt_boxes).squeeze(0)
            best_iou, best_gt_idx = ious.max(dim=0)

            if best_iou.item() >= iou_threshold and not matched[best_gt_idx]:
                matched[best_gt_idx] = True
                tps.append(1)
                fps.append(0)
            else:
                tps.append(0)
                fps.append(1)

        if not tps:
            per_class_ap[cls] = 0.0
            continue

        cum_tp = np.cumsum(np.asarray(tps))
        cum_fp = np.cumsum(np.asarray(fps))
        recalls = cum_tp / (n_gt + 1e-8)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-8)
        per_class_ap[cls] = _compute_ap_from_pr(precisions, recalls)

    return {
        'mAP': float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0,
        'per_class_ap': per_class_ap,
    }


def _compute_detection_quality_summary(
    all_clean_preds,
    all_adv_preds,
    all_gts,
    conf_threshold=0.3,
    iou_threshold=0.5,
):
    thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]

    clean_pr = [
        _precision_recall_and_mean_iou(
            pred,
            gt,
            iou_threshold=iou_threshold,
            conf_threshold=conf_threshold,
        )
        for pred, gt in zip(all_clean_preds, all_gts)
    ]
    adv_pr = [
        _precision_recall_and_mean_iou(
            pred,
            gt,
            iou_threshold=iou_threshold,
            conf_threshold=conf_threshold,
        )
        for pred, gt in zip(all_adv_preds, all_gts)
    ]

    def aggregate_pr(rows):
        tp = sum(row['true_positives'] for row in rows)
        fp = sum(row['false_positives'] for row in rows)
        fn = sum(row['false_negatives'] for row in rows)
        iou_sum = sum(row['matched_iou_sum'] for row in rows)
        iou_count = sum(row['matched_iou_count'] for row in rows)
        return {
            'precision': float(tp / (tp + fp + 1e-8)),
            'recall': float(tp / (tp + fn + 1e-8)),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'mean_matched_iou': float(iou_sum / (iou_count + 1e-8)),
            'matched_iou_sum': float(iou_sum),
            'matched_iou_count': int(iou_count),
        }

    clean_summary = aggregate_pr(clean_pr)
    adv_summary = aggregate_pr(adv_pr)

    clean_map50 = _compute_map_at_iou(all_clean_preds, all_gts, 0.5)
    adv_map50 = _compute_map_at_iou(all_adv_preds, all_gts, 0.5)

    clean_maps = [_compute_map_at_iou(all_clean_preds, all_gts, thr)['mAP'] for thr in thresholds]
    adv_maps = [_compute_map_at_iou(all_adv_preds, all_gts, thr)['mAP'] for thr in thresholds]

    clean_summary['mAP50'] = clean_map50['mAP']
    clean_summary['mAP50_95'] = float(np.mean(clean_maps)) if clean_maps else 0.0
    clean_summary['per_class_ap50'] = clean_map50['per_class_ap']

    adv_summary['mAP50'] = adv_map50['mAP']
    adv_summary['mAP50_95'] = float(np.mean(adv_maps)) if adv_maps else 0.0
    adv_summary['per_class_ap50'] = adv_map50['per_class_ap']

    return {
        'clean': clean_summary,
        'adversarial': adv_summary,
        'drop': {
            'precision': clean_summary['precision'] - adv_summary['precision'],
            'recall': clean_summary['recall'] - adv_summary['recall'],
            'mAP50': clean_summary['mAP50'] - adv_summary['mAP50'],
            'mAP50_95': clean_summary['mAP50_95'] - adv_summary['mAP50_95'],
            'mean_matched_iou': clean_summary['mean_matched_iou'] - adv_summary['mean_matched_iou'],
        },
        'iou_thresholds_for_mAP50_95': thresholds,
        'precision_recall_iou_threshold': float(iou_threshold),
        'precision_recall_conf_threshold': float(conf_threshold),
    }


def _image_ssim(clean, adv, window_size=11):
    """
    Lightweight SSIM for tensors in [0, 1].
    Returns average SSIM over the image batch.
    """
    clean = clean.detach().float().clamp(0, 1)
    adv = adv.detach().float().clamp(0, 1)

    channels = clean.shape[1]
    padding = window_size // 2
    window = torch.ones(
        (channels, 1, window_size, window_size),
        dtype=clean.dtype,
        device=clean.device,
    ) / float(window_size * window_size)

    mu_x = F.conv2d(clean, window, padding=padding, groups=channels)
    mu_y = F.conv2d(adv, window, padding=padding, groups=channels)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(clean * clean, window, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(adv * adv, window, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(clean * adv, window, padding=padding, groups=channels) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2) + 1e-8
    )
    return float(ssim_map.mean().item())


class OptionalImageQualityMetrics:
    """Computes image similarity metrics while keeping optional dependencies isolated."""

    def __init__(self, device='cuda', compute_lpips=False):
        self.device = device
        self.lpips_model = None

        if compute_lpips:
            try:
                import lpips
                self.lpips_model = lpips.LPIPS(net='alex').to(device).eval()
                print("[Metrics] LPIPS enabled")
            except Exception as e:
                print(f"[Warning] LPIPS disabled: {e}")
                print("[Warning] Install it in the attack environment with: pip install lpips")

    def compute_pair(self, clean, adv):
        clean = clean.detach().float()
        adv = adv.detach().float()
        if clean.dim() == 3:
            clean = clean.unsqueeze(0)
        if adv.dim() == 3:
            adv = adv.unsqueeze(0)

        clean = clean.clamp(0, 1)
        adv = adv.clamp(0, 1)

        metrics = {
            'ssim': _image_ssim(clean.to(self.device), adv.to(self.device)),
        }

        if self.lpips_model is not None:
            with torch.no_grad():
                # LPIPS expects images normalized to [-1, 1].
                lpips_value = self.lpips_model(
                    clean.to(self.device) * 2 - 1,
                    adv.to(self.device) * 2 - 1,
                )
            metrics['lpips'] = float(lpips_value.mean().item())
        else:
            metrics['lpips'] = None

        return metrics

    def summarize(self, per_image_metrics):
        summary = {}
        for key in ('ssim', 'lpips'):
            values = [m[key] for m in per_image_metrics if m.get(key) is not None]
            summary[key] = float(np.mean(values)) if values else None

        return summary


def _build_union_object_mask(boxes, height, width, device, dtype):
    """Build one binary mask containing every normalized object box."""
    union_mask = torch.zeros((1, 1, height, width), device=device, dtype=dtype)
    for box in boxes.detach().float().cpu():
        x1, y1, x2, y2 = box.tolist()
        px1 = max(0, min(width, int(x1 * width)))
        py1 = max(0, min(height, int(y1 * height)))
        px2 = max(0, min(width, int(x2 * width)))
        py2 = max(0, min(height, int(y2 * height)))
        if px2 > px1 and py2 > py1:
            union_mask[:, :, py1:py2, px1:px2] = 1.0
    return union_mask


def _compute_final_thermal_plausibility(
    clean, adv, boxes, plausibility_evaluator
):
    """
    Evaluate the final, accumulated adversarial frame against its clean image.

    These metrics are deliberately computed after adaptive strategy selection
    and after all selected per-object attacks have been composed. Smoothness is
    measured for both the clean and final adversarial frame, and their change
    is recorded. Histogram consistency is the full-frame KL divergence from
    the clean intensity histogram to the final adversarial intensity histogram.
    Boundary deviation is measured once around the union of all attacked object
    boxes, using the clean and final adversarial frame.
    """
    if plausibility_evaluator is None:
        return None

    clean = clean.detach()
    adv = adv.detach()
    if clean.dim() == 3:
        clean = clean.unsqueeze(0)
    if adv.dim() == 3:
        adv = adv.unsqueeze(0)

    device = adv.device
    clean = clean.to(device=device, dtype=adv.dtype).clamp(0, 1)
    adv = adv.clamp(0, 1)
    full_mask = torch.ones(
        (adv.shape[0], 1, adv.shape[-2], adv.shape[-1]),
        dtype=adv.dtype,
        device=device,
    )
    union_object_mask = _build_union_object_mask(
        boxes=boxes,
        height=adv.shape[-2],
        width=adv.shape[-1],
        device=device,
        dtype=adv.dtype,
    )

    with torch.no_grad():
        clean_smoothness = plausibility_evaluator.smoothness_loss(clean, full_mask)
        adv_smoothness = plausibility_evaluator.smoothness_loss(adv, full_mask)
        histogram_kl = plausibility_evaluator.histogram_loss(
            adv, clean, full_mask
        )
        boundary_deviation = plausibility_evaluator.boundary_loss(
            adv, clean, union_object_mask
        )

    clean_smoothness = float(clean_smoothness.item())
    adv_smoothness = float(adv_smoothness.item())
    return {
        'clean_spatial_smoothness': clean_smoothness,
        'adversarial_spatial_smoothness': adv_smoothness,
        'spatial_smoothness_change': adv_smoothness - clean_smoothness,
        'histogram_kl_divergence': float(histogram_kl.item()),
        'boundary_deviation': float(boundary_deviation.item()),
        'scope': {
            'smoothness': 'full_image',
            'histogram': 'full_image_clean_vs_final_adversarial',
            'boundary': 'union_of_attacked_object_boundaries_clean_vs_final_adversarial',
        },
    }


def _summarize_final_thermal_plausibility(per_image_metrics):
    """Aggregate final-frame plausibility metrics across evaluated images."""
    valid = [row for row in per_image_metrics if row is not None]
    if not valid:
        return None

    summary = {
        'num_images': len(valid),
        'scope': valid[0]['scope'],
    }
    for key in (
        'clean_spatial_smoothness',
        'adversarial_spatial_smoothness',
        'spatial_smoothness_change',
        'histogram_kl_divergence',
        'boundary_deviation',
    ):
        values = np.asarray([row[key] for row in valid], dtype=np.float64)
        summary[key] = {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'min': float(values.min()),
            'max': float(values.max()),
        }
    return summary


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

    # Always load Faster R-CNN because it is needed either as the target/loop
    # detector or as the feature extractor for GAN class priors.
    print(f"[Setup] Loading Faster R-CNN detector...")
    fasterrcnn_detector = FasterRCNNWrapper(
        num_flir_classes=NUM_CLASSES,
        pretrained=True,
        device=device,
    )
    if args.detector_ckpt:
        fasterrcnn_detector.model.load_state_dict(
            torch.load(args.detector_ckpt, map_location=device)
        )
        print(f"[Faster R-CNN] Loaded checkpoint <- {args.detector_ckpt}")
    fasterrcnn_detector.model.eval()

    yolo_detectors = {}
    rtdetr_detectors = {}

    def get_yolo_detector(model_path, conf_threshold, role):
        key = (model_path, conf_threshold)
        if key not in yolo_detectors:
            print(f"[Setup] Loading {role} detector: YOLO <- {model_path}")
            yolo_detectors[key] = YOLOWrapper(
                model_path=model_path,
                num_classes=NUM_CLASSES,
                conf_threshold=conf_threshold,
                device=device,
            )
        return yolo_detectors[key]

    def get_rtdetr_detector(model_path, conf_threshold, role):
        key = (model_path, conf_threshold)
        if key not in rtdetr_detectors:
            print(f"[Setup] Loading {role} detector: RT-DETR <- {model_path}")
            rtdetr_detectors[key] = RTDETRWrapper(
                model_path=model_path,
                num_classes=NUM_CLASSES,
                conf_threshold=conf_threshold,
                device=device,
            )
        return rtdetr_detectors[key]

    if args.loop_detector == 'yolo':
        loop_yolo_ckpt = args.loop_yolo_ckpt or args.yolo_ckpt
        loop_yolo_conf = args.loop_yolo_conf if args.loop_yolo_conf is not None else args.yolo_conf
        loop_detector = get_yolo_detector(loop_yolo_ckpt, loop_yolo_conf, "detector-in-the-loop")
        loop_detector_name = f"yolo:{loop_yolo_ckpt}"
    else:
        loop_detector = fasterrcnn_detector
        loop_detector_name = "fasterrcnn"

    if args.target_detector == 'yolo':
        target_detector = get_yolo_detector(args.yolo_ckpt, args.yolo_conf, "target")
        target_detector_name = f"yolo:{args.yolo_ckpt}"
    elif args.target_detector == 'rtdetr':
        target_detector = get_rtdetr_detector(args.rtdetr_ckpt, args.rtdetr_conf, "target")
        target_detector_name = f"rtdetr:{args.rtdetr_ckpt}"
    else:
        target_detector = fasterrcnn_detector
        target_detector_name = "fasterrcnn"

    print(f"[Setup] Detector-in-the-loop for guidance: {loop_detector_name}")
    print(f"[Setup] Target detector for success metrics: {target_detector_name}")

    # Build class prior extractor
    print(f"[Setup] Building class priors...")
    class_prior = ThermalClassPrior(
        generator=gan_trainer.G,
        feature_extractor=fasterrcnn_detector.get_feature_extractor(),
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
                unet, scheduler, class_emb, diffusion_meta = load_object_diffusion_checkpoint(
                    args.diffusion_ckpt, device=device
                )
                saved_names = diffusion_meta.get("class_embedding_config", {}).get("class_names")
                if saved_names is not None and list(saved_names) != list(FLIR_CLASS_NAMES):
                    raise ValueError(
                        "Diffusion checkpoint class order does not match this dataset: "
                        f"checkpoint={saved_names}, expected={FLIR_CLASS_NAMES}"
                    )
                obj_editor = ThermalObjectEditor(
                    unet=unet,
                    scheduler=scheduler,
                    class_embeddings=class_emb,
                    device=device,
                    inpaint_start_t=500,
                )
                # Context attacks are analytic in this runner. A random scene
                # embedding must never be paired with an object-trained U-Net.
                ctx_editor = None
                print(
                    "[Setup] Restored trained U-Net and class embeddings; "
                    f"object crop size={unet.img_size}"
                )
            else:
                if args.diffusion_ckpt:
                    raise FileNotFoundError(
                        f"Diffusion checkpoint not found: {args.diffusion_ckpt}. "
                        "Run train_diffusion.py first or omit --diffusion_ckpt to use FGSM."
                    )
                else:
                    print("[Setup] No diffusion checkpoint supplied; object branch will use FGSM fallback when needed")
                print("[Setup] Context branch will use physics-informed contrast/smoothing/cooling attack")

            guide = AdversarialGuide(loop_detector, loss_scale=1.0)
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
            attack_engine.eval_detector = target_detector
            print(f"[Setup] Full attack engine ready | context_attack={args.context_attack}")
            return attack_engine, loop_detector, loop_detector_name, target_detector, target_detector_name
        except Exception as e:
            if args.diffusion_ckpt:
                raise RuntimeError(
                    "Diffusion was explicitly requested but could not be initialized"
                ) from e
            print(f"[Warning] Full attack engine initialization failed: {e}")
            print("[Warning] Falling back to SimpleAttackWrapper")

    if SIMPLE_FALLBACK_AVAILABLE:
        attack_engine = SimpleAttackWrapper(
            detector=loop_detector,
            gan_generator=gan_trainer.G,
            class_prior=class_prior,
            device=device,
        )
        attack_engine.eval_detector = target_detector
        return attack_engine, loop_detector, loop_detector_name, target_detector, target_detector_name

    print("[Error] No attack engine available. Check joint_attack.py or simple_attack_wrapper.py.")
    sys.exit(1)


def mode_profile(args):
    """Profile mode: determine best attack strategies per class."""
    
    print("\n" + "="*60)
    print("  MODE: PROFILE - Building Strategy Profile")
    print("="*60)
    
    # Build attack engine
    attack_engine, loop_detector, loop_detector_name, target_detector, target_detector_name = build_attack_engine(args)
    
    # Build dataloader
    val_loader = build_dataloader(
        root=args.data_root,
        split='val',  # Always use validation for profiling
        batch_size=1,
        num_workers=4,
        img_size=FLIR_NATIVE_SIZE,
        normalize=False,
        return_masks=False,
        classes=args.attack_classes,
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
    attack_engine, loop_detector, loop_detector_name, target_detector, target_detector_name = build_attack_engine(args)
    eval_conf, eval_iou = _resolve_eval_thresholds(args)
    print(f"[Setup] Final evaluation thresholds: confidence >= {eval_conf:.3f}, IoU > {eval_iou:.3f}")
    
    # Build dataloader
    test_loader = build_dataloader(
        root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=4,
        img_size=FLIR_NATIVE_SIZE,
        normalize=False,
        return_masks=False,
        classes=args.attack_classes,
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

    image_quality_evaluator = OptionalImageQualityMetrics(
        device=args.device,
        compute_lpips=args.compute_lpips,
    )
    
    # Attack loop
    all_results = []
    all_clean_preds = []
    all_adv_preds = []
    all_gts = []
    all_image_quality_metrics = []
    all_final_thermal_plausibility = []
    total_target_clean_detected_objects = 0
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
                verbose=args.verbose,
                verbose_refinement=args.verbose_refinement,
            )
            
            adv_image = attack_result['adv_image']

            # Evaluate plausibility only after the final strategies have been
            # selected and accumulated into the image-level adversarial result.
            final_thermal_plausibility = _compute_final_thermal_plausibility(
                clean=image.unsqueeze(0),
                adv=adv_image,
                boxes=boxes,
                plausibility_evaluator=getattr(attack_engine, 'plaus_loss', None),
            )
            
            # Get target detector predictions on clean and adversarial images.
            # The detector-in-the-loop may be Faster R-CNN while this target
            # detector can be YOLO.
            with torch.no_grad():
                clean_dets = target_detector.detect(image.unsqueeze(0).to(args.device))
                adv_dets = target_detector.detect(adv_image)

            H, W = image.shape[-2], image.shape[-1]
            clean_pred_metrics = _prepare_predictions_for_metrics(clean_dets[0] if clean_dets else None)
            adv_pred_metrics = _prepare_predictions_for_metrics(adv_dets[0] if adv_dets else None)
            gt_metrics = _prepare_gt_for_metrics(boxes, labels, H, W)

            clean_image_quality = _precision_recall_and_mean_iou(
                clean_pred_metrics,
                gt_metrics,
                iou_threshold=eval_iou,
                conf_threshold=eval_conf,
            )
            adv_image_quality = _precision_recall_and_mean_iou(
                adv_pred_metrics,
                gt_metrics,
                iou_threshold=eval_iou,
                conf_threshold=eval_conf,
            )
            image_iou_drop = (
                clean_image_quality['mean_matched_iou'] -
                adv_image_quality['mean_matched_iou']
            )
            image_quality_metrics = image_quality_evaluator.compute_pair(
                clean=image.unsqueeze(0),
                adv=adv_image.detach().cpu(),
            )

            all_clean_preds.append(clean_pred_metrics)
            all_adv_preds.append(adv_pred_metrics)
            all_gts.append(gt_metrics)
            all_image_quality_metrics.append(image_quality_metrics)
            all_final_thermal_plausibility.append(final_thermal_plausibility)

            target_object_results = evaluate_target_detector_success(
                attack_result=attack_result,
                clean_dets=clean_dets[0] if clean_dets else None,
                adv_dets=adv_dets[0] if adv_dets else None,
                boxes=boxes,
                H=H,
                W=W,
                conf_threshold=eval_conf,
                iou_threshold=eval_iou,
            )
            target_summary = summarize_target_success(
                per_object_target_results=target_object_results,
                labels=labels,
            )
            total_target_clean_detected_objects += sum(
                1 for item in target_object_results
                if item['clean_confidence'] >= eval_conf
            )

            object_successes = [
                bool(item['success'])
                for item in target_object_results
            ]
            num_successful_objects = sum(object_successes)
            image_attack_success = num_successful_objects > 0

            # Compute metrics
            result_metrics = {
                'image_id': _to_python_number(img_id),
                'num_objects': len(boxes),
                'num_successful_objects': num_successful_objects,
                'object_attack_success_rate': target_summary['object_attack_success_rate'],
                'image_attack_success': image_attack_success,
                'overall_success_rate': target_summary['object_attack_success_rate'],
                'per_class_success': target_summary['per_class_success'],
                'target_detector': target_detector_name,
                'detector_in_the_loop': loop_detector_name,
                'target_object_results': target_object_results,
                'strategies_used': [
                    str(strat) for _, strat, _ in attack_result['strategies_used']
                ],
                'n_det_clean': len(clean_dets[0]['boxes']) if clean_dets else 0,
                'n_det_adv': len(adv_dets[0]['boxes']) if adv_dets else 0,
                'clean_detector_quality': clean_image_quality,
                'adversarial_detector_quality': adv_image_quality,
                'iou_drop': float(image_iou_drop),
                'image_quality': image_quality_metrics,
                'final_thermal_plausibility': final_thermal_plausibility,
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
                    title=f"Image {num_processed} | Target success: {target_summary['object_attack_success_rate']:.1%}",
                    save_path=str(img_out_path)
                )
                plt.close('all')
            
            num_processed += 1
    
    # Aggregate and print results
    print("\n" + "="*60)
    print("  ATTACK RESULTS SUMMARY")
    print("="*60)
    print(f"Target detector: {target_detector_name}")
    print(f"Detector-in-the-loop: {loop_detector_name}")
    
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
    print(f"Target-clean detected objects: {total_target_clean_detected_objects}/{total_objects} "
          f"(confidence >= {eval_conf:.3f} and IoU > {eval_iou:.3f})")
    print(f"Object-level Attack Success Rate: {object_level_asr:.1%} "
          f"({total_successful_objects}/{total_objects} objects)")
    print(f"Image-level Attack Success Rate: {image_level_asr:.1%} "
          f"({sum(r['image_attack_success'] for r in all_results)}/{len(all_results)} images with >=1 success)")
    print(f"Mean per-image object success rate: {overall_asr:.1%}")

    detection_quality = _compute_detection_quality_summary(
        all_clean_preds=all_clean_preds,
        all_adv_preds=all_adv_preds,
        all_gts=all_gts,
        conf_threshold=eval_conf,
        iou_threshold=eval_iou,
    ) if all_results else {}

    if detection_quality:
        clean_q = detection_quality['clean']
        adv_q = detection_quality['adversarial']
        drop_q = detection_quality['drop']

        print("\nClean vs Adversarial Detector Quality:")
        print(f"  Precision:  clean={clean_q['precision']:.3f} | adv={adv_q['precision']:.3f} | drop={drop_q['precision']:.3f}")
        print(f"  Recall:     clean={clean_q['recall']:.3f} | adv={adv_q['recall']:.3f} | drop={drop_q['recall']:.3f}")
        print(f"  mAP50:      clean={clean_q['mAP50']:.3f} | adv={adv_q['mAP50']:.3f} | drop={drop_q['mAP50']:.3f}")
        print(f"  mAP50-95:   clean={clean_q['mAP50_95']:.3f} | adv={adv_q['mAP50_95']:.3f} | drop={drop_q['mAP50_95']:.3f}")
        print(f"  Mean IoU:   clean={clean_q['mean_matched_iou']:.3f} | adv={adv_q['mean_matched_iou']:.3f} | drop={drop_q['mean_matched_iou']:.3f}")

    image_quality_summary = image_quality_evaluator.summarize(all_image_quality_metrics)
    if image_quality_summary:
        print("\nClean vs Adversarial Image Quality:")
        print(f"  LPIPS:   {_format_optional_metric(image_quality_summary.get('lpips'))}")
        print(f"  SSIM:    {_format_optional_metric(image_quality_summary.get('ssim'))}")

    final_thermal_plausibility_summary = _summarize_final_thermal_plausibility(
        all_final_thermal_plausibility
    )
    if final_thermal_plausibility_summary:
        clean_smooth_summary = final_thermal_plausibility_summary['clean_spatial_smoothness']
        adv_smooth_summary = final_thermal_plausibility_summary['adversarial_spatial_smoothness']
        smooth_change_summary = final_thermal_plausibility_summary['spatial_smoothness_change']
        hist_summary = final_thermal_plausibility_summary['histogram_kl_divergence']
        boundary_summary = final_thermal_plausibility_summary['boundary_deviation']
        print("\nFinal Adversarial Thermal Plausibility (image-level evaluation):")
        print(
            "  Clean spatial smoothness (L1 TV): "
            f"mean={clean_smooth_summary['mean']:.6f} | "
            f"std={clean_smooth_summary['std']:.6f}"
        )
        print(
            "  Adversarial spatial smoothness (L1 TV): "
            f"mean={adv_smooth_summary['mean']:.6f} | "
            f"std={adv_smooth_summary['std']:.6f} | "
            f"min={adv_smooth_summary['min']:.6f} | "
            f"max={adv_smooth_summary['max']:.6f}"
        )
        print(
            "  Spatial smoothness change (adversarial - clean): "
            f"mean={smooth_change_summary['mean']:.6f} | "
            f"std={smooth_change_summary['std']:.6f}"
        )
        print(
            "  Histogram KL (clean || adversarial): "
            f"mean={hist_summary['mean']:.6f} | std={hist_summary['std']:.6f} | "
            f"min={hist_summary['min']:.6f} | max={hist_summary['max']:.6f}"
        )
        print(
            "  Object-boundary deviation (clean vs adversarial): "
            f"mean={boundary_summary['mean']:.6f} | "
            f"std={boundary_summary['std']:.6f} | "
            f"min={boundary_summary['min']:.6f} | "
            f"max={boundary_summary['max']:.6f}"
        )
    else:
        print("\nFinal Adversarial Thermal Plausibility: N/A "
              "(full plausibility evaluator unavailable)")
    
    # Per-class breakdown
    per_class_asr = {}
    for cls in range(NUM_CLASSES):
        class_successes = []
        for r in all_results:
            if cls in r['per_class_success']:
                class_successes.append(r['per_class_success'][cls])
        if class_successes:
            per_class_asr[cls] = float(np.mean(class_successes))
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
                'target_detector': target_detector_name,
                'detector_in_the_loop': loop_detector_name,
                'eval_conf_threshold': float(eval_conf),
                'eval_iou_threshold': float(eval_iou),
                'num_objects': total_objects,
                'target_clean_detected_objects': total_target_clean_detected_objects,
                'num_successful_objects': total_successful_objects,
                'object_level_attack_success_rate': float(object_level_asr),
                'image_level_attack_success_rate': float(image_level_asr),
                'overall_success_rate': float(overall_asr),
                'overall_success_rate_definition': 'mean per-image object success rate',
                'per_class_success_rate': per_class_asr,
                'strategy_usage': strategy_counts,
                'detector_quality': detection_quality,
                'image_quality': image_quality_summary,
                'final_thermal_plausibility': final_thermal_plausibility_summary,
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
