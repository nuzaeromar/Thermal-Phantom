"""
Stage 9: Evaluation Protocol

Measures:
  - mAP drop (clean vs adversarial)
  - Attack Success Rate (ASR)
  - False Negative Rate (FNR)
  - Targeted Misclassification Rate (TMR)
  - Confidence Suppression
  - Box IoU degradation
  - Thermal Realism Score (BRISQUE-proxy + histogram similarity)

Ablation variants:
  A. Object manipulation only
  B. Context manipulation only
  C. Object + Context (full)
  D. Without GAN prior
  E. Without diffusion (FGSM only)
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional
import json
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.flir_dataset import FLIR_CLASS_NAMES, NUM_CLASSES


# ── IoU utilities ─────────────────────────────────────────────────────────────

def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU. boxes: (N,4) xyxy."""
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(0) * \
              (boxes_a[:, 3] - boxes_a[:, 1]).clamp(0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(0) * \
              (boxes_b[:, 3] - boxes_b[:, 1]).clamp(0)

    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-8)


# ── AP computation ────────────────────────────────────────────────────────────

def compute_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """Compute AP using 11-point interpolation."""
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        mask = recalls >= thr
        p = precisions[mask].max() if mask.any() else 0.0
        ap += p / 11.0
    return ap


def compute_map(
    all_preds: List[Dict],     # list of {'boxes': (N,4), 'scores': (N,), 'labels': (N,)}
    all_gts:   List[Dict],     # list of {'boxes': (N,4), 'labels': (N,)}
    num_classes: int = 4,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.0,
) -> Dict:
    """
    Compute per-class AP and mAP.
    Returns dict with class APs and 'mAP' key.
    """
    # Aggregate by class
    class_preds = defaultdict(list)  # cls -> [(score, tp)]
    class_n_gt  = defaultdict(int)

    for pred, gt in zip(all_preds, all_gts):
        gt_boxes  = gt["boxes"]
        gt_labels = gt["labels"]
        pr_boxes  = pred.get("boxes",  torch.zeros(0, 4))
        pr_scores = pred.get("scores", torch.zeros(0))
        pr_labels = pred.get("labels", torch.zeros(0, dtype=torch.long))

        # Filter by confidence
        keep = pr_scores >= conf_threshold
        pr_boxes  = pr_boxes[keep]
        pr_scores = pr_scores[keep]
        pr_labels = pr_labels[keep]

        for c in range(num_classes):
            gt_mask = (gt_labels == c)
            pr_mask = (pr_labels == c)
            class_n_gt[c] += gt_mask.sum().item()

            c_gt_boxes = gt_boxes[gt_mask]
            c_pr_boxes = pr_boxes[pr_mask]
            c_pr_scores= pr_scores[pr_mask]

            if c_pr_boxes.numel() == 0:
                continue

            # Sort by score descending
            order      = c_pr_scores.argsort(descending=True)
            c_pr_boxes = c_pr_boxes[order]
            c_pr_scores= c_pr_scores[order]

            matched = torch.zeros(len(c_gt_boxes), dtype=torch.bool)

            for pr_box, score in zip(c_pr_boxes, c_pr_scores):
                if c_gt_boxes.numel() == 0:
                    class_preds[c].append((score.item(), 0))
                    continue
                ious = box_iou(pr_box.unsqueeze(0), c_gt_boxes).squeeze(0)
                best_iou, best_j = ious.max(0) if ious.numel() > 0 else (torch.tensor(0.), 0)
                if best_iou >= iou_threshold and not matched[best_j]:
                    matched[best_j] = True
                    class_preds[c].append((score.item(), 1))
                else:
                    class_preds[c].append((score.item(), 0))

    # Compute AP per class
    aps = {}
    for c in range(num_classes):
        preds_c = sorted(class_preds[c], key=lambda x: -x[0])
        n_gt    = class_n_gt[c]
        if n_gt == 0 or len(preds_c) == 0:
            aps[c] = 0.0
            continue
        tps  = np.array([x[1] for x in preds_c])
        fps  = 1 - tps
        cum_tp = np.cumsum(tps)
        cum_fp = np.cumsum(fps)
        recalls    = cum_tp / (n_gt + 1e-8)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-8)
        aps[c] = compute_ap(precisions, recalls)

    mAP = np.mean(list(aps.values()))
    return {"per_class_ap": aps, "mAP": mAP}


# ── Thermal Realism Score ─────────────────────────────────────────────────────

def thermal_realism_score(
    clean: torch.Tensor,          # (1, 3, H, W)
    adv:   torch.Tensor,          # (1, 3, H, W)
    obj_mask: torch.Tensor,       # (1, 1, H, W)
    n_bins: int = 32,
) -> Dict:
    """
    Proxy thermal realism score:
    1. Smoothness ratio: TV of adv vs clean in edited region
    2. Histogram similarity: chi-squared between edited region histograms
    3. Gradient magnitude consistency
    """
    clean = clean.cpu()
    adv   = adv.cpu()
    mask  = obj_mask.cpu().expand_as(clean)

    # 1. TV smoothness ratio
    def tv(img):
        dy = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs()
        dx = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs()
        return dy.mean() + dx.mean()

    tv_clean = tv(clean).item()
    tv_adv   = tv(adv).item()
    tv_ratio  = tv_adv / (tv_clean + 1e-8)    # 1.0 = same smoothness

    # 2. Histogram similarity (in masked region)
    def hist_vec(img, mask, bins=32):
        m = mask.bool().flatten()
        vals = img.mean(0).flatten()[m]
        h, _ = np.histogram(vals.numpy(), bins=bins, range=(0, 1))
        h = h / (h.sum() + 1e-8)
        return h

    hist_c = hist_vec(clean[0], obj_mask[0, 0])
    hist_a = hist_vec(adv[0],   obj_mask[0, 0])
    chi2   = 0.5 * np.sum((hist_c - hist_a) ** 2 / (hist_c + hist_a + 1e-8))

    # 3. Gradient magnitude consistency
    def grad_mag(img):
        dy = (img[:, :, 1:, :] - img[:, :, :-1, :])
        dx = (img[:, :, :, 1:] - img[:, :, :, :-1])
        return (dy ** 2 + 1e-8).sqrt().mean() + (dx ** 2 + 1e-8).sqrt().mean()

    gm_clean = grad_mag(clean).item()
    gm_adv   = grad_mag(adv).item()
    gm_ratio  = gm_adv / (gm_clean + 1e-8)

    # Overall realism score: closer to 1 = more realistic
    score = 1 / (1 + abs(tv_ratio - 1) + chi2 + abs(gm_ratio - 1))

    return {
        "realism_score":   score,
        "tv_ratio":        tv_ratio,
        "hist_chi2":       chi2,
        "grad_mag_ratio":  gm_ratio,
    }


# ── Comprehensive Evaluator ───────────────────────────────────────────────────

class AttackEvaluator:
    """
    Runs all evaluation metrics over a set of per-image attack results.
    """

    CLASS_NAMES = FLIR_CLASS_NAMES   # ["person", "bicycle", "car", "dog"]

    def __init__(self, num_classes: int = NUM_CLASSES):
        self.num_classes = num_classes

    def aggregate(self, results: List[Dict]) -> Dict:
        """Aggregate per-image metrics into a final summary."""
        if not results:
            return {}

        asr     = np.mean([r.get("attack_success",      0) for r in results])
        fnr     = np.mean([r.get("false_negative_rate", 0) for r in results])
        conf_d  = np.mean([r.get("conf_drop",           0) for r in results])
        plaus   = np.mean([r.get("plaus_loss",           0) for r in results])
        pert    = np.mean([r.get("perturbation",         0) for r in results])
        n_clean = np.mean([r.get("n_det_clean",          0) for r in results])
        n_adv   = np.mean([r.get("n_det_adv",            0) for r in results])

        summary = {
            "n_images":              len(results),
            "attack_success_rate":   asr,
            "false_negative_rate":   fnr,
            "mean_conf_drop":        conf_d,
            "mean_plausibility_loss":plaus,
            "mean_perturbation_linf":pert,
            "mean_n_det_clean":      n_clean,
            "mean_n_det_adv":        n_adv,
        }

        # Realism scores (if available)
        real_scores = [r["realism_score"] for r in results if "realism_score" in r]
        if real_scores:
            summary["mean_realism_score"] = np.mean(real_scores)

        return summary

    def print_summary(self, summary: Dict):
        print("\n" + "=" * 55)
        print("  THERMAL MIRAGE — Attack Evaluation Summary")
        print("=" * 55)
        print(f"  Images evaluated          : {summary.get('n_images', 0)}")
        print(f"  Attack Success Rate (ASR) : {summary.get('attack_success_rate', 0):.3f}")
        print(f"  False Negative Rate (FNR) : {summary.get('false_negative_rate', 0):.3f}")
        print(f"  Mean Confidence Drop      : {summary.get('mean_conf_drop', 0):.4f}")
        print(f"  Mean Plausibility Loss    : {summary.get('mean_plausibility_loss', 0):.4f}")
        print(f"  Mean L∞ Perturbation      : {summary.get('mean_perturbation_linf', 0):.4f}")
        if "mean_realism_score" in summary:
            print(f"  Mean Realism Score        : {summary.get('mean_realism_score', 0):.4f}")
        print("=" * 55)

    def save_json(self, summary: Dict, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Eval] Saved results -> {path}")


# ── Ablation Experiment Runner ────────────────────────────────────────────────

class AblationRunner:
    """
    Runs five ablation variants and compares their evaluation summaries.

    Variants:
        A: object_only    - object branch only, no context
        B: context_only   - context branch only, no object edit
        C: full           - object + context (proposed)
        D: no_gan_prior   - full but without GAN prior
        E: fgsm_only      - no diffusion, FGSM fallback only
    """

    VARIANTS = {
        "A_object_only":    dict(use_analytic_ctx=False, n_refinement_steps=0),
        "B_context_only":   dict(tgt_class=None,          n_refinement_steps=0),
        "C_full":           dict(),
        "D_no_gan_prior":   dict(use_gan_prior=False),
        "E_fgsm_only":      dict(tgt_class=None, use_gan_prior=False,
                                 use_analytic_ctx=False, n_refinement_steps=5),
    }

    def __init__(self, base_config, attack_factory, runner_factory, evaluator):
        """
        attack_factory(config) -> ThermalMirageAttack
        runner_factory(attack) -> BatchAttackRunner
        """
        self.base_config    = base_config
        self.attack_factory = attack_factory
        self.runner_factory = runner_factory
        self.evaluator      = evaluator

    def run_all(self, batches: List[Dict], verbose: bool = True) -> Dict[str, Dict]:
        from dataclasses import replace
        all_summaries = {}

        for name, overrides in self.VARIANTS.items():
            print(f"\n[Ablation] Running variant: {name}")
            cfg    = replace(self.base_config, **overrides)
            attack = self.attack_factory(cfg)
            runner = self.runner_factory(attack)

            results = []
            for batch in batches:
                res = runner.run_batch(batch, verbose=False)
                results.extend(res)

            summary = self.evaluator.aggregate(results)
            all_summaries[name] = summary
            if verbose:
                self.evaluator.print_summary(summary)

        return all_summaries
