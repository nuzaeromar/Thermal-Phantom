"""
Stages 6, 7, 8: Joint Adversarial Attack Pipeline

ThermalMirageAttack orchestrates:
  1. GAN prior extraction (class-transition direction)
  2. Object manipulation (diffusion inpainting)
  3. Context manipulation (scene degradation)
  4. Detector-in-the-loop guidance
  5. Thermal plausibility constraints
  6. Final composition and evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class AttackConfig:
    # Attack mode
    mode: str = "hide"               # 'hide' | 'misclassify' | 'background'
    wrong_class: Optional[int] = None   # target class for misclassification

    # Source / target class
    src_class: int = 0               # 0=person, 1=bicycle, 2=car, 3=dog
    tgt_class: Optional[int] = None  # None = background

    # Scene degradation
    scene_condition: str = "night"   # 'night' | 'low_contrast' | 'cluttered' | 'foggy_thermal'
    use_analytic_ctx: bool = True    # use fast analytic context degradation
    context_strength: float = 0.35   # strength of analytic degradation

    # Diffusion parameters
    obj_start_t: int = 500
    ctx_start_t: int = 300
    guidance_scale: float = 0.08
    n_refinement_steps: int = 3      # detector-in-the-loop refinement iterations

    # Loss weights
    lambda_det: float  = 1.0         # detection suppression
    lambda_cls: float  = 0.5         # wrong-class encouragement
    lambda_real: float = 0.3         # thermal plausibility
    lambda_ctx: float  = 0.4         # context salience drop

    # Constraint
    max_perturbation: float = 0.15   # max L∞ pixel change
    use_gan_prior: bool = True        # use GAN prior for object branch

    # Eval
    save_intermediate: bool = False


# ── Main Attack Class ─────────────────────────────────────────────────────────

class ThermalMirageAttack:
    """
    Full joint adversarial attack for FLIR thermal detection.

    Combines:
      - ThermalClassPrior   (GAN prior)
      - ThermalObjectEditor (diffusion inpainting)
      - ThermalContextEditor (context degradation)
      - AdversarialGuide    (detector gradient guidance)
      - ThermalPlausibilityLoss (physical constraints)

    Usage:
        attack = ThermalMirageAttack(
            gan_prior    = prior,
            obj_editor   = obj_editor,
            ctx_editor   = ctx_editor,
            guide        = guide,
            plaus_loss   = plaus_loss,
            config       = AttackConfig(mode='hide', src_class=0),
        )
        result = attack.run(image, obj_mask, ctx_mask, boxes, labels)
    """

    def __init__(
        self,
        gan_prior,              # ThermalClassPrior or None
        obj_editor,             # ThermalObjectEditor or None
        ctx_editor,             # ThermalContextEditor
        guide,                  # AdversarialGuide
        plaus_loss,             # ThermalPlausibilityLoss
        config: AttackConfig = None,
        device: str = "cuda",
    ):
        if device == "cuda" and not torch.cuda.is_available():
            print("[ThermalMirageAttack] CUDA unavailable, using CPU.")
            device = "cpu"
        self.prior       = gan_prior
        self.obj_editor  = obj_editor
        self.ctx_editor  = ctx_editor
        self.guide       = guide
        self.plaus_loss  = plaus_loss
        self.config      = config or AttackConfig()
        self.device      = torch.device(device)
    
    def run(
        self,
        image: torch.Tensor,         # (1, 3, H, W) in [0,1]
        obj_mask: torch.Tensor,      # (1, 1, H, W)
        ctx_mask: torch.Tensor,      # (1, 1, H, W)
        boxes: torch.Tensor,         # (N, 4) normalized xyxy
        labels: torch.Tensor,        # (N,)
        verbose: bool = True) -> Dict:
        """
        Returns:
            dict with keys:
              'adv_image'    : adversarial image (1,3,H,W) in [0,1]
              'obj_edited'   : after object edit only
              'ctx_edited'   : after context edit only
              'plaus_loss'   : plausibility loss value
              'perturbation' : L∞ change from original
              'attack_losses': list of detector loss values per refinement step
        """
        cfg    = self.config
        image  = image.to(self.device)
        obj_m  = obj_mask.to(self.device)
        ctx_m  = ctx_mask.to(self.device)
        boxes  = boxes.to(self.device)
        labels = labels.to(self.device)

        result = {}

        # ── Step 1: GAN prior for target class ─────────────────────────────
        gan_prior_imgs = None
        if cfg.use_gan_prior and self.prior is not None and cfg.tgt_class is not None:
            try:
                gan_prior_imgs = self.prior.sample_target_images(
                    target_class=cfg.tgt_class, n=16
                )
                if verbose:
                    print(f"[Attack] GAN prior sampled for class {cfg.tgt_class}")
            except Exception as e:
                if verbose:
                    print(f"[Attack] GAN prior failed: {e}, skipping.")

        # ── Step 2: Build detector guidance function ─────────────────────────
        guidance_fn = self.guide.make_guidance_fn(
            target_boxes  = boxes,
            target_labels = labels,
            mode          = cfg.mode,
            wrong_class   = cfg.wrong_class,
        )

        # ── Step 3: Object manipulation branch ─────────────────────────────
        if self.obj_editor is not None and cfg.tgt_class is not None:
            if verbose:
                print(f"[Attack] Object branch: class {cfg.src_class} → {cfg.tgt_class}")
            obj_edited = self.obj_editor.edit_object(
                image          = image,
                obj_mask       = obj_m,
                target_class   = cfg.tgt_class,
                guidance_fn    = guidance_fn,
                guidance_scale = cfg.guidance_scale,
                gan_prior_images = gan_prior_imgs,
            )
        else:
            # Use FGSM-style fast object perturbation as fallback
            obj_edited = self._fgsm_object_perturb(image, obj_m, boxes, labels, cfg)

        result["obj_edited"] = obj_edited.detach().cpu()

        # ── Step 4: Context manipulation branch ────────────────────────────
        if verbose:
            print(f"[Attack] Context branch: mode={cfg.scene_condition}")

        if cfg.use_analytic_ctx:
            from models.diffusion_editor import ThermalContextEditor
            ctx_edited = ThermalContextEditor.apply_analytic_degradation(
                image         = obj_edited,
                ctx_mask      = ctx_m,
                obj_mask      = obj_m,
                mode          = cfg.scene_condition if cfg.scene_condition in
                                  ["night","low_contrast","blur","clutter"] else "night",
                strength      = cfg.context_strength,
            )
        else:
            ctx_edited = self.ctx_editor.degrade_context(
                image           = obj_edited,
                ctx_mask        = ctx_m,
                obj_mask        = obj_m,
                scene_condition = cfg.scene_condition,
                guidance_fn     = guidance_fn,
                guidance_scale  = cfg.guidance_scale * cfg.lambda_ctx,
            )

        result["ctx_edited"] = ctx_edited.detach().cpu()

        # ── Step 5: Fuse branches ───────────────────────────────────────────
        fused = self._fuse(obj_edited, ctx_edited, obj_m, ctx_m)

        # ── Step 6: Detector-in-the-loop refinement ────────────────────────
        attack_losses = []
        adv = fused.detach().requires_grad_(True)

        for step in range(cfg.n_refinement_steps):
            try:
                det_loss = self.guide.detector.attack_loss(
                    adv, boxes, labels, mode=cfg.mode, wrong_class=cfg.wrong_class
                )
                attack_losses.append(det_loss.item() if hasattr(det_loss, 'item') else float(det_loss))

                grad = torch.autograd.grad(det_loss, adv)[0]
                # Gradient ascent on attack loss (adversarial direction)
                step_size = 0.02 / (cfg.n_refinement_steps)
                delta = step_size * grad.sign()
                adv = (adv + delta).detach().clamp(0, 1).requires_grad_(True)

                if verbose:
                    print(f"  [Refine step {step+1}] det_loss={attack_losses[-1]:.4f}")
            except Exception as e:
                if verbose:
                    print(f"  [Refine step {step+1}] skipped: {e}")

        adv = adv.detach()

        # ── Step 7: Enforce L∞ perturbation budget ─────────────────────────
        delta_total = adv - image
        delta_total = delta_total.clamp(-cfg.max_perturbation, cfg.max_perturbation)
        adv = (image + delta_total).clamp(0, 1)

        # Force thermal consistency: all RGB channels identical
        adv_gray = adv.mean(dim=1, keepdim=True)  # (B, 1, H, W) - average across RGB
        adv = adv_gray.repeat(1, 3, 1, 1)         # (B, 3, H, W) - replicate to RGB

        if verbose:
            # Verify consistency
            r_g_diff = (adv[:, 0] - adv[:, 1]).abs().max().item()
            print(f"[Attack] Thermal consistency enforced (R-G diff: {r_g_diff:.6f})")

        # ── Step 8: Thermal plausibility ────────────────────────────────────
        plaus, plaus_components = self.plaus_loss(adv, image, obj_m)

        if verbose:
            print(f"[Attack] Plausibility loss: {plaus.item():.4f} | {plaus_components}")

        result["adv_image"]    = adv.cpu()
        result["plaus_loss"]   = plaus.item()
        result["plaus_detail"] = plaus_components
        result["attack_losses"]= attack_losses
        result["perturbation"] = (adv - image).abs().max().item()

        return result

    def _fuse(
        self,
        obj_edited: torch.Tensor,
        ctx_edited: torch.Tensor,
        obj_mask:   torch.Tensor,
        ctx_mask:   torch.Tensor) -> torch.Tensor:
        """
        Fuse object and context branches.
        Object edits dominate inside obj_mask.
        Context edits dominate inside ctx_mask \ obj_mask.
        Original image is used elsewhere.
        """
        m_obj = obj_mask.expand_as(obj_edited)
        m_ctx = (ctx_mask * (1 - obj_mask)).expand_as(ctx_edited)

        fused  = m_obj * obj_edited           # object region from obj branch
        fused += m_ctx * ctx_edited           # context region from ctx branch
        # Fill remaining with ctx_edited (it preserves outside already)
        outside = 1 - m_obj - m_ctx
        fused += outside.clamp(0, 1) * ctx_edited
        return fused.clamp(0, 1)

    def _fgsm_object_perturb(
        self,
        image: torch.Tensor,
        obj_mask: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        cfg: AttackConfig,
        eps: float = 0.08) -> torch.Tensor:
        """
        Fast fallback: FGSM perturbation inside object mask.
        Used when diffusion object editor is not available.
        """
        img_adv = image.detach().clone().requires_grad_(True)
        try:
            loss = self.guide.detector.attack_loss(
                img_adv, boxes, labels, mode=cfg.mode, wrong_class=cfg.wrong_class
            )
            grad  = torch.autograd.grad(loss, img_adv)[0]
            delta = eps * grad.sign() * obj_mask.expand_as(image)
            result = (image + delta).clamp(0, 1)
            return result.detach()
        except Exception:
            return image.clone()

    def attack_single_object(
        self,
        image: torch.Tensor,
        box: torch.Tensor,
        src_class: int,
        tgt_class: int,
        attack_mode: str = 'hide') -> Tuple[torch.Tensor, Dict]:
        """
        Attack a single object in an image.
        
        Args:
            image: (1, 3, H, W) or (3, H, W) clean image
            box: (4,) tensor - [x1, y1, x2, y2] in normalized [0,1] coordinates
            src_class: Source object class (0-3)
            tgt_class: Target class (-1 for hide, 0-3 for misclassify)
            attack_mode: 'hide' or 'misclassify'
        
        Returns:
            adv_image: (1, 3, H, W) adversarial image
            info: Dict with attack metrics
        """
        # Ensure proper shape
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        box = box.to(self.device)
        B, C, H, W = image.shape
        
        # Get initial detection confidence
        with torch.no_grad():
            orig_dets = self.detector.detect(image)
            orig_conf = self._get_box_confidence(orig_dets[0], box, H, W)
            orig_class = src_class
        
        # Build spatial masks for this object
        obj_mask, ctx_mask = self._build_masks_from_box(box, H, W)
        
        # Initialize adversarial perturbation
        delta = torch.zeros_like(image, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=0.01)
        
        # Get GAN prior for target class (if misclassifying)
        if attack_mode == 'misclassify' and tgt_class >= 0:
            with torch.no_grad():
                gan_prior = self.class_prior.get_prior(tgt_class)
                # Resize to match object region
                x1, y1, x2, y2 = self._denormalize_box(box, H, W)
                obj_h, obj_w = y2 - y1, x2 - x1
                if obj_h > 0 and obj_w > 0:
                    gan_prior_resized = F.interpolate(
                        gan_prior.unsqueeze(0),
                        size=(obj_h, obj_w),
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    gan_prior_resized = None
        else:
            gan_prior_resized = None
        
        # Attack optimization loop
        best_adv = image.clone()
        best_conf = orig_conf
        
        for step in range(100):  # attack_iters
            adv = torch.clamp(image + delta, 0, 1)
            
            # Get detector outputs
            dets = self.detector.detect(adv)
            
            # Compute losses
            loss = 0.0
            
            # L1: Detection/Classification loss
            if attack_mode == 'hide':
                # Suppress detection confidence
                det_loss = self._compute_hide_loss(dets[0], box, H, W)
            else:
                # Encourage misclassification to target
                det_loss = self._compute_misclassify_loss(
                    dets[0], box, tgt_class, H, W
                )
            loss += 1.0 * det_loss
            
            # L2: GAN prior alignment (for misclassification)
            if gan_prior_resized is not None:
                x1, y1, x2, y2 = self._denormalize_box(box, H, W)
                adv_crop = adv[:, :, y1:y2, x1:x2]
                # Scale to [-1, 1] for GAN
                adv_crop_scaled = adv_crop * 2 - 1
                gan_prior_scaled = gan_prior_resized * 2 - 1
                prior_loss = F.mse_loss(adv_crop_scaled, gan_prior_scaled)
                loss += 0.5 * prior_loss
            
            # L3: Thermal plausibility (smooth, bounded perturbations)
            thermal_loss = self._thermal_plausibility_loss(delta, obj_mask)
            loss += 0.3 * thermal_loss
            
            # L4: Smoothness in object region
            smooth_loss = self._smoothness_loss(delta, obj_mask)
            loss += 0.2 * smooth_loss
            
            # L5: Minimize perturbation magnitude
            pert_loss = (delta * obj_mask).abs().mean()
            loss += 0.1 * pert_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Project perturbation to valid range
            with torch.no_grad():
                # Clamp to max perturbation (L-infinity constraint)
                delta.data = torch.clamp(delta.data, -0.15, 0.15)
                # Apply only to object region
                delta.data = delta.data * obj_mask
            
            # Track best adversarial so far
            if step % 10 == 0:
                with torch.no_grad():
                    curr_conf = self._get_box_confidence(dets[0], box, H, W)
                    if curr_conf < best_conf:
                        best_conf = curr_conf
                        best_adv = adv.clone()
        
        # Final adversarial image
        adv_image = best_adv
        
        # Evaluate attack success
        with torch.no_grad():
            final_dets = self.detector.detect(adv_image)
            adv_conf = self._get_box_confidence(final_dets[0], box, H, W)
            pred_class = self._get_predicted_class(final_dets[0], box, H, W)
            
            if attack_mode == 'hide':
                success = adv_conf < 0.3  # Detection suppressed
            else:
                success = (pred_class == tgt_class)  # Misclassified correctly
            
            perturbation = (adv_image - image).abs().max().item()
        
        info = {
            'original_confidence': orig_conf,
            'adversarial_confidence': adv_conf,
            'predicted_class': pred_class,
            'perturbation_linf': perturbation,
            'success': success,
            'attack_mode': attack_mode,
            'target_class': tgt_class
        }
        
        return adv_image, info

    
    def _build_masks_from_box(
        self,
        box: torch.Tensor,
        H: int,
        W: int,
        context_scale: float = 1.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build object and context masks from normalized bounding box.
        
        Returns:
            obj_mask: (1, 1, H, W) - binary mask for object
            ctx_mask: (1, 1, H, W) - binary mask for expanded context
        """
        x1, y1, x2, y2 = self._denormalize_box(box, H, W)
        
        # Object mask
        obj_mask = torch.zeros(1, 1, H, W, device=box.device)
        obj_mask[:, :, y1:y2, x1:x2] = 1.0
        
        # Context mask (expanded)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = (x2 - x1) * context_scale
        h = (y2 - y1) * context_scale
        
        cx1 = max(0, int(cx - w / 2))
        cy1 = max(0, int(cy - h / 2))
        cx2 = min(W, int(cx + w / 2))
        cy2 = min(H, int(cy + h / 2))
        
        ctx_mask = torch.zeros(1, 1, H, W, device=box.device)
        ctx_mask[:, :, cy1:cy2, cx1:cx2] = 1.0
        
        return obj_mask, ctx_mask
    
    def _denormalize_box(self, box: torch.Tensor, H: int, W: int) -> Tuple[int, int, int, int]:
        """Convert normalized box [0,1] to pixel coordinates."""
        x1, y1, x2, y2 = box.tolist()
        x1 = int(x1 * W)
        y1 = int(y1 * H)
        x2 = int(x2 * W)
        y2 = int(y2 * H)
        return x1, y1, x2, y2
    
    def _get_box_confidence(
        self,
        detections: Dict,
        query_box: torch.Tensor,
        H: int,
        W: int) -> float:
        """
        Get detection confidence for box matching query_box.
        
        Args:
            detections: {'boxes': (N,4), 'scores': (N,), 'labels': (N,)}
            query_box: (4,) normalized xyxy
            H, W: image dimensions
        
        Returns:
            confidence score (0-1)
        """
        if len(detections['boxes']) == 0:
            return 0.0
        
        # Normalize detected boxes to [0,1]
        det_boxes = detections['boxes'].clone()
        det_boxes[:, [0, 2]] /= W
        det_boxes[:, [1, 3]] /= H
        
        # Compute IoU with query box
        ious = self._box_iou_vectorized(
            query_box.unsqueeze(0),
            det_boxes
        )
        
        best_iou, best_idx = ious.max(dim=1)
        
        if best_iou.item() > 0.5:
            return detections['scores'][best_idx].item()
        return 0.0
    
    def _get_predicted_class(
        self,
        detections: Dict,
        query_box: torch.Tensor,
        H: int,
        W: int) -> int:
        """Get predicted class for detection matching query_box."""
        if len(detections['boxes']) == 0:
            return -1
        
        det_boxes = detections['boxes'].clone()
        det_boxes[:, [0, 2]] /= W
        det_boxes[:, [1, 3]] /= H
        
        ious = self._box_iou_vectorized(
            query_box.unsqueeze(0),
            det_boxes
        )
        
        best_iou, best_idx = ious.max(dim=1)
        
        if best_iou.item() > 0.5:
            return detections['labels'][best_idx].item()
        return -1
    
    def _box_iou_vectorized(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between two sets of boxes.
        
        Args:
            boxes1: (M, 4) xyxy
            boxes2: (N, 4) xyxy
        
        Returns:
            ious: (M, N)
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (M, N, 2)
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (M, N, 2)
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2[None, :] - inter
        iou = inter / (union + 1e-6)
        
        return iou
    
    def _compute_hide_loss(
        self,
        detections: Dict,
        query_box: torch.Tensor,
        H: int,
        W: int) -> torch.Tensor:
        """
        Loss to suppress detection of object at query_box.
        Minimizes detection confidence.
        """
        if len(detections['boxes']) == 0:
            return torch.tensor(0.0, device=query_box.device)
        
        # Find detection matching query box
        det_boxes = detections['boxes'].clone()
        det_boxes[:, [0, 2]] /= W
        det_boxes[:, [1, 3]] /= H
        
        ious = self._box_iou_vectorized(
            query_box.unsqueeze(0),
            det_boxes
        )
        
        best_iou, best_idx = ious.max(dim=1)
        
        if best_iou.item() > 0.3:
            # Matched a detection - minimize its confidence
            conf = detections['scores'][best_idx]
            return conf  # Want to minimize this
        else:
            # No detection found - attack successful
            return torch.tensor(0.0, device=query_box.device)
    
    def _compute_misclassify_loss(
        self,
        detections: Dict,
        query_box: torch.Tensor,
        target_class: int,
        H: int,
        W: int) -> torch.Tensor:
        """
        Loss to misclassify object to target_class.
        Maximizes probability of target class.
        """
        if len(detections['boxes']) == 0:
            # No detection - encourage detection as target class
            return torch.tensor(1.0, device=query_box.device)
        
        det_boxes = detections['boxes'].clone()
        det_boxes[:, [0, 2]] /= W
        det_boxes[:, [1, 3]] /= H
        
        ious = self._box_iou_vectorized(
            query_box.unsqueeze(0),
            det_boxes
        )
        
        best_iou, best_idx = ious.max(dim=1)
        
        if best_iou.item() > 0.3:
            # Found matching detection
            pred_class = detections['labels'][best_idx].item()
            conf = detections['scores'][best_idx]
            
            if pred_class == target_class:
                # Already misclassified correctly - minimize loss
                return -conf  # Negative to encourage this
            else:
                # Wrong class - penalize
                return conf  # Maximize confusion
        else:
            # No detection
            return torch.tensor(1.0, device=query_box.device)
    
    def _thermal_plausibility_loss(
        self,
        delta: torch.Tensor,
        obj_mask: torch.Tensor) -> torch.Tensor:
        """
        Ensure perturbations are thermally plausible:
        - Bounded intensity changes
        - Smooth gradients
        """
        # L1: Bounded intensity (already handled by clipping, but add soft constraint)
        intensity_loss = (delta * obj_mask).abs().mean()
        
        # L2: Local consistency (neighboring pixels should have similar perturbations)
        dy = delta[:, :, 1:, :] - delta[:, :, :-1, :]
        dx = delta[:, :, :, 1:] - delta[:, :, :, :-1]
        gradient_loss = (dy.abs().mean() + dx.abs().mean())
        
        return intensity_loss + 0.5 * gradient_loss
    
    def _smoothness_loss(
        self,
        delta: torch.Tensor,
        obj_mask: torch.Tensor) -> torch.Tensor:
        """Total variation loss for smoothness."""
        # Masked total variation
        masked_delta = delta * obj_mask
        
        tv_h = (masked_delta[:, :, 1:, :] - masked_delta[:, :, :-1, :]).abs()
        tv_w = (masked_delta[:, :, :, 1:] - masked_delta[:, :, :, :-1]).abs()
        
        return tv_h.mean() + tv_w.mean()


            


# ── Batch attack runner ───────────────────────────────────────────────────────

class BatchAttackRunner:
    """
    Runs ThermalMirageAttack over a DataLoader.
    Saves results and computes per-image success metrics.
    """

    def __init__(
        self,
        attack: ThermalMirageAttack,
        detector_fn,               # callable: image -> [{'boxes':..,'scores':..,'labels':..}]
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.25,
    ):
        self.attack    = attack
        self.detect    = detector_fn
        self.iou_thr   = iou_threshold
        self.conf_thr  = conf_threshold

    def run_batch(
        self,
        batch: Dict,
        verbose: bool = False,
    ) -> List[Dict]:
        """
        Process one batch from the DataLoader.
        Returns list of per-image result dicts.
        """
        images    = batch["images"]          # (B, 3, H, W)
        images_raw= batch["images_raw"]
        boxes_list= batch["boxes"]
        labels_list=batch["labels"]
        obj_masks = batch.get("obj_masks")
        ctx_masks = batch.get("ctx_masks")

        B = images.shape[0]
        results = []

        for i in range(B):
            img   = images_raw[i:i+1]        # (1, 3, H, W)
            boxes = boxes_list[i]
            lbls  = labels_list[i]

            if obj_masks is not None:
                obj_m = obj_masks[i:i+1]
                ctx_m = ctx_masks[i:i+1]
            else:
                # Fallback: create uniform masks
                H, W  = img.shape[2:]
                obj_m = torch.ones(1, 1, H, W) * 0.5
                ctx_m = torch.ones(1, 1, H, W)

            if boxes.numel() == 0:
                continue

            # Run attack
            try:
                res = self.attack.run(img, obj_m, ctx_m, boxes, lbls, verbose=verbose)
            except Exception as e:
                print(f"[BatchRunner] Image {i} attack failed: {e}")
                continue

            # Evaluate clean vs adversarial detections
            adv_img = res["adv_image"].to(images.device)
            clean_dets = self.detect(img)
            adv_dets   = self.detect(adv_img)

            metrics = self._compute_metrics(clean_dets[0], adv_dets[0], boxes, lbls)
            res.update(metrics)
            res["img_path"]   = batch["img_paths"][i]
            res["image_id"]   = batch["image_ids"][i]
            results.append(res)

        return results

    def _compute_metrics(
        self,
        clean_det: Dict,
        adv_det:   Dict,
        gt_boxes:  torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> Dict:
        """
        Compute per-image attack metrics:
          - attack_success_rate : fraction of GT objects suppressed after attack
          - conf_drop           : mean confidence drop
          - false_negative_rate : % of objects that disappear
        """
        clean_scores = clean_det.get("scores", torch.zeros(0))
        adv_scores   = adv_det.get("scores",   torch.zeros(0))

        n_clean = (clean_scores > self.conf_thr).sum().item()
        n_adv   = (adv_scores   > self.conf_thr).sum().item()
        n_gt    = len(gt_boxes)

        conf_clean = clean_scores[clean_scores > self.conf_thr].mean().item() if n_clean > 0 else 0.0
        conf_adv   = adv_scores[  adv_scores   > self.conf_thr].mean().item() if n_adv   > 0 else 0.0

        attack_success = 1.0 if (n_adv < n_clean and n_clean > 0) else 0.0
        fn_rate = max(0.0, (n_clean - n_adv) / (n_gt + 1e-8))

        return {
            "n_det_clean":        n_clean,
            "n_det_adv":          n_adv,
            "conf_clean":         conf_clean,
            "conf_adv":           conf_adv,
            "conf_drop":          conf_clean - conf_adv,
            "attack_success":     attack_success,
            "false_negative_rate": fn_rate,
        }
