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

    # Scene degradation / physics-informed context attack
    scene_condition: str = "sequential"
    # Valid physics modes: sequential | contrast | smooth | adaptive_cool | cool
    # Legacy analytic modes are still supported: night | low_contrast | blur | clutter
    use_analytic_ctx: bool = True
    context_strength: float = 0.08   # also used as context L_inf epsilon when physics mode is active
    context_tau: float = 20.0        # decay parameter for M_soft(p)=exp(-d/tau)
    context_epsilon_tv: float = 0.02 # TV smoothing threshold
    context_alpha: float = 0.45      # contrast reduction strength
    context_beta: float = 0.65       # thermal smoothing strength
    context_gamma: float = 0.45      # adaptive cooling strength

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

        physics_modes = {
            "physics_context", "sequential", "sequential_context",
            "contrast", "contrast_reduction",
            "smooth", "thermal_smoothing",
            "cool", "context_cooling", "adaptive_cool",
        }

        if cfg.use_analytic_ctx or self.ctx_editor is None or (cfg.scene_condition in physics_modes):
            from models.diffusion_editor import ThermalContextEditor
            if cfg.scene_condition in physics_modes:
                ctx_edited = ThermalContextEditor.apply_physics_context_attack(
                    image=obj_edited,
                    ctx_mask=ctx_m,
                    obj_mask=obj_m,
                    mode=cfg.scene_condition,
                    epsilon=cfg.context_strength,
                    epsilon_tv=cfg.context_epsilon_tv,
                    tau=cfg.context_tau,
                    alpha=cfg.context_alpha,
                    beta=cfg.context_beta,
                    gamma=cfg.context_gamma,
                    adaptive_cooling=True,
                )
            else:
                ctx_edited = ThermalContextEditor.apply_analytic_degradation(
                    image=obj_edited,
                    ctx_mask=ctx_m,
                    obj_mask=obj_m,
                    mode=cfg.scene_condition if cfg.scene_condition in ["night", "low_contrast", "blur", "clutter"] else "night",
                    strength=cfg.context_strength,
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
            print(f"[Attack] Intermediate per-object plausibility: "
                  f"{plaus.item():.4f} | {plaus_components}")

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
        Context edits dominate inside ctx_mask excluding obj_mask.
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
        attack_mode: str = "hide",
        verbose: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Attack one object using the full object+context ThermalMirage pipeline.

        This method is the interface expected by attack_selector.py. It builds
        the object/context masks from one normalized bounding box, runs the
        object-level branch and the physics-informed context branch, and falls
        back to FGSM only if the full pipeline is unavailable or fails.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device).clamp(0, 1)
        box = box.to(self.device).float()
        _, _, H, W = image.shape

        metric_detector = getattr(self, "eval_detector", self.guide.detector)
        with torch.no_grad():
            original_dets = metric_detector.detect(image)
            orig_conf = self._get_box_confidence(original_dets[0], box, H, W)

        obj_mask, ctx_mask = self._build_masks_from_box(box, H, W)
        labels = torch.tensor([int(src_class)], dtype=torch.long, device=self.device)
        boxes = box.unsqueeze(0)

        # Temporarily adapt the attack config to this object's selected strategy.
        old_cfg = self.config
        self.config = AttackConfig(
            mode=attack_mode,
            wrong_class=int(tgt_class) if attack_mode == "misclassify" and tgt_class >= 0 else None,
            src_class=int(src_class),
            tgt_class=int(tgt_class) if attack_mode == "misclassify" and tgt_class >= 0 else None,
            scene_condition=getattr(old_cfg, "scene_condition", "sequential"),
            use_analytic_ctx=getattr(old_cfg, "use_analytic_ctx", True),
            context_strength=getattr(old_cfg, "context_strength", 0.08),
            context_tau=getattr(old_cfg, "context_tau", 20.0),
            context_epsilon_tv=getattr(old_cfg, "context_epsilon_tv", 0.02),
            context_alpha=getattr(old_cfg, "context_alpha", 0.45),
            context_beta=getattr(old_cfg, "context_beta", 0.65),
            context_gamma=getattr(old_cfg, "context_gamma", 0.45),
            obj_start_t=getattr(old_cfg, "obj_start_t", 500),
            ctx_start_t=getattr(old_cfg, "ctx_start_t", 300),
            guidance_scale=getattr(old_cfg, "guidance_scale", 0.08),
            n_refinement_steps=getattr(old_cfg, "n_refinement_steps", 3),
            lambda_det=getattr(old_cfg, "lambda_det", 1.0),
            lambda_cls=getattr(old_cfg, "lambda_cls", 0.5),
            lambda_real=getattr(old_cfg, "lambda_real", 0.3),
            lambda_ctx=getattr(old_cfg, "lambda_ctx", 0.4),
            max_perturbation=getattr(old_cfg, "max_perturbation", 0.15),
            use_gan_prior=getattr(old_cfg, "use_gan_prior", True),
            save_intermediate=getattr(old_cfg, "save_intermediate", False),
        )

        full_error = None
        try:
            result = self.run(
                image=image,
                obj_mask=obj_mask,
                ctx_mask=ctx_mask,
                boxes=boxes,
                labels=labels,
                verbose=verbose,
            )
            adv_image = result["adv_image"].to(self.device).clamp(0, 1)
            backend = "object_context"
        except Exception as e:
            full_error = str(e)
            adv_image = self._fgsm_single_object_fallback(
                image=image,
                box=box,
                obj_mask=obj_mask,
                src_class=src_class,
                tgt_class=tgt_class,
                attack_mode=attack_mode,
            )
            result = {}
            backend = "fgsm_fallback"
        finally:
            self.config = old_cfg

        with torch.no_grad():
            final_dets = metric_detector.detect(adv_image)
            adv_conf = self._get_box_confidence(final_dets[0], box, H, W)
            pred_class = self._get_predicted_class(final_dets[0], box, H, W)
            success = self._single_object_success(adv_conf, pred_class, tgt_class, attack_mode)
            perturbation = (adv_image - image).abs().max().item()

        info = {
            "original_confidence": float(orig_conf),
            "adversarial_confidence": float(adv_conf),
            "predicted_class": int(pred_class),
            "perturbation_linf": float(perturbation),
            "success": bool(success),
            "attack_mode": attack_mode,
            "target_class": int(tgt_class),
            "backend": backend,
            "context_attack": getattr(old_cfg, "scene_condition", "sequential"),
            "plaus_loss": result.get("plaus_loss", None),
            "full_attack_error": full_error,
        }
        return adv_image.detach(), info

    def _single_object_success(
        self,
        adv_conf: float,
        pred_class: int,
        tgt_class: int,
        attack_mode: str,
        hide_threshold: float = 0.3,
    ) -> bool:
        if attack_mode == "hide":
            return adv_conf < hide_threshold
        return pred_class == int(tgt_class)

    def _fgsm_single_object_fallback(
        self,
        image: torch.Tensor,
        box: torch.Tensor,
        obj_mask: torch.Tensor,
        src_class: int,
        tgt_class: int,
        attack_mode: str,
        steps: int = 20,
        step_size: float = 0.01,
        epsilon: float = 0.10,
    ) -> torch.Tensor:
        """
        Emergency FGSM/PGD fallback used only when object+context attack fails.
        The perturbation is constrained to the object box and clipped to [0,1].
        """
        detector = self.guide.detector
        labels = torch.tensor([int(src_class)], dtype=torch.long, device=self.device)
        boxes = box.unsqueeze(0)
        delta = torch.zeros_like(image, device=self.device, requires_grad=True)
        mask = obj_mask.expand_as(image).to(self.device)

        for _ in range(steps):
            adv = (image + delta).clamp(0, 1)
            try:
                loss = detector.attack_loss(
                    adv,
                    boxes,
                    labels,
                    mode=attack_mode,
                    wrong_class=int(tgt_class) if attack_mode == "misclassify" and tgt_class >= 0 else None,
                )
                if delta.grad is not None:
                    delta.grad.zero_()
                grad = torch.autograd.grad(loss, delta, retain_graph=False, allow_unused=True)[0]
                if grad is None:
                    break
                with torch.no_grad():
                    delta += step_size * grad.sign() * mask
                    delta.clamp_(-epsilon, epsilon)
                    delta *= mask
            except Exception:
                break

        return (image + delta.detach()).clamp(0, 1)

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
        
        # Normalize detected boxes to [0,1]. Detector wrappers may return
        # CPU tensors even when the attack image/box is on CUDA, so align
        # devices before computing IoU.
        device = query_box.device
        det_boxes = detections['boxes'].detach().clone().to(device)
        det_boxes[:, [0, 2]] /= W
        det_boxes[:, [1, 3]] /= H
        query = query_box.to(device).unsqueeze(0)
        
        # Compute IoU with query box
        ious = self._box_iou_vectorized(
            query,
            det_boxes
        )
        
        best_iou, best_idx = ious.max(dim=1)
        
        if best_iou.item() > 0.5:
            scores = detections['scores'].detach().to(device)
            return scores[best_idx.item()].item()
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
        
        device = query_box.device
        det_boxes = detections['boxes'].detach().clone().to(device)
        det_boxes[:, [0, 2]] /= W
        det_boxes[:, [1, 3]] /= H
        query = query_box.to(device).unsqueeze(0)
        
        ious = self._box_iou_vectorized(
            query,
            det_boxes
        )
        
        best_iou, best_idx = ious.max(dim=1)
        
        if best_iou.item() > 0.5:
            labels = detections['labels'].detach().to(device)
            return labels[best_idx.item()].item()
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
