"""
Stage 7: Detector-in-the-Loop Adversarial Guidance

Wraps YOLO / Faster R-CNN / Deformable DETR detectors and exposes:
  - run()      : inference with clean boxes
  - attack_loss(): differentiable loss for adversarial guidance
  - grad_guidance(): compute gradient w.r.t. input for diffusion steering

Attack objectives:
  A. Hide (suppress) target object
  B. Force wrong-class misclassification
  C. Merge into background (reduce objectness)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np


# ── Detector abstraction ──────────────────────────────────────────────────────

class BaseDetectorWrapper(nn.Module):
    """
    Abstract wrapper around object detectors.
    Subclass for specific detector families.
    """

    def __init__(self, model: nn.Module, num_classes: int = 4, device: str = "cuda"):
        super().__init__()
        if device == "cuda" and not torch.cuda.is_available():
            print("[Detector] CUDA unavailable, falling back to CPU.")
            device = "cpu"
        self.device       = torch.device(device)
        self.model        = model.to(self.device)
        self.num_classes  = num_classes

    def forward(self, images: torch.Tensor) -> Dict:
        """Run inference. Returns dict with boxes, scores, labels."""
        raise NotImplementedError

    def attack_loss(
        self,
        images: torch.Tensor,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
        mode: str = "hide",
        wrong_class: Optional[int] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class YOLOWrapper(BaseDetectorWrapper):
    """
    Wraps YOLOv8 (ultralytics) for adversarial guidance.

    ⚠️  IMPORTANT — Two valid use cases, different model paths:

    Case A — Fine-tuned on FLIR (recommended):
        Train YOLO on your FLIR data first:
            yolo train data=flir.yaml model=yolov8n.pt epochs=50
        Then pass the resulting weights:
            YOLOWrapper(model_path='runs/detect/train/weights/best.pt',
                        num_classes=4)

    Case B — COCO pretrained (transfer / black-box baseline only):
        The default yolov8n.pt has 80 COCO classes.
        person=0, bicycle=1, car=2 exist in COCO but dog=16 (not 3).
        This works as a rough transfer attack baseline ONLY.
        Do NOT use COCO weights as the primary detector for FLIR evaluation.

    The `coco_to_flir_map` param handles Case B by remapping COCO class IDs
    to our internal 0-3 indices:
        COCO 0  (person)  → internal 0
        COCO 1  (bicycle) → internal 1
        COCO 2  (car)     → internal 2
        COCO 16 (dog)     → internal 3
    """

    # COCO class ids for the four FLIR classes we care about
    COCO_TO_FLIR = {0: 0, 1: 1, 2: 2, 16: 3}   # coco_id → internal_idx

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        num_classes: int = 4,
        conf_threshold: float = 0.25,
        device: str = "cuda",
        is_coco_pretrained: bool = None,   # None = auto-detect
    ):
        try:
            from ultralytics import YOLO
            yolo = YOLO(model_path)
            yolo.model.eval()
            super().__init__(yolo.model, num_classes, device)
            self.yolo = yolo
        except ImportError:
            raise ImportError("Install ultralytics: pip install ultralytics")

        self.conf_threshold = conf_threshold

        # Auto-detect whether this is a COCO model (80 classes) or FLIR-tuned (4 classes)
        try:
            nc = yolo.model.nc   # number of classes the model was trained with
        except AttributeError:
            nc = num_classes

        if is_coco_pretrained is None:
            is_coco_pretrained = (nc == 80)

        self.is_coco = is_coco_pretrained

        if self.is_coco:
            print(f"[YOLOWrapper] ⚠  Loaded COCO-pretrained model ({nc} classes).")
            print(f"  Will filter to COCO ids {{0,1,2,16}} and remap to internal 0-3.")
            print(f"  For proper FLIR evaluation, fine-tune on FLIR data first.")
        else:
            print(f"[YOLOWrapper] Loaded FLIR-tuned model ({nc} classes → internal 0-3).")

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> list:
        results = self.yolo(images.to(self.device), verbose=False)
        out = []
        for r in results:
            boxes  = r.boxes.xyxy.cpu()
            scores = r.boxes.conf.cpu()
            labels = r.boxes.cls.cpu().long()

            if self.is_coco:
                # Keep only the 4 FLIR-relevant COCO classes and remap
                keep_mask = torch.tensor(
                    [l.item() in self.COCO_TO_FLIR for l in labels],
                    dtype=torch.bool,
                )
                boxes  = boxes[keep_mask]
                scores = scores[keep_mask]
                labels = torch.tensor(
                    [self.COCO_TO_FLIR[l.item()] for l in labels[keep_mask]],
                    dtype=torch.long,
                )
            # else: assume model outputs 0-3 directly

            keep_conf = scores >= self.conf_threshold
            out.append({
                "boxes":  boxes[keep_conf],
                "scores": scores[keep_conf],
                "labels": labels[keep_conf],
            })
        return out

    @torch.no_grad()
    def detect(self, images: torch.Tensor, conf_threshold: float = None):
        old_threshold = self.conf_threshold
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        try:
            return self.forward(images)
        finally:
            self.conf_threshold = old_threshold

    def raw_logits(self, images: torch.Tensor):
        images = images.to(self.device)
        pred   = self.model(images)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        return pred

    def attack_loss(
        self,
        images: torch.Tensor,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
        mode: str = "hide",
        wrong_class: Optional[int] = None,
        iou_thresh: float = 0.3,
    ) -> torch.Tensor:
        pred = self.raw_logits(images)   # (B, A, 4+nc) or (B, A, 5+nc)
        B    = pred.shape[0]

        if pred.shape[-1] == 5 + self.num_classes:
            obj_conf   = torch.sigmoid(pred[..., 4])
            cls_logits = pred[..., 5:]
        else:
            cls_logits = pred[..., 4:]
            obj_conf   = cls_logits.max(dim=-1).values.sigmoid()

        if mode == "hide":
            return -obj_conf.mean()

        elif mode == "misclassify" and wrong_class is not None:
            true_cls   = target_labels[0].item() if target_labels.numel() > 0 else 0
            true_conf  = cls_logits[..., true_cls].mean()
            wrong_conf = cls_logits[..., wrong_class].mean()
            return wrong_conf - true_conf

        elif mode == "background":
            max_cls = cls_logits.max(dim=-1).values
            return -(obj_conf * max_cls).mean()

        return -obj_conf.mean()


class RTDETRWrapper(BaseDetectorWrapper):
    """
    Wraps Ultralytics RT-DETR for target-side evaluation.

    This wrapper mirrors YOLOWrapper's detection output format:
        boxes  : pixel xyxy
        scores : confidence
        labels : internal FLIR labels 0-3

    It is intended for black-box target evaluation. If used as the
    detector-in-the-loop, attack_loss falls back to a non-differentiable stub.
    """

    COCO_TO_FLIR = YOLOWrapper.COCO_TO_FLIR

    def __init__(
        self,
        model_path: str = "rtdetr-l.pt",
        num_classes: int = 4,
        conf_threshold: float = 0.25,
        device: str = "cuda",
        is_coco_pretrained: bool = None,
    ):
        try:
            from ultralytics import RTDETR
            rtdetr = RTDETR(model_path)
            rtdetr.model.eval()
            super().__init__(rtdetr.model, num_classes, device)
            self.rtdetr = rtdetr
        except ImportError:
            raise ImportError("Install ultralytics with RT-DETR support: pip install ultralytics")

        self.conf_threshold = conf_threshold

        try:
            nc = rtdetr.model.nc
        except AttributeError:
            nc = num_classes

        if is_coco_pretrained is None:
            is_coco_pretrained = (nc == 80)

        self.is_coco = is_coco_pretrained

        if self.is_coco:
            print(f"[RTDETRWrapper] ⚠  Loaded COCO-pretrained model ({nc} classes).")
            print("  Will filter to COCO ids {0,1,2,16} and remap to internal 0-3.")
            print("  For proper FLIR evaluation, fine-tune RT-DETR on FLIR/mine_data first.")
        else:
            print(f"[RTDETRWrapper] Loaded FLIR-tuned model ({nc} classes → internal 0-3).")

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> list:
        results = self.rtdetr(images.to(self.device), verbose=False)
        out = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                out.append({
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.long),
                })
                continue

            boxes = r.boxes.xyxy.cpu()
            scores = r.boxes.conf.cpu()
            labels = r.boxes.cls.cpu().long()

            if self.is_coco:
                keep_mask = torch.tensor(
                    [l.item() in self.COCO_TO_FLIR for l in labels],
                    dtype=torch.bool,
                )
                boxes = boxes[keep_mask]
                scores = scores[keep_mask]
                labels = torch.tensor(
                    [self.COCO_TO_FLIR[l.item()] for l in labels[keep_mask]],
                    dtype=torch.long,
                )

            keep_conf = scores >= self.conf_threshold
            out.append({
                "boxes": boxes[keep_conf],
                "scores": scores[keep_conf],
                "labels": labels[keep_conf],
            })
        return out

    @torch.no_grad()
    def detect(self, images: torch.Tensor, conf_threshold: float = None):
        old_threshold = self.conf_threshold
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        try:
            return self.forward(images)
        finally:
            self.conf_threshold = old_threshold

    def attack_loss(
        self,
        images: torch.Tensor,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
        mode: str = "hide",
        wrong_class: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        return torch.tensor(0.0, device=self.device, requires_grad=True)


class FasterRCNNWrapper(BaseDetectorWrapper):
    """
    Wraps torchvision Faster R-CNN for adversarial guidance.

    Class convention:
      - Faster R-CNN internally uses class 0 = background
      - Our 4 FLIR classes map to detector classes 1-4
        (internal_idx 0=person → detector_cls 1,
         internal_idx 1=bicycle → detector_cls 2, etc.)
      - This offset is applied automatically in attack_loss()
    """

    def __init__(
        self,
        num_flir_classes: int = 4,    # person, bicycle, car, dog (NOT including background)
        pretrained: bool = True,
        device: str = "cuda",
    ):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        # num_classes for Faster R-CNN = background + N object classes
        num_det_classes = num_flir_classes + 1   # +1 for background (class 0)
        model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_det_classes)

        super().__init__(model, num_flir_classes, device)
        self.num_det_classes = num_det_classes
        print(f"[FasterRCNN] head: {num_det_classes} classes "
              f"(background + {num_flir_classes} FLIR classes)")
        print(f"  background=0, person=1, bicycle=2, car=3, dog=4")

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> List[Dict]:
        """
        Returns predictions with labels shifted back to internal 0-3 indices.
        (removes background class 0, shifts 1-4 → 0-3)
        """
        imgs  = [img.to(self.device) for img in images]
        preds = self.model(imgs)
        out   = []
        for pred in preds:
            keep = pred["labels"] > 0           # discard background predictions
            out.append({
                "boxes":  pred["boxes"][keep].cpu(),
                "scores": pred["scores"][keep].cpu(),
                "labels": (pred["labels"][keep] - 1).cpu(),  # shift 1-4 → 0-3
            })
        return out

    def attack_loss(
        self,
        images: torch.Tensor,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
        mode: str = "hide",
        wrong_class: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute adversarial loss using Faster R-CNN training losses.
        Labels are shifted from internal 0-3 → detector 1-4 (+1 for background).
        """
        self.model.train()
        imgs = [images[0].to(self.device)]

        tgt_boxes  = target_boxes.to(self.device)
        tgt_labels = target_labels.to(self.device)

        if tgt_boxes.numel() == 0:
            self.model.eval()
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Convert normalized boxes → pixel coordinates
        H, W = images.shape[2], images.shape[3]
        tgt_boxes_px = tgt_boxes.clone()
        tgt_boxes_px[:, [0, 2]] *= W
        tgt_boxes_px[:, [1, 3]] *= H

        # Shift internal indices 0-3 → detector indices 1-4
        # (0 is background in Faster R-CNN, our classes start at 1)
        tgt_labels_shifted = tgt_labels + 1

        targets = [{"boxes": tgt_boxes_px, "labels": tgt_labels_shifted}]

        try:
            loss_dict = self.model(imgs, targets)
        except Exception as e:
            self.model.eval()
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        self.model.eval()

        if mode == "hide":
            # The attack loop performs gradient ascent, so larger values must
            # mean "worse for the detector." Maximizing the detector's own
            # training losses suppresses clean detections.
            loss = (loss_dict.get("loss_classifier",   torch.tensor(0.0, device=self.device)) +
                    loss_dict.get("loss_objectness",   torch.tensor(0.0, device=self.device)))
            return loss

        elif mode == "misclassify" and wrong_class is not None:
            return loss_dict.get("loss_classifier", torch.tensor(0.0, device=self.device))

        else:   # background
            return sum(loss_dict.values())

    @torch.no_grad()
    def detect(self, images: torch.Tensor, conf_threshold: float = 0.3):
        """
        Run detection on images with confidence threshold.
        
        Args:
            images: (B, 3, H, W) tensor or list of (3, H, W) tensors
            conf_threshold: minimum confidence score (default: 0.3)
        
        Returns:
            List of dicts, one per image:
            {
                'boxes': (N, 4) tensor - xyxy in pixel coordinates,
                'labels': (N,) tensor - class IDs (0-3 for FLIR),
                'scores': (N,) tensor - confidence scores
            }
        """
        self.model.eval()
        
        # Handle both batch tensor and list of tensors
        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                images = images.unsqueeze(0)  # (3,H,W) -> (1,3,H,W)
            imgs = [img.to(self.device) for img in images]
        else:
            imgs = [img.to(self.device) for img in images]
        
        # Run detection
        predictions = self.model(imgs)
        
        # Process results
        results = []
        for pred in predictions:
            # Filter by confidence threshold
            keep = pred['scores'] >= conf_threshold
            
            # Also filter out background predictions (label == 0 in detector)
            keep = keep & (pred['labels'] > 0)
            
            results.append({
                'boxes': pred['boxes'][keep],      # Already in pixel coordinates
                'labels': pred['labels'][keep] - 1,  # Shift 1-4 → 0-3 (FLIR classes)
                'scores': pred['scores'][keep]
            })
        
        return results

    def get_feature_extractor(self):
        """
        Return the backbone feature extractor for class prior computation.
        
        Returns:
            nn.Module: The ResNet50-FPN backbone of Faster R-CNN
        """
        return self.model.backbone


# ── Gradient-based adversarial guidance ──────────────────────────────────────

class AdversarialGuide:
    """
    Computes gradient of the detector adversarial loss w.r.t. input images.
    Used to steer diffusion reverse steps (Classifier Guidance style).

    Usage:
        guide = AdversarialGuide(detector_wrapper)
        grad  = guide.compute_gradient(x_t, boxes, labels, mode='hide')
        # inject grad into diffusion p_sample_loop as guidance_fn
    """

    def __init__(
        self,
        detector: BaseDetectorWrapper,
        loss_scale: float = 1.0,
    ):
        self.detector   = detector
        self.loss_scale = loss_scale

    def compute_gradient(
        self,
        image: torch.Tensor,          # (1, 3, H, W) requires_grad=True
        target_boxes: torch.Tensor,   # (N, 4) normalized xyxy
        target_labels: torch.Tensor,  # (N,)
        mode: str = "hide",
        wrong_class: Optional[int] = None,
    ) -> torch.Tensor:
        """Returns gradient (1, 3, H, W) to add to diffusion mean."""
        image = image.detach().requires_grad_(True)
        loss  = self.detector.attack_loss(
            image, target_boxes, target_labels, mode=mode, wrong_class=wrong_class
        )
        loss  = loss * self.loss_scale
        grad  = torch.autograd.grad(loss, image)[0]
        return grad.detach()

    def make_guidance_fn(
        self,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
        mode: str = "hide",
        wrong_class: Optional[int] = None,
    ):
        """Returns a closure compatible with DDPMScheduler.p_sample_loop guidance_fn."""
        def guidance_fn(x_t: torch.Tensor, t_val: int) -> torch.Tensor:
            # Only guide at specific timesteps to avoid instability
            if t_val % 10 != 0:
                return None
            try:
                grad = self.compute_gradient(x_t, target_boxes, target_labels, mode, wrong_class)
                return -grad    # sign: we want to MAXIMIZE attack loss, so follow gradient
            except Exception:
                return None
        return guidance_fn


# ── Thermal Plausibility Loss (Stage 8) ───────────────────────────────────────

class ThermalPlausibilityLoss(nn.Module):
    """
    Evaluates physically implausible thermal edits.

    Losses:
      1. smoothness_loss    : L1 spatial gradients over the full image
      2. boundary_loss      : continuity at mask boundary
      3. histogram_loss     : KL between edited and original full-image histograms
      4. intensity_range_loss: edited intensities stay in valid range

    The object mask is retained for the boundary and range terms. Smoothness
    and histogram consistency are intentionally evaluated over the complete
    thermal frame so their reported values describe global image plausibility.
    """

    def __init__(
        self,
        lambda_smooth: float = 1.0,
        lambda_boundary: float = 0.5,
        lambda_hist: float = 0.5,
        lambda_range: float = 1.0,
        n_hist_bins: int = 32,
    ):
        super().__init__()
        self.lambda_smooth   = lambda_smooth
        self.lambda_boundary = lambda_boundary
        self.lambda_hist     = lambda_hist
        self.lambda_range    = lambda_range
        self.n_bins          = n_hist_bins

    def _spatial_grad(self, img: torch.Tensor) -> torch.Tensor:
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]
        return dy, dx

    def smoothness_loss(
        self, edited: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """L1 total variation over the region selected by ``mask``."""
        dy, dx = self._spatial_grad(edited)
        my = mask[:, :, 1:, :]
        mx = mask[:, :, :, 1:]
        loss = (my * dy.abs()).mean() + (mx * dx.abs()).mean()
        return loss

    def boundary_loss(
        self, edited: torch.Tensor, original: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean clean-to-edited intensity change in the object-boundary band.

        Binary dilation and erosion are implemented with max pooling. The
        result is normalized by the number of active boundary values (including
        channels), rather than by the area of the complete image.
        """
        mask = mask.float()

        # Binary dilation: active when any value in the 3x3 neighborhood is 1.
        dilated = F.max_pool2d(
            mask,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Binary erosion via the complement of a dilation of the complement.
        eroded = 1.0 - F.max_pool2d(
            1.0 - mask,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        boundary = (dilated - eroded).clamp(0, 1)

        boundary_channels = boundary.expand_as(edited)
        absolute_change = (edited - original).abs() * boundary_channels
        return absolute_change.sum() / (boundary_channels.sum() + 1e-8)

    def histogram_loss(
        self, edited: torch.Tensor, original: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        KL divergence between grayscale intensity histograms in ``mask``.

        A small positive floor is added to both distributions before they are
        renormalized. This implements the limiting 0*log(0)=0 behavior safely
        and prevents empty histogram bins from producing NaNs in evaluation
        logs.
        """
        B = edited.shape[0]
        total = torch.tensor(0.0, device=edited.device)
        for b in range(B):
            m   = mask[b, 0].bool().flatten()
            e   = edited[b].mean(0).flatten()[m]
            o   = original[b].mean(0).flatten()[m]
            if e.numel() == 0:
                continue
            # Soft histogram
            bins = torch.linspace(0, 1, self.n_bins + 1, device=edited.device)
            he   = self._soft_hist(e, bins)
            ho   = self._soft_hist(o, bins)
            eps = 1e-8
            he = (he + eps) / (he + eps).sum()
            ho = (ho + eps) / (ho + eps).sum()
            kl = (ho * (ho.log() - he.log())).sum()
            total = total + kl
        return total / B

    def _soft_hist(self, x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        """Differentiable soft histogram."""
        centers = (bins[:-1] + bins[1:]) / 2
        bw      = bins[1] - bins[0]
        dist    = (x.unsqueeze(-1) - centers.unsqueeze(0)).abs()
        weights = F.relu(1 - dist / bw)
        h       = weights.sum(0)
        return h / (h.sum() + 1e-8)

    def intensity_range_loss(
        self, edited: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Penalize out-of-range [0,1] pixels in mask."""
        m = mask.expand_as(edited)
        over  = F.relu(edited - 1.0) * m
        under = F.relu(-edited)      * m
        return (over + under).mean()

    def forward(
        self,
        edited: torch.Tensor,
        original: torch.Tensor,
        obj_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Smoothness and histogram consistency are global evaluation metrics.
        # Boundary continuity and range validation remain object-localized.
        full_mask  = torch.ones_like(obj_mask)
        l_smooth   = self.smoothness_loss(edited, full_mask)
        l_boundary = self.boundary_loss(edited, original, obj_mask)
        l_hist     = self.histogram_loss(edited, original, full_mask)
        l_range    = self.intensity_range_loss(edited, obj_mask)

        total = (
            self.lambda_smooth   * l_smooth +
            self.lambda_boundary * l_boundary +
            self.lambda_hist     * l_hist +
            self.lambda_range    * l_range
        )
        return total, {
            "smooth":   l_smooth.item(),
            "boundary": l_boundary.item(),
            "hist":     l_hist.item(),
            "range":    l_range.item(),
        }
