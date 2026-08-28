"""
simple_attack_wrapper.py

Simplified attack wrapper for multi-class profiling.
This is a temporary solution until you fully implement attack_single_object().

Place this file in your project root directory.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict


class SimpleAttackWrapper:
    """
    Simplified attack for multi-class profiling.
    This uses basic FGSM-style attacks for testing the multi-class framework.
    
    Once you're ready, replace this with your full ThermalMirageAttack implementation
    by adding the attack_single_object() method to your attack/joint_attack.py.
    """
    
    def __init__(self, detector, gan_generator, class_prior, device="cuda"):
        self.detector = detector
        self.G = gan_generator
        self.class_prior = class_prior
        self.device = device
        
        print("[SimpleAttackWrapper] Using simplified attack for testing")
        print("  This is a placeholder - implement attack_single_object() for full functionality")
    
    def attack_single_object(
        self,
        image: torch.Tensor,
        box: torch.Tensor,
        src_class: int,
        tgt_class: int,
        attack_mode: str = 'hide'
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Simplified single-object attack using FGSM.
        
        Args:
            image: (1, 3, H, W) or (3, H, W) input image
            box: (4,) normalized xyxy coordinates
            src_class: source class (0-3)
            tgt_class: target class (-1 for hide, 0-3 for misclassify)
            attack_mode: 'hide' or 'misclassify'
        
        Returns:
            adv_image: (1, 3, H, W) adversarial image
            info: attack metrics dict
        """
        # Ensure proper shape
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        box = box.to(self.device)
        B, C, H, W = image.shape
        
        # Get original detection
        metric_detector = getattr(self, "eval_detector", self.detector)
        with torch.no_grad():
            orig_dets = metric_detector.detect(image)
            orig_conf = self._get_box_confidence(orig_dets[0], box, H, W)
        
        # Build mask for this object
        obj_mask = self._build_mask_from_box(box, H, W)
        
        # Simple FGSM attack
        delta = torch.zeros_like(image, requires_grad=True)
        
        # Multiple iterations for better results
        lr = 0.01
        for step in range(20):  # Fast: only 20 iterations
            adv = torch.clamp(image + delta, 0, 1)
            
            # Get detections
            dets = self.detector.detect(adv)
            
            # Compute loss
            if attack_mode == 'hide':
                loss = self._compute_hide_loss_simple(dets[0], box, H, W)
            else:
                loss = self._compute_misclass_loss_simple(dets[0], box, tgt_class, H, W)
            
            # Backward
            if delta.grad is not None:
                delta.grad.zero_()
            if loss.requires_grad:
                loss.backward()
            
            # Update
            with torch.no_grad():
                if delta.grad is not None:
                    delta.data = delta.data + lr * delta.grad.sign()
                    delta.data = torch.clamp(delta.data, -0.1, 0.1)
                    delta.data = delta.data * obj_mask
        
        # Final adversarial image
        adv_image = torch.clamp(image + delta, 0, 1)
        
        # Evaluate
        with torch.no_grad():
            final_dets = metric_detector.detect(adv_image)
            adv_conf = self._get_box_confidence(final_dets[0], box, H, W)
            pred_class = self._get_predicted_class(final_dets[0], box, H, W)
            
            if attack_mode == 'hide':
                success = adv_conf < 0.3
            else:
                success = (pred_class == tgt_class)
            
            perturbation = delta.abs().max().item()
        
        info = {
            'original_confidence': orig_conf,
            'adversarial_confidence': adv_conf,
            'predicted_class': pred_class,
            'perturbation_linf': perturbation,
            'success': success,
            'attack_mode': attack_mode,
            'target_class': tgt_class
        }
        
        return adv_image.detach(), info
    
    def _build_mask_from_box(self, box, H, W):
        """Build binary mask from normalized bounding box."""
        x1, y1, x2, y2 = box.tolist()
        x1 = int(x1 * W)
        y1 = int(y1 * H)
        x2 = int(x2 * W)
        y2 = int(y2 * H)
        
        mask = torch.zeros(1, 1, H, W, device=box.device)
        mask[:, :, y1:y2, x1:x2] = 1.0
        return mask
    
    def _get_box_confidence(self, detections, query_box, H, W):
        """Get confidence for detection matching query_box."""
        if len(detections['boxes']) == 0:
            return 0.0
        
        device = query_box.device
        det_boxes = detections['boxes'].detach().clone().to(device)
        det_boxes[:, [0, 2]] /= W
        det_boxes[:, [1, 3]] /= H
        
        ious = self._box_iou(query_box.to(device).unsqueeze(0), det_boxes)
        best_iou, best_idx = ious.max(dim=1)
        
        if best_iou.item() > 0.5:
            scores = detections['scores'].detach().to(device)
            return scores[best_idx.item()].item()
        return 0.0
    
    def _get_predicted_class(self, detections, query_box, H, W):
        """Get predicted class for detection matching query_box."""
        if len(detections['boxes']) == 0:
            return -1
        
        device = query_box.device
        det_boxes = detections['boxes'].detach().clone().to(device)
        det_boxes[:, [0, 2]] /= W
        det_boxes[:, [1, 3]] /= H
        
        ious = self._box_iou(query_box.to(device).unsqueeze(0), det_boxes)
        best_iou, best_idx = ious.max(dim=1)
        
        if best_iou.item() > 0.5:
            labels = detections['labels'].detach().to(device)
            return labels[best_idx.item()].item()
        return -1
    
    def _box_iou(self, boxes1, boxes2):
        """Compute IoU between boxes."""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2[None, :] - inter
        return inter / (union + 1e-6)
    
    def _compute_hide_loss_simple(self, detections, box, H, W):
        """Simple loss to suppress detection."""
        conf = self._get_box_confidence(detections, box, H, W)
        # Maximize confidence (we'll flip sign in gradient update)
        return torch.tensor(conf, device=self.device, requires_grad=False)
    
    def _compute_misclass_loss_simple(self, detections, box, tgt_class, H, W):
        """Simple loss for misclassification."""
        pred_class = self._get_predicted_class(detections, box, H, W)
        if pred_class == tgt_class:
            return torch.tensor(0.0, device=self.device, requires_grad=False)
        else:
            return torch.tensor(1.0, device=self.device, requires_grad=False)


if __name__ == "__main__":
    print("SimpleAttackWrapper - Temporary attack implementation for testing")
    print("\nThis wrapper provides basic FGSM-style attacks for:")
    print("  - Testing the multi-class framework")
    print("  - Profiling attack strategies")
    print("\nFor production use, implement attack_single_object() in your")
    print("attack/joint_attack.py file using the full ThermalMirage pipeline.")
