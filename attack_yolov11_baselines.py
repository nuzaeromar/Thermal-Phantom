"""
Baseline adversarial attacks for a FLIR-trained YOLOv11 detector.

Attacks:
  - PGD, epsilon=16/255 by default
  - BIM
  - C&W-style L2 attack

Metrics:
  - object-level ASR
  - clean/adversarial precision, recall, mAP50, mAP50-95, mean IoU
  - SSIM and optional LPIPS
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


FLIR_CLASS_NAMES = ["person", "bicycle", "car", "dog"]
FLIR_CAT_TO_INTERNAL = {1: 0, 2: 1, 3: 2, 17: 3}


def parse_args():
    parser = argparse.ArgumentParser(description="PGD/BIM/CW attacks on YOLOv11 FLIR")
    parser.add_argument("--attack", type=str, required=True, choices=["pgd", "bim", "cw", "all"])
    parser.add_argument("--data_root", type=str, required=True, help="Path to FLIR_ADAS_1_3")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--yolo_ckpt", type=str, required=True, help="YOLOv11 FLIR checkpoint")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images to attack; omit for all")
    parser.add_argument("--imgsz", type=int, default=640, help="Square resize used for YOLO and metrics")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for metric detections")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU threshold for matching detections to GT")
    parser.add_argument("--epsilon", type=float, default=16 / 255, help="PGD/BIM L_inf budget; 16/255 by default")
    parser.add_argument("--alpha", type=float, default=2 / 255, help="PGD/BIM step size")
    parser.add_argument("--steps", type=int, default=20, help="PGD/BIM/CW optimization steps")
    parser.add_argument("--cw_lr", type=float, default=0.01, help="C&W Adam learning rate")
    parser.add_argument("--cw_c", type=float, default=5.0, help="C&W attack loss weight")
    parser.add_argument("--cw_l2", type=float, default=0.05, help="C&W L2 penalty weight")
    parser.add_argument("--compute_lpips", action="store_true", help="Compute LPIPS if lpips is installed")
    parser.add_argument("--save_images", action="store_true", help="Save adversarial images")
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save every Nth adversarial image when --save_images is used; 0 disables image saving",
    )
    parser.add_argument("--out_dir", type=str, default="results/yolo11_baseline_attacks")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def find_flir_annotation(root: Path, split: str) -> Path:
    candidates = [
        root / split / "thermal_annotations.json",
        root / split / "thermal_annotations",
        root / split / "thermal annotations.json",
        root / "annotations" / f"instances_{split}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No FLIR annotation file found for split={split} under {root}")


class FLIRDetectionDataset:
    def __init__(self, data_root: str, split: str, imgsz: int):
        self.root = Path(data_root)
        self.split = split
        self.imgsz = imgsz
        self.image_dir = self.root / split / "thermal_8_bit"
        self.ann_path = find_flir_annotation(self.root, split)

        with open(self.ann_path, "r") as f:
            coco = json.load(f)

        self.images = []
        self.anns_by_image = {}
        for image_info in coco["images"]:
            img_id = image_info["id"]
            file_name = image_info["file_name"]
            path = self._resolve_image_path(file_name)
            if path.exists():
                self.images.append({
                    "id": img_id,
                    "file_name": file_name,
                    "path": path,
                    "width": image_info.get("width"),
                    "height": image_info.get("height"),
                })
                self.anns_by_image[img_id] = []

        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            if img_id in self.anns_by_image and cat_id in FLIR_CAT_TO_INTERNAL:
                self.anns_by_image[img_id].append(ann)

        self.to_tensor = transforms.ToTensor()
        print(f"[FLIR] annotations: {self.ann_path}")
        print(f"[FLIR] images: {self.image_dir}")
        print(f"[FLIR] loaded images with files: {len(self.images)}")

    def _resolve_image_path(self, file_name: str) -> Path:
        candidates = [
            self.root / self.split / file_name,
            self.image_dir / file_name,
            self.image_dir / Path(file_name).name,
        ]
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    def __len__(self):
        return len(self.images)

    def __iter__(self) -> Iterable[Dict]:
        for item in self.images:
            image = Image.open(item["path"]).convert("RGB")
            orig_w, orig_h = image.size
            resized = image.resize((self.imgsz, self.imgsz), Image.BILINEAR)
            tensor = self.to_tensor(resized).unsqueeze(0)

            boxes = []
            labels = []
            for ann in self.anns_by_image[item["id"]]:
                x, y, w, h = ann["bbox"]
                boxes.append([
                    x * self.imgsz / orig_w,
                    y * self.imgsz / orig_h,
                    (x + w) * self.imgsz / orig_w,
                    (y + h) * self.imgsz / orig_h,
                ])
                labels.append(FLIR_CAT_TO_INTERNAL[ann["category_id"]])

            yield {
                "image_id": item["id"],
                "file_name": item["file_name"],
                "image": tensor,
                "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
                "labels": torch.tensor(labels, dtype=torch.long),
            }


class YOLOv11Target:
    def __init__(self, ckpt: str, device: str, conf: float):
        if device == "cuda" and not torch.cuda.is_available():
            print("[YOLOv11] CUDA unavailable, falling back to CPU")
            device = "cpu"
        self.device = torch.device(device)
        self.conf = conf

        from ultralytics import YOLO
        self.eval_yolo = YOLO(ckpt)
        self.eval_yolo.model.to(self.device).eval()
        self.attack_yolo = YOLO(ckpt)
        self.attack_yolo.model.to(self.device).eval()
        print(f"[YOLOv11] loaded target: {ckpt}")

    @torch.no_grad()
    def detect(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        results = self.eval_yolo.predict(
            source=image.to(self.device),
            conf=self.conf,
            verbose=False,
            device=str(self.device),
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return empty_prediction()
        boxes = results[0].boxes
        return {
            "boxes": boxes.xyxy.detach().float().cpu(),
            "scores": boxes.conf.detach().float().cpu(),
            "labels": boxes.cls.detach().long().cpu(),
        }

    def suppress_loss(self, image: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor, topk: int = 20) -> torch.Tensor:
        """Mean target-class confidence near GT boxes. Minimize this to hide objects."""
        raw = self.attack_yolo.model(image.to(self.device))
        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        if raw.dim() != 3:
            return image.sum() * 0.0

        if raw.shape[1] <= raw.shape[2]:
            pred_boxes = raw[:, :4, :].permute(0, 2, 1)
            cls_raw = raw[:, 4:, :].permute(0, 2, 1)
        else:
            pred_boxes = raw[:, :, :4]
            cls_raw = raw[:, :, 4:]

        cls_scores = cls_raw.sigmoid() if cls_raw.max().item() > 1.0 or cls_raw.min().item() < 0.0 else cls_raw
        centers = pred_boxes[0, :, :2]
        if centers.max().item() <= 1.5:
            centers = centers * image.shape[-1]

        losses = []
        for box, label in zip(boxes.to(centers.device), labels.to(centers.device)):
            label = int(label.item())
            if cls_scores.shape[-1] <= label:
                continue
            x1, y1, x2, y2 = box
            inside = (
                (centers[:, 0] >= x1) & (centers[:, 0] <= x2) &
                (centers[:, 1] >= y1) & (centers[:, 1] <= y2)
            )
            selected = cls_scores[0, inside, label]
            if selected.numel() == 0:
                selected = cls_scores[0, :, label]
            k = min(topk, selected.numel())
            losses.append(selected.topk(k).values.mean())

        if not losses:
            return image.sum() * 0.0
        return torch.stack(losses).mean()


def empty_prediction():
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "scores": torch.zeros((0,), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.long),
    }


def pgd_attack(model: YOLOv11Target, image: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor, args):
    clean = image.detach().to(model.device)
    adv = (clean + torch.empty_like(clean).uniform_(-args.epsilon, args.epsilon)).clamp(0, 1)

    for _ in range(args.steps):
        adv = adv.detach().requires_grad_(True)
        loss = model.suppress_loss(adv, boxes, labels)
        grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
        adv = adv - args.alpha * grad.sign()
        adv = torch.max(torch.min(adv, clean + args.epsilon), clean - args.epsilon)
        adv = adv.clamp(0, 1)

    return adv.detach()


def bim_attack(model: YOLOv11Target, image: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor, args):
    clean = image.detach().to(model.device)
    adv = clean.clone()

    for _ in range(args.steps):
        adv = adv.detach().requires_grad_(True)
        loss = model.suppress_loss(adv, boxes, labels)
        grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
        adv = adv - args.alpha * grad.sign()
        adv = torch.max(torch.min(adv, clean + args.epsilon), clean - args.epsilon)
        adv = adv.clamp(0, 1)

    return adv.detach()


def cw_attack(model: YOLOv11Target, image: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor, args):
    clean = image.detach().to(model.device).clamp(0, 1)
    clean_tanh = torch.atanh((clean * 2 - 1).clamp(-0.999999, 0.999999))
    w = clean_tanh.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=args.cw_lr)

    for _ in range(args.steps):
        adv = (torch.tanh(w) + 1) / 2
        attack_loss = model.suppress_loss(adv, boxes, labels)
        l2 = (adv - clean).pow(2).mean()
        loss = args.cw_c * attack_loss + args.cw_l2 * l2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return ((torch.tanh(w.detach()) + 1) / 2).clamp(0, 1)


def run_attack(model, image, boxes, labels, attack_name, args):
    if attack_name == "pgd":
        return pgd_attack(model, image, boxes, labels, args)
    if attack_name == "bim":
        return bim_attack(model, image, boxes, labels, args)
    if attack_name == "cw":
        return cw_attack(model, image, boxes, labels, args)
    raise ValueError(f"Unknown attack: {attack_name}")


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0)
    lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-8)


def matched_detection(pred: Dict[str, torch.Tensor], box: torch.Tensor, label: int, iou_threshold: float) -> Tuple[bool, float, float]:
    if len(pred["boxes"]) == 0:
        return False, 0.0, 0.0
    same_class = pred["labels"] == label
    if same_class.sum().item() == 0:
        return False, 0.0, 0.0
    boxes = pred["boxes"][same_class]
    scores = pred["scores"][same_class]
    ious = box_iou(box.unsqueeze(0), boxes).squeeze(0)
    best_iou, best_idx = ious.max(dim=0)
    if best_iou.item() < iou_threshold:
        return False, float(scores[best_idx].item()), float(best_iou.item())
    return True, float(scores[best_idx].item()), float(best_iou.item())


def precision_recall_mean_iou(pred: Dict[str, torch.Tensor], gt: Dict[str, torch.Tensor], iou_threshold: float, conf: float):
    keep = pred["scores"] >= conf
    pred_boxes = pred["boxes"][keep]
    pred_scores = pred["scores"][keep]
    pred_labels = pred["labels"][keep]
    gt_boxes = gt["boxes"]
    gt_labels = gt["labels"]

    matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)
    tp = 0
    matched_ious = []
    if len(pred_boxes) > 0:
        order = pred_scores.argsort(descending=True)
        for box, label in zip(pred_boxes[order], pred_labels[order]):
            candidates = (gt_labels == label) & (~matched_gt)
            if candidates.sum().item() == 0:
                continue
            candidate_idx = torch.where(candidates)[0]
            ious = box_iou(box.unsqueeze(0), gt_boxes[candidate_idx]).squeeze(0)
            best_iou, best_local = ious.max(dim=0)
            if best_iou.item() >= iou_threshold:
                matched_gt[candidate_idx[best_local]] = True
                tp += 1
                matched_ious.append(float(best_iou.item()))

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return {
        "precision": float(tp / (tp + fp + 1e-8)),
        "recall": float(tp / (tp + fn + 1e-8)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "iou_sum": float(sum(matched_ious)),
        "iou_count": int(len(matched_ious)),
    }


def compute_ap(precisions, recalls):
    if len(precisions) == 0:
        return 0.0
    precisions = np.asarray(precisions)
    recalls = np.asarray(recalls)
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_map_at_iou(preds, gts, iou_threshold, num_classes=4):
    aps = {}
    for cls in range(num_classes):
        cls_preds = []
        n_gt = 0
        for image_idx, (pred, gt) in enumerate(zip(preds, gts)):
            n_gt += int((gt["labels"] == cls).sum().item())
            mask = pred["labels"] == cls
            for box, score in zip(pred["boxes"][mask], pred["scores"][mask]):
                cls_preds.append((image_idx, float(score.item()), box))
        if n_gt == 0:
            aps[cls] = 0.0
            continue
        cls_preds.sort(key=lambda item: item[1], reverse=True)
        matched = {
            i: torch.zeros(int((gt["labels"] == cls).sum().item()), dtype=torch.bool)
            for i, gt in enumerate(gts)
        }
        tps, fps = [], []
        for image_idx, _, box in cls_preds:
            gt_boxes = gts[image_idx]["boxes"][gts[image_idx]["labels"] == cls]
            if len(gt_boxes) == 0:
                tps.append(0)
                fps.append(1)
                continue
            ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
            best_iou, best_idx = ious.max(dim=0)
            if best_iou.item() >= iou_threshold and not matched[image_idx][best_idx]:
                matched[image_idx][best_idx] = True
                tps.append(1)
                fps.append(0)
            else:
                tps.append(0)
                fps.append(1)
        if not tps:
            aps[cls] = 0.0
            continue
        cum_tp = np.cumsum(np.asarray(tps))
        cum_fp = np.cumsum(np.asarray(fps))
        recalls = cum_tp / (n_gt + 1e-8)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-8)
        aps[cls] = compute_ap(precisions, recalls)
    return {"mAP": float(np.mean(list(aps.values()))), "per_class_ap": aps}


def summarize_detector_metrics(clean_preds, adv_preds, gts, iou_threshold, conf):
    def aggregate(rows):
        tp = sum(r["tp"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        iou_sum = sum(r["iou_sum"] for r in rows)
        iou_count = sum(r["iou_count"] for r in rows)
        return {
            "precision": float(tp / (tp + fp + 1e-8)),
            "recall": float(tp / (tp + fn + 1e-8)),
            "mean_iou": float(iou_sum / (iou_count + 1e-8)),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }

    clean_rows = [precision_recall_mean_iou(p, g, iou_threshold, conf) for p, g in zip(clean_preds, gts)]
    adv_rows = [precision_recall_mean_iou(p, g, iou_threshold, conf) for p, g in zip(adv_preds, gts)]
    clean = aggregate(clean_rows)
    adv = aggregate(adv_rows)

    thresholds = [round(0.50 + i * 0.05, 2) for i in range(10)]
    clean_map50 = compute_map_at_iou(clean_preds, gts, 0.50)
    adv_map50 = compute_map_at_iou(adv_preds, gts, 0.50)
    clean["mAP50"] = clean_map50["mAP"]
    adv["mAP50"] = adv_map50["mAP"]
    clean["mAP50_95"] = float(np.mean([compute_map_at_iou(clean_preds, gts, t)["mAP"] for t in thresholds]))
    adv["mAP50_95"] = float(np.mean([compute_map_at_iou(adv_preds, gts, t)["mAP"] for t in thresholds]))

    return {
        "clean": clean,
        "adversarial": adv,
        "drop": {
            "precision": clean["precision"] - adv["precision"],
            "recall": clean["recall"] - adv["recall"],
            "mAP50": clean["mAP50"] - adv["mAP50"],
            "mAP50_95": clean["mAP50_95"] - adv["mAP50_95"],
            "mean_iou": clean["mean_iou"] - adv["mean_iou"],
        },
    }


def image_ssim(clean, adv, window_size=11):
    clean = clean.detach().float().clamp(0, 1)
    adv = adv.detach().float().clamp(0, 1)
    channels = clean.shape[1]
    window = torch.ones((channels, 1, window_size, window_size), dtype=clean.dtype, device=clean.device)
    window = window / float(window_size * window_size)
    pad = window_size // 2
    mu_x = F.conv2d(clean, window, padding=pad, groups=channels)
    mu_y = F.conv2d(adv, window, padding=pad, groups=channels)
    sigma_x = F.conv2d(clean * clean, window, padding=pad, groups=channels) - mu_x.pow(2)
    sigma_y = F.conv2d(adv * adv, window, padding=pad, groups=channels) - mu_y.pow(2)
    sigma_xy = F.conv2d(clean * adv, window, padding=pad, groups=channels) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return float(ssim.mean().item())


def spatial_variation(image: torch.Tensor) -> torch.Tensor:
    """Full-frame anisotropic L1 total variation used by run_multiclass.py."""
    image = image.detach().float().clamp(0, 1)
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    return dy.abs().mean() + dx.abs().mean()


def soft_histogram(image: torch.Tensor, n_bins: int = 32) -> torch.Tensor:
    """Differentiable-style triangular grayscale histogram over the full image."""
    values = image.detach().float().clamp(0, 1).mean(dim=1).flatten()
    bins = torch.linspace(0, 1, n_bins + 1, device=values.device, dtype=values.dtype)
    centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    distances = (values.unsqueeze(-1) - centers.unsqueeze(0)).abs()
    histogram = F.relu(1 - distances / bin_width).sum(dim=0)
    eps = 1e-8
    return (histogram + eps) / (histogram + eps).sum()


def histogram_kl_divergence(clean: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    """Compute D_KL(h_clean || h_adv) using full-frame 32-bin histograms."""
    clean_hist = soft_histogram(clean, n_bins=32)
    adv_hist = soft_histogram(adv, n_bins=32)
    return (clean_hist * (clean_hist.log() - adv_hist.log())).sum()


def union_object_mask(boxes: torch.Tensor, height: int, width: int,
                      device, dtype) -> torch.Tensor:
    """Create the union of all attacked-object bounding-box masks."""
    mask = torch.zeros((1, 1, height, width), device=device, dtype=dtype)
    for box in boxes.detach().float().cpu():
        x1, y1, x2, y2 = box.tolist()
        x1 = max(0, min(width, int(x1)))
        y1 = max(0, min(height, int(y1)))
        x2 = max(0, min(width, int(x2)))
        y2 = max(0, min(height, int(y2)))
        if x2 > x1 and y2 > y1:
            mask[:, :, y1:y2, x1:x2] = 1
    return mask


def object_boundary_deviation(clean: torch.Tensor, adv: torch.Tensor,
                              boxes: torch.Tensor) -> torch.Tensor:
    """Mean absolute clean-to-adversarial change in the union boundary band."""
    mask = union_object_mask(
        boxes, adv.shape[-2], adv.shape[-1], adv.device, adv.dtype
    )
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = 1 - F.max_pool2d(1 - mask, kernel_size=3, stride=1, padding=1)
    boundary = (dilated - eroded).clamp(0, 1)
    boundary_channels = boundary.expand_as(adv)
    absolute_change = (adv - clean).abs() * boundary_channels
    return absolute_change.sum() / (boundary_channels.sum() + 1e-8)


def thermal_evaluation_metrics(clean: torch.Tensor, adv: torch.Tensor,
                               boxes: torch.Tensor) -> Dict[str, float]:
    """Evaluate the final adversarial frame against its corresponding clean frame."""
    with torch.no_grad():
        clean_sv = spatial_variation(clean)
        adv_sv = spatial_variation(adv)
        histogram_kl = histogram_kl_divergence(clean, adv)
        boundary_deviation = object_boundary_deviation(clean, adv, boxes)
    return {
        "clean_spatial_variation": float(clean_sv.item()),
        "adversarial_spatial_variation": float(adv_sv.item()),
        "spatial_variation_change": float((adv_sv - clean_sv).item()),
        "histogram_kl_divergence": float(histogram_kl.item()),
        "object_boundary_deviation": float(boundary_deviation.item()),
    }


def summarize_thermal_metrics(rows):
    if not rows:
        return None
    summary = {
        "num_images": len(rows),
        "scope": {
            "spatial_variation": "full_image_clean_and_final_adversarial",
            "histogram": "full_image_32_bin_D_KL(clean||adversarial)",
            "boundary": "union_of_ground_truth_object_box_boundaries",
        },
    }
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


class ImageQuality:
    def __init__(self, device: str, compute_lpips: bool):
        self.device = device
        self.lpips_model = None
        if compute_lpips:
            try:
                import lpips
                self.lpips_model = lpips.LPIPS(net="alex").to(device).eval()
                print("[Metrics] LPIPS enabled")
            except Exception as exc:
                print(f"[Warning] LPIPS disabled: {exc}")
                print("[Warning] Install with: pip install lpips")

    def compute(self, clean, adv):
        clean = clean.to(self.device).clamp(0, 1)
        adv = adv.to(self.device).clamp(0, 1)
        out = {"ssim": image_ssim(clean, adv), "lpips": None}
        if self.lpips_model is not None:
            with torch.no_grad():
                out["lpips"] = float(self.lpips_model(clean * 2 - 1, adv * 2 - 1).mean().item())
        return out

    @staticmethod
    def summarize(rows):
        return {
            "ssim": float(np.mean([r["ssim"] for r in rows])) if rows else None,
            "lpips": float(np.mean([r["lpips"] for r in rows if r["lpips"] is not None]))
            if any(r["lpips"] is not None for r in rows) else None,
        }


def save_tensor_image(tensor: torch.Tensor, path: Path):
    arr = tensor.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
    Image.fromarray((arr * 255).astype(np.uint8)).save(path, quality=95)


def fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


def run_one_attack(attack_name: str, args, dataset, detector, quality):
    out_dir = Path(args.out_dir) / attack_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_images_enabled = bool(args.save_images and args.save_every > 0)
    save_warning_printed = False
    if save_images_enabled:
        (out_dir / "adv_imgs").mkdir(exist_ok=True)

    clean_preds, adv_preds, gts = [], [], []
    quality_rows = []
    thermal_rows = []
    per_image_results = []
    total_clean_detected_objects = 0
    total_adv_matched_clean_detected_objects = 0
    total_successful_objects = 0
    total_gt_objects = 0
    total_clean_detections = 0
    total_adv_detections = 0
    images_with_any_success = 0

    processed = 0
    iterator = tqdm(dataset, total=len(dataset), desc=f"{attack_name.upper()} attack")

    for sample in iterator:
        if args.num_images is not None and processed >= args.num_images:
            break
        if len(sample["boxes"]) == 0:
            continue

        clean = sample["image"].to(detector.device)
        boxes = sample["boxes"]
        labels = sample["labels"]
        gt = {"boxes": boxes.cpu(), "labels": labels.cpu()}

        clean_pred = detector.detect(clean)
        adv = run_attack(detector, clean, boxes, labels, attack_name, args)
        adv_pred = detector.detect(adv)

        object_results = []
        clean_detected = 0
        adv_matched_clean_detected = 0
        successful = 0
        for obj_idx, (box, label) in enumerate(zip(boxes, labels)):
            clean_match, clean_score, clean_iou = matched_detection(clean_pred, box, int(label.item()), args.iou)
            adv_match, adv_score, adv_iou = matched_detection(adv_pred, box, int(label.item()), args.iou)
            success = bool(clean_match and not adv_match)
            clean_detected += int(clean_match)
            adv_matched_clean_detected += int(clean_match and adv_match)
            successful += int(success)
            object_results.append({
                "box_idx": int(obj_idx),
                "class": FLIR_CLASS_NAMES[int(label.item())],
                "clean_detected": bool(clean_match),
                "adversarial_detected": bool(adv_match),
                "success": success,
                "clean_score": float(clean_score),
                "adversarial_score": float(adv_score),
                "clean_iou": float(clean_iou),
                "adversarial_iou": float(adv_iou),
            })

        clean_preds.append(clean_pred)
        adv_preds.append(adv_pred)
        gts.append(gt)
        image_quality = quality.compute(clean, adv)
        quality_rows.append(image_quality)
        thermal_metrics = thermal_evaluation_metrics(clean, adv, boxes)
        thermal_rows.append(thermal_metrics)

        total_clean_detected_objects += clean_detected
        total_adv_matched_clean_detected_objects += adv_matched_clean_detected
        total_successful_objects += successful
        total_gt_objects += len(boxes)
        total_clean_detections += int(len(clean_pred["boxes"]))
        total_adv_detections += int(len(adv_pred["boxes"]))
        images_with_any_success += int(successful > 0)

        per_image_results.append({
            "image_id": sample["image_id"],
            "file_name": sample["file_name"],
            "num_gt_objects": int(len(boxes)),
            "num_clean_detections": int(len(clean_pred["boxes"])),
            "num_adversarial_detections": int(len(adv_pred["boxes"])),
            "num_clean_detected_objects": int(clean_detected),
            "num_adversarial_matched_clean_detected_objects": int(adv_matched_clean_detected),
            "num_successful_objects": int(successful),
            "object_level_asr_clean_detected_denominator": float(successful / (clean_detected + 1e-8)),
            "adversarial_recall_on_clean_detected_objects": float(
                adv_matched_clean_detected / (clean_detected + 1e-8)
            ),
            "object_level_asr_all_gt_denominator": float(successful / (len(boxes) + 1e-8)),
            "image_attack_success_any_object": bool(successful > 0),
            "objects": object_results,
            "image_quality": image_quality,
            "thermal_evaluation": thermal_metrics,
        })

        if save_images_enabled and processed % args.save_every == 0:
            stem = Path(sample["file_name"]).stem
            try:
                save_tensor_image(adv, out_dir / "adv_imgs" / f"{stem}_{attack_name}.png")
            except OSError as exc:
                save_images_enabled = False
                if not save_warning_printed:
                    print(
                        f"\n[Warning] Could not save adversarial images ({exc}). "
                        f"Disabling image saving for the rest of {attack_name.upper()}."
                    )
                    save_warning_printed = True

        processed += 1

    detector_summary = summarize_detector_metrics(clean_preds, adv_preds, gts, args.iou, args.conf) if processed else {}
    image_quality_summary = quality.summarize(quality_rows)
    thermal_summary = summarize_thermal_metrics(thermal_rows)
    object_asr_clean_detected = float(total_successful_objects / (total_clean_detected_objects + 1e-8))
    adv_recall_on_clean_detected = float(
        total_adv_matched_clean_detected_objects / (total_clean_detected_objects + 1e-8)
    )
    object_asr_all_gt = float(total_successful_objects / (total_gt_objects + 1e-8))
    image_asr_any = float(images_with_any_success / (processed + 1e-8))

    print("\n" + "=" * 70)
    print(f"YOLOv11 {attack_name.upper()} ATTACK SUMMARY")
    print("=" * 70)
    print(f"Images processed: {processed}")
    print(f"Object-level ASR (clean-detected denominator): {object_asr_clean_detected:.1%} "
          f"({total_successful_objects}/{total_clean_detected_objects})")
    print(f"Adversarial recall on clean-detected objects: {adv_recall_on_clean_detected:.1%} "
          f"({total_adv_matched_clean_detected_objects}/{total_clean_detected_objects})")
    print(f"Object-level ASR (all GT denominator): {object_asr_all_gt:.1%} "
          f"({total_successful_objects}/{total_gt_objects})")
    print(f"Image-level ASR (any object succeeds): {image_asr_any:.1%} "
          f"({images_with_any_success}/{processed})")
    print(f"Raw detections: clean={total_clean_detections} | adversarial={total_adv_detections}")

    if detector_summary:
        clean_s = detector_summary["clean"]
        adv_s = detector_summary["adversarial"]
        print("\nClean vs Adversarial Detection Metrics:")
        print(f"  Precision: clean={clean_s['precision']:.3f} | adv={adv_s['precision']:.3f}")
        print(f"  Recall:    clean={clean_s['recall']:.3f} | adv={adv_s['recall']:.3f}")
        print(f"  mAP50:     clean={clean_s['mAP50']:.3f} | adv={adv_s['mAP50']:.3f}")
        print(f"  mAP50-95:  clean={clean_s['mAP50_95']:.3f} | adv={adv_s['mAP50_95']:.3f}")
        print(f"  Mean IoU:  clean={clean_s['mean_iou']:.3f} | adv={adv_s['mean_iou']:.3f}")

    print("\nClean vs Adversarial Image Quality:")
    print(f"  SSIM:  {fmt(image_quality_summary['ssim'])}")
    print(f"  LPIPS: {fmt(image_quality_summary['lpips'])}")

    if thermal_summary is not None:
        clean_sv = thermal_summary["clean_spatial_variation"]
        adv_sv = thermal_summary["adversarial_spatial_variation"]
        delta_sv = thermal_summary["spatial_variation_change"]
        hist_kl = thermal_summary["histogram_kl_divergence"]
        boundary = thermal_summary["object_boundary_deviation"]
        print("\nFinal-frame Thermal Evaluation Metrics:")
        print(
            "  Clean spatial variation:      "
            f"mean={clean_sv['mean']:.6f} | std={clean_sv['std']:.6f} | "
            f"min={clean_sv['min']:.6f} | max={clean_sv['max']:.6f}"
        )
        print(
            "  Adversarial spatial variation: "
            f"mean={adv_sv['mean']:.6f} | std={adv_sv['std']:.6f} | "
            f"min={adv_sv['min']:.6f} | max={adv_sv['max']:.6f}"
        )
        print(
            "  Spatial variation change (adv-clean): "
            f"mean={delta_sv['mean']:.6f} | std={delta_sv['std']:.6f} | "
            f"min={delta_sv['min']:.6f} | max={delta_sv['max']:.6f}"
        )
        print(
            "  Histogram KL D_KL(clean||adv): "
            f"mean={hist_kl['mean']:.6f} | std={hist_kl['std']:.6f} | "
            f"min={hist_kl['min']:.6f} | max={hist_kl['max']:.6f}"
        )
        print(
            "  Object boundary deviation:    "
            f"mean={boundary['mean']:.6f} | std={boundary['std']:.6f} | "
            f"min={boundary['min']:.6f} | max={boundary['max']:.6f}"
        )

    output = {
        "summary": {
            "attack": attack_name,
            "num_images": processed,
            "target_model": f"yolov11:{args.yolo_ckpt}",
            "epsilon": args.epsilon if attack_name in {"pgd", "bim"} else None,
            "object_level_asr_clean_detected_denominator": object_asr_clean_detected,
            "object_level_asr_all_gt_denominator": object_asr_all_gt,
            "image_level_asr_any_object": image_asr_any,
            "num_successful_objects": total_successful_objects,
            "num_clean_detected_objects": total_clean_detected_objects,
            "num_adversarial_matched_clean_detected_objects": total_adv_matched_clean_detected_objects,
            "num_gt_objects": total_gt_objects,
            "adversarial_recall_on_clean_detected_objects": adv_recall_on_clean_detected,
            "num_clean_detections": total_clean_detections,
            "num_adversarial_detections": total_adv_detections,
            "detector_quality": detector_summary,
            "image_quality": image_quality_summary,
            "thermal_evaluation": thermal_summary,
        },
        "per_image_results": per_image_results,
    }
    with open(out_dir / "attack_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_dir / 'attack_results.json'}")
    return output["summary"]


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    dataset = FLIRDetectionDataset(args.data_root, args.split, args.imgsz)
    detector = YOLOv11Target(args.yolo_ckpt, args.device, args.conf)
    quality = ImageQuality(str(detector.device), args.compute_lpips)

    attacks = ["pgd", "bim", "cw"] if args.attack == "all" else [args.attack]
    start = time.time()
    summaries = {}
    for attack_name in attacks:
        summaries[attack_name] = run_one_attack(attack_name, args, dataset, detector, quality)

    if len(summaries) > 1:
        with open(Path(args.out_dir) / "all_attack_summaries.json", "w") as f:
            json.dump(summaries, f, indent=2)
    print(f"\nTotal runtime: {(time.time() - start) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
