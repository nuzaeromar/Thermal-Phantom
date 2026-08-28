"""
thermal_mirage/dataset.py
FLIR thermal dataset loader — supports full images and object crops.

Expected directory layout (FLIR ADAS v2 or the common 3/4-class subset):
    flir_root/
        images_thermal_train/   *.jpg  (or *.png)
        images_thermal_val/
        annotations/
            train.json   (COCO-format)
            val.json
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from config import FLIR_CLASSES, GAN_CROP_SIZE, IMG_SIZE


# ─── Utility ─────────────────────────────────────────────────────────────────

def load_thermal_image(path: str) -> torch.Tensor:
    """Load a thermal PNG/JPG as a single-channel float tensor in [0, 1]."""
    img = Image.open(path).convert("L")           # grayscale
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    return TF.to_tensor(img)                       # (1, H, W)


def expand_box(box, scale: float, img_w: int, img_h: int):
    """Expand a bounding box by `scale` around its centre, clipped to image."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h   = (x2 - x1) * scale, (y2 - y1) * scale
    nx1 = max(0, int(cx - w / 2))
    ny1 = max(0, int(cy - h / 2))
    nx2 = min(img_w, int(cx + w / 2))
    ny2 = min(img_h, int(cy + h / 2))
    return nx1, ny1, nx2, ny2


def build_masks(box, context_scale: float, img_w: int, img_h: int):
    """
    Returns:
        obj_mask     (H, W) bool tensor – object region
        context_mask (H, W) bool tensor – expanded context region
    """
    H, W = img_h, img_w
    x1, y1, x2, y2 = [int(v) for v in box]
    cx1, cy1, cx2, cy2 = expand_box(box, context_scale, W, H)

    obj_mask = torch.zeros(H, W, dtype=torch.bool)
    obj_mask[y1:y2, x1:x2] = True

    ctx_mask = torch.zeros(H, W, dtype=torch.bool)
    ctx_mask[cy1:cy2, cx1:cx2] = True

    return obj_mask, ctx_mask


# ─── Full-scene dataset ───────────────────────────────────────────────────────

class FLIRSceneDataset(Dataset):
    """
    Returns full thermal frames with all annotations.
    Each item: {image, boxes, labels, image_id}
    """

    def __init__(self, root: str, split: str = "train"):
        self.root  = Path(root)
        self.split = split
        ann_path   = self.root / "annotations" / f"{split}.json"

        with open(ann_path) as f:
            coco = json.load(f)

        # Build id → filename map
        self.id2file = {img["id"]: img["file_name"] for img in coco["images"]}
        self.img_ids = list(self.id2file.keys())

        # Build id → list of (box_xywh, category_id)
        self.annotations: dict = {img_id: [] for img_id in self.img_ids}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id in self.annotations:
                self.annotations[img_id].append(
                    (ann["bbox"], ann["category_id"])
                )

        # Map COCO category ids to 0-indexed FLIR class ids
        # (assumes categories are sorted by id in the JSON)
        cat_ids = sorted({c["id"] for c in coco["categories"]})
        self.cat_remap = {cid: i for i, cid in enumerate(cat_ids)}

        img_dir_name = (
            "images_thermal_train" if split == "train"
            else "images_thermal_val"
        )
        self.img_dir = self.root / img_dir_name

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_path = self.img_dir / self.id2file[img_id]

        image = load_thermal_image(str(img_path))   # (1, H, W)
        H, W  = image.shape[1], image.shape[2]

        boxes, labels = [], []
        for (bx, by, bw, bh), cat_id in self.annotations[img_id]:
            # COCO xywh → xyxy, scaled to IMG_SIZE
            sx = IMG_SIZE[0] / W
            sy = IMG_SIZE[1] / H
            x1 = bx * sx
            y1 = by * sy
            x2 = (bx + bw) * sx
            y2 = (by + bh) * sy
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_remap.get(cat_id, 0))

        return {
            "image":    image,
            "boxes":    torch.tensor(boxes,  dtype=torch.float32),
            "labels":   torch.tensor(labels, dtype=torch.long),
            "image_id": img_id,
        }


# ─── Crop dataset for GAN training ───────────────────────────────────────────

class FLIRCropDataset(Dataset):
    """
    Returns square-cropped thermal object patches labelled by class.
    Used to train the class-conditional thermal GAN.
    """

    def __init__(self, root: str, split: str = "train",
                 crop_size: int = GAN_CROP_SIZE):
        self.scene_ds  = FLIRSceneDataset(root, split)
        self.crop_size = crop_size
        self.transform = T.Compose([
            T.Resize((crop_size, crop_size)),
        ])

        # Pre-index all (scene_idx, box_idx) pairs
        self.index = []
        for s_idx in range(len(self.scene_ds)):
            item   = self.scene_ds[s_idx]
            n_objs = item["boxes"].shape[0]
            for b_idx in range(n_objs):
                self.index.append((s_idx, b_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s_idx, b_idx = self.index[idx]
        item  = self.scene_ds[s_idx]
        image = item["image"]          # (1, H, W)
        box   = item["boxes"][b_idx]   # xyxy
        label = item["labels"][b_idx]

        x1, y1, x2, y2 = [max(0, int(v)) for v in box]
        crop = image[:, y1:y2, x1:x2]

        # Resize to square crop
        crop_pil = TF.to_pil_image(crop)
        crop_pil = crop_pil.resize((self.crop_size, self.crop_size), Image.BILINEAR)
        crop_t   = TF.to_tensor(crop_pil)   # (1, crop_size, crop_size)

        # Map to [-1, 1] for GAN
        crop_t = crop_t * 2.0 - 1.0

        return crop_t, label


# ─── DataLoader helpers ───────────────────────────────────────────────────────

def get_scene_loader(root, split="train", batch_size=4, shuffle=True):
    def collate_fn(batch):
        images   = torch.stack([b["image"]    for b in batch])
        image_ids = [b["image_id"] for b in batch]
        boxes    = [b["boxes"]    for b in batch]   # list of (N_i, 4)
        labels   = [b["labels"]   for b in batch]   # list of (N_i,)
        return {"images": images, "boxes": boxes,
                "labels": labels, "image_ids": image_ids}

    ds = FLIRSceneDataset(root, split)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate_fn)


def get_crop_loader(root, split="train", batch_size=64, shuffle=True):
    ds = FLIRCropDataset(root, split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
