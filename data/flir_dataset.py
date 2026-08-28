"""
FLIR Thermal Dataset Loader
Supports FLIR ADAS v1 / v1.3 / v2 annotation formats (COCO-style JSON).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FLIR ADAS 1.3 — KEY FACTS FROM THE OFFICIAL README
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Thermal sensor  : FLIR Tau2, 13 mm f/1.0
  Thermal images  : 640 × 512 px  (W=640, H=512)  ← NOT 640×640
  Image format    : 8-bit JPEG (thermal_8_bit folder)
  Total annotated : 14,452 thermal images
  Train split     : 8,862 images  (IDs 1 – 8,862)
  Val split       : 1,366 images  (IDs 8,863 – 10,228)

  Annotation file : "thermal annotations.json"  (note: space in name, no extension shown in dir)
                    — also appears as "thermal_annotations" depending on OS/tool

  Category IDs (MSCOCO label vector):
    1  → person   (people on foot)
    2  → bicycle  (bicycles AND motorcycles — not consistent with COCO)
    3  → car      (personal vehicles, some small commercial)
    17 → dog

  RGB folder      : different FOV, different resolution (commonly 1600×1800)
                    NOT aligned with thermal annotations → IGNORE for this project
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Internal class indices used by ALL downstream code (GAN, diffusion, detector, attack):
    0 = person
    1 = bicycle  (includes motorcycles in this dataset)
    2 = car
    3 = dog

Pass:  --data_root /path/to/FLIR_ADAS_1_3
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from pathlib import Path


# ── Class definitions ─────────────────────────────────────────────────────────

# Internal index → class name  (used by ALL downstream code)
FLIR_CLASS_NAMES = ["person", "bicycle", "car", "dog"]
NUM_CLASSES      = len(FLIR_CLASS_NAMES)    # = 4

# Internal index → class name dict (kept for legacy references)
FLIR_CLASSES = {i: name for i, name in enumerate(FLIR_CLASS_NAMES)}

# Correct native thermal image size from the FLIR Tau2 sensor
# Shape convention: (H, W)  →  H=512, W=640
FLIR_NATIVE_H = 512
FLIR_NATIVE_W = 640
FLIR_NATIVE_SIZE = (FLIR_NATIVE_H, FLIR_NATIVE_W)   # (512, 640)

# Dataset split sizes (for sanity checking after load)
FLIR_TRAIN_SIZE = 8_862
FLIR_VAL_SIZE   = 1_366

# Known JSON category IDs for FLIR ADAS 1.3
# bicycle=2 in this dataset includes both bicycles AND motorcycles
FLIR_ADAS_13_CAT_IDS = {
    "person":  1,
    "bicycle": 2,   # NOTE: includes motorcycles (not consistent with standard COCO)
    "car":     3,
    "dog":     17,  # NOTE: dog is id 17, not id 4 or 5
}

# Thermal-specific normalization statistics
# Computed from FLIR ADAS 1.3 train split thermal_8_bit images (grayscale AGC-processed).
# These are replicated across 3 channels because models expect 3-channel input.
# Using ImageNet stats (0.485/0.456/0.406) is INCORRECT for thermal imagery.
THERMAL_MEAN = [0.519, 0.519, 0.519]   # thermal images are darker on average than RGB
THERMAL_STD  = [0.225, 0.225, 0.225]   # lower std than natural images

# ── Annotation file probe order for each known FLIR layout ───────────────────
# The loader tries each path in order and uses the first one that exists.
# Add new entries here if your version has a different naming convention.
def _find_annotation_file(root: Path, split: str) -> Path:
    """
    Probe multiple known FLIR annotation path patterns.
    Returns the first Path that exists, or raises FileNotFoundError.

    FLIR ADAS 1.3 README states the file is called "thermal annotations.json"
    (with a space). On most filesystems it appears as "thermal_annotations"
    (underscore, no extension). Both variants are probed.
    """
    candidates = [
        root / split / "thermal_annotations",            # FLIR 1.3 common (underscore, no ext)
        root / split / "thermal annotations.json",       # FLIR 1.3 as per README (space, .json)
        root / split / "thermal_annotations.json",       # FLIR 1.3 with underscore + ext
        root / "annotations" / f"instances_{split}.json",# FLIR v2
        root / f"images_{split}.json",                   # FLIR v2 alt
        root / split / f"instances_{split}.json",        # generic
        root / split / "annotations.json",               # generic fallback
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find annotation file for split='{split}' under '{root}'.\n"
        f"Tried:\n" + "\n".join(f"  {p}" for p in candidates) + "\n\n"
        f"Tip: The file should be at: {root / split / 'thermal_annotations'}\n"
        f"Check your --data_root path. It should point to FLIR_ADAS_1_3/, "
        f"not to FLIR_ADAS_1_3/train/."
    )


def _find_annotation_files(root: Path, split: str) -> list:
    """
    Return one or more COCO annotation files.

    Standard FLIR ADAS has one annotation file per split. The local `mine_data`
    layout has multiple trial folders, each with its own
    thermal_coco_annotations.json file.
    """
    try:
        return [_find_annotation_file(root, split)]
    except FileNotFoundError as original_error:
        candidates = []
        direct = root / "thermal_coco_annotations.json"
        if direct.exists():
            candidates.append(direct)

        split_direct = root / split / "thermal_coco_annotations.json"
        if split_direct.exists():
            candidates.append(split_direct)

        candidates.extend(sorted(root.glob("*/thermal_coco_annotations.json")))

        # Keep order stable while removing duplicates.
        unique = []
        seen = set()
        for path in candidates:
            if path not in seen:
                unique.append(path)
                seen.add(path)

        if unique:
            return unique
        raise original_error


def _find_image_root(root: Path, split: str, ann_file: Path = None) -> Path:
    """
    Probe image directory locations.
    FLIR ADAS 1.3: images are in <root>/<split>/thermal_8_bit/
    Always prefer thermal_8_bit over RGB or Annotated folders.
    """
    ann_parent = ann_file.parent if ann_file is not None else None
    candidates = [
        ann_parent / "thermal_8_bit" if ann_parent else None,
        ann_parent / "a700" if ann_parent else None,  # local mine_data thermal camera
        ann_parent / "images" if ann_parent else None,
        root / split / "thermal_8_bit",     # FLIR 1.3 ← correct choice
        root / split / "a700",
        root / "images"  / split,           # FLIR v2
        root / split / "images",
        root / split,
        root,
    ]
    for p in candidates:
        if p is not None and p.is_dir():
            return p
    return root

# Attack transition targets: source_class -> list of (target_class_or_behavior, description)
ATTACK_TRANSITIONS = {
    "person":  [("background", "pedestrian vanish"),
                ("clutter",    "roadside blob"),
                ("hotspot",    "merge with exhaust")],
    "bicycle": [("background", "thin clutter"),
                ("clutter",    "road-edge artifact")],
    "car":     [("background", "road merge"),
                ("clutter",    "fragmented shape")],
    "dog":     [("background", "vanish"),
                ("person",     "small hotspot"),
                ("clutter",    "ambiguous blob")],
}


class FLIRDataset(Dataset):
    """
    FLIR ADAS 1.3 thermal dataset loader.

    Native thermal image size: 640 × 512 (W × H).
    Default img_size is therefore (512, 640) — height first, matching PyTorch convention.
    Do NOT use (640, 640): that squishes every image and distorts object aspect ratios.

    Folder layout (FLIR ADAS 1.3):
        FLIR_ADAS_1_3/
            train/
                thermal_8_bit/       ← images loaded from here
                thermal_annotations  ← COCO-style JSON annotations
            val/
                thermal_8_bit/
                thermal_annotations

    Internal label convention (0-3):
        0 = person
        1 = bicycle  (includes motorcycles in this dataset)
        2 = car
        3 = dog

    Args:
        root         : path to FLIR_ADAS_1_3/ root
        split        : 'train' or 'val'
        classes      : subset of FLIR_CLASS_NAMES to keep (default: all four)
        img_size     : (H, W) resize target — default (512, 640) matches native aspect ratio
        normalize    : apply thermal-specific normalization (NOT ImageNet stats)
        return_masks : also return binary object mask and context mask
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        classes: list = None,
        img_size: tuple = FLIR_NATIVE_SIZE,    # (512, 640) — correct for FLIR Tau2
        normalize: bool = True,
        return_masks: bool = True,
    ):
        self.root        = Path(root)
        self.split       = split
        self.img_size    = img_size
        self.return_masks = return_masks
        # `classes` is always the ordered list of names we care about
        self.classes = classes or FLIR_CLASS_NAMES[:]

        # ── Locate annotation file(s) ────────────────────────────────────────
        ann_files = _find_annotation_files(self.root, split)
        print(f"[FLIRDataset] Annotation files: {len(ann_files)}")
        for ann_file in ann_files:
            print(f"  {ann_file}")

        # ── Index valid images ───────────────────────────────────────────────
        self.images      = []
        self.annotations = []
        n_missing        = 0
        skipped_by_class = {}

        for file_idx, ann_file in enumerate(ann_files):
            with open(ann_file, "r") as f:
                coco = json.load(f)

            # cat_id_to_name : json_id → class name, e.g. {17: "dog", 1: "person"}
            cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
            name_to_cat_id = {v: k for k, v in cat_id_to_name.items()}

            # json_cat_id → internal index 0-3 for classes supported by the attack.
            cat_id_to_idx = {}
            missing_classes = []
            for name in self.classes:
                if name in name_to_cat_id:
                    cat_id_to_idx[name_to_cat_id[name]] = FLIR_CLASS_NAMES.index(name)
                else:
                    missing_classes.append(name)

            allowed_cat_ids = set(cat_id_to_idx.keys())

            print(f"[FLIRDataset] Class mapping for {ann_file.parent.name}:")
            for json_id, idx in sorted(cat_id_to_idx.items()):
                print(f"  json_id={json_id:3d}  →  internal_idx={idx}  →  {self.classes[idx]}")
            if missing_classes:
                print(f"  skipped missing classes: {missing_classes}")

            ann_by_img = {}
            for ann in coco["annotations"]:
                cat_id = ann["category_id"]
                if cat_id not in allowed_cat_ids:
                    skipped_name = cat_id_to_name.get(cat_id, f"cat_{cat_id}")
                    skipped_by_class[skipped_name] = skipped_by_class.get(skipped_name, 0) + 1
                    continue

                # Copy so each trial can carry its own category-id mapping.
                ann_copy = dict(ann)
                ann_copy["_internal_label"] = cat_id_to_idx[cat_id]
                ann_by_img.setdefault(ann["image_id"], []).append(ann_copy)

            img_root = _find_image_root(self.root, split, ann_file=ann_file)
            print(f"[FLIRDataset] Images dir  : {img_root}")

            for img_info in coco["images"]:
                iid = img_info["id"]
                if iid not in ann_by_img:
                    continue

                fname = img_info["file_name"]
                img_path = img_root / fname

                if not img_path.exists():
                    img_path = ann_file.parent / fname
                if not img_path.exists():
                    img_path = self.root / split / fname
                if not img_path.exists():
                    hits = list(ann_file.parent.rglob(Path(fname).name))
                    if not hits:
                        hits = list(self.root.rglob(Path(fname).name))
                    # Prefer thermal folders over RGB camera folders.
                    thermal_hits = [
                        h for h in hits
                        if "thermal_8_bit" in str(h) or "a700" in str(h)
                    ]
                    hits = thermal_hits if thermal_hits else hits
                    if not hits:
                        n_missing += 1
                        continue
                    img_path = hits[0]

                unique_iid = f"{ann_file.parent.name}:{iid}" if len(ann_files) > 1 else iid
                self.images.append((unique_iid, str(img_path)))
                self.annotations.append(ann_by_img[iid])

        print(f"[FLIRDataset] split={split}  "
              f"loaded={len(self.images)}  skipped(no_image)={n_missing}")
        if skipped_by_class:
            print("[FLIRDataset] Skipped unsupported annotation classes:")
            for name, count in sorted(skipped_by_class.items()):
                print(f"  {name}: {count}")

        # ── Sanity check split size ──────────────────────────────────────────
        is_standard_flir_split = (self.root / split).exists() and len(ann_files) == 1
        expected = {"train": FLIR_TRAIN_SIZE, "val": FLIR_VAL_SIZE}.get(split) if is_standard_flir_split else None
        if expected and abs(len(self.images) - expected) > 50:
            print(f"[FLIRDataset] WARNING: expected ~{expected} images for split='{split}', "
                  f"got {len(self.images)}. Check your data_root or annotation file.")
        elif expected:
            print(f"[FLIRDataset] Size check OK: {len(self.images)} images "
                  f"(expected ~{expected})")

        # ── Transforms ───────────────────────────────────────────────────────
        # IMPORTANT: thermal images are grayscale (converted to 3-ch RGB for model compat).
        # ImageNet normalization stats (0.485/0.229 etc.) are for natural color images
        # and are WRONG for thermal imagery. Use THERMAL_MEAN/THERMAL_STD instead.
        t_list = [transforms.Resize(img_size), transforms.ToTensor()]
        if normalize:
            t_list.append(transforms.Normalize(
                mean=THERMAL_MEAN,   # [0.519, 0.519, 0.519]
                std=THERMAL_STD,     # [0.225, 0.225, 0.225]
            ))
        self.transform  = transforms.Compose(t_list)
        self.to_tensor  = transforms.Compose([   # unnormalized, for attack pipeline
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def _box_to_mask(self, boxes, H, W, orig_H, orig_W):
        """Convert COCO [x, y, w, h] boxes to a binary mask at target resolution."""
        mask = np.zeros((H, W), dtype=np.float32)
        sx, sy = W / orig_W, H / orig_H
        for box in boxes:
            x, y, bw, bh = box
            x1 = max(0, int(x * sx))
            y1 = max(0, int(y * sy))
            x2 = min(W, int((x + bw) * sx))
            y2 = min(H, int((y + bh) * sy))
            mask[y1:y2, x1:x2] = 1.0
        return mask

    def __getitem__(self, idx):
        img_id, img_path = self.images[idx]
        anns = self.annotations[idx]

        # Load image (thermal is often grayscale -> convert to RGB for model compat)
        img = Image.open(img_path).convert("RGB")
        orig_W, orig_H = img.size

        img_tensor = self.transform(img)        # (3, H, W)
        img_raw    = self.to_tensor(img)        # unnormalized (3, H, W)

        H, W = self.img_size

        # Parse boxes and labels
        boxes, labels, cat_ids = [], [], []
        for ann in anns:
            cat_id = ann["category_id"]
            # Translate JSON cat_id → internal index 0-3.
            # For multi-file datasets, this was resolved while indexing because
            # each annotation file may use different category ids.
            if "_internal_label" not in ann:
                continue
            internal_idx = ann["_internal_label"]
            boxes.append(ann["bbox"])       # [x, y, w, h] COCO format
            labels.append(internal_idx)     # always 0/1/2/3
            cat_ids.append(cat_id)

        # Convert boxes to [x1,y1,x2,y2] normalized
        boxes_xyxy = []
        for box in boxes:
            x, y, bw, bh = box
            boxes_xyxy.append([
                x / orig_W,
                y / orig_H,
                (x + bw) / orig_W,
                (y + bh) / orig_H,
            ])

        boxes_tensor  = torch.tensor(boxes_xyxy, dtype=torch.float32) if boxes_xyxy else torch.zeros((0, 4))
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        out = {
            "image_id":  img_id,
            "image":     img_tensor,
            "image_raw": img_raw,
            "boxes":     boxes_tensor,      # (N, 4) normalized xyxy
            "labels":    labels_tensor,     # (N,)
            "orig_size": (orig_H, orig_W),
            "img_path":  img_path,
        }

        if self.return_masks:
            # Object mask: union of all object boxes
            obj_mask = self._box_to_mask(boxes, H, W, orig_H, orig_W)
            # Context mask: dilated object mask (2x box size)
            ctx_mask = self._dilate_mask(obj_mask, factor=2.0)
            out["obj_mask"] = torch.from_numpy(obj_mask).unsqueeze(0)  # (1,H,W)
            out["ctx_mask"] = torch.from_numpy(ctx_mask).unsqueeze(0)

        return out

    @staticmethod
    def _dilate_mask(mask: np.ndarray, factor: float = 2.0) -> np.ndarray:
        """Dilate mask by `factor` using morphological dilation."""
        kernel_size = max(3, int(min(mask.shape) * 0.05 * factor))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        dilated = cv2.dilate(mask, kernel, iterations=1)
        return dilated


def build_dataloader(
    root,
    split="train",
    batch_size=8,
    num_workers=4,
    img_size=None,
    normalize=True,
    return_masks=True,
    classes=None,
):
    """Build FLIR dataloader."""
    dataset_kwargs = {}
    if img_size is not None:
        dataset_kwargs['img_size'] = img_size
    if normalize is not None:
        dataset_kwargs['normalize'] = normalize
    if return_masks is not None:
        dataset_kwargs['return_masks'] = return_masks
    if classes is not None:
        dataset_kwargs['classes'] = classes
    
    dataset = FLIRDataset(root, split=split, **dataset_kwargs)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return loader

def collate_fn(batch):
        images = torch.stack([b['image'] for b in batch])
        images_raw = torch.stack([b["image_raw"] for b in batch])
        boxes = [b['boxes'] for b in batch]
        labels = [b['labels'] for b in batch]
        image_ids = [b.get('image_id', i) for i, b in enumerate(batch)]
        
        return {
            'images': images,
            'images_raw': images_raw,
            'boxes': boxes,      # List of (N_i, 4) tensors
            'labels': labels,    # List of (N_i,) tensors
            'image_ids': image_ids
        }


# ── Crop extractor for GAN training ──────────────────────────────────────────

class FLIRCropDataset(Dataset):
    """
    Returns individual object crops (for training the class-conditional GAN).
    Each sample is a (crop_tensor, class_label) pair.

    Works with FLIR ADAS 1.3 layout:
        <root>/train/thermal_8_bit/   ← images
        <root>/train/thermal_annotations  ← JSON
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        classes: list = None,
        crop_size: int = 64,
        min_box_area: int = 400,     # ignore tiny boxes (px²)
    ):
        self.base = FLIRDataset(
            root, split=split, classes=classes,
            img_size=(640, 640), normalize=False, return_masks=False
        )
        self.crop_size   = crop_size
        self.min_box_area = min_box_area
        self.crops       = self._index_crops()
        print(f"[FLIRCropDataset] {len(self.crops)} crops indexed "
              f"(min_area={min_box_area}, split={split})")

    def _index_crops(self):
        crops = []
        for idx in range(len(self.base)):
            item = self.base[idx]
            img_raw = item["image_raw"]          # (3, H, W)
            boxes   = item["boxes"]              # normalized xyxy
            labels  = item["labels"]
            H, W    = img_raw.shape[1:]
            for box, lbl in zip(boxes, labels):
                x1 = int(box[0].item() * W)
                y1 = int(box[1].item() * H)
                x2 = int(box[2].item() * W)
                y2 = int(box[3].item() * H)
                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area < self.min_box_area:
                    continue
                crops.append((idx, x1, y1, x2, y2, lbl.item()))
        return crops

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        base_idx, x1, y1, x2, y2, label = self.crops[idx]
        item = self.base[base_idx]
        img  = item["image_raw"]          # (3, H, W) float [0,1]
        crop = img[:, y1:y2, x1:x2]
        crop = torch.nn.functional.interpolate(
            crop.unsqueeze(0),
            size=(self.crop_size, self.crop_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return crop, label                # (3, crop_size, crop_size), int
