"""
Fine-tune Ultralytics RT-DETR on FLIR-style thermal detection data.

This script prepares COCO-style FLIR annotations into Ultralytics YOLO format,
trains RT-DETR, validates it on the val split, and copies the best checkpoint to
checkpoints/rtdetr_flir.pt by default.
"""

import argparse
import json
import os
import shutil
from pathlib import Path


FLIR_CLASS_NAMES = ["person", "bicycle", "car", "dog"]


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune RT-DETR on FLIR thermal data")
    parser.add_argument("--data_root", type=str, required=True, help="Path to FLIR_ADAS_1_3")
    parser.add_argument("--prepared_root", type=str, default="prepared_rtdetr_flir")
    parser.add_argument("--model", type=str, default="rtdetr-l.pt", help="Starting RT-DETR checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="runs/rtdetr_flir")
    parser.add_argument("--name", type=str, default="rtdetr_flir")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default="rtdetr_flir.pt")
    parser.add_argument("--link_mode", type=str, default="symlink", choices=["symlink", "hardlink", "copy"])
    parser.add_argument("--include_unannotated_images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--exist_ok", action="store_true")
    return parser.parse_args()


def find_annotation_file(root: Path, split: str) -> Path:
    candidates = [
        root / split / "thermal_annotations",
        root / split / "thermal annotations.json",
        root / split / "thermal_annotations.json",
        root / "annotations" / f"instances_{split}.json",
        root / split / "annotations.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No annotation file found for split={split} under {root}")


def find_image_path(root: Path, split: str, file_name: str) -> Path:
    candidates = [
        root / split / file_name,
        root / split / "thermal_8_bit" / file_name,
        root / split / "thermal_8_bit" / Path(file_name).name,
        root / split / "images" / file_name,
        root / "images" / split / file_name,
    ]
    for path in candidates:
        if path.exists():
            return path

    hits = list((root / split).rglob(Path(file_name).name))
    thermal_hits = [h for h in hits if "thermal_8_bit" in str(h)]
    hits = thermal_hits if thermal_hits else hits
    if hits:
        return hits[0]

    raise FileNotFoundError(f"Image {file_name} not found for split={split}")


def link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def coco_box_to_yolo(bbox, width, height):
    x, y, w, h = bbox
    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height
    nw = w / width
    nh = h / height
    vals = [cx, cy, nw, nh]
    vals = [min(1.0, max(0.0, float(v))) for v in vals]
    return vals


def prepare_split(data_root: Path, prepared_root: Path, split: str, link_mode: str, include_unannotated: bool):
    ann_path = find_annotation_file(data_root, split)
    print(f"[Prepare] {split} annotations: {ann_path}")

    with open(ann_path, "r") as f:
        coco = json.load(f)

    cat_name_by_id = {cat["id"]: cat["name"] for cat in coco["categories"]}
    cat_id_to_internal = {
        cat_id: FLIR_CLASS_NAMES.index(name)
        for cat_id, name in cat_name_by_id.items()
        if name in FLIR_CLASS_NAMES
    }

    anns_by_image = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        if ann["category_id"] not in cat_id_to_internal:
            continue
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    out_img_dir = prepared_root / "images" / split
    out_lbl_dir = prepared_root / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_objects = 0
    n_missing = 0

    for image_info in coco["images"]:
        image_id = image_info["id"]
        anns = anns_by_image.get(image_id, [])
        if not anns and not include_unannotated:
            continue

        file_name = image_info["file_name"]
        try:
            src_img = find_image_path(data_root, split, file_name)
        except FileNotFoundError:
            n_missing += 1
            continue

        dst_name = Path(file_name).name
        dst_img = out_img_dir / dst_name
        link_or_copy(src_img, dst_img, link_mode)

        width = float(image_info.get("width", 0) or 0)
        height = float(image_info.get("height", 0) or 0)
        if width <= 0 or height <= 0:
            from PIL import Image
            with Image.open(src_img) as im:
                width, height = im.size

        label_lines = []
        for ann in anns:
            cls_id = cat_id_to_internal[ann["category_id"]]
            cx, cy, w, h = coco_box_to_yolo(ann["bbox"], width, height)
            if w <= 0 or h <= 0:
                continue
            label_lines.append(f"{cls_id} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}")

        (out_lbl_dir / f"{Path(dst_name).stem}.txt").write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
        n_images += 1
        n_objects += len(label_lines)

    print(f"[Prepare] {split}: images={n_images}, objects={n_objects}, missing_images={n_missing}")
    return n_images, n_objects


def write_data_yaml(prepared_root: Path) -> Path:
    yaml_path = prepared_root / "flir_rtdetr.yaml"
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(FLIR_CLASS_NAMES))
    yaml_path.write_text(
        f"path: {prepared_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(FLIR_CLASS_NAMES)}\n"
        f"names:\n{names}\n"
    )
    print(f"[Prepare] data yaml: {yaml_path}")
    return yaml_path


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    prepared_root = Path(args.prepared_root)
    prepared_root.mkdir(parents=True, exist_ok=True)

    prepare_split(data_root, prepared_root, "train", args.link_mode, args.include_unannotated_images)
    prepare_split(data_root, prepared_root, "val", args.link_mode, args.include_unannotated_images)
    data_yaml = write_data_yaml(prepared_root)

    if args.prepare_only:
        print("[Done] Dataset prepared only.")
        return

    from ultralytics import RTDETR

    model = RTDETR(args.model)
    print(f"[Train] Starting RT-DETR fine-tuning from {args.model}")
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )

    print("[Val] Running validation on FLIR val split")
    model.val(data=str(data_yaml), imgsz=args.imgsz, device=args.device, split="val")

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    source_ckpt = best if best.exists() else last

    if not source_ckpt.exists():
        raise FileNotFoundError(f"No trained checkpoint found under {save_dir / 'weights'}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dst = checkpoint_dir / args.checkpoint_name
    shutil.copy2(source_ckpt, dst)
    print(f"[Done] Best RT-DETR checkpoint copied to: {dst}")


if __name__ == "__main__":
    main()
