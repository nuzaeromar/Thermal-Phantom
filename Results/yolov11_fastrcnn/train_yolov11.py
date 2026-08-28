"""
Fine-tune YOLOv11 on the FLIR ADAS thermal dataset.

This script mirrors fasterR-CNN.py at a high level, but Ultralytics YOLO
expects a YOLO-format dataset:

    prepared/
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt
      labels/val/*.txt
      flir_yolo.yaml

The script converts FLIR COCO-style thermal annotations into YOLO labels,
links images into the expected layout, trains YOLOv11, and saves the best
checkpoint to checkpoints/yolo11_flir.pt.

Usage:
    python3 train_yolov11.py \
      --data_root /home/exouser/Downloads/thermal_mirage/FLIR_ADAS_1_3 \
      --model yolo11n.pt \
      --epochs 50 \
      --batch 16 \
      --device cuda
"""

import argparse
import json
import shutil
from pathlib import Path

# Keep the same internal class convention as the rest of the project:
# 0=person, 1=bicycle, 2=car, 3=dog.
FLIR_CLASS_NAMES = ["person", "bicycle", "car", "dog"]
NUM_CLASSES = len(FLIR_CLASS_NAMES)
FLIR_NAME_TO_YOLO_ID = {name: idx for idx, name in enumerate(FLIR_CLASS_NAMES)}


def find_annotation_file(root: Path, split: str) -> Path:
    candidates = [
        root / split / "thermal_annotations.json",
        root / split / "thermal_annotations",
        root / split / "thermal annotations.json",
        root / "annotations" / f"instances_{split}.json",
        root / split / "annotations.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find annotations for split='{split}' under {root}"
    )


def find_image_root(root: Path, split: str) -> Path:
    candidates = [
        root / split / "thermal_8_bit",
        root / split / "images",
        root / "images" / split,
        root / split,
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"Could not find image directory for split='{split}' under {root}"
    )


def link_or_copy_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            shutil.copy2(src, dst)
            return

    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def coco_bbox_to_yolo_line(bbox, img_w: float, img_h: float, class_id: int) -> str:
    x, y, w, h = bbox

    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(img_w), x + max(0.0, w))
    y2 = min(float(img_h), y + max(0.0, h))

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 1.0 or bh <= 1.0:
        return ""

    xc = (x1 + bw / 2.0) / img_w
    yc = (y1 + bh / 2.0) / img_h
    nw = bw / img_w
    nh = bh / img_h

    return f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"


def prepare_split(
    data_root: Path,
    prepared_root: Path,
    split: str,
    link_mode: str,
    include_unannotated_images: bool,
) -> int:
    ann_path = find_annotation_file(data_root, split)
    image_root = find_image_root(data_root, split)

    with open(ann_path, "r") as f:
        coco = json.load(f)

    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        name = cat_id_to_name.get(ann["category_id"])
        if name not in FLIR_NAME_TO_YOLO_ID:
            continue
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    out_img_dir = prepared_root / "images" / split
    out_lbl_dir = prepared_root / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_boxes = 0
    n_missing = 0

    seen_files = set()

    for img in coco["images"]:
        img_id = img["id"]
        file_name = Path(img["file_name"]).name
        seen_files.add(file_name)
        src = image_root / file_name
        if not src.exists():
            hits = list(data_root.rglob(file_name))
            thermal_hits = [p for p in hits if "thermal_8_bit" in str(p)]
            hits = thermal_hits or hits
            if not hits:
                n_missing += 1
                continue
            src = hits[0]

        dst = out_img_dir / file_name
        link_or_copy_image(src, dst, link_mode)

        img_w = float(img.get("width", 0))
        img_h = float(img.get("height", 0))
        if img_w <= 0 or img_h <= 0:
            raise ValueError(f"Image {file_name} has invalid size in annotations")

        label_lines = []
        for ann in anns_by_image.get(img_id, []):
            class_name = cat_id_to_name[ann["category_id"]]
            class_id = FLIR_NAME_TO_YOLO_ID[class_name]
            line = coco_bbox_to_yolo_line(ann["bbox"], img_w, img_h, class_id)
            if line:
                label_lines.append(line)

        (out_lbl_dir / f"{dst.stem}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else "")
        )
        n_images += 1
        n_boxes += len(label_lines)

    n_unannotated = 0
    if include_unannotated_images:
        image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
        for src in sorted(image_root.iterdir()):
            if not src.is_file() or src.suffix.lower() not in image_exts:
                continue
            if src.name in seen_files:
                continue

            dst = out_img_dir / src.name
            link_or_copy_image(src, dst, link_mode)
            (out_lbl_dir / f"{dst.stem}.txt").write_text("")
            n_images += 1
            n_unannotated += 1

    print(
        f"[Prepare] {split}: images={n_images}, boxes={n_boxes}, "
        f"missing_images={n_missing}, unannotated_images={n_unannotated}, "
        f"labels={out_lbl_dir}"
    )
    return n_images


def write_dataset_yaml(prepared_root: Path) -> Path:
    yaml_path = prepared_root / "flir_yolo.yaml"
    names = ", ".join(f"{idx}: {name}" for idx, name in enumerate(FLIR_CLASS_NAMES))
    yaml_text = (
        f"path: {prepared_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {NUM_CLASSES}\n"
        f"names: {{{names}}}\n"
    )
    yaml_path.write_text(yaml_text)
    print(f"[Prepare] YAML: {yaml_path}")
    return yaml_path


def prepare_yolo_dataset(args) -> Path:
    data_root = Path(args.data_root).resolve()
    prepared_root = Path(args.prepared_root).resolve()
    prepared_root.mkdir(parents=True, exist_ok=True)

    prepare_split(
        data_root,
        prepared_root,
        "train",
        args.link_mode,
        args.include_unannotated_images,
    )
    prepare_split(
        data_root,
        prepared_root,
        "val",
        args.link_mode,
        args.include_unannotated_images,
    )
    return write_dataset_yaml(prepared_root)


def train_yolo(args, yaml_path: Path) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is required. Install it with: pip install ultralytics"
        ) from exc

    print(f"[Setup] Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    print("[Train] Starting YOLOv11 fine-tuning")
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        pretrained=True,
        exist_ok=args.exist_ok,
    )

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        dst = ckpt_dir / "yolo11_flir.pt"
        shutil.copy2(best, dst)
        print(f"[Done] Best YOLO checkpoint copied to: {dst}")
    else:
        print(f"[Done] Training finished. Check weights under: {save_dir / 'weights'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv11 on FLIR ADAS")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/exouser/Downloads/thermal_mirage/FLIR_ADAS_1_3",
        help="Path to FLIR_ADAS_1_3 root",
    )
    parser.add_argument(
        "--prepared_root",
        type=str,
        default="yolo_flir_dataset",
        help="Where to write YOLO-format labels, image links, and YAML",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Starting YOLOv11 checkpoint, e.g. yolo11n.pt or yolo11s.pt",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="runs/yolo_flir")
    parser.add_argument("--name", type=str, default="yolo11_flir")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--link_mode",
        type=str,
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to place images into YOLO dataset layout",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only create YOLO dataset files; do not train",
    )
    parser.add_argument(
        "--include_unannotated_images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include every image in thermal_8_bit, creating empty labels when no valid annotation exists",
    )
    parser.add_argument(
        "--exist_ok",
        action="store_true",
        help="Allow Ultralytics to reuse the same project/name directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    yaml_path = prepare_yolo_dataset(args)
    if args.prepare_only:
        print("[Done] Dataset prepared; training skipped.")
    else:
        train_yolo(args, yaml_path)
