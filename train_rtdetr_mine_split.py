"""
Fine-tune RT-DETR on mine_data with a small train/held-out split.

By default this trains on 30 person-annotated thermal images from mine_data and
validates on the remaining person-annotated thermal images. The prepared output
uses Ultralytics YOLO-format labels so RT-DETR can train and report precision,
recall, mAP50, and mAP50-95 on the held-out split.
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path


DEFAULT_CLASSES = ["person"]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train RT-DETR on 30 mine_data samples and test on the rest")
    parser.add_argument("--data_root", type=str, default="mine_data", help="Path to mine_data")
    parser.add_argument("--prepared_root", type=str, default="prepared_rtdetr_mine_30")
    parser.add_argument("--model", type=str, default="rtdetr-l.pt", help="Starting RT-DETR checkpoint")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="Classes to train/evaluate")
    parser.add_argument("--train_samples", type=int, default=30, help="Number of annotated images to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="runs/rtdetr_mine")
    parser.add_argument("--name", type=str, default="rtdetr_mine_person_30")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default="rtdetr_mine_person_30.pt")
    parser.add_argument("--link_mode", type=str, default="symlink", choices=["symlink", "hardlink", "copy"])
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--exist_ok", action="store_true")
    return parser.parse_args()


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
    return [min(1.0, max(0.0, float(v))) for v in vals]


def find_trials(data_root: Path):
    return sorted(path for path in data_root.iterdir() if path.is_dir() and (path / "thermal_coco_annotations.json").exists())


def collect_annotated_samples(data_root: Path, selected_classes):
    samples = []

    for trial_dir in find_trials(data_root):
        ann_path = trial_dir / "thermal_coco_annotations.json"
        image_root = trial_dir / "a700"
        with open(ann_path, "r") as f:
            coco = json.load(f)

        cat_name_by_id = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
        cat_id_to_internal = {
            cat_id: selected_classes.index(name)
            for cat_id, name in cat_name_by_id.items()
            if name in selected_classes
        }

        anns_by_image = {}
        for ann in coco.get("annotations", []):
            if ann.get("iscrowd", 0):
                continue
            if ann.get("category_id") not in cat_id_to_internal:
                continue
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        for image_info in coco.get("images", []):
            anns = anns_by_image.get(image_info["id"], [])
            if not anns:
                continue

            image_path = image_root / Path(image_info["file_name"]).name
            if not image_path.exists():
                continue

            sample_id = f"{trial_dir.name}_{image_info['id']}"
            samples.append({
                "sample_id": sample_id,
                "trial": trial_dir.name,
                "image_path": image_path,
                "image_info": image_info,
                "annotations": anns,
                "cat_id_to_internal": cat_id_to_internal,
            })

    return sorted(samples, key=lambda item: item["sample_id"])


def write_split(samples, prepared_root: Path, split: str, selected_classes, link_mode: str):
    image_dir = prepared_root / "images" / split
    label_dir = prepared_root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    n_objects = 0
    for sample in samples:
        src_img = sample["image_path"]
        dst_name = f"{sample['sample_id']}_{src_img.name}"
        dst_img = image_dir / dst_name
        link_or_copy(src_img, dst_img, link_mode)

        image_info = sample["image_info"]
        width = float(image_info.get("width", 0) or 0)
        height = float(image_info.get("height", 0) or 0)
        if width <= 0 or height <= 0:
            from PIL import Image
            with Image.open(src_img) as im:
                width, height = im.size

        lines = []
        for ann in sample["annotations"]:
            cls_id = sample["cat_id_to_internal"][ann["category_id"]]
            cx, cy, w, h = coco_box_to_yolo(ann["bbox"], width, height)
            if w <= 0 or h <= 0:
                continue
            lines.append(f"{cls_id} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}")

        (label_dir / f"{Path(dst_name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_objects += len(lines)

    print(f"[Prepare] {split}: images={len(samples)}, objects={n_objects}")
    return len(samples), n_objects


def write_data_yaml(prepared_root: Path, selected_classes):
    yaml_path = prepared_root / "mine_rtdetr.yaml"
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(selected_classes))
    yaml_path.write_text(
        f"path: {prepared_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(selected_classes)}\n"
        f"names:\n{names}\n"
    )
    print(f"[Prepare] data yaml: {yaml_path}")
    return yaml_path


def print_validation_summary(val_results):
    box = getattr(val_results, "box", None)
    if box is None:
        print(f"[Val] raw metrics: {val_results}", flush=True)
        return

    precision = getattr(box, "mp", None)
    recall = getattr(box, "mr", None)
    map50 = getattr(box, "map50", None)
    map50_95 = getattr(box, "map", None)

    print("[Val] Held-out accuracy metrics:", flush=True)
    if precision is not None:
        print(f"  Precision: {precision:.4f}", flush=True)
    if recall is not None:
        print(f"  Recall:    {recall:.4f}", flush=True)
    if map50 is not None:
        print(f"  mAP50:     {map50:.4f}", flush=True)
    if map50_95 is not None:
        print(f"  mAP50-95:  {map50_95:.4f}", flush=True)


def prepare_dataset(args):
    data_root = Path(args.data_root)
    prepared_root = Path(args.prepared_root)
    if prepared_root.exists():
        shutil.rmtree(prepared_root)
    prepared_root.mkdir(parents=True, exist_ok=True)

    selected_classes = list(args.classes)
    samples = collect_annotated_samples(data_root, selected_classes)
    if len(samples) <= args.train_samples:
        raise ValueError(
            f"Need more than {args.train_samples} annotated images for a held-out test split; "
            f"found {len(samples)}."
        )

    rng = random.Random(args.seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    train_samples = shuffled[:args.train_samples]
    val_samples = shuffled[args.train_samples:]

    print(f"[Prepare] selected classes: {selected_classes}")
    print(f"[Prepare] annotated images found: {len(samples)}")
    print(f"[Prepare] train images: {len(train_samples)}")
    print(f"[Prepare] held-out val images: {len(val_samples)}")

    write_split(train_samples, prepared_root, "train", selected_classes, args.link_mode)
    write_split(val_samples, prepared_root, "val", selected_classes, args.link_mode)
    data_yaml = write_data_yaml(prepared_root, selected_classes)
    return data_yaml


def main():
    print("[Start] train_rtdetr_mine_split.py", flush=True)
    args = parse_args()
    data_yaml = prepare_dataset(args)

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

    print("[Val] Running validation on held-out mine_data split")
    val_results = model.val(data=str(data_yaml), imgsz=args.imgsz, device=args.device, split="val")
    print_validation_summary(val_results)

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
    print(f"[Done] Checkpoint copied to: {dst}")


if __name__ == "__main__":
    main()
