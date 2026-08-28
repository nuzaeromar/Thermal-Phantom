"""Standalone ThermalMirage component and context ablation runner.

This script intentionally leaves the existing attack and evaluation files
unchanged. Object edits use the class-conditioned GAN prior and diffusion
editor with Faster R-CNN guidance. Faster R-CNN is also used for optional
post-composition refinement; YOLOv11 is used only as the target model for final
attack-success and detection-quality evaluation.

The ablation uses a fixed deterministic strategy instead of online adaptive
selection. Object diffusion maps each source class to the next FLIR class;
context manipulation and optional final refinement suppress detector evidence.
Object ASR counts a clean-correct object that becomes hidden or misclassified.
This avoids adaptive selection becoming an additional source of feedback.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import run_multiclass as rm
from data.flir_dataset import (
    FLIR_CLASS_NAMES,
    FLIR_NATIVE_SIZE,
    NUM_CLASSES,
    build_dataloader,
)
from models.diffusion_editor import ThermalContextEditor


@dataclass(frozen=True)
class AblationVariant:
    name: str
    use_object: bool = False
    context_mode: str = "none"
    use_detector_refinement: bool = False


COMPONENT_VARIANTS = [
    AblationVariant("Clean"),
    AblationVariant("Object only", use_object=True),
    AblationVariant("Context only", context_mode="sequential"),
    AblationVariant("Object + context", use_object=True, context_mode="sequential"),
    AblationVariant("Object + detector", use_object=True, use_detector_refinement=True),
    AblationVariant("Context + detector", context_mode="sequential", use_detector_refinement=True),
]

CONTEXT_VARIANTS = [
    AblationVariant("None"),
    AblationVariant("Contrast", context_mode="contrast"),
    AblationVariant("Smoothing", context_mode="smooth"),
    AblationVariant("Adaptive cooling", context_mode="adaptive_cool"),
    AblationVariant("Sequential", context_mode="sequential"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="ThermalMirage ablations: Faster R-CNN loop, YOLOv11 target"
    )
    parser.add_argument("--suite", choices=["components", "context", "all"], default="all")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--gan_ckpt", required=True)
    parser.add_argument("--detector_ckpt", required=True,
                        help="FLIR Faster R-CNN checkpoint used for gradients")
    parser.add_argument("--diffusion_ckpt", required=True,
                        help="Trained thermal diffusion checkpoint for object editing")
    parser.add_argument("--yolo_ckpt", required=True,
                        help="FLIR-trained YOLOv11 checkpoint used as target model")
    parser.add_argument("--yolo_conf", type=float, default=0.25)
    parser.add_argument("--eval_conf", type=float, default=None)
    parser.add_argument("--eval_iou", type=float, default=0.5)
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="results/ablation_yolo11")

    parser.add_argument("--context_strength", type=float, default=0.12)
    parser.add_argument("--context_tau", type=float, default=20.0)
    parser.add_argument("--epsilon_tv", type=float, default=0.02)
    parser.add_argument("--context_alpha", type=float, default=0.45)
    parser.add_argument("--context_beta", type=float, default=0.65)
    parser.add_argument("--context_gamma", type=float, default=0.45)
    parser.add_argument("--max_perturbation", type=float, default=0.20)
    parser.add_argument("--n_refinement_steps", type=int, default=8)
    parser.add_argument("--diffusion_steps", type=int, default=200)
    parser.add_argument("--guidance_scale", type=float, default=0.08)

    parser.add_argument(
        "--asr_denominator",
        choices=["all", "clean_detected"],
        default="all",
        help="Use 'all' to match the existing run_multiclass summary definition",
    )
    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    # Attributes expected by run_multiclass.build_attack_engine().
    parser.set_defaults(
        target_detector="yolo",
        loop_detector="fasterrcnn",
        loop_yolo_ckpt=None,
        loop_yolo_conf=None,
        rtdetr_ckpt="rtdetr-l.pt",
        rtdetr_conf=0.25,
        context_attack="sequential",
    )
    return parser.parse_args()


def _unique_variants(suite: str) -> List[AblationVariant]:
    requested = []
    if suite in {"components", "all"}:
        requested.extend(COMPONENT_VARIANTS)
    if suite in {"context", "all"}:
        requested.extend(CONTEXT_VARIANTS)

    unique = []
    seen = set()
    for variant in requested:
        key = (variant.use_object, variant.context_mode, variant.use_detector_refinement)
        if key not in seen:
            unique.append(variant)
            seen.add(key)
    return unique


def _variant_key(variant: AblationVariant):
    return (variant.use_object, variant.context_mode, variant.use_detector_refinement)


def _new_accumulator():
    return {
        "predictions": [],
        "ground_truth": [],
        "ssim": [],
        "lpips": [],
        "thermal": [],
        "successes": 0,
        "eligible_successes": 0,
        "objects": 0,
        "eligible_objects": 0,
        "successful_images": 0,
        "images": 0,
        "confidence_drops": [],
    }


def _finalize_image(clean, candidate, max_perturbation):
    """Apply the same final full-image budget and grayscale constraint."""
    delta = (candidate - clean).clamp(-max_perturbation, max_perturbation)
    adv = (clean + delta).clamp(0, 1)
    gray = adv.mean(dim=1, keepdim=True)
    return gray.repeat(1, 3, 1, 1)


def _apply_object_attack(image, box, label, engine, args):
    """GAN-prior-conditioned diffusion edit with Faster R-CNN guidance."""
    _, _, height, width = image.shape
    obj_mask, _ = engine._build_masks_from_box(box, height, width)
    src_class = int(label)
    # The thermal GAN/diffusion model has no background embedding. Use a fixed,
    # deterministic wrong-class mapping to create a semantic object transition.
    target_class = (src_class + 1) % NUM_CLASSES
    target_labels = torch.tensor(
        [src_class], dtype=torch.long, device=image.device
    )
    guidance_fn = engine.guide.make_guidance_fn(
        target_boxes=box.unsqueeze(0),
        target_labels=target_labels,
        mode="misclassify",
        wrong_class=target_class,
    )
    gan_prior_images = engine.prior.sample_target_images(
        target_class=target_class,
        n=16,
    )
    return engine.obj_editor.edit_object(
        image=image,
        obj_mask=obj_mask,
        target_class=target_class,
        guidance_fn=guidance_fn,
        guidance_scale=args.guidance_scale,
        n_steps=args.diffusion_steps,
        gan_prior_images=gan_prior_images,
    )


def _apply_context_attack(image, box, engine, args, mode):
    _, _, height, width = image.shape
    obj_mask, ctx_mask = engine._build_masks_from_box(box, height, width)
    return ThermalContextEditor.apply_physics_context_attack(
        image=image,
        ctx_mask=ctx_mask,
        obj_mask=obj_mask,
        mode=mode,
        epsilon=args.context_strength,
        epsilon_tv=args.epsilon_tv,
        tau=args.context_tau,
        alpha=args.context_alpha,
        beta=args.context_beta,
        gamma=args.context_gamma,
        adaptive_cooling=True,
    )


def _refine_with_detector(image, clean, boxes, labels, engine, steps, budget):
    """Post-composition Faster R-CNN detector-in-the-loop refinement."""
    if steps <= 0:
        return image

    adv = image.detach().requires_grad_(True)
    step_size = 0.02 / steps
    for _ in range(steps):
        try:
            loss = engine.guide.detector.attack_loss(
                adv, boxes, labels, mode="hide", wrong_class=None
            )
            grad = torch.autograd.grad(loss, adv)[0]
            adv = (adv + step_size * grad.sign()).detach().clamp(0, 1)
            adv.requires_grad_(True)
        except Exception:
            break
    return _finalize_image(clean, adv.detach(), budget)


def generate_variant(clean, boxes, labels, engine, args, variant):
    if not variant.use_object and variant.context_mode == "none" \
            and not variant.use_detector_refinement:
        return clean.clone()

    current = clean.clone()
    for box, label in zip(boxes, labels):
        if variant.use_object:
            current = _apply_object_attack(
                current, box, int(label.item()), engine, args
            )
        if variant.context_mode != "none":
            current = _apply_context_attack(
                current, box, engine, args, variant.context_mode
            )

    current = _finalize_image(clean, current, args.max_perturbation)
    if variant.use_detector_refinement:
        current = _refine_with_detector(
            current,
            clean,
            boxes,
            labels,
            engine,
            args.n_refinement_steps,
            args.max_perturbation,
        )
    return current.detach()


def _evaluate_attack(clean_dets, adv_dets, boxes, labels, height, width, conf, iou):
    """Object ASR: a clean-correct object becomes hidden or misclassified."""
    rows = []
    for box, label in zip(boxes, labels):
        source_class = int(label.item())
        clean_conf, clean_class = rm._matched_detection(
            clean_dets, box, height, width, iou_threshold=iou
        )
        adv_conf, adv_class = rm._matched_detection(
            adv_dets, box, height, width, iou_threshold=iou
        )
        eligible = clean_conf >= conf and clean_class == source_class
        hidden = adv_conf < conf
        misclassified = adv_conf >= conf and adv_class != source_class
        success = eligible and (hidden or misclassified)
        rows.append({
            "eligible": eligible,
            "success": success,
            "hidden": hidden,
            "misclassified": misclassified,
            "confidence_drop": clean_conf - adv_conf,
        })
    return rows


def _aggregate_variant(acc, args):
    map50 = rm._compute_map_at_iou(
        acc["predictions"], acc["ground_truth"], 0.5, num_classes=NUM_CLASSES
    )["mAP"] if acc["images"] else 0.0

    denominator = (
        acc["eligible_objects"]
        if args.asr_denominator == "clean_detected"
        else acc["objects"]
    )
    successes = (
        acc["eligible_successes"]
        if args.asr_denominator == "clean_detected"
        else acc["successes"]
    )
    thermal = rm._summarize_final_thermal_plausibility(acc["thermal"])
    smooth_change = thermal["spatial_smoothness_change"]["mean"] if thermal else 0.0

    return {
        "object_asr": float(successes / denominator) if denominator else 0.0,
        "eligible_object_asr": (
            float(acc["eligible_successes"] / acc["eligible_objects"])
            if acc["eligible_objects"] else 0.0
        ),
        "image_asr": float(acc["successful_images"] / acc["images"])
        if acc["images"] else 0.0,
        "confidence_drop": float(np.mean(acc["confidence_drops"]))
        if acc["confidence_drops"] else 0.0,
        "map50": float(map50),
        "ssim": float(np.mean(acc["ssim"])) if acc["ssim"] else 0.0,
        "lpips": float(np.mean(acc["lpips"])) if acc["lpips"] else None,
        "histogram_kl": thermal["histogram_kl_divergence"]["mean"] if thermal else 0.0,
        "delta_tv": smooth_change,
        "abs_delta_tv": abs(smooth_change),
        "boundary_deviation": thermal["boundary_deviation"]["mean"] if thermal else 0.0,
        "num_images": acc["images"],
        "num_objects": acc["objects"],
        "num_eligible_objects": acc["eligible_objects"],
        "asr_denominator": args.asr_denominator,
        "thermal_detail": thermal,
    }


def _format_table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def _print_component_table(results):
    rows = []
    for variant in COMPONENT_VARIANTS:
        metric = results[_variant_key(variant)]
        rows.append([
            variant.name,
            f"{metric['object_asr']:.4f}",
            f"{metric['image_asr']:.4f}",
            f"{metric['map50']:.4f}",
            f"{metric['ssim']:.4f}",
            f"{metric['histogram_kl']:.6f}",
            f"{metric['abs_delta_tv']:.6f}",
            f"{metric['boundary_deviation']:.6f}",
        ])
    print("\nCOMPONENT ABLATION (Faster R-CNN loop -> YOLOv11 target)")
    _format_table(
        ["Variant", "Object ASR", "Image ASR", "mAP50", "SSIM",
         "Hist. KL", "|Delta TV|", "Boundary dev."],
        rows,
    )


def _print_context_table(results):
    rows = []
    for variant in CONTEXT_VARIANTS:
        metric = results[_variant_key(variant)]
        rows.append([
            variant.name,
            f"{metric['object_asr']:.4f}",
            f"{metric['confidence_drop']:.4f}",
            f"{metric['map50']:.4f}",
            f"{metric['ssim']:.4f}",
            f"{metric['histogram_kl']:.6f}",
            f"{metric['delta_tv']:.6f}",
            f"{metric['boundary_deviation']:.6f}",
        ])
    print("\nCONTEXT-OPERATOR ABLATION (context only, YOLOv11 target)")
    _format_table(
        ["Context operator", "ASR", "Confidence drop", "mAP50", "SSIM",
         "Hist. KL", "Delta TV", "Boundary dev."],
        rows,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is False")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_conf = args.eval_conf if args.eval_conf is not None else args.yolo_conf

    engine, _, loop_name, target_detector, target_name = rm.build_attack_engine(args)
    if getattr(engine, "obj_editor", None) is None:
        raise RuntimeError(
            "The diffusion object editor was not initialized. Verify that "
            "--diffusion_ckpt exists and matches ThermalUNet."
        )
    print(f"[Ablation] Detector-in-the-loop: {loop_name}")
    print(f"[Ablation] Target detector: {target_name}")
    print("[Ablation] Object strategy: fixed cyclic class transition")
    print("[Ablation] Object ASR: clean-correct objects hidden or misclassified")

    loader = build_dataloader(
        root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=FLIR_NATIVE_SIZE,
        normalize=False,
        return_masks=False,
    )
    image_quality = rm.OptionalImageQualityMetrics(
        device=args.device, compute_lpips=args.compute_lpips
    )

    variants = _unique_variants(args.suite)
    accumulators = {_variant_key(v): _new_accumulator() for v in variants}
    processed = 0

    for batch_idx, batch in enumerate(loader):
        for image, boxes, labels in zip(batch["images"], batch["boxes"], batch["labels"]):
            if args.num_images is not None and processed >= args.num_images:
                break
            if len(boxes) == 0:
                continue

            clean = image.unsqueeze(0).to(args.device)
            boxes_device = boxes.to(args.device)
            labels_device = labels.to(args.device)
            height, width = image.shape[-2:]

            with torch.no_grad():
                clean_detection_batch = target_detector.detect(clean)
            clean_dets = clean_detection_batch[0] if clean_detection_batch else None
            gt = rm._prepare_gt_for_metrics(boxes, labels, height, width)

            generated: Dict[tuple, torch.Tensor] = {}
            for variant in variants:
                key = _variant_key(variant)
                if key not in generated:
                    generated[key] = generate_variant(
                        clean, boxes_device, labels_device, engine, args, variant
                    )
                adv = generated[key]

                with torch.no_grad():
                    adv_detection_batch = target_detector.detect(adv)
                adv_dets = adv_detection_batch[0] if adv_detection_batch else None
                attack_rows = _evaluate_attack(
                    clean_dets, adv_dets, boxes, labels, height, width,
                    eval_conf, args.eval_iou,
                )
                pair_quality = image_quality.compute_pair(clean.cpu(), adv.cpu())
                thermal = rm._compute_final_thermal_plausibility(
                    clean=clean,
                    adv=adv,
                    boxes=boxes,
                    plausibility_evaluator=getattr(engine, "plaus_loss", None),
                )

                acc = accumulators[key]
                acc["predictions"].append(rm._prepare_predictions_for_metrics(adv_dets))
                acc["ground_truth"].append(gt)
                acc["ssim"].append(pair_quality["ssim"])
                if pair_quality.get("lpips") is not None:
                    acc["lpips"].append(pair_quality["lpips"])
                acc["thermal"].append(thermal)
                acc["objects"] += len(attack_rows)
                eligible = [row for row in attack_rows if row["eligible"]]
                acc["eligible_objects"] += len(eligible)
                acc["successes"] += sum(row["success"] for row in attack_rows)
                acc["eligible_successes"] += sum(row["success"] for row in eligible)
                acc["successful_images"] += int(any(row["success"] for row in attack_rows))
                acc["images"] += 1
                acc["confidence_drops"].extend(
                    row["confidence_drop"] for row in attack_rows
                )

            processed += 1
            if args.verbose:
                print(f"[Ablation] Processed image {processed}")

        if args.num_images is not None and processed >= args.num_images:
            break

    results = {key: _aggregate_variant(acc, args) for key, acc in accumulators.items()}

    print("\n" + "=" * 72)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 72)
    print(f"Images: {processed} | loop={loop_name} | target={target_name}")
    print(f"ASR denominator: {args.asr_denominator}")
    if args.suite in {"components", "all"}:
        _print_component_table(results)
    if args.suite in {"context", "all"}:
        _print_context_table(results)

    serializable = {
        "configuration": vars(args),
        "detector_in_the_loop": loop_name,
        "target_detector": target_name,
        "objective": "fixed_cyclic_object_transition; success=hidden_or_misclassified",
        "results": {
            variant.name: results[_variant_key(variant)]
            for variant in (
                COMPONENT_VARIANTS + CONTEXT_VARIANTS
                if args.suite == "all"
                else COMPONENT_VARIANTS if args.suite == "components"
                else CONTEXT_VARIANTS
            )
        },
    }
    output_path = out_dir / "ablation_results.json"
    with open(output_path, "w") as handle:
        json.dump(serializable, handle, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
