"""
Visualization utilities for ThermalMirage attack results.

Provides:
  - side_by_side_plot   : clean vs adversarial with detection overlays
  - mask_overlay        : show object/context masks
  - attack_progress     : show diffusion editing steps
  - class_prior_grid    : GAN sampled thermal crops per class
  - metrics_bar_chart   : ablation comparison bar charts
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from typing import List, Dict, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.flir_dataset import FLIR_CLASS_NAMES, NUM_CLASSES

CLASS_NAMES = FLIR_CLASS_NAMES   # ["person", "bicycle", "car", "dog"]

FLIR_CLASS_COLORS = {
    "person":  "#FF4444",
    "bicycle": "#44AAFF",
    "car":     "#44FF44",
    "dog":     "#FFAA44",
    "unknown": "#AAAAAA",
}


def tensor_to_np(img: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) or (3,H,W) float [0,1] -> (H,W,3) uint8."""
    if img.dim() == 4:
        img = img.squeeze(0)
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img.clip(0, 1) * 255).astype(np.uint8)
    
    if img.shape[2] == 3 and np.allclose(img[:, :, 0], img[:, :, 1], atol=10):
        gray = img[:, :, 0].astype(np.float32) / 255.0
        return gray  # return 2D for imshow with cmap
    return img

from torchvision.ops import nms
def draw_boxes(
    ax: plt.Axes,
    boxes: torch.Tensor,       # (N, 4) normalized xyxy or pixel xyxy
    labels: torch.Tensor,      # (N,)
    scores: Optional[torch.Tensor] = None,
    img_hw: Tuple[int, int] = (640, 640),
    normalized: bool = True,
    class_names: list = None,
):
    """Draw bounding boxes on matplotlib axes."""
    if class_names is None:
        class_names = CLASS_NAMES

    if scores is not None and len(boxes) > 0:
        keep_indices = nms(boxes, scores, iou_threshold=0.5)
        boxes = boxes[keep_indices]
        labels = labels[keep_indices]
        scores = scores[keep_indices]

    H, W = img_hw
    for i, (box, lbl) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box.tolist()
        if normalized:
            x1, x2 = x1 * W, x2 * W
            y1, y2 = y1 * H, y2 * H
        name  = class_names[int(lbl)] if int(lbl) < len(class_names) else "unknown"
        color = FLIR_CLASS_COLORS.get(name, "#AAAAAA")
        rect  = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        label = name if scores is None else f"{name} {scores[i]:.2f}"
        ax.text(x1, y1 - 4, label, color=color, fontsize=8,
                bbox=dict(facecolor="black", alpha=0.5, pad=1))


def side_by_side_plot(
    clean: torch.Tensor,              # (1,3,H,W)
    adv:   torch.Tensor,              # (1,3,H,W)
    gt_boxes: torch.Tensor,           # (N,4)
    gt_labels: torch.Tensor,          # (N,)
    clean_dets: Optional[Dict] = None,
    adv_dets:   Optional[Dict] = None,
    title: str = "",
    save_path: Optional[str] = None,
    cmap: str = "inferno",
) -> plt.Figure:
    """
    Three-panel figure: [Clean + GT boxes] | [Adv + adv detections] | [Difference map]
    """
    H, W = clean.shape[2], clean.shape[3]
    clean_np = tensor_to_np(clean)
    adv_np   = tensor_to_np(adv)

    diff = (adv[0] - clean[0]).abs().mean(0).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title or "ThermalMirage: Clean vs Adversarial", fontsize=13)

    # Panel 1: Clean
    ax = axes[0]
    ax.imshow(clean_np, cmap=cmap if clean_np.ndim == 2 else None)
    draw_boxes(ax, gt_boxes, gt_labels, img_hw=(H, W), normalized=True)
    ax.set_title("Clean (GT boxes)")
    ax.axis("off")

    # Panel 2: Adversarial
    ax = axes[1]
    ax.imshow(adv_np, cmap=cmap if adv_np.ndim == 2 else None)
    if adv_dets is not None and len(adv_dets.get("boxes", [])) > 0:
        draw_boxes(
            ax,
            adv_dets["boxes"],
            adv_dets["labels"],
            adv_dets.get("scores"),
            img_hw=(H, W),
            normalized=False,  # detector output usually in pixel space
        )
    ax.set_title("Adversarial (pred boxes)")
    ax.axis("off")

    # Panel 3: Difference heatmap
    ax = axes[2]
    im = ax.imshow(diff, cmap="hot", norm=Normalize(vmin=0, vmax=0.3))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"|Δ| map (L∞={diff.max():.3f})")
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def mask_overlay_plot(
    image: torch.Tensor,
    obj_mask: torch.Tensor,
    ctx_mask: torch.Tensor,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Show image with object mask (red) and context mask (blue) overlay."""
    img_np = tensor_to_np(image)
    H, W   = image.shape[2], image.shape[3]

    obj_np = obj_mask[0, 0].cpu().numpy()
    ctx_np = ctx_mask[0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cmap = "inferno" if img_np.ndim == 2 else None

    axes[0].imshow(img_np, cmap=cmap)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(img_np, cmap=cmap)
    axes[1].imshow(obj_np, alpha=0.4, cmap="Reds")
    axes[1].set_title("Object Mask (red)")
    axes[1].axis("off")

    axes[2].imshow(img_np, cmap=cmap)
    axes[2].imshow(ctx_np, alpha=0.3, cmap="Blues")
    axes[2].imshow(obj_np, alpha=0.35, cmap="Reds")
    axes[2].set_title("Context (blue) + Object (red)")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def class_prior_grid(
    gan_prior,
    class_names: list = CLASS_NAMES,
    n_per_class: int = 8,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Sample GAN crops for each class and display as a grid."""
    n_cls = len(class_names)
    fig, axes = plt.subplots(n_cls, n_per_class, figsize=(n_per_class * 1.5, n_cls * 1.5))
    fig.suptitle("GAN Thermal Priors per Class", fontsize=12)

    for c, name in enumerate(class_names):
        try:
            imgs = gan_prior.sample_target_images(c, n=n_per_class)  # (n, 3, H, W)
            for j in range(n_per_class):
                ax  = axes[c][j]
                img = imgs[j].permute(1, 2, 0).cpu().numpy()
                img = (img.clip(0, 1) * 255).astype(np.uint8)
                gray = img[:, :, 0] if img.shape[2] == 3 else img
                ax.imshow(gray, cmap="inferno")
                ax.axis("off")
                if j == 0:
                    ax.set_ylabel(name, fontsize=10)
        except Exception as e:
            print(f"[Viz] class {name} failed: {e}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def ablation_bar_chart(
    summaries: Dict[str, Dict],
    metrics: List[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing ablation variants on selected metrics."""
    if metrics is None:
        metrics = ["attack_success_rate", "false_negative_rate", "mean_conf_drop"]

    variant_names = list(summaries.keys())
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 0.8, len(variant_names)))

    for mi, metric in enumerate(metrics):
        ax     = axes[mi]
        values = [summaries[v].get(metric, 0) for v in variant_names]
        bars   = ax.bar(variant_names, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(metric.replace("_", " ").title(), fontsize=10)
        ax.set_ylim(0, max(values) * 1.2 + 0.05)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Ablation Study — ThermalMirage", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


def attack_progress_plot(
    stages: Dict[str, torch.Tensor],   # {'clean': ..., 'obj_edited': ..., 'ctx_edited': ..., 'final': ...}
    save_path: Optional[str] = None,
    cmap: str = "inferno",
) -> plt.Figure:
    """Show editing progression: clean -> obj_edit -> ctx_edit -> final adv."""
    stage_names = list(stages.keys())
    n = len(stage_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, stage_names):
        img = tensor_to_np(stages[name])
        ax.imshow(img, cmap=cmap if img.ndim == 2 else None)
        ax.set_title(name.replace("_", " ").title(), fontsize=10)
        ax.axis("off")

    fig.suptitle("Attack Editing Stages", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig
