"""
Thermal Mirage: GAN-Guided Diffusion Adversarial Attack on FLIR Thermal Detection
Configuration file
"""

import torch

# ─── Dataset ────────────────────────────────────────────────────────────────
FLIR_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "dog"}
FLIR_NUM_CLASSES = len(FLIR_CLASSES)
IMG_SIZE = (640, 512)          # FLIR native resolution (W, H)
THERMAL_INTENSITY_RANGE = (0.0, 1.0)

# ─── Attack targets (source → target class id, or -1 = background) ──────────
ATTACK_TRANSITIONS = {
    "person":  [("background", -1), ("clutter",  -1)],
    "bicycle": [("background", -1), ("clutter",  -1)],
    "car":     [("background", -1), ("clutter",  -1)],
    "dog":     [("background", -1), ("person",    0)],
}

# ─── GAN (class-conditional thermal object GAN) ──────────────────────────────
GAN_LATENT_DIM     = 128
GAN_EMBED_DIM      = 64
GAN_CROP_SIZE      = 96          # square crops for object GAN
GAN_TRAIN_EPOCHS   = 100
GAN_LR_G           = 2e-4
GAN_LR_D           = 2e-4
GAN_BATCH_SIZE     = 64

# ─── Diffusion (score-based / DDPM-style lightweight model) ──────────────────
DIFF_TIMESTEPS     = 1000
DIFF_BETA_START    = 1e-4
DIFF_BETA_END      = 0.02
DIFF_GUIDANCE_SCALE = 7.5
DIFF_INFER_STEPS   = 50

# ─── Attack optimization ─────────────────────────────────────────────────────
ATTACK_ITERS       = 100
ATTACK_LR          = 1e-3
CONTEXT_EXPAND     = 1.5        # context mask = 1.5× object-box side

# Loss weights  λ_det  λ_cls  λ_gan  λ_thermal  λ_consistency
LAMBDA_DET         = 1.0
LAMBDA_CLS         = 0.8
LAMBDA_GAN         = 0.5
LAMBDA_THERMAL     = 0.3
LAMBDA_CONSISTENCY = 0.2

# Context branch weights
LAMBDA_SALIENCY    = 0.6
LAMBDA_CONTRAST    = 0.4
LAMBDA_SCENE_REAL  = 0.3

# Thermal plausibility
MAX_INTENSITY_DELTA = 0.25      # max pixel change allowed
SMOOTHNESS_KERNEL   = 5         # Gaussian kernel size for smoothness loss

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
