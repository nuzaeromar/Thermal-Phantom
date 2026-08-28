"""
Stages 4 & 5: Diffusion-Based Thermal Image Editor

Implements:
  - ThermalDiffusionEditor : lightweight DDPM for inpainting object regions
  - ContextDiffusionEditor : scene-level thermal context degradation

The diffusion model is a lightweight UNet conditioned on:
  - class embedding (object branch)
  - scene condition embedding (context branch)

For the full pipeline these are guided during reverse diffusion using
detector gradients (Stage 7).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


# ── Sinusoidal timestep embedding ─────────────────────────────────────────────

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    ).float()
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)   # (B, dim)


# ── UNet building blocks ──────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block with timestep + condition injection."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1   = nn.Conv2d(in_ch,  out_ch, 3, 1, 1)
        self.conv2   = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.norm1   = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2   = nn.GroupNorm(min(8, out_ch), out_ch)
        self.t_proj  = nn.Linear(t_dim, out_ch * 2)   # scale + shift
        self.drop    = nn.Dropout(dropout)
        self.skip    = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        t = self.t_proj(F.silu(t_emb))[:, :, None, None]
        scale, shift = t.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.drop(F.silu(self.conv2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention at bottleneck."""

    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + h


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# ── Lightweight Thermal UNet ──────────────────────────────────────────────────

class ThermalUNet(nn.Module):
    """
    DDPM UNet for 3-channel thermal images at 64x64 or 256x256.
    Conditioned on timestep + (class_id or scene_cond).

    channel_mult controls resolution stages.
    """

    def __init__(
        self,
        img_channels: int  = 3,
        base_ch: int       = 64,
        channel_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions: tuple = (8,),
        cond_dim: int      = 64,
        t_dim: int         = 256,
        dropout: float     = 0.1,
        img_size: int      = 64,
    ):
        super().__init__()
        self.t_dim = t_dim
        self.img_size = img_size
        self.model_config = {
            "img_channels": int(img_channels),
            "base_ch": int(base_ch),
            "channel_mult": tuple(int(v) for v in channel_mult),
            "num_res_blocks": int(num_res_blocks),
            "attn_resolutions": tuple(int(v) for v in attn_resolutions),
            "cond_dim": int(cond_dim),
            "t_dim": int(t_dim),
            "dropout": float(dropout),
            "img_size": int(img_size),
        }

        # Timestep embedding MLP
        self.t_embed = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        # Condition embedding (class or scene)
        self.cond_proj = nn.Linear(cond_dim, t_dim)

        # Encoder
        self.in_conv  = nn.Conv2d(img_channels, base_ch, 3, 1, 1)
        channels_enc = []
        ch = base_ch
        res = img_size
        self.enc_resblocks = nn.ModuleList()
        self.enc_attentions = nn.ModuleList()
        self.downs = nn.ModuleList()

        for stage_idx, mult in enumerate(channel_mult):
            out_ch = base_ch * mult
            stage_resblocks = nn.ModuleList()
            stage_attentions = nn.ModuleList()
            for _ in range(num_res_blocks):
                stage_resblocks.append(ResBlock(ch, out_ch, t_dim, dropout))
                channels_enc.append(out_ch)
                stage_attentions.append(
                    AttentionBlock(out_ch) if res in attn_resolutions else nn.Identity()
                )
                ch = out_ch
            self.enc_resblocks.append(stage_resblocks)
            self.enc_attentions.append(stage_attentions)
            if stage_idx < len(channel_mult) - 1:
                self.downs.append(Downsample(ch))
                res //= 2

        # Bottleneck
        self.mid1   = ResBlock(ch, ch, t_dim, dropout)
        self.mid_attn = AttentionBlock(ch) if res in attn_resolutions else nn.Identity()
        self.mid2   = ResBlock(ch, ch, t_dim, dropout)

        # Decoder
        self.dec_resblocks = nn.ModuleList()
        self.dec_attentions = nn.ModuleList()
        self.ups = nn.ModuleList()

        for reverse_idx, mult in enumerate(reversed(channel_mult)):
            if reverse_idx > 0:
                self.ups.append(Upsample(ch))
                res *= 2
            out_ch = base_ch * mult
            stage_resblocks = nn.ModuleList()
            stage_attentions = nn.ModuleList()
            for _ in range(num_res_blocks):
                skip_ch = channels_enc.pop()
                stage_resblocks.append(ResBlock(ch + skip_ch, out_ch, t_dim, dropout))
                stage_attentions.append(
                    AttentionBlock(out_ch) if res in attn_resolutions else nn.Identity()
                )
                ch = out_ch
            self.dec_resblocks.append(stage_resblocks)
            self.dec_attentions.append(stage_attentions)

        self.out_norm = nn.GroupNorm(min(8, ch), ch)
        self.out_conv = nn.Conv2d(ch, img_channels, 3, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        x        : (B, C, H, W)  noisy image
        t        : (B,)           timestep index
        cond_emb : (B, cond_dim)  class or scene conditioning
        Returns  : (B, C, H, W)  predicted noise
        """
        # Fuse timestep + conditioning
        t_emb = timestep_embedding(t, self.t_dim)
        t_emb = self.t_embed(t_emb) + self.cond_proj(cond_emb)

        # Encoder
        h = self.in_conv(x)
        skips = []
        for stage_idx, (resblocks, attentions) in enumerate(
            zip(self.enc_resblocks, self.enc_attentions)
        ):
            for resblock, attention in zip(resblocks, attentions):
                h = resblock(h, t_emb)
                h = attention(h)
                skips.append(h)
            if stage_idx < len(self.downs):
                h = self.downs[stage_idx](h)

        # Bottleneck
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        # Decoder
        for stage_idx, (resblocks, attentions) in enumerate(
            zip(self.dec_resblocks, self.dec_attentions)
        ):
            if stage_idx > 0:
                h = self.ups[stage_idx - 1](h)
            for resblock, attention in zip(resblocks, attentions):
                skip = skips.pop()
                if skip.shape[2:] != h.shape[2:]:
                    skip = F.interpolate(skip, size=h.shape[2:], mode="nearest")
                h = resblock(torch.cat([h, skip], dim=1), t_emb)
                h = attention(h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


def load_object_diffusion_checkpoint(checkpoint_path, device="cpu"):
    """Restore every learned component required for class-conditioned editing."""
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    required = {"unet", "class_embedding", "model_config", "scheduler_config"}
    if not isinstance(ckpt, dict) or not required.issubset(ckpt):
        missing = sorted(required.difference(ckpt.keys() if isinstance(ckpt, dict) else []))
        raise ValueError(
            f"Incomplete diffusion checkpoint {checkpoint_path}. Missing: {missing}. "
            "Train it with train_diffusion.py so U-Net and class embeddings are saved together."
        )

    model_config = dict(ckpt["model_config"])
    for key in ("channel_mult", "attn_resolutions"):
        if key in model_config:
            model_config[key] = tuple(model_config[key])
    unet = ThermalUNet(**model_config).to(device)
    unet.load_state_dict(ckpt["unet"])

    embedding_config = ckpt.get("class_embedding_config", {})
    num_classes = int(embedding_config.get("num_classes", ckpt["class_embedding"]["weight"].shape[0]))
    cond_dim = int(embedding_config.get("cond_dim", model_config["cond_dim"]))
    class_embedding = nn.Embedding(num_classes, cond_dim).to(device)
    class_embedding.load_state_dict(ckpt["class_embedding"])

    scheduler_config = dict(ckpt["scheduler_config"])
    scheduler = DDPMScheduler(**scheduler_config)
    unet.eval()
    class_embedding.eval()
    return unet, scheduler, class_embedding, ckpt


# ── DDPM Noise Scheduler ──────────────────────────────────────────────────────

class DDPMScheduler:
    """
    Linear beta schedule DDPM.
    Supports q_sample (forward) and p_sample (reverse, one step).
    """

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        betas  = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register = lambda name, val: setattr(self, name, val)
        self.register("betas",    betas)
        self.register("alphas",   alphas)
        self.register("acp",      alphas_cumprod)             # ā_t
        self.register("acp_prev", alphas_cumprod_prev)        # ā_{t-1}
        self.register("sqrt_acp",         alphas_cumprod.sqrt())
        self.register("sqrt_one_minus_acp", (1 - alphas_cumprod).sqrt())
        self.register("posterior_var",
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod))

    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape: tuple):
        """Gather values at timesteps t, broadcast to shape."""
        vals = a.to(t.device)[t]
        return vals.reshape(-1, *([1] * (len(shape) - 1)))

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward process: x_t = sqrt(ā_t)*x0 + sqrt(1-ā_t)*noise"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_acp      = self._extract(self.sqrt_acp,          t, x0.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_acp, t, x0.shape)
        return sqrt_acp * x0 + sqrt_one_minus * noise, noise

    @torch.no_grad()
    def p_sample(
        self,
        model: ThermalUNet,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_emb: torch.Tensor,
        guidance_grad: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.05,
    ) -> torch.Tensor:
        """
        One reverse diffusion step: x_{t-1} | x_t.
        Optionally adds classifier / detector guidance gradient.
        """
        betas      = self._extract(self.betas,    t, x_t.shape)
        acp        = self._extract(self.acp,      t, x_t.shape)
        acp_prev   = self._extract(self.acp_prev, t, x_t.shape)
        alphas     = self._extract(self.alphas,   t, x_t.shape)
        sqrt_recip_alpha = (1.0 / alphas).sqrt()

        # Predict noise
        eps_pred = model(x_t, t, cond_emb)

        # DDPM mean estimate
        mean = sqrt_recip_alpha * (
            x_t - betas * eps_pred / self._extract(self.sqrt_one_minus_acp, t, x_t.shape)
        )

        # Add adversarial guidance
        if guidance_grad is not None:
            mean = mean + guidance_scale * guidance_grad

        # Variance
        var   = self._extract(self.posterior_var, t, x_t.shape).clamp(min=1e-20)
        noise = torch.randn_like(x_t) if t[0] > 0 else torch.zeros_like(x_t)
        return mean + var.sqrt() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: ThermalUNet,
        shape: tuple,
        cond_emb: torch.Tensor,
        device: str = "cuda",
        guidance_fn=None,
        guidance_scale: float = 0.05,
        start_t: Optional[int] = None,
        x_start: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full reverse diffusion loop.
        If start_t and x_start given, starts from noised x_start (for inpainting).
        """
        T_start = start_t if start_t is not None else self.T - 1

        if x_start is not None and start_t is not None:
            t_tensor = torch.full((shape[0],), start_t, dtype=torch.long, device=device)
            x, _ = self.q_sample(x_start.to(device), t_tensor)
        else:
            x = torch.randn(shape, device=device)

        for t_val in reversed(range(T_start + 1)):
            t_tensor = torch.full((shape[0],), t_val, dtype=torch.long, device=device)
            grad = None
            if guidance_fn is not None and t_val % 5 == 0:
                grad = guidance_fn(x, t_val)
            x = self.p_sample(model, x, t_tensor, cond_emb, grad, guidance_scale)

        return x


# ── Object Manipulation Editor (Stage 4) ─────────────────────────────────────

class ThermalObjectEditor:
    """
    Inpaints the object region to shift toward target class semantics.

    Wraps a diffusion model + scheduler with:
    - mask-based inpainting (keep background, edit object)
    - GAN prior target features for guided editing
    - detector-in-the-loop gradient guidance (injected externally)
    """

    def __init__(
        self,
        unet: ThermalUNet,
        scheduler: DDPMScheduler,
        class_embeddings: nn.Embedding,
        device: str = "cuda",
        inpaint_start_t: int = 500,
    ):
        # Safe device resolution
        if device == "cuda" and not torch.cuda.is_available():
            print("[ThermalObjectEditor] CUDA unavailable, using CPU.")
            device = "cpu"
        self.device          = torch.device(device)
        self.unet            = unet.to(self.device)
        self.scheduler       = scheduler
        self.class_emb       = class_embeddings.to(self.device)
        self.inpaint_start_t = inpaint_start_t

    def edit_object(
        self,
        image: torch.Tensor,           # (1, 3, H, W) in [0,1]
        obj_mask: torch.Tensor,        # (1, 1, H, W) binary
        target_class: int,
        guidance_fn=None,
        guidance_scale: float = 0.1,
        n_steps: int = 200,
        gan_prior_images: Optional[torch.Tensor] = None,  # (k, 3, h, w)
    ) -> torch.Tensor:
        """
        Returns adversarially edited image (1, 3, H, W) in [0,1].
        Object region is inpainted toward target_class semantics.
        Background is preserved via mask composition.
        """
        image = image.to(self.device)
        mask = obj_mask.to(self.device, dtype=image.dtype)
        output = image.clone()
        model_size = int(self.unet.img_size)
        start_t = min(int(self.inpaint_start_t), int(n_steps), self.scheduler.T - 1)

        # Object crops preserve local detail and keep diffusion cost independent
        # of the detector's full-frame input resolution.
        for batch_idx in range(image.shape[0]):
            support = mask[batch_idx, 0] > 0.5
            if not support.any():
                continue
            ys, xs = torch.where(support)
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            pad = max(2, int(round(0.10 * max(y2 - y1, x2 - x1))))
            y1, y2 = max(0, y1 - pad), min(image.shape[-2], y2 + pad)
            x1, x2 = max(0, x1 - pad), min(image.shape[-1], x2 + pad)

            crop = image[batch_idx:batch_idx + 1, :, y1:y2, x1:x2]
            crop_mask = mask[batch_idx:batch_idx + 1, :, y1:y2, x1:x2]
            crop = F.interpolate(crop, (model_size, model_size), mode="bilinear", align_corners=False)
            crop_mask = F.interpolate(crop_mask, (model_size, model_size), mode="nearest")
            x0_scaled = crop * 2 - 1

            if gan_prior_images is not None:
                prior = gan_prior_images.mean(dim=0, keepdim=True).to(self.device)
                prior = F.interpolate(prior, (model_size, model_size), mode="bilinear", align_corners=False)
                prior_scaled = prior * 2 - 1
                x0_scaled = crop_mask * prior_scaled + (1 - crop_mask) * x0_scaled

            cid = torch.tensor([target_class], dtype=torch.long, device=self.device)
            c_emb = self.class_emb(cid)

            def crop_guidance(x_t, t_val):
                if guidance_fn is None:
                    return None
                crop_01 = ((x_t + 1) / 2).clamp(0, 1)
                crop_back = F.interpolate(
                    crop_01, size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False
                )
                mask_back = mask[batch_idx:batch_idx + 1, :, y1:y2, x1:x2]
                full = image[batch_idx:batch_idx + 1].clone()
                full_region = full[:, :, y1:y2, x1:x2]
                full[:, :, y1:y2, x1:x2] = (
                    mask_back * crop_back + (1 - mask_back) * full_region
                )
                full_grad = guidance_fn(full, t_val)
                if full_grad is None:
                    return None
                grad_crop = full_grad[:, :, y1:y2, x1:x2]
                return F.interpolate(
                    grad_crop, (model_size, model_size), mode="bilinear", align_corners=False
                ) * crop_mask

            edited_scaled = self.scheduler.p_sample_loop(
                model=self.unet,
                shape=x0_scaled.shape,
                cond_emb=c_emb,
                device=str(self.device),
                guidance_fn=crop_guidance,
                guidance_scale=guidance_scale,
                start_t=start_t,
                x_start=x0_scaled,
            )
            edited_crop = ((edited_scaled + 1) / 2).clamp(0, 1)
            edited_crop = crop_mask * edited_crop + (1 - crop_mask) * crop
            edited_back = F.interpolate(
                edited_crop, size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False
            )
            original_region = output[batch_idx:batch_idx + 1, :, y1:y2, x1:x2]
            mask_back = mask[batch_idx:batch_idx + 1, :, y1:y2, x1:x2]
            output[batch_idx:batch_idx + 1, :, y1:y2, x1:x2] = (
                mask_back * edited_back + (1 - mask_back) * original_region
            )

        return output.clamp(0, 1)


# ── Context Editor (Stage 5) ──────────────────────────────────────────────────

SCENE_CONDITIONS = {
    "day":          0,
    "night":        1,
    "low_contrast": 2,
    "cluttered":    3,
    "foggy_thermal": 4,
}


class ThermalContextEditor:
    """
    Degrades the scene context (Stage 5) to reduce detector salience.

    Modes:
      "night"        : lower global thermal intensity
      "low_contrast" : reduce local contrast around objects
      "cluttered"    : add background heat structure
      "foggy_thermal": smooth-blur the scene

    These degrade detector confidence without changing object identity.
    """

    def __init__(
        self,
        unet: ThermalUNet,
        scheduler: DDPMScheduler,
        scene_embeddings: nn.Embedding,
        device: str = "cuda",
        ctx_start_t: int = 300,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            print("[ThermalContextEditor] CUDA unavailable, using CPU.")
            device = "cpu"
        self.device        = torch.device(device)
        self.unet          = unet.to(self.device)
        self.scheduler     = scheduler
        self.scene_emb     = scene_embeddings.to(self.device)
        self.ctx_start_t   = ctx_start_t

    def degrade_context(
        self,
        image: torch.Tensor,           # (1, 3, H, W) in [0,1]
        ctx_mask: torch.Tensor,        # (1, 1, H, W) context region
        obj_mask: torch.Tensor,        # (1, 1, H, W) object region (preserve)
        scene_condition: str = "night",
        guidance_fn=None,
        guidance_scale: float = 0.05,
    ) -> torch.Tensor:
        """
        Degrades context region while leaving object region mostly unchanged.
        Returns edited image (1, 3, H, W) in [0,1].
        """
        image = image.to(self.device)
        ctx   = ctx_mask.to(self.device)
        obj   = obj_mask.to(self.device)

        # Pure context mask (exclude object region)
        pure_ctx = (ctx * (1 - obj)).clamp(0, 1)

        cond_id = SCENE_CONDITIONS.get(scene_condition, 0)
        cid     = torch.full((1,), cond_id, dtype=torch.long, device=self.device)
        c_emb   = self.scene_emb(cid)

        x0_scaled = image * 2 - 1

        def ctx_guidance(x_t, t_val):
            t_tensor = torch.full((1,), t_val, dtype=torch.long, device=self.device)
            # Preserve non-context region
            bg_noise, _ = self.scheduler.q_sample(x0_scaled, t_tensor)
            x_t = pure_ctx * x_t + (1 - pure_ctx) * bg_noise
            if guidance_fn is not None:
                return guidance_fn(x_t, t_val)
            return None

        edited_scaled = self.scheduler.p_sample_loop(
            model         = self.unet,
            shape         = image.shape,
            cond_emb      = c_emb,
            device        = str(self.device),
            guidance_fn   = ctx_guidance,
            guidance_scale= guidance_scale,
            start_t       = self.ctx_start_t,
            x_start       = x0_scaled,
        )

        # Composite: ctx edits in context, original outside
        edited_scaled = pure_ctx * edited_scaled + (1 - pure_ctx) * x0_scaled
        return ((edited_scaled + 1) / 2).clamp(0, 1)

    @staticmethod
    def _gaussian_kernel1d(kernel_size: int, sigma: float, device, dtype):
        """Create a normalized 1D Gaussian kernel."""
        if kernel_size % 2 == 0:
            kernel_size += 1
        radius = kernel_size // 2
        x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        return kernel / (kernel.sum() + 1e-8)

    @staticmethod
    def _gaussian_blur_tensor(x: torch.Tensor, kernel_size: int = 15, sigma: float = 5.0) -> torch.Tensor:
        """
        Differentiable Gaussian blur for BxCxHxW tensors.
        Uses separable depthwise convolution and keeps the same spatial size.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_1d = ThermalContextEditor._gaussian_kernel1d(
            kernel_size, sigma, x.device, x.dtype
        )
        c = x.shape[1]
        pad = kernel_size // 2
        kx = kernel_1d.view(1, 1, 1, kernel_size).repeat(c, 1, 1, 1)
        ky = kernel_1d.view(1, 1, kernel_size, 1).repeat(c, 1, 1, 1)
        x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), kx, groups=c)
        x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"), ky, groups=c)
        return x

    @staticmethod
    def _masked_total_variation(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Total variation of a masked perturbation."""
        if mask.shape[1] == 1 and delta.shape[1] != 1:
            mask = mask.expand_as(delta)
        masked = delta * mask
        tv_h = (masked[:, :, 1:, :] - masked[:, :, :-1, :]).abs().mean()
        tv_w = (masked[:, :, :, 1:] - masked[:, :, :, :-1]).abs().mean()
        return tv_h + tv_w

    @staticmethod
    def _soft_distance_mask(
        obj_mask: torch.Tensor,
        ctx_mask: torch.Tensor,
        tau: float = 20.0,
    ) -> torch.Tensor:
        """
        Build M_soft(p)=exp(-d(p,M_o)/tau) inside the surrounding context.

        The perturbation is strongest close to the object boundary and decays
        smoothly into the background. The object itself is excluded so the
        method performs a context-level edit.
        """
        obj = (obj_mask.detach() > 0.5).float()
        ctx = (ctx_mask.detach() > 0.5).float()
        pure_ctx = (ctx * (1.0 - obj)).clamp(0, 1)
        b, _, h, w = obj.shape
        masks = []

        for i in range(b):
            obj_np = obj[i, 0].cpu().numpy().astype("uint8")
            pure_np = pure_ctx[i, 0].cpu().numpy().astype("float32")
            if obj_np.sum() == 0 or pure_np.sum() == 0:
                masks.append(torch.zeros(1, h, w, device=obj_mask.device, dtype=obj_mask.dtype))
                continue

            try:
                import cv2
                kernel = np.ones((3, 3), np.uint8)
                eroded = cv2.erode(obj_np, kernel, iterations=1)
                boundary = (obj_np - eroded).clip(0, 1).astype("uint8")
                if boundary.sum() == 0:
                    boundary = obj_np
                # distanceTransform returns distance to nearest zero. Make boundary zero.
                dist_input = (1 - boundary).astype("uint8")
                dist = cv2.distanceTransform(dist_input, cv2.DIST_L2, 3)
            except Exception:
                # Fallback: approximate a boundary distance with a smooth coordinate grid.
                yy, xx = np.mgrid[0:h, 0:w]
                ys, xs = np.where(obj_np > 0)
                cy, cx = float(ys.mean()), float(xs.mean())
                dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

            soft_np = np.exp(-dist / max(float(tau), 1e-6)).astype("float32") * pure_np
            masks.append(torch.from_numpy(soft_np).to(device=obj_mask.device, dtype=obj_mask.dtype).unsqueeze(0))

        return torch.stack(masks, dim=0)

    @staticmethod
    def _project_physical_constraints(
        image: torch.Tensor,
        original: torch.Tensor,
        soft_mask: torch.Tensor,
        raw_delta: torch.Tensor,
        epsilon: float = 0.08,
        epsilon_tv: float = 0.02,
        smooth_kernel: int = 9,
        smooth_sigma: float = 3.0,
    ) -> torch.Tensor:
        """
        Enforce the physical plausibility constraints:
          ||M_soft ⊙ Δθ(x)||_∞ <= epsilon
          TV(M_soft ⊙ Δθ(x)) <= epsilon_tv approximately via smoothing
          0 <= x' <= 1
        """
        if soft_mask.shape[1] == 1 and raw_delta.shape[1] != 1:
            mask = soft_mask.expand_as(raw_delta)
        else:
            mask = soft_mask

        masked_delta = (mask * raw_delta).clamp(-epsilon, epsilon)

        # Smooth the perturbation if its total variation is too large.
        # This is a projection-like practical enforcement for thermal imagery.
        for _ in range(3):
            tv = ThermalContextEditor._masked_total_variation(masked_delta, mask)
            if tv.item() <= epsilon_tv:
                break
            blurred_delta = ThermalContextEditor._gaussian_blur_tensor(
                masked_delta, kernel_size=smooth_kernel, sigma=smooth_sigma
            )
            masked_delta = (mask * blurred_delta).clamp(-epsilon, epsilon)

        return (original + masked_delta).clamp(0, 1)

    @staticmethod
    def apply_physics_context_attack(
        image: torch.Tensor,
        ctx_mask: torch.Tensor,
        obj_mask: torch.Tensor,
        mode: str = "sequential",
        epsilon: float = 0.08,
        epsilon_tv: float = 0.02,
        tau: float = 20.0,
        alpha: float = 0.45,
        beta: float = 0.65,
        gamma: float = 0.45,
        blur_kernel: int = 15,
        blur_sigma: float = 5.0,
        adaptive_cooling: bool = True,
    ) -> torch.Tensor:
        """
        Physics-informed context-level attacks for thermal images.

        Implements:
          x' = x + M_soft ⊙ Δθ(x)

        Available modes:
          - "contrast": Δ_contrast(x)=α(μ_o - x)
          - "smooth":   Δ_smooth(x)=β(Blur(x)-x)
          - "cool":     Δ_cool(x)=-γ or adaptive cooling
          - "adaptive_cool": adaptive context cooling only
          - "sequential": contrast → smoothing → adaptive cooling

        The perturbation is restricted to the distance-aware context mask and
        projected using L∞, TV-smoothing, and valid intensity constraints.
        """
        img = image.clone()
        original = image.clone()
        obj = obj_mask.to(img.device, dtype=img.dtype)
        ctx = ctx_mask.to(img.device, dtype=img.dtype)
        soft = ThermalContextEditor._soft_distance_mask(obj, ctx, tau=tau).to(img.device, dtype=img.dtype)

        if soft.sum() <= 0:
            return img.clamp(0, 1)

        def contrast_step(x: torch.Tensor) -> torch.Tensor:
            obj_3 = obj.expand_as(x)
            mu_o = (x * obj_3).sum(dim=(1, 2, 3), keepdim=True) / (obj_3.sum(dim=(1, 2, 3), keepdim=True) + 1e-6)
            delta = alpha * (mu_o - x)
            return ThermalContextEditor._project_physical_constraints(
                x, x, soft, delta, epsilon=epsilon, epsilon_tv=epsilon_tv
            )

        def smooth_step(x: torch.Tensor) -> torch.Tensor:
            blurred = ThermalContextEditor._gaussian_blur_tensor(x, kernel_size=blur_kernel, sigma=blur_sigma)
            delta = beta * (blurred - x)
            return ThermalContextEditor._project_physical_constraints(
                x, x, soft, delta, epsilon=epsilon, epsilon_tv=epsilon_tv
            )

        def cool_step(x: torch.Tensor) -> torch.Tensor:
            if adaptive_cooling or mode == "adaptive_cool":
                local_mean = ThermalContextEditor._gaussian_blur_tensor(x, kernel_size=blur_kernel, sigma=blur_sigma)
                delta = -gamma * torch.clamp(x - local_mean, min=0.0)
            else:
                delta = -gamma * torch.ones_like(x)
            return ThermalContextEditor._project_physical_constraints(
                x, x, soft, delta, epsilon=epsilon, epsilon_tv=epsilon_tv
            )

        mode = (mode or "sequential").lower()
        if mode in {"physics_context", "sequential", "sequential_context"}:
            img = contrast_step(img)
            img = smooth_step(img)
            img = cool_step(img)
        elif mode in {"contrast", "contrast_reduction", "low_contrast_physics"}:
            img = contrast_step(img)
        elif mode in {"smooth", "thermal_smoothing", "blur_physics"}:
            img = smooth_step(img)
        elif mode in {"cool", "context_cooling", "adaptive_cool"}:
            img = cool_step(img)
        else:
            # Unknown mode: use full sequential physical context attack.
            img = contrast_step(img)
            img = smooth_step(img)
            img = cool_step(img)

        # Final global budget relative to the original context image.
        total_delta = (img - original).clamp(-epsilon, epsilon)
        if soft.shape[1] == 1 and total_delta.shape[1] != 1:
            total_delta = total_delta * soft.expand_as(total_delta)
        else:
            total_delta = total_delta * soft
        return (original + total_delta).clamp(0, 1)

    @staticmethod
    def apply_analytic_degradation(
        image: torch.Tensor,
        ctx_mask: torch.Tensor,
        obj_mask: torch.Tensor,
        mode: str = "night",
        strength: float = 0.3,
    ) -> torch.Tensor:
        """
        Fast analytic context degradation (no diffusion required).

        In addition to the older modes, this method now supports the
        physics-informed context attacks:
          "sequential", "physics_context", "contrast", "smooth",
          "cool", and "adaptive_cool".
        """
        physics_modes = {
            "physics_context", "sequential", "sequential_context",
            "contrast", "contrast_reduction", "smooth", "thermal_smoothing",
            "cool", "context_cooling", "adaptive_cool"
        }
        if (mode or "").lower() in physics_modes:
            return ThermalContextEditor.apply_physics_context_attack(
                image=image,
                ctx_mask=ctx_mask,
                obj_mask=obj_mask,
                mode=mode,
                epsilon=max(0.01, min(float(strength), 0.25)),
                epsilon_tv=0.02,
                tau=20.0,
                alpha=0.45,
                beta=0.65,
                gamma=0.45,
                adaptive_cooling=True,
            )

        img = image.clone()
        ctx  = ctx_mask.clone()
        obj  = obj_mask.clone()
        pure = (ctx * (1 - obj)).clamp(0, 1)

        if mode == "night":
            # Darken thermal by attenuating intensity
            img = img * (1 - pure * strength)

        elif mode == "low_contrast":
            # Blend toward mean intensity in context region
            mean = (img * pure).sum() / (pure.sum() + 1e-6)
            blended = img * (1 - strength) + mean * strength
            img = pure * blended + (1 - pure) * img

        elif mode == "blur":
            blurred_t = ThermalContextEditor._gaussian_blur_tensor(img, kernel_size=15, sigma=5.0)
            pure_3 = pure.expand_as(img)
            img = pure_3 * blurred_t * strength + img * (1 - pure_3 * strength)

        elif mode == "clutter":
            noise = torch.randn_like(img) * 0.1 * strength
            img = (img + pure * noise).clamp(0, 1)

        return img.clamp(0, 1)
