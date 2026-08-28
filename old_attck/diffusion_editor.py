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
        self.t_dim    = t_dim
        self.img_size = img_size

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
        channels_enc  = []
        ch = base_ch
        res = img_size
        self.enc_blocks = nn.ModuleList()
        self.downs      = nn.ModuleList()

        for mult in channel_mult:
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                block = ResBlock(ch, out_ch, t_dim, dropout)
                self.enc_blocks.append(block)
                channels_enc.append(out_ch)
                if res in attn_resolutions:
                    self.enc_blocks.append(AttentionBlock(out_ch))
                    channels_enc.append(out_ch)
                ch = out_ch
            if mult != channel_mult[-1]:
                self.downs.append(Downsample(ch))
                res //= 2

        # Bottleneck
        self.mid1   = ResBlock(ch, ch, t_dim, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid2   = ResBlock(ch, ch, t_dim, dropout)

        # Decoder
        self.dec_blocks = nn.ModuleList()
        self.ups        = nn.ModuleList()

        for mult in reversed(channel_mult):
            out_ch = base_ch * mult
            for i in range(num_res_blocks + 1):
                skip_ch = channels_enc.pop() if channels_enc else 0
                block = ResBlock(ch + skip_ch, out_ch, t_dim, dropout)
                self.dec_blocks.append(block)
                if res in attn_resolutions:
                    self.dec_blocks.append(AttentionBlock(out_ch))
                ch = out_ch
            if mult != channel_mult[0]:
                self.ups.append(Upsample(ch))
                res *= 2

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
        skips = [h]
        down_iter = iter(self.downs)

        enc_idx = 0
        prev_mult = None
        for mult in [1, 2, 4, 8]:   # channel_mult
            for _ in range(2):      # num_res_blocks
                block = self.enc_blocks[enc_idx]
                enc_idx += 1
                if isinstance(block, ResBlock):
                    h = block(h, t_emb)
                    skips.append(h)
                elif isinstance(block, AttentionBlock):
                    h = block(h)
                    skips.append(h)
            if mult != 8:
                try:
                    h = next(down_iter)(h)
                except StopIteration:
                    pass

        # Bottleneck
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        # Decoder
        up_iter  = iter(self.ups)
        dec_idx  = 0
        for mult in reversed([1, 2, 4, 8]):
            for i in range(3):          # num_res_blocks+1
                skip = skips.pop() if skips else torch.zeros_like(h)
                # Resize skip if needed
                if skip.shape[2:] != h.shape[2:]:
                    skip = F.interpolate(skip, size=h.shape[2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
                block = self.dec_blocks[dec_idx]
                dec_idx += 1
                if isinstance(block, ResBlock):
                    h = block(h, t_emb)
                elif isinstance(block, AttentionBlock):
                    h = block(h)
            if mult != 1:
                try:
                    h = next(up_iter)(h)
                except StopIteration:
                    pass

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


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
        image  = image.to(self.device)
        mask   = obj_mask.to(self.device)
        B, C, H, W = image.shape

        # Scale to [-1, 1] for diffusion
        x0_scaled  = image * 2 - 1

        # Class conditioning
        cid   = torch.full((B,), target_class, dtype=torch.long, device=self.device)
        c_emb = self.class_emb(cid)                   # (B, emb_dim)

        # Start from partially noised x0 (inpainting style)
        start_t = min(self.inpaint_start_t, n_steps)

        # If GAN prior images given, blend them into the masked region as a style hint
        if gan_prior_images is not None:
            prior = gan_prior_images.mean(dim=0, keepdim=True).to(self.device)
            prior_up = F.interpolate(prior, size=(H, W), mode="bilinear", align_corners=False)
            prior_scaled = prior_up * 2 - 1
            # Composite: prior in object region, original elsewhere
            x0_scaled = mask * prior_scaled + (1 - mask) * x0_scaled

        # Reverse diffusion with inpainting mask
        def masked_guidance(x_t, t_val):
            """Re-compose mask at each step to keep background clean."""
            t_tensor = torch.full((B,), t_val, dtype=torch.long, device=self.device)
            # Keep background from original
            bg_noise, _ = self.scheduler.q_sample(x0_scaled, t_tensor)
            x_t = mask * x_t + (1 - mask) * bg_noise
            if guidance_fn is not None:
                return guidance_fn(x_t, t_val)
            return None

        edited_scaled = self.scheduler.p_sample_loop(
            model         = self.unet,
            shape         = (B, C, H, W),
            cond_emb      = c_emb,
            device        = str(self.device),
            guidance_fn   = masked_guidance,
            guidance_scale= guidance_scale,
            start_t       = start_t,
            x_start       = x0_scaled,
        )

        # Final mask composition (clean background boundary)
        edited_scaled = mask * edited_scaled + (1 - mask) * x0_scaled

        # Back to [0, 1]
        edited = (edited_scaled + 1) / 2
        return edited.clamp(0, 1)


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
    def apply_analytic_degradation(
        image: torch.Tensor,
        ctx_mask: torch.Tensor,
        obj_mask: torch.Tensor,
        mode: str = "night",
        strength: float = 0.3,
    ) -> torch.Tensor:
        """
        Fast analytic context degradation (no diffusion required).
        Use as baseline or when diffusion model is not yet trained.

        Modes:
          "night"        : darken context region
          "low_contrast" : blend toward local mean
          "blur"         : Gaussian blur in context region
          "clutter"      : add Gaussian noise to context
        """
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
            import torchvision.transforms.functional as TF
            import cv2, numpy as np
            img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            blurred = cv2.GaussianBlur(img_np, (15, 15), 5)
            blurred_t = torch.from_numpy(blurred).permute(2, 0, 1).unsqueeze(0)
            pure_3 = pure.expand_as(img)
            img = pure_3 * blurred_t * strength + img * (1 - pure_3 * strength)

        elif mode == "clutter":
            noise = torch.randn_like(img) * 0.1 * strength
            img = (img + pure * noise).clamp(0, 1)

        return img.clamp(0, 1)
