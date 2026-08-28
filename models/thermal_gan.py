"""
Stage 1 & 2: Class-Conditional Thermal GAN
- Generator: noise + class embedding -> thermal crop
- Discriminator: crop -> real/fake + class
- Provides class-prior features and latent transition directions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from pathlib import Path
from collections import OrderedDict

# ── Building blocks ──────────────────────────────────────────────────────────

class ConditionalBatchNorm2d(nn.Module):
    """BatchNorm whose affine params are conditioned on class embedding."""

    def __init__(self, num_features: int, num_classes: int, emb_dim: int = 64):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma_fc = nn.Linear(emb_dim, num_features)
        self.beta_fc  = nn.Linear(emb_dim, num_features)

    def forward(self, x, class_emb):
        out    = self.bn(x)
        gamma  = self.gamma_fc(class_emb).unsqueeze(-1).unsqueeze(-1)
        beta   = self.beta_fc(class_emb).unsqueeze(-1).unsqueeze(-1)
        return gamma * out + beta


class ResBlockGen(nn.Module):
    """Residual block for generator with conditional BN."""

    def __init__(self, ch: int, num_classes: int, emb_dim: int = 64, upsample: bool = False):
        super().__init__()
        self.upsample = upsample
        self.conv1  = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv2  = nn.Conv2d(ch, ch, 3, 1, 1)
        self.cbn1   = ConditionalBatchNorm2d(ch, num_classes, emb_dim)
        self.cbn2   = ConditionalBatchNorm2d(ch, num_classes, emb_dim)

    def forward(self, x, emb):
        h = F.relu(self.cbn1(x, emb))
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = F.relu(self.cbn2(h, emb))
        h = self.conv2(h)
        return h + x


class ResBlockDis(nn.Module):
    """Residual block for discriminator (spectral norm)."""

    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False):
        super().__init__()
        self.downsample = downsample
        self.conv1  = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.conv2  = nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.skip   = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

    def forward(self, x):
        h = F.leaky_relu(x, 0.2)
        h = self.conv1(h)
        h = F.leaky_relu(h, 0.2)
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
            x = F.avg_pool2d(self.skip(x), 2)
        else:
            x = self.skip(x)
        return h + x


# ── Generator ─────────────────────────────────────────────────────────────────

class ThermalGenerator(nn.Module):
    """
    Class-conditional GAN generator.
    z: (B, z_dim)  class_id: (B,)  ->  image: (B, 3, crop_size, crop_size)

    Architecture: z -> linear -> (4x4 base) -> 4 upsample ResBlocks -> RGB
    For crop_size=64: 4->8->16->32->64
    """

    def __init__(
        self,
        num_classes: int = 4,
        z_dim: int = 128,
        ch: int = 256,
        emb_dim: int = 64,
        out_channels: int = 3,
    ):
        super().__init__()
        self.z_dim   = z_dim
        self.ch      = ch
        self.emb_dim = emb_dim

        self.class_emb = nn.Embedding(num_classes, emb_dim)
        self.z_proj    = nn.Linear(z_dim + emb_dim, ch * 4 * 4)

        self.blocks = nn.ModuleList([
            ResBlockGen(ch, num_classes, emb_dim, upsample=True),   # 4->8
            ResBlockGen(ch, num_classes, emb_dim, upsample=True),   # 8->16
            ResBlockGen(ch, num_classes, emb_dim, upsample=True),   # 16->32
            ResBlockGen(ch, num_classes, emb_dim, upsample=True),   # 32->64
        ])

        self.bn_out  = nn.BatchNorm2d(ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, 1, 1)

    def forward(self, z: torch.Tensor, class_id: torch.Tensor) -> torch.Tensor:
        emb = self.class_emb(class_id)                 # (B, emb_dim)
        h   = torch.cat([z, emb], dim=1)              # (B, z_dim+emb_dim)
        h   = self.z_proj(h).view(-1, self.ch, 4, 4)  # (B, ch, 4, 4)
        for block in self.blocks:
            h = block(h, emb)
        h = F.relu(self.bn_out(h))
        h = torch.tanh(self.conv_out(h))               # (B, 3, 64, 64) in [-1,1]
        return h

    @torch.no_grad()
    def get_class_prior(self, class_id: int, n_samples: int = 64, device="cpu"):
        """Sample `n_samples` crops for a given class -> mean embedding in feature space."""
        c = torch.full((n_samples,), class_id, dtype=torch.long, device=device)
        z = torch.randn(n_samples, self.z_dim, device=device)
        imgs = self.forward(z, c)       # (n, 3, H, W)
        return imgs


# ── Discriminator ─────────────────────────────────────────────────────────────

class ThermalDiscriminator(nn.Module):
    """
    Projection discriminator with spectral normalization.
    Input: (B, 3, crop_size, crop_size)  Output: (B,) real/fake logit
    """

    def __init__(self, num_classes: int = 4, ch_base: int = 64):
        super().__init__()
        # 64->32->16->8->4->pool
        self.blocks = nn.Sequential(
            ResBlockDis(3,           ch_base,     downsample=True),   # 64->32
            ResBlockDis(ch_base,     ch_base * 2, downsample=True),   # 32->16
            ResBlockDis(ch_base * 2, ch_base * 4, downsample=True),   # 16->8
            ResBlockDis(ch_base * 4, ch_base * 8, downsample=True),   # 8->4
        )
        self.fc         = nn.utils.spectral_norm(nn.Linear(ch_base * 8, 1))
        self.class_emb  = nn.utils.spectral_norm(nn.Embedding(num_classes, ch_base * 8))

    def forward(self, x: torch.Tensor, class_id: torch.Tensor) -> torch.Tensor:
        h   = self.blocks(x)
        h   = F.relu(h)
        h   = h.mean(dim=[2, 3])              # global average pool (B, ch)
        out = self.fc(h)                       # (B, 1)
        # projection conditioning
        emb = self.class_emb(class_id)        # (B, ch)
        out = out + (h * emb).sum(dim=1, keepdim=True)
        return out.squeeze(1)                  # (B,)


# ── Loss functions ────────────────────────────────────────────────────────────

def hinge_loss_gen(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def hinge_loss_dis(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


# ── Trainer ───────────────────────────────────────────────────────────────────

class ThermalGANTrainer:
    """
    Trains the class-conditional GAN on FLIR object crops.

    Usage:
        trainer = ThermalGANTrainer(num_classes=4, device='cuda')
        trainer.train(crop_loader, epochs=100)
        trainer.save('checkpoints/thermal_gan.pt')
    """

    def __init__(
        self,
        num_classes: int = 4,
        z_dim: int = 128,
        lr_g: float = 1e-4,
        lr_d: float = 2e-3,
        n_critic: int = 4,
        device: str = "cuda",
    ):
        # Safely resolve device — fall back to CPU if CUDA unavailable
        if device == "cuda" and not torch.cuda.is_available():
            print("[ThermalGANTrainer] CUDA not available, falling back to CPU.")
            device = "cpu"
        try:
            self.device = torch.device(device)
            # Force-initialise to catch driver errors early
            torch.zeros(1, device=self.device)
        except RuntimeError as e:
            print(f"[ThermalGANTrainer] Device '{device}' failed ({e}), "
                  f"falling back to CPU.")
            self.device = torch.device("cpu")

        self.n_critic = n_critic
        self.z_dim    = z_dim

        self.G = ThermalGenerator(num_classes=num_classes, z_dim=z_dim).to(self.device)
        self.D = ThermalDiscriminator(num_classes=num_classes).to(self.device)

        self.opt_G = Adam(self.G.parameters(), lr=lr_g, betas=(0.0, 0.999))
        self.opt_D = Adam(self.D.parameters(), lr=lr_d, betas=(0.0, 0.999))

        self.history = {"d_loss": [], "g_loss": [], "d_real": [], "d_fake": []}
        print(f"[ThermalGANTrainer] Running on {self.device}")

    def train_step(self, real_crops: torch.Tensor, labels: torch.Tensor):
        B = real_crops.size(0)
        real_crops = real_crops.to(self.device)
        labels     = labels.to(self.device)

        # Scale real images to [-1, 1]
        real_scaled = real_crops * 2 - 1

        # ── Train D (n_critic steps) ──
        d_loss_acc = 0.0
        for _ in range(self.n_critic):
            z = torch.randn(B, self.z_dim, device=self.device)
            with torch.no_grad():
                fake = self.G(z, labels)
            real_logits = self.D(real_scaled, labels)
            D_real = torch.sigmoid(real_logits).mean().item()
            fake_logits = self.D(fake.detach(), labels)
            D_fake = torch.sigmoid(fake_logits).mean().item()
            d_loss = hinge_loss_dis(real_logits, fake_logits)

            self.opt_D.zero_grad()
            d_loss.backward()
            self.opt_D.step()
            d_loss_acc += d_loss.item()

        # ── Train G ──
        z = torch.randn(B, self.z_dim, device=self.device)
        fake       = self.G(z, labels)
        fake_logits = self.D(fake, labels)
        g_loss     = hinge_loss_gen(fake_logits)

        self.opt_G.zero_grad()
        g_loss.backward()
        self.opt_G.step()

        return {
            'd_loss': d_loss_acc / self.n_critic,
            'g_loss': g_loss.item(),
            'd_real': D_real,
            'd_fake': D_fake,
        }

    def train(self, loader, epochs: int = 100, log_every: int = 50):
        print(f"[GAN] Training for {epochs} epochs on {self.device}")
        for epoch in range(epochs):
            epoch_metrics = {k: [] for k in self.history.keys()}
         
            for batch in loader:
                crops, labels = batch
                metrics = self.train_step(crops, labels)
                for k, v in metrics.items():
                    epoch_metrics[k].append(v)

            # Average metrics for this epoch
            for k in ['d_loss', 'g_loss', 'd_real', 'd_fake']:
                self.history[k].append(np.mean(epoch_metrics[k]))

            if (epoch + 1) % log_every == 0:
                print(f"  Epoch {epoch+1:>4d}/{epochs}  "
                      f"D={self.history['d_loss'][-1]:.4f}  "
                      f"G={self.history['g_loss'][-1]:.4f}  "
                      f"D(real)={self.history['d_real'][-1]:.3f}  "
                      f"D(fake)={self.history['d_fake'][-1]:.3f}")

        print("[GAN] Training complete.")

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
        }, path)
        print(f"[GAN] Saved checkpoint -> {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.D.load_state_dict(ckpt["D"])
        print(f"[GAN] Loaded checkpoint <- {path}")


# ── Class-Prior Extractor (Stage 2) ──────────────────────────────────────────

class ThermalClassPrior:
    """
    Wraps a trained Generator to extract class-specific feature priors
    and latent transition directions for the attack pipeline.
    """

    def __init__(
        self,
        generator: ThermalGenerator,
        feature_extractor: nn.Module,
        num_classes: int = 4,
        z_dim: int = 128,
        device: str = "cuda",
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.G    = generator.eval()
        self.feat = feature_extractor.eval()
        self.num_classes = num_classes
        self.z_dim  = z_dim
        self.device = torch.device(device)
        self.G.to(self.device)
        self.feat.to(self.device)
        self._priors: dict  = {}
        self._latents: dict = {}

    @torch.no_grad()
    def build_priors(self, n_samples: int = 512):
        """Compute mean feature embedding per class from GAN samples."""
        from collections import OrderedDict
        
        for c in range(self.num_classes):
            cid = torch.full((n_samples,), c, dtype=torch.long, device=self.device)
            z   = torch.randn(n_samples, self.z_dim, device=self.device)
            imgs = self.G(z, cid)
            imgs_01 = (imgs + 1) / 2
            
            feats = self.feat(imgs_01)
            
            # Handle OrderedDict from ResNet50-FPN
            if isinstance(feats, (dict, OrderedDict)):
                feat_key = list(feats.keys())[-1]
                feats = feats[feat_key]
                if c == 0:
                    print(f"  [Prior] Using feature level '{feat_key}' from backbone")
            
            if feats.dim() > 2:
                feats = feats.mean(dim=[2, 3])
            
            self._priors[c] = feats.mean(dim=0).cpu()
            self._latents[c] = z.mean(dim=0).cpu()
            print(f"  [Prior] class {c}: mean feat shape = {self._priors[c].shape}")

    def get_prior(self, class_id: int) -> torch.Tensor:
        """Return mean feature prior for `class_id`."""
        if class_id not in self._priors:
            raise RuntimeError("Call build_priors() first.")
        return self._priors[class_id].to(self.device)

    def get_transition_direction(
        self,
        src_class: int,
        tgt_class: int,
    ) -> torch.Tensor:
        """
        Returns the latent direction from src -> tgt class.
        Used to guide diffusion inpainting toward the target semantic.
        """
        src = self.get_prior(src_class)
        tgt = self.get_prior(tgt_class)
        direction = tgt - src
        direction = F.normalize(direction, dim=0)
        return direction

    def sample_target_images(
        self,
        target_class: int,
        n: int = 16,
    ) -> torch.Tensor:
        """Sample GAN images of target class for visual reference / style transfer."""
        cid = torch.full((n,), target_class, dtype=torch.long, device=self.device)
        z   = torch.randn(n, self.z_dim, device=self.device)
        with torch.no_grad():
            imgs = self.G(z, cid)
        return (imgs + 1) / 2   # [0,1]
