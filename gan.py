"""
thermal_mirage/gan.py
Stage 1 – Class-conditional thermal object GAN.
Stage 2 – Class-transition prior extraction.

Architecture: lightweight DC-GAN with class embedding injection
              (suitable for 96×96 thermal crops).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path

from config import (
    FLIR_NUM_CLASSES, GAN_LATENT_DIM, GAN_EMBED_DIM,
    GAN_CROP_SIZE, GAN_LR_G, GAN_LR_D, GAN_BATCH_SIZE,
    GAN_TRAIN_EPOCHS, DEVICE,
)


# ─── Generator ───────────────────────────────────────────────────────────────

class ThermalGenerator(nn.Module):
    """
    z (latent_dim,) + class embedding → (1, H, W) thermal crop in [-1,1].
    Architecture: MLP projection → series of transposed convolutions.
    """

    def __init__(self, latent_dim=GAN_LATENT_DIM,
                 embed_dim=GAN_EMBED_DIM,
                 num_classes=FLIR_NUM_CLASSES,
                 crop_size=GAN_CROP_SIZE):
        super().__init__()
        self.latent_dim = latent_dim
        self.crop_size  = crop_size

        # Class embedding
        self.class_embed = nn.Embedding(num_classes, embed_dim)

        in_ch = latent_dim + embed_dim
        # 6×6 → 12 → 24 → 48 → 96
        self.init_size = crop_size // 16   # 6 for 96px
        self.proj = nn.Linear(in_ch, 512 * self.init_size * self.init_size)

        self.net = nn.Sequential(
            nn.BatchNorm2d(512),
            # 6 → 12
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 12 → 24
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 24 → 48
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 48 → 96
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c   = self.class_embed(labels)           # (B, embed_dim)
        inp = torch.cat([z, c], dim=1)           # (B, latent+embed)
        x   = self.proj(inp)
        x   = x.view(x.size(0), 512,
                     self.init_size, self.init_size)
        return self.net(x)                       # (B, 1, H, W)

    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        """Convenience: sample from N(0,1)."""
        z = torch.randn(labels.size(0), self.latent_dim, device=labels.device)
        return self.forward(z, labels)


# ─── Discriminator ───────────────────────────────────────────────────────────

class ThermalDiscriminator(nn.Module):
    """
    (1, H, W) crop + class label → real/fake score.
    Projection discriminator style.
    """

    def __init__(self, embed_dim=GAN_EMBED_DIM,
                 num_classes=FLIR_NUM_CLASSES,
                 crop_size=GAN_CROP_SIZE):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, embed_dim)

        # 96 → 48 → 24 → 12 → 6
        self.feat = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )
        feat_spatial = crop_size // 16   # 6 for 96px
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(512 + embed_dim, 1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        h = self.feat(x)
        h = self.pool(h).view(h.size(0), -1)    # (B, 512)
        c = self.class_embed(labels)             # (B, embed_dim)
        return self.fc(torch.cat([h, c], dim=1)).squeeze(1)


# ─── Training ─────────────────────────────────────────────────────────────────

class ThermalGANTrainer:
    def __init__(self, save_dir: str = "checkpoints"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.G = ThermalGenerator().to(DEVICE)
        self.D = ThermalDiscriminator().to(DEVICE)

        self.opt_G = optim.Adam(self.G.parameters(),
                                lr=GAN_LR_G, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(),
                                lr=GAN_LR_D, betas=(0.5, 0.999))
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self, dataloader: DataLoader):
        print(f"Training thermal GAN on {DEVICE} …")
        for epoch in range(GAN_TRAIN_EPOCHS):
            g_losses, d_losses = [], []

            for real_crops, labels in dataloader:
                real_crops = real_crops.to(DEVICE)
                labels     = labels.to(DEVICE)
                B          = real_crops.size(0)
                real_lbl   = torch.ones(B, device=DEVICE)
                fake_lbl   = torch.zeros(B, device=DEVICE)

                # ── Discriminator step ──────────────────────────────────────
                self.opt_D.zero_grad()
                z         = torch.randn(B, GAN_LATENT_DIM, device=DEVICE)
                fake_crops = self.G(z, labels).detach()
                d_real = self.D(real_crops, labels)
                d_fake = self.D(fake_crops, labels)
                d_loss = (self.criterion(d_real, real_lbl) +
                          self.criterion(d_fake, fake_lbl)) / 2
                d_loss.backward()
                self.opt_D.step()

                # ── Generator step ──────────────────────────────────────────
                self.opt_G.zero_grad()
                z         = torch.randn(B, GAN_LATENT_DIM, device=DEVICE)
                fake_crops = self.G(z, labels)
                g_loss = self.criterion(self.D(fake_crops, labels), real_lbl)
                g_loss.backward()
                self.opt_G.step()

                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d} | "
                      f"G: {sum(g_losses)/len(g_losses):.4f}  "
                      f"D: {sum(d_losses)/len(d_losses):.4f}")

        self.save()
        print("GAN training complete.")

    def save(self):
        torch.save(self.G.state_dict(), self.save_dir / "gan_G.pt")
        torch.save(self.D.state_dict(), self.save_dir / "gan_D.pt")
        print(f"Saved GAN to {self.save_dir}")

    def load(self):
        self.G.load_state_dict(torch.load(
            self.save_dir / "gan_G.pt", map_location=DEVICE))
        self.D.load_state_dict(torch.load(
            self.save_dir / "gan_D.pt", map_location=DEVICE))
        self.G.eval()
        self.D.eval()
        print(f"Loaded GAN from {self.save_dir}")


# ─── Stage 2: Class-transition prior extraction ───────────────────────────────

class ClassTransitionPrior:
    """
    Computes a target-class thermal feature prior by aggregating
    latent samples from the GAN generator.
    Also provides latent-space interpolation between class prototypes.
    """

    def __init__(self, generator: ThermalGenerator,
                 n_samples: int = 256):
        self.G        = generator.eval()
        self.n_samples = n_samples

    @torch.no_grad()
    def class_prototype(self, class_id: int) -> torch.Tensor:
        """Mean generated image for a class – acts as a thermal prototype."""
        labels = torch.full((self.n_samples,), class_id,
                            dtype=torch.long, device=DEVICE)
        samples = self.G.sample(labels)       # (N, 1, H, W) in [-1,1]
        return samples.mean(dim=0, keepdim=True)   # (1, 1, H, W)

    @torch.no_grad()
    def latent_transition(self, src_class: int, tgt_class: int,
                          steps: int = 8) -> list:
        """
        Interpolate between mean latent codes of two classes.
        Returns a list of generated images along the path.
        """
        # Sample mean latent for each class
        n    = 128
        lbls = torch.full((n,), src_class, dtype=torch.long, device=DEVICE)
        z_src = torch.randn(n, self.G.latent_dim, device=DEVICE)
        z_src_mean = z_src.mean(0, keepdim=True)

        lbls_t = torch.full((n,), tgt_class, dtype=torch.long, device=DEVICE)
        z_tgt  = torch.randn(n, self.G.latent_dim, device=DEVICE)
        z_tgt_mean = z_tgt.mean(0, keepdim=True)

        alphas = torch.linspace(0, 1, steps)
        frames = []
        src_lbl = torch.tensor([src_class], device=DEVICE)
        tgt_lbl = torch.tensor([tgt_class], device=DEVICE)
        for a in alphas:
            z_interp = (1 - a) * z_src_mean + a * z_tgt_mean
            lbl_use  = src_lbl if a < 0.5 else tgt_lbl
            img      = self.G(z_interp, lbl_use)
            frames.append(img)
        return frames

    @torch.no_grad()
    def target_features(self, target_class_id: int,
                        detector_encoder=None) -> torch.Tensor:
        """
        If a detector encoder is supplied, return mean detector features
        for the target class (used to compute L_prior loss later).
        Otherwise return mean pixel prototype.
        """
        proto = self.class_prototype(target_class_id)
        if detector_encoder is not None:
            with torch.no_grad():
                feats = detector_encoder(proto)
            return feats
        return proto
