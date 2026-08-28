import torch, matplotlib.pyplot as plt
from models.thermal_gan import ThermalGANTrainer, ThermalClassPrior
import torchvision.models as tvm

device = "cuda"
trainer = ThermalGANTrainer(num_classes=4, device=device)
trainer.load("checkpoints/thermal_gan.pt")
trainer.G.eval()

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
class_names = ["person", "bicycle", "car", "dog"]

for c, name in enumerate(class_names):
    cid  = torch.full((8,), c, dtype=torch.long, device=device)
    z    = torch.randn(8, 128, device=device)
    with torch.no_grad():
        imgs = trainer.G(z, cid)          # (8, 3, 64, 64) in [-1,1]
    imgs = (imgs + 1) / 2                 # → [0,1]
    for j in range(8):
        ax = axes[c][j]
        ax.imshow(imgs[j, 0].cpu().numpy(), cmap="inferno")
        ax.axis("off")
        if j == 0:
            ax.set_ylabel(name, fontsize=10)

plt.suptitle("GAN Thermal Priors per Class")
plt.tight_layout()
plt.savefig("checkpoints/gan_priors_check.png", dpi=120)
print("Saved: checkpoints/gan_priors_check.png")
