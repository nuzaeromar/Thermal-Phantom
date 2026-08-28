import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from data.flir_dataset import build_dataloader, FLIR_NATIVE_SIZE, NUM_CLASSES
from attack.detector_wrapper import FasterRCNNWrapper
from pathlib import Path

device    = "cuda"
data_root = "/home/exouser/Downloads/thermal_mirage/FLIR_ADAS_1_3"
out_dir   = Path("checkpoints")
out_dir.mkdir(exist_ok=True)

# Build detector (head replaced with 5-class: background + 4 FLIR)
detector = FasterRCNNWrapper(num_flir_classes=NUM_CLASSES,
                              pretrained=True, device=device)
detector.model.train()

# Dataloader — use NORMALIZED images for detector training
train_loader = build_dataloader(
    root         = data_root,
    split        = "train",
    batch_size   = 4,
    num_workers  = 4,
    img_size     = FLIR_NATIVE_SIZE,
    normalize    = True,          # use thermal normalization stats
    return_masks = False,
)

optimizer = SGD(detector.model.parameters(), lr=0.005,
                momentum=0.9, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

print(f"Training detector for 20 epochs on {len(train_loader.dataset)} images")

for epoch in range(20):
    epoch_loss = 0.0
    for batch in train_loader:
        images    = list(img.to(device) for img in batch["images"])
        boxes_list  = batch["boxes"]
        labels_list = batch["labels"]

        targets = []
        for boxes, labels in zip(boxes_list, labels_list):
            # Convert normalized xyxy → pixel coords
            H, W = FLIR_NATIVE_SIZE
            b = boxes.clone()
            b[:, [0,2]] *= W
            b[:, [1,3]] *= H
            targets.append({
                "boxes":  b.to(device),
                "labels": (labels + 1).to(device),   # +1 for background class
            })

        loss_dict  = detector.model(images, targets)
        total_loss = sum(loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(detector.model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += total_loss.item()

    scheduler.step()
    avg = epoch_loss / len(train_loader)
    print(f"  Epoch {epoch+1:>3d}/20  loss={avg:.4f}")

torch.save(detector.model.state_dict(),
           str(out_dir / "fasterrcnn_flir.pt"))
print("Saved: checkpoints/fasterrcnn_flir.pt")