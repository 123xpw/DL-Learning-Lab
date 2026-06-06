import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.dataset import HazeDataset
from src.model import AODNet

import math

def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 20 * math.log10(1.0 / math.sqrt(mse.item())) if mse > 0 else float("inf")

# ── 超参数 ────────────────────────────────────────────────────────
EPOCHS     = 30
BATCH_SIZE = 16
LR         = 1e-4
PATCH_SIZE = 256
VOC_IMGS   = "../segformer-voc/data/VOCdevkit/VOC2012/JPEGImages"
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"

if __name__ == "__main__":

    train_dataset = HazeDataset(VOC_IMGS, patch_size=PATCH_SIZE, split="train")
    val_dataset   = HazeDataset(VOC_IMGS, patch_size=PATCH_SIZE, split="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=1,          shuffle=False, num_workers=2)

    model     = AODNet().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"device: {DEVICE} | params: {total_params:,} | train: {len(train_dataset)} | val: {len(val_dataset)}")

    for epoch in range(1, EPOCHS + 1):

        model.train()
        total_loss = 0.0
        for hazy, clean in train_loader:
            hazy, clean = hazy.to(DEVICE), clean.to(DEVICE)

            pred  = model(hazy)
            loss  = criterion(pred, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        psnr_list = []
        with torch.no_grad():
            for hazy, clean in val_loader:
                hazy, clean = hazy.to(DEVICE), clean.to(DEVICE)
                pred = model(hazy)
                psnr_list.append(psnr(pred.squeeze(0), clean.squeeze(0)))

        print(f"Epoch {epoch:02d}/{EPOCHS} | loss: {avg_loss:.6f} | val PSNR: {sum(psnr_list)/len(psnr_list):.2f} dB")

    torch.save(model.state_dict(), "aodnet.pth")
    print("模型已保存：aodnet.pth")
