import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.dataset import SRDataset
from src.model import SRCNN
from src.evaluate import psnr

# ── 超参数 ────────────────────────────────────────────────────────
EPOCHS     = 30
BATCH_SIZE = 16
LR         = 1e-4
PATCH_SIZE = 128
SCALE      = 2
VOC_IMGS   = "../segformer-voc/data/VOCdevkit/VOC2012/JPEGImages"
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"

if __name__ == "__main__":

    # ── 数据 ──────────────────────────────────────────────────────
    train_dataset = SRDataset(VOC_IMGS, scale=SCALE, patch_size=PATCH_SIZE, split="train")
    val_dataset   = SRDataset(VOC_IMGS, scale=SCALE, patch_size=PATCH_SIZE, split="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=1,          shuffle=False, num_workers=2)

    # ── 模型 ──────────────────────────────────────────────────────
    model     = SRCNN().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()   # 像素级均方误差

    print(f"device: {DEVICE} | train: {len(train_dataset)} | val: {len(val_dataset)}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ── 训练循环 ──────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):

        model.train()
        total_loss = 0.0
        for lr_imgs, hr_imgs in train_loader:
            lr_imgs = lr_imgs.to(DEVICE)
            hr_imgs = hr_imgs.to(DEVICE)

            preds = model(lr_imgs)
            loss  = criterion(preds, hr_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 验证
        model.eval()
        psnr_list = []
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.to(DEVICE)
                hr_imgs = hr_imgs.to(DEVICE)
                preds   = model(lr_imgs)
                psnr_list.append(psnr(preds.squeeze(0), hr_imgs.squeeze(0)))

        avg_psnr = sum(psnr_list) / len(psnr_list)
        print(f"Epoch {epoch:02d}/{EPOCHS} | loss: {avg_loss:.6f} | val PSNR: {avg_psnr:.2f} dB")

    torch.save(model.state_dict(), "srcnn.pth")
    print("模型已保存：srcnn.pth")
