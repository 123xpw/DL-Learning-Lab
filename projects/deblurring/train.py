import math
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import BlurDataset
from src.model   import SimpleUNet

# ── 超参数 ────────────────────────────────────────────────────────────
EPOCHS     = 30
BATCH_SIZE = 8
LR         = 1e-3
PATCH_SIZE = 128
BASE       = 16
VOC_IMGS   = "../segformer-voc/data/VOCdevkit/VOC2012/JPEGImages"
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
CKPT_EVERY = 5
CKPT_PATH  = "checkpoint.pth"


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 20 * math.log10(1.0 / math.sqrt(mse.item())) if mse > 0 else float("inf")


if __name__ == "__main__":
    train_ds = BlurDataset(VOC_IMGS, patch_size=PATCH_SIZE, split="train")
    val_ds   = BlurDataset(VOC_IMGS, patch_size=PATCH_SIZE, split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False,
                              num_workers=0, pin_memory=False)

    model     = SimpleUNet(base=BASE).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = CharbonnierLoss()
    start_epoch = 1

    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"从 checkpoint 恢复，继续从 epoch {start_epoch} 开始")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"device: {DEVICE} | params: {total_params:,} | train: {len(train_ds)} | val: {len(val_ds)}")

    for epoch in range(start_epoch, EPOCHS + 1):

        # ── 训练 ──────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for blurry, clean in train_loader:
            blurry, clean = blurry.to(DEVICE), clean.to(DEVICE)

            pred = model(blurry)           # 直接预测干净图
            loss = criterion(pred, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # ── 验证 ──────────────────────────────────────────────────────
        model.eval()
        psnr_blurry_list  = []
        psnr_deblur_list  = []
        with torch.no_grad():
            for blurry, clean in val_loader:
                blurry, clean = blurry.to(DEVICE), clean.to(DEVICE)
                pred = model(blurry)
                psnr_blurry_list.append(psnr(blurry, clean))
                psnr_deblur_list.append(psnr(pred,   clean))

        val_blurry = sum(psnr_blurry_list) / len(psnr_blurry_list)
        val_deblur = sum(psnr_deblur_list) / len(psnr_deblur_list)
        delta      = val_deblur - val_blurry

        print(f"Epoch {epoch:02d}/{EPOCHS} | loss: {avg_loss:.4f} "
              f"| blurry: {val_blurry:.2f} dB → deblurred: {val_deblur:.2f} dB "
              f"| delta: {delta:+.2f} dB")

        if epoch % CKPT_EVERY == 0:
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, CKPT_PATH)
            print(f"  → checkpoint 已保存（epoch {epoch}）")

    torch.save(model.state_dict(), "deblurring.pth")
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)
    print("训练完成，模型已保存：deblurring.pth")
