import math
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import NoisyDataset
from src.model   import DnCNN

# ── 超参数 ────────────────────────────────────────────────────────────
EPOCHS     = 30
BATCH_SIZE = 8
LR         = 1e-3
PATCH_SIZE = 128
N_FEAT     = 64
DEPTH      = 9
VOC_IMGS   = "../segformer-voc/data/VOCdevkit/VOC2012/JPEGImages"
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
CKPT_EVERY = 5
CKPT_PATH  = "checkpoint.pth"


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 20 * math.log10(1.0 / math.sqrt(mse.item())) if mse > 0 else float("inf")


if __name__ == "__main__":
    train_ds = NoisyDataset(VOC_IMGS, patch_size=PATCH_SIZE, split="train")
    val_ds   = NoisyDataset(VOC_IMGS, patch_size=PATCH_SIZE, split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False,
                              num_workers=0, pin_memory=False)

    model     = DnCNN(n_feat=N_FEAT, depth=DEPTH).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.MSELoss()
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
    print(f"val 噪声基线 PSNR ≈ 20.17 dB（sigma=25/255）")

    for epoch in range(start_epoch, EPOCHS + 1):

        # ── 训练 ──────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            actual_noise = noisy - clean          # ground truth noise

            pred_noise = model(noisy)             # 残差学习：预测噪声
            loss = criterion(pred_noise, actual_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # ── 验证 ──────────────────────────────────────────────────────
        model.eval()
        psnr_noisy_list    = []
        psnr_denoised_list = []
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                pred_noise = model(noisy)
                denoised   = (noisy - pred_noise).clamp(0, 1)
                psnr_noisy_list.append(psnr(noisy, clean))
                psnr_denoised_list.append(psnr(denoised, clean))

        val_noisy    = sum(psnr_noisy_list)    / len(psnr_noisy_list)
        val_denoised = sum(psnr_denoised_list) / len(psnr_denoised_list)
        delta        = val_denoised - val_noisy

        print(f"Epoch {epoch:02d}/{EPOCHS} | loss: {avg_loss:.6f} "
              f"| noisy: {val_noisy:.2f} dB → denoised: {val_denoised:.2f} dB "
              f"| delta: {delta:+.2f} dB")

        if epoch % CKPT_EVERY == 0:
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, CKPT_PATH)
            print(f"  → checkpoint 已保存（epoch {epoch}）")

    torch.save(model.state_dict(), "denoising.pth")
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)
    print("训练完成，模型已保存：denoising.pth")
