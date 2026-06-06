import math
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import RainDataset
from src.model   import SimpleMPRNet
from src.losses  import CharbonnierLoss

# ── 超参数 ────────────────────────────────────────────────────────
EPOCHS     = 30
BATCH_SIZE = 8
LR         = 2e-4
PATCH_SIZE = 128
N_FEAT     = 16
VOC_IMGS   = "../segformer-voc/data/VOCdevkit/VOC2012/JPEGImages"
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
CKPT_EVERY = 5
CKPT_PATH  = "checkpoint.pth"


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 20 * math.log10(1.0 / math.sqrt(mse.item())) if mse > 0 else float("inf")


if __name__ == "__main__":
    train_ds = RainDataset(VOC_IMGS, patch_size=PATCH_SIZE, split="train")
    val_ds   = RainDataset(VOC_IMGS, patch_size=PATCH_SIZE, split="val")

    # num_workers=0: 避免 macOS DataLoader spawn 崩溃
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False,
                              num_workers=0, pin_memory=False)

    model     = SimpleMPRNet(n_feat=N_FEAT).to(DEVICE)
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

        # ── 训练 ────────────────────────────────────────────────
        model.train()
        total_loss = total_l1 = total_l2 = 0.0
        for rainy, clean in train_loader:
            rainy, clean = rainy.to(DEVICE), clean.to(DEVICE)

            pred1, pred2 = model(rainy)
            l1   = criterion(pred1, clean)   # stage 1 supervision
            l2   = criterion(pred2, clean)   # stage 2 supervision
            loss = l1 + l2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_l1   += l1.item()
            total_l2   += l2.item()

        scheduler.step()
        n_batch   = len(train_loader)
        avg_loss  = total_loss / n_batch
        avg_l1    = total_l1   / n_batch
        avg_l2    = total_l2   / n_batch

        # ── 验证 ────────────────────────────────────────────────
        model.eval()
        psnr_list = []
        with torch.no_grad():
            for rainy, clean in val_loader:
                rainy, clean = rainy.to(DEVICE), clean.to(DEVICE)
                _, pred2 = model(rainy)
                psnr_list.append(psnr(pred2, clean))
        val_psnr = sum(psnr_list) / len(psnr_list)

        print(f"Epoch {epoch:02d}/{EPOCHS} | loss: {avg_loss:.4f} | val PSNR: {val_psnr:.2f} dB")
        print(f"  stage1: {avg_l1:.4f}  stage2: {avg_l2:.4f}  lr: {scheduler.get_last_lr()[0]:.2e}")

        if epoch % CKPT_EVERY == 0:
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, CKPT_PATH)
            print(f"  → checkpoint 已保存（epoch {epoch}）")

    torch.save(model.state_dict(), "deraining.pth")
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)
    print("训练完成，模型已保存：deraining.pth")
