import math
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.dataset import LowLightDataset
from src.model   import DCENet
from src.losses  import (SpatialConsistencyLoss, ExposureControlLoss,
                          ColorConstancyLoss, IlluminationSmoothnessLoss)

# ── 超参数 ────────────────────────────────────────────────────────
EPOCHS        = 30
BATCH_SIZE    = 8
LR            = 1e-4
PATCH_SIZE    = 256
VOC_IMGS      = "../segformer-voc/data/VOCdevkit/VOC2012/JPEGImages"
DEVICE        = "mps" if torch.backends.mps.is_available() else "cpu"
CKPT_EVERY    = 5          # 每 5 epoch 保存一次 checkpoint
CKPT_PATH     = "checkpoint.pth"

# Loss 权重（来自原论文）
W_SPA = 1
W_EXP = 10
W_COL = 5
W_TV  = 20


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 20 * math.log10(1.0 / math.sqrt(mse.item())) if mse > 0 else float("inf")


if __name__ == "__main__":
    train_ds = LowLightDataset(VOC_IMGS, patch_size=PATCH_SIZE, split="train")
    val_ds   = LowLightDataset(VOC_IMGS, patch_size=PATCH_SIZE, split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False, num_workers=2)

    model     = DCENet().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    start_epoch = 1

    # 有 checkpoint 则从断点继续
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"从 checkpoint 恢复，继续从 epoch {start_epoch} 开始")

    L_spa = SpatialConsistencyLoss().to(DEVICE)
    L_exp = ExposureControlLoss().to(DEVICE)
    L_col = ColorConstancyLoss()
    L_tv  = IlluminationSmoothnessLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"device: {DEVICE} | params: {total_params:,} | train: {len(train_ds)} | val: {len(val_ds)}")
    print(f"loss weights → spa:{W_SPA}  exp:{W_EXP}  col:{W_COL}  tv:{W_TV}")

    for epoch in range(start_epoch, EPOCHS + 1):

        model.train()
        total_loss = total_spa = total_exp = total_col = total_tv = 0.0
        for dark, _ in train_loader:
            dark = dark.to(DEVICE)
            enhanced, alphas = model(dark)

            l_spa = L_spa(enhanced, dark)
            l_exp = L_exp(enhanced)
            l_col = L_col(enhanced)
            l_tv  = L_tv(alphas)
            loss  = W_SPA*l_spa + W_EXP*l_exp + W_COL*l_col + W_TV*l_tv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_spa  += (W_SPA * l_spa).item()
            total_exp  += (W_EXP * l_exp).item()
            total_col  += (W_COL * l_col).item()
            total_tv   += (W_TV  * l_tv).item()

        avg_loss = total_loss / len(train_loader)
        avg_spa  = total_spa  / len(train_loader)
        avg_exp  = total_exp  / len(train_loader)
        avg_col  = total_col  / len(train_loader)
        avg_tv   = total_tv   / len(train_loader)

        # 验证：用 GT clean 计算 PSNR（仅评估用，训练不依赖）
        model.eval()
        psnr_list = []
        with torch.no_grad():
            for dark, clean in val_loader:
                dark, clean = dark.to(DEVICE), clean.to(DEVICE)
                enhanced, _ = model(dark)
                psnr_list.append(psnr(enhanced, clean))

        val_psnr = sum(psnr_list) / len(psnr_list)
        print(f"Epoch {epoch:02d}/{EPOCHS} | loss: {avg_loss:.4f} | val PSNR: {val_psnr:.2f} dB")
        print(f"  spa:{avg_spa:.4f}  exp:{avg_exp:.4f}  col:{avg_col:.4f}  tv:{avg_tv:.4f}")

        if epoch % CKPT_EVERY == 0:
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()}, CKPT_PATH)
            print(f"  → checkpoint 已保存（epoch {epoch}）")

    torch.save(model.state_dict(), "zerodce.pth")
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)
    print("训练完成，模型已保存：zerodce.pth")
