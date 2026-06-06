import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import SegFormer
from src.dataset import VOC2012Dataset, NUM_CLASSES, IGNORE_INDEX

# ── 超参数 ────────────────────────────────────────────────────────────────
EPOCHS     = 40
BATCH_SIZE = 4
LR         = 6e-5       # AdamW + SegFormer 论文推荐值
WEIGHT_DECAY = 0.01
IMG_SIZE   = 512
DATA_DIR   = "./data"   # 本地改为 "../segformer-voc/data"
CKPT_EVERY = 5
CKPT_PATH  = "checkpoint.pth"

# MPS 的 Transformer backward 有已知兼容问题，本地用 CPU 验证
# 完整训练请上传到 Colab 使用 GPU
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# ── mIoU 计算 ─────────────────────────────────────────────────────────────
def compute_miou(inter, union):
    valid = union > 0
    if valid.sum() == 0:
        return 0.0
    return (inter[valid] / (union[valid] + 1e-10)).mean().item()


def update_iou_stats(preds_argmax, targets, inter, union):
    valid = targets != IGNORE_INDEX
    for c in range(NUM_CLASSES):
        pred_c   = preds_argmax == c
        target_c = targets == c
        inter[c] += (pred_c & target_c & valid).sum().item()
        union[c] += ((pred_c | target_c) & valid).sum().item()


# ── 主训练循环 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_ds = VOC2012Dataset(DATA_DIR, split="train", size=IMG_SIZE)
    val_ds   = VOC2012Dataset(DATA_DIR, split="val",   size=IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=2,           shuffle=False,
                              num_workers=0, pin_memory=False)

    model     = SegFormer(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Poly LR schedule：lr = lr0 * (1 - epoch/max_epoch)^0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: (1 - ep / EPOCHS) ** 0.9
    )
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    start_epoch = 1

    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"从 checkpoint 恢复，继续从 epoch {start_epoch}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"device: {DEVICE} | params: {total_params:,} | "
          f"train: {len(train_ds)} | val: {len(val_ds)}")

    best_miou = 0.0

    for epoch in range(start_epoch, EPOCHS + 1):

        # ── 训练 ──────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            logits = model(imgs)                    # [B, 21, H, W]
            loss   = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        cur_lr   = optimizer.param_groups[0]['lr']

        # ── 验证 ──────────────────────────────────────────────────────
        model.eval()
        inter = torch.zeros(NUM_CLASSES)
        union = torch.zeros(NUM_CLASSES)

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                logits = model(imgs)
                preds  = logits.argmax(1).cpu()
                update_iou_stats(preds, masks.cpu(), inter, union)

        miou = compute_miou(inter, union)
        best_miou = max(best_miou, miou)

        print(f"Epoch {epoch:02d}/{EPOCHS} | loss: {avg_loss:.4f} "
              f"| mIoU: {miou:.4f} | best: {best_miou:.4f} | lr: {cur_lr:.2e}")

        if epoch % CKPT_EVERY == 0:
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, CKPT_PATH)
            print(f"  → checkpoint 已保存（epoch {epoch}）")

    torch.save(model.state_dict(), "segformer_scratch.pth")
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)
    print(f"训练完成。最佳 mIoU: {best_miou:.4f}")
