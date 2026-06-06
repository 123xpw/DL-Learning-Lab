import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # MPS 不支持的算子自动回退到 CPU

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.dataset import VOC2012Dataset, IGNORE_INDEX
from src.model import build_model
from src.evaluate import compute_miou

# ── 超参数 ────────────────────────────────────────────────────────
EPOCHS     = 20
BATCH_SIZE = 4
LR         = 6e-5
IMG_SIZE   = 512
DEVICE     = "cpu"  # MPS 反向传播存在非连续张量 bug，暂用 CPU

# macOS 的多进程机制（spawn）要求训练代码放在这个块里
# 否则 DataLoader 的 worker 进程启动时会反复 import 主脚本导致崩溃
if __name__ == "__main__":

    # ── 数据 ──────────────────────────────────────────────────────
    train_dataset = VOC2012Dataset("./data", split="train", size=IMG_SIZE)
    val_dataset   = VOC2012Dataset("./data", split="val",   size=IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=1,          shuffle=False, num_workers=2)

    # ── 模型 ──────────────────────────────────────────────────────
    model = build_model().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    print(f"device: {DEVICE} | train: {len(train_dataset)} | val: {len(val_dataset)}")

    # ── 训练循环 ──────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):

        # 训练
        model.train()
        total_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            outputs = model(pixel_values=imgs)
            logits  = outputs.logits  # (B, 21, H/4, W/4)

            # 上采样到原始尺寸，再算 loss
            logits_up = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits_up, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 验证
        model.eval()
        miou_list = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(DEVICE)

                outputs   = model(pixel_values=imgs)
                logits_up = F.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                preds     = logits_up.argmax(dim=1).squeeze(0).cpu()  # (H, W)
                masks     = masks.squeeze(0)                          # (H, W)

                miou_list.append(compute_miou(preds, masks))

        avg_miou = sum(miou_list) / len(miou_list)
        print(f"Epoch {epoch:02d}/{EPOCHS} | loss: {avg_loss:.4f} | val mIoU: {avg_miou:.4f}")

    # ── 保存 ──────────────────────────────────────────────────────
    torch.save(model.state_dict(), "segformer_voc.pth")
    print("模型已保存：segformer_voc.pth")
