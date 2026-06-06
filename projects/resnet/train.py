import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms

from src.model import ResNet18

# ── 超参数 ────────────────────────────────────────────────────────────
EPOCHS     = 30
BATCH_SIZE = 128
LR         = 0.1        # SGD + ResNet 标准起始 lr
MOMENTUM   = 0.9
WEIGHT_DECAY = 5e-4
DATA_DIR   = "./data"
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
CKPT_EVERY = 5
CKPT_PATH  = "checkpoint.pth"

# ── 数据增强 ──────────────────────────────────────────────────────────
# 训练集：随机裁剪 + 随机翻转（经典 CIFAR-10 增强）
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),  # CIFAR-10 均值/标准差
])

# 验证集：只做归一化，不做随机增强
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


if __name__ == "__main__":
    train_ds = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_transform)
    val_ds   = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=256,         shuffle=False,
                              num_workers=0, pin_memory=False)

    model     = ResNet18(num_classes=10).to(DEVICE)
    optimizer = SGD(model.parameters(), lr=LR,
                    momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 1

    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"从 checkpoint 恢复，继续从 epoch {start_epoch} 开始")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"device: {DEVICE} | params: {total_params:,} | "
          f"train: {len(train_ds)} | val: {len(val_ds)}")

    for epoch in range(start_epoch, EPOCHS + 1):

        # ── 训练 ──────────────────────────────────────────────────────
        model.train()
        total_loss = correct = total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            logits = model(imgs)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        scheduler.step()
        train_acc  = correct / total * 100
        avg_loss   = total_loss / len(train_loader)

        # ── 验证 ──────────────────────────────────────────────────────
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total * 100

        print(f"Epoch {epoch:02d}/{EPOCHS} | loss: {avg_loss:.4f} "
              f"| train acc: {train_acc:.1f}% | val acc: {val_acc:.1f}%")

        if epoch % CKPT_EVERY == 0:
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, CKPT_PATH)
            print(f"  → checkpoint 已保存（epoch {epoch}）")

    torch.save(model.state_dict(), "resnet18.pth")
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)
    print("训练完成，模型已保存：resnet18.pth")
