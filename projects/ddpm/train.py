import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.noise_scheduler import NoiseScheduler
from src.unet import UNet


DEVICE    = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
T         = 1000   # 总扩散步数
BATCH     = 64
EPOCHS    = 20
LR        = 2e-4
IMG_SIZE  = 28


def get_dataloader():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 像素值从[0,1]映射到[-1,1]
    ])
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)


def train():
    scheduler = NoiseScheduler(T=T)
    model     = UNet(in_ch=1, base_ch=32, time_dim=128, num_classes=11).to(DEVICE)
    # num_classes=11：0-9 是真实类别，10 是"无条件"占位符（CFG 用）
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader    = get_dataloader()

    print(f"device: {DEVICE} | params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for x0, label in loader:
            x0    = x0.to(DEVICE)
            label = label.to(DEVICE)                        # [B] 数字类别 0-9

            t     = torch.randint(0, T, (x0.shape[0],)).to(DEVICE)
            noise = torch.randn_like(x0)
            xt    = scheduler.add_noise(x0, t, noise)

            # 训练时有 20% 概率丢掉类别（CFG：让模型同时学有条件和无条件）
            mask  = (torch.rand(label.shape[0], device=DEVICE) > 0.2)
            label_in = torch.where(mask, label, torch.full_like(label, 10))

            predicted_noise = model(xt, t, label_in)
            loss = nn.functional.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:02d}/{EPOCHS} | loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "ddpm_mnist.pth")
    print("模型已保存：ddpm_mnist.pth")


if __name__ == "__main__":
    train()
