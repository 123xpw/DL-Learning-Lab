import os
import math
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T


def synthesize_haze(img_tensor, t=None, A=1.0):
    """
    用大气散射模型合成有雾图像。

    I = J * t + A * (1 - t)

    img_tensor : (3, H, W) 清晰图，值域 [0,1]
    t          : 透射率，None 时随机取 [0.3, 0.8]
    A          : 大气光强度，默认 1.0（纯白雾）
    """
    if t is None:
        t = random.uniform(0.3, 0.8)
    hazy = img_tensor * t + A * (1 - t)
    return hazy.clamp(0, 1), t


class HazeDataset(Dataset):
    def __init__(self, img_dir, patch_size=256, split="train"):
        self.patch_size = patch_size
        self.split      = split

        paths = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith(".jpg")
        ])

        # 90% 训练，10% 验证
        split_idx = int(len(paths) * 0.9)
        if split == "train":
            self.paths = paths[:split_idx]
        else:
            self.paths = paths[split_idx:]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")

        # 图片小于 patch_size 时先放大，避免裁剪报错
        # 用 ceil 向上取整，防止 int() 截断后仍差 1 像素
        w, h = img.size
        if w < self.patch_size or h < self.patch_size:
            scale = self.patch_size / min(w, h)
            img = img.resize((math.ceil(w * scale), math.ceil(h * scale)), Image.BICUBIC)

        # 裁剪
        if self.split == "train":
            i, j, h, w = T.RandomCrop.get_params(img, (self.patch_size, self.patch_size))
            clean = TF.crop(img, i, j, h, w)
            if random.random() > 0.5:
                clean = TF.hflip(clean)
        else:
            clean = TF.center_crop(img, self.patch_size)

        clean = TF.to_tensor(clean)          # (3, H, W) 清晰图
        # val 集用固定 t=0.5，确保每次评估结果可比较
        fixed_t = None if self.split == "train" else 0.5
        hazy, t = synthesize_haze(clean, t=fixed_t)

        return hazy, clean
