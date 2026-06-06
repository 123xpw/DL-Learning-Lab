import os
import math
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T


def synthesize_dark(img_tensor, gamma=None):
    """
    用 gamma 矫正模拟低光照：img^gamma
    gamma > 1 → 变暗；gamma 越大越暗。
    """
    if gamma is None:
        gamma = random.uniform(2.0, 5.0)
    return img_tensor ** gamma, gamma


class LowLightDataset(Dataset):
    def __init__(self, img_dir, patch_size=256, split="train"):
        self.patch_size = patch_size
        self.split      = split

        paths = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith(".jpg")
        ])

        split_idx = int(len(paths) * 0.9)
        self.paths = paths[:split_idx] if split == "train" else paths[split_idx:]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")

        w, h = img.size
        if w < self.patch_size or h < self.patch_size:
            scale = self.patch_size / min(w, h)
            img = img.resize((math.ceil(w * scale), math.ceil(h * scale)), Image.BICUBIC)

        if self.split == "train":
            i, j, h, w = T.RandomCrop.get_params(img, (self.patch_size, self.patch_size))
            clean = TF.crop(img, i, j, h, w)
            if random.random() > 0.5:
                clean = TF.hflip(clean)
        else:
            clean = TF.center_crop(img, self.patch_size)

        clean = TF.to_tensor(clean)

        # 验证集用固定 gamma=3.0，保证评估可复现
        fixed_gamma = None if self.split == "train" else 3.0
        dark, _ = synthesize_dark(clean, gamma=fixed_gamma)

        return dark, clean   # 训练时只用 dark；clean 仅用于验证 PSNR
