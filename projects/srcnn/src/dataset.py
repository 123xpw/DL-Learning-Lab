import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random

class SRDataset(Dataset):
    def __init__(self, img_dir, scale=2, patch_size=128, split="train"):
        """
        img_dir    : 高清图所在目录（用 VOC2012 的 JPEGImages）
        scale      : 放大倍数（2 表示 LR 是 HR 的一半分辨率）
        patch_size : 每次随机裁剪的 HR patch 大小
        split      : "train" 随机裁剪 + 翻转；"val" 固定中心裁剪
        """
        self.scale      = scale
        self.patch_size = patch_size
        self.split      = split

        self.paths = [
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith(".jpg")
        ]
        self.paths.sort()

        # 训练集用前 1300 张，验证集用后 164 张
        if split == "train":
            self.paths = self.paths[:1300]
        else:
            self.paths = self.paths[1300:]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")

        # ── 裁剪出一个 HR patch ──────────────────────────────────
        if self.split == "train":
            # 随机裁剪
            i, j, h, w = T.RandomCrop.get_params(img, (self.patch_size, self.patch_size))
            hr = TF.crop(img, i, j, h, w)
            # 随机水平翻转（数据增强）
            if random.random() > 0.5:
                hr = TF.hflip(hr)
        else:
            # 中心裁剪，结果固定可复现
            hr = TF.center_crop(img, self.patch_size)

        lr_size = self.patch_size // self.scale

        # LR：把 HR 缩小 → 再放大回 patch_size（模拟低清图）
        lr = hr.resize((lr_size, lr_size), Image.BICUBIC)        # 缩小
        lr = lr.resize((self.patch_size, self.patch_size), Image.BICUBIC)  # 放大回来

        # 转 tensor，值域 [0,1]
        hr = TF.to_tensor(hr)  # (3, 128, 128)
        lr = TF.to_tensor(lr)  # (3, 128, 128)  ← 和 HR 同尺寸，但细节模糊

        return lr, hr
