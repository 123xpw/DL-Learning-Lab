import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def make_motion_kernel(length, angle_deg):
    """
    Line kernel of given length at given angle.
    Returns float32 numpy array [length, length], sum=1.
    """
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    angle  = np.radians(angle_deg)
    for i in range(length):
        offset = i - center
        x = int(round(center + offset * np.cos(angle)))
        y = int(round(center + offset * np.sin(angle)))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    s = kernel.sum()
    return kernel / s if s > 0 else kernel


def apply_blur(img_tensor, kernel_np):
    """
    img_tensor: [3, H, W] float32 in [0, 1]
    kernel_np:  [k, k] float32
    Returns blurred tensor [3, H, W].
    """
    k = kernel_np.shape[0]
    pad = k // 2
    kt = torch.from_numpy(kernel_np).float()
    kt = kt.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)  # [3, 1, k, k]
    blurred = F.conv2d(img_tensor.unsqueeze(0), kt,
                       padding=pad, groups=3).squeeze(0)
    return blurred.clamp(0, 1)


class BlurDataset(Dataset):
    """
    VOC2012 images with synthetic motion blur.
    - split="train": 2000 images, random length/angle each call
    - split="val":   200 images,  fixed blur per index (reproducible PSNR)
    """
    LEN_RANGE   = (7, 21)
    ANGLE_RANGE = (0, 180)

    def __init__(self, voc_root, patch_size=128, split="train", seed=42):
        super().__init__()
        self.patch_size = patch_size
        self.split      = split

        all_imgs = sorted([
            os.path.join(voc_root, f)
            for f in os.listdir(voc_root)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        rng = random.Random(seed)
        rng.shuffle(all_imgs)

        n_val = 200
        if split == "val":
            self.paths = all_imgs[:n_val]
        else:
            self.paths = all_imgs[n_val: n_val + 2000]

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        W_img, H_img = img.size

        if H_img >= self.patch_size and W_img >= self.patch_size:
            top  = random.randint(0, H_img - self.patch_size)
            left = random.randint(0, W_img - self.patch_size)
            img  = img.crop((left, top,
                             left + self.patch_size, top + self.patch_size))
        else:
            img = img.resize((self.patch_size, self.patch_size), Image.BILINEAR)

        clean = self.to_tensor(img)  # [3, H, W] in [0, 1]

        if self.split == "val":
            rng    = random.Random(idx + 9999)
            length = rng.randint(*self.LEN_RANGE)
            angle  = rng.uniform(*self.ANGLE_RANGE)
        else:
            length = random.randint(*self.LEN_RANGE)
            angle  = random.uniform(*self.ANGLE_RANGE)

        if length % 2 == 0:   # 偶数 kernel 会导致输出尺寸 +1，强制转奇数
            length += 1

        kernel = make_motion_kernel(length, angle)
        blurry = apply_blur(clean, kernel)
        return blurry, clean
