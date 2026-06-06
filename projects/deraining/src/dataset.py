import os
import math
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def generate_rain(H, W, seed=None):
    """
    Vectorized synthetic rain generation.
    Returns rain layer [3, H, W] in [0, 1].
    seed: if set, produces deterministic output (used for fixed val set).
    """
    if seed is not None:
        saved = random.getstate()
        random.seed(seed)

    angle     = random.uniform(70, 110)     # near-vertical (90° = straight down)
    n_streaks = random.randint(80, 250)
    length    = random.randint(12, 28)
    intensity = random.uniform(0.25, 0.50)

    if seed is not None:
        random.setstate(saved)

    rain = torch.zeros(H, W)
    rad  = math.radians(angle)
    sin_a = math.sin(rad)   # vertical component (dy per step)
    cos_a = math.cos(rad)   # horizontal component (dx per step)

    # Starting positions [n_streaks]
    x0s = torch.randint(0, W, (n_streaks,))
    y0s = torch.randint(0, H, (n_streaks,))

    # Per-step offsets [length]
    ts  = torch.arange(length, dtype=torch.float32)
    dys = (ts * sin_a).long()
    dxs = (ts * cos_a).long()

    # Pixel positions [n_streaks, length]
    ys = (y0s.unsqueeze(1) + dys.unsqueeze(0)).clamp(0, H - 1)
    xs = (x0s.unsqueeze(1) + dxs.unsqueeze(0)).clamp(0, W - 1)

    rain[ys.reshape(-1), xs.reshape(-1)] = intensity
    return rain.unsqueeze(0).expand(3, -1, -1).clone()   # [3, H, W]


class RainDataset(Dataset):
    """
    VOC2012 images with synthetic rain streaks.
    - split="train": 2000 images, random rain each call
    - split="val":   200 images,  fixed rain per index (reproducible PSNR)
    """
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
        img   = Image.open(self.paths[idx]).convert("RGB")
        W_img, H_img = img.size   # PIL returns (width, height)

        # Random/center crop to patch_size
        if H_img >= self.patch_size and W_img >= self.patch_size:
            top  = random.randint(0, H_img - self.patch_size)
            left = random.randint(0, W_img - self.patch_size)
            img  = img.crop((left, top,
                             left + self.patch_size, top + self.patch_size))
        else:
            img = img.resize((self.patch_size, self.patch_size), Image.BILINEAR)

        clean = self.to_tensor(img)   # [3, H, W] in [0, 1]

        rain_seed = idx if self.split == "val" else None
        rain      = generate_rain(self.patch_size, self.patch_size, seed=rain_seed)
        rainy     = (clean + rain).clamp(0, 1)

        return rainy, clean
