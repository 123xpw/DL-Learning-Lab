import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class NoisyDataset(Dataset):
    """
    VOC2012 images with synthetic Gaussian noise.
    - split="train": 2000 images, random sigma in [15, 50]/255 each call
    - split="val":   200 images,  fixed sigma=25/255 (reproducible PSNR)
    """
    SIGMA_TRAIN = (15, 50)
    SIGMA_VAL   = 25

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
            rng   = torch.Generator()
            rng.manual_seed(idx)
            sigma = self.SIGMA_VAL / 255.0
            noise = torch.randn(clean.shape, generator=rng) * sigma
        else:
            sigma = random.uniform(*self.SIGMA_TRAIN) / 255.0
            noise = torch.randn(clean.shape) * sigma

        noisy = (clean + noise).clamp(0, 1)
        return noisy, clean
