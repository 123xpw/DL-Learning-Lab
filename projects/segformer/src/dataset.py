import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
NUM_CLASSES  = 21
IGNORE_INDEX = 255


class VOC2012Dataset(Dataset):
    def __init__(self, root, split="train", size=512):
        self.size  = size
        self.split = split
        self.img_dir  = os.path.join(root, "VOCdevkit/VOC2012/JPEGImages")
        self.mask_dir = os.path.join(root, "VOCdevkit/VOC2012/SegmentationClass")

        split_file = os.path.join(
            root, "VOCdevkit/VOC2012/ImageSets/Segmentation", f"{split}.txt"
        )
        with open(split_file) as f:
            self.ids = [l.strip() for l in f]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img  = Image.open(os.path.join(self.img_dir,  f"{name}.jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, f"{name}.png"))

        if self.split == "train":
            # 随机缩放 [0.5, 2.0]
            scale = random.uniform(0.5, 2.0)
            new_h = int(self.size * scale)
            new_w = int(self.size * scale)
            img  = img.resize((new_w, new_h), Image.BILINEAR)
            mask = mask.resize((new_w, new_h), Image.NEAREST)

            # 如果缩放后比 size 小，先 padding（图片填 0，mask 填 255 即 ignore）
            img, mask = self._pad_if_needed(img, mask)

            # 随机裁剪到精确 size×size
            i, j = self._get_random_crop_params(img)
            img  = TF.crop(img,  i, j, self.size, self.size)
            mask = TF.crop(mask, i, j, self.size, self.size)

            # 随机水平翻转
            if random.random() > 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)
        else:
            img  = img.resize((self.size, self.size), Image.BILINEAR)
            mask = mask.resize((self.size, self.size), Image.NEAREST)

        img  = TF.to_tensor(img)
        img  = TF.normalize(img, mean=[0.485, 0.456, 0.406],
                                  std =[0.229, 0.224, 0.225])
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return img, mask

    def _pad_if_needed(self, img, mask):
        from PIL import ImageOps
        w, h = img.size
        pad_h = max(0, self.size - h)
        pad_w = max(0, self.size - w)
        if pad_h > 0 or pad_w > 0:
            img  = ImageOps.expand(img,  (0, 0, pad_w, pad_h), fill=0)
            mask = ImageOps.expand(mask, (0, 0, pad_w, pad_h), fill=255)
        return img, mask

    def _get_random_crop_params(self, img):
        w, h = img.size   # 此时已保证 h >= size, w >= size
        i = random.randint(0, h - self.size)
        j = random.randint(0, w - self.size)
        return i, j
