import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch

# VOC2012 的 21 个类别（顺序即类别 id）
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

NUM_CLASSES = 21   # 20 个物体 + 背景
IGNORE_INDEX = 255 # 边界像素，计算 loss 时忽略


class VOC2012Dataset(Dataset):
    def __init__(self, root, split="train", size=512):
        """
        root  : data/ 目录（里面有 VOCdevkit/）
        split : "train" 或 "val"
        size  : 输入图像 resize 到的边长
        """
        self.size = size
        self.img_dir  = os.path.join(root, "VOCdevkit/VOC2012/JPEGImages")
        self.mask_dir = os.path.join(root, "VOCdevkit/VOC2012/SegmentationClass")

        split_file = os.path.join(
            root, "VOCdevkit/VOC2012/ImageSets/Segmentation", f"{split}.txt"
        )
        with open(split_file) as f:
            self.ids = [line.strip() for line in f]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]

        img  = Image.open(os.path.join(self.img_dir,  f"{name}.jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, f"{name}.png"))

        # resize：图用双线性，mask 必须用 NEAREST（像素值是类别 id，不能插值）
        img  = img.resize((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.NEAREST)

        img  = TF.to_tensor(img)               # (3, H, W)，值域 [0,1]
        img  = TF.normalize(img,               # ImageNet 均值/标准差
                            mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225])

        mask = torch.as_tensor(np.array(mask), dtype=torch.long)  # (H, W)，值 = 类别 id

        return img, mask
