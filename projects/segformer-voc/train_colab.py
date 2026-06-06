# 在 Colab 第一个 cell 运行：
# !pip install transformers==5.9.0 -q
# !python train_colab.py

import os
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torchvision.transforms.functional as TF
import torchvision.datasets as tv_datasets
from transformers import SegformerForSemanticSegmentation

# ── 超参数 ────────────────────────────────────────────────────────
EPOCHS     = 20
BATCH_SIZE = 8      # GPU 显存够，batch 可以更大
LR         = 6e-5
IMG_SIZE   = 512
NUM_CLASSES  = 21
IGNORE_INDEX = 255
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT  = "/content/data"

# ── 数据集 ────────────────────────────────────────────────────────
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

class VOC2012Dataset(Dataset):
    def __init__(self, root, split="train", size=512):
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
        img  = img.resize((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.NEAREST)
        img  = TF.to_tensor(img)
        img  = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return img, mask

def compute_miou(preds, targets):
    iou_list = []
    for cls in range(NUM_CLASSES):
        pred_mask = (preds == cls)
        true_mask = (targets == cls)
        valid     = (targets != IGNORE_INDEX)
        intersection = (pred_mask & true_mask & valid).sum().item()
        union        = ((pred_mask | true_mask) & valid).sum().item()
        if union == 0:
            continue
        iou_list.append(intersection / union)
    return sum(iou_list) / len(iou_list) if iou_list else 0.0

# ── 下载数据（首次运行） ───────────────────────────────────────────
print("下载 VOC2012...")
tv_datasets.VOCSegmentation(root=DATA_ROOT, year="2012", image_set="train", download=True)
print("下载完成")

# ── 数据加载 ──────────────────────────────────────────────────────
train_dataset = VOC2012Dataset(DATA_ROOT, split="train", size=IMG_SIZE)
val_dataset   = VOC2012Dataset(DATA_ROOT, split="val",   size=IMG_SIZE)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader    = DataLoader(val_dataset,   batch_size=1,          shuffle=False, num_workers=2)

# ── 模型 ──────────────────────────────────────────────────────────
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

print(f"device: {DEVICE} | train: {len(train_dataset)} | val: {len(val_dataset)}")

# ── 训练循环 ──────────────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):

    model.train()
    total_loss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        outputs   = model(pixel_values=imgs)
        logits_up = F.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        loss = criterion(logits_up, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    model.eval()
    miou_list = []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(DEVICE)
            outputs   = model(pixel_values=imgs)
            logits_up = F.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            preds     = logits_up.argmax(dim=1).squeeze(0).cpu()
            miou_list.append(compute_miou(preds, masks.squeeze(0)))

    avg_miou = sum(miou_list) / len(miou_list)
    print(f"Epoch {epoch:02d}/{EPOCHS} | loss: {avg_loss:.4f} | val mIoU: {avg_miou:.4f}")

# ── 保存 ──────────────────────────────────────────────────────────
torch.save(model.state_dict(), "/content/segformer_voc.pth")
print("模型已保存：/content/segformer_voc.pth")
