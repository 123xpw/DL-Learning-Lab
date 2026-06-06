from src.dataset import VOC2012Dataset, VOC_CLASSES, IGNORE_INDEX
import torch

dataset = VOC2012Dataset(root="./data", split="train", size=512)
print(f"训练集大小: {len(dataset)}")

img, mask = dataset[0]
print(f"图像 shape : {img.shape}   dtype: {img.dtype}")
print(f"Mask shape : {mask.shape}  dtype: {mask.dtype}")

ids = mask.unique().tolist()
print(f"Mask 中出现的类别 id: {ids}")
for i in ids:
    if i == IGNORE_INDEX:
        print(f"  {i} → 边界（忽略）")
    else:
        print(f"  {i} → {VOC_CLASSES[i]}")
