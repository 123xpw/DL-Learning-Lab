import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import re

from src.dataset import VOC2012Dataset, VOC_CLASSES, NUM_CLASSES, IGNORE_INDEX
from src.model import build_model


def remap_keys(old_sd):
    """
    Colab(transformers 4.x) 保存的权重键名 → 本地(transformers 5.x) 的键名
    主要变化：
      segformer.encoder.* → segformer.stages.*
      decode_head.linear_c.{i}.* → decode_head.linear_projections.{i}.*
    """
    new_sd = {}
    for k, v in old_sd.items():

        # patch_embeddings: encoder.patch_embeddings.{i}.X → stages.{i}.patch_embeddings.X
        m = re.match(r'segformer\.encoder\.patch_embeddings\.(\d+)\.(.*)', k)
        if m:
            new_sd[f'segformer.stages.{m.group(1)}.patch_embeddings.{m.group(2)}'] = v
            continue

        # layer_norm: encoder.layer_norm.{i}.X → stages.{i}.layer_norm.X
        m = re.match(r'segformer\.encoder\.layer_norm\.(\d+)\.(.*)', k)
        if m:
            new_sd[f'segformer.stages.{m.group(1)}.layer_norm.{m.group(2)}'] = v
            continue

        # block: encoder.block.{i}.{j}.X → stages.{i}.blocks.{j}.X_renamed
        m = re.match(r'segformer\.encoder\.block\.(\d+)\.(\d+)\.(.*)', k)
        if m:
            i, j, rest = m.group(1), m.group(2), m.group(3)
            rest = rest.replace('layer_norm_1.',              'layernorm_before.')
            rest = rest.replace('layer_norm_2.',              'layernorm_after.')
            rest = rest.replace('attention.self.query.',      'attention.q_proj.')
            rest = rest.replace('attention.self.key.',        'attention.k_proj.')
            rest = rest.replace('attention.self.value.',      'attention.v_proj.')
            rest = rest.replace('attention.self.sr.',         'attention.sequence_reduction.sequence_reduction.')
            rest = rest.replace('attention.self.layer_norm.', 'attention.sequence_reduction.layer_norm.')
            rest = rest.replace('attention.output.dense.',    'attention.o_proj.')
            rest = rest.replace('mlp.dense1.',                'mlp.fc1.')
            rest = rest.replace('mlp.dense2.',                'mlp.fc2.')
            new_sd[f'segformer.stages.{i}.blocks.{j}.{rest}'] = v
            continue

        # decode_head.linear_c.{i}.X → decode_head.linear_projections.{i}.X
        m = re.match(r'decode_head\.linear_c\.(\d+)\.(.*)', k)
        if m:
            new_sd[f'decode_head.linear_projections.{m.group(1)}.{m.group(2)}'] = v
            continue

        new_sd[k] = v

    return new_sd

# VOC2012 官方调色板（每个类别 id 对应一种颜色）
VOC_COLORMAP = [
    (0,   0,   0),    # 0  background
    (128, 0,   0),    # 1  aeroplane
    (0,   128, 0),    # 2  bicycle
    (128, 128, 0),    # 3  bird
    (0,   0,   128),  # 4  boat
    (128, 0,   128),  # 5  bottle
    (0,   128, 128),  # 6  bus
    (128, 128, 128),  # 7  car
    (64,  0,   0),    # 8  cat
    (192, 0,   0),    # 9  chair
    (64,  128, 0),    # 10 cow
    (192, 128, 0),    # 11 diningtable
    (64,  0,   128),  # 12 dog
    (192, 0,   128),  # 13 horse
    (64,  128, 128),  # 14 motorbike
    (192, 128, 128),  # 15 person
    (0,   64,  0),    # 16 pottedplant
    (128, 64,  0),    # 17 sheep
    (0,   192, 0),    # 18 sofa
    (128, 192, 0),    # 19 train
    (0,   64,  128),  # 20 tvmonitor
]

def mask_to_rgb(mask):
    """把类别 id 的 mask 转成 RGB 彩色图"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(VOC_COLORMAP):
        rgb[mask == cls_id] = color
    return rgb

def denormalize(img_tensor):
    """把标准化过的 tensor 还原成可显示的图像"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    img = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)

def visualize(num_samples=4, save_path="predictions.png"):
    device = "cpu"

    # 加载模型
    model = build_model()
    old_sd = torch.load("segformer_voc.pth", map_location="cpu")
    model.load_state_dict(remap_keys(old_sd))
    model.eval()

    dataset = VOC2012Dataset("./data", split="val", size=512)

    # 固定挑几张有代表性的图（不随机，结果可复现）
    indices = [0, 10, 50, 100, 200, 300][:num_samples]

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))
    fig.suptitle("SegFormer-B0 Predictions (VOC2012 val)", fontsize=14)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            img_tensor, true_mask = dataset[idx]

            # predict
            logits = model(pixel_values=img_tensor.unsqueeze(0))
            logits_up = F.interpolate(logits.logits, size=(512, 512), mode="bilinear", align_corners=False)
            pred_mask = logits_up.argmax(dim=1).squeeze(0).numpy()
            true_mask_np = true_mask.numpy()

            axes[row, 0].imshow(denormalize(img_tensor))
            axes[row, 0].set_title("Image")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(mask_to_rgb(true_mask_np))
            axes[row, 1].set_title("Ground Truth")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(mask_to_rgb(pred_mask))
            axes[row, 2].set_title("Prediction")
            axes[row, 2].axis("off")

    # 图例
    patches = [
        mpatches.Patch(color=[c/255 for c in VOC_COLORMAP[i]], label=VOC_CLASSES[i])
        for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=7, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"已保存：{save_path}")
    plt.show()

if __name__ == "__main__":
    visualize(num_samples=4)
