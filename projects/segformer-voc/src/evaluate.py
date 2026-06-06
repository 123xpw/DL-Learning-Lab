import torch
from src.dataset import NUM_CLASSES, IGNORE_INDEX


def compute_miou(preds, targets):
    """
    preds  : (H, W) 预测类别 id（已 argmax）
    targets: (H, W) 真实类别 id
    返回当前图片的 mIoU（float）
    """
    iou_list = []
    for cls in range(NUM_CLASSES):
        pred_mask = (preds == cls)
        true_mask = (targets == cls)
        valid     = (targets != IGNORE_INDEX)

        intersection = (pred_mask & true_mask & valid).sum().item()
        union        = ((pred_mask | true_mask) & valid).sum().item()

        if union == 0:
            continue  # 这个类在当前图中不存在，跳过
        iou_list.append(intersection / union)

    return sum(iou_list) / len(iou_list) if iou_list else 0.0
