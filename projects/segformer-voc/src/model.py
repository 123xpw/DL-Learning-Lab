from transformers import SegformerForSemanticSegmentation
from src.dataset import NUM_CLASSES


def build_model():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",          # 最小版本，适合入门
        num_labels=NUM_CLASSES,   # 21 类
        ignore_mismatched_sizes=True,  # 替换原来的分类头
    )
    return model
