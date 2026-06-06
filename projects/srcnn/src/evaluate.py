import torch
import math


def psnr(pred, target):
    """
    PSNR（峰值信噪比），单位 dB，越高越好。
    衡量预测图和真实图的像素差异。

    pred, target: (3, H, W) 或 (B, 3, H, W)，值域 [0, 1]

    经验参考：
      < 25 dB : 明显模糊
      25-30 dB: 一般质量
      30-35 dB: 较好
      > 35 dB : 接近原图
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))
