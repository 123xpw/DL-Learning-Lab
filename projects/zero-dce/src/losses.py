import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialConsistencyLoss(nn.Module):
    """
    空间一致性损失：增强前后相邻区域的亮度差异应保持一致。
    防止模型"乱改"局部对比度，保留原图的空间结构。
    """
    def __init__(self):
        super().__init__()
        kl = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).view(1, 1, 3, 3)
        kr = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).view(1, 1, 3, 3)
        ku = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).view(1, 1, 3, 3)
        kd = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).view(1, 1, 3, 3)
        self.register_buffer("kl", kl)
        self.register_buffer("kr", kr)
        self.register_buffer("ku", ku)
        self.register_buffer("kd", kd)
        self.pool = nn.AvgPool2d(4)

    def forward(self, enhanced, original):
        e = self.pool(enhanced.mean(dim=1, keepdim=True))
        o = self.pool(original.mean(dim=1, keepdim=True))
        loss = sum(
            (F.conv2d(e, k, padding=1) - F.conv2d(o, k, padding=1)) ** 2
            for k in [self.kl, self.kr, self.ku, self.kd]
        )
        return loss.mean()


class ExposureControlLoss(nn.Module):
    """
    曝光控制损失：把图像分成小块，让每块的平均亮度接近目标值 E。
    E=0.6 表示"正常曝光"大约在 60% 亮度左右。
    防止模型把整张图拉得过亮或仍然过暗。
    """
    def __init__(self, patch_size=16, target=0.6):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.E    = target

    def forward(self, enhanced):
        gray = enhanced.mean(dim=1, keepdim=True)
        mean = self.pool(gray)
        return ((mean - self.E) ** 2).mean()


class ColorConstancyLoss(nn.Module):
    """
    色彩恒常损失：增强后三个通道的平均值应该接近相等（灰色世界假设）。
    防止模型增强时偏色——把图拉亮的同时保持颜色平衡。
    """
    def forward(self, enhanced):
        r = enhanced[:, 0].mean()
        g = enhanced[:, 1].mean()
        b = enhanced[:, 2].mean()
        return (r - g) ** 2 + (r - b) ** 2 + (g - b) ** 2


class IlluminationSmoothnessLoss(nn.Module):
    """
    光照平滑损失：让 alpha 曲线在空间上平滑变化（Total Variation）。
    防止增强图出现块状伪影——相邻像素的增强强度不应突变。
    """
    def forward(self, alphas):
        tv_h = (alphas[:, :, 1:, :] - alphas[:, :, :-1, :]).abs().mean()
        tv_w = (alphas[:, :, :, 1:] - alphas[:, :, :, :-1]).abs().mean()
        return tv_h + tv_w
