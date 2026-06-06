import torch
import torch.nn as nn


class AODNet(nn.Module):
    """
    AOD-Net: All-in-One Dehazing Network (Li et al., 2017)

    不直接预测清晰图，而是预测一个中间变量 K，
    再用物理公式还原清晰图：

        J = (1/K) * I - A/K + A

    这样网络的输出天然满足雾的物理约束。
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,  8, kernel_size=1)
        self.conv2 = nn.Conv2d(8,  8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(16, 8, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(torch.cat([x1, x2], dim=1)))
        x4 = self.relu(self.conv4(torch.cat([x2, x3], dim=1)))
        K  = self.relu(self.conv5(torch.cat([x3, x4], dim=1))) + 0.1  # 防止 K≈0 导致除法爆炸

        # 物理公式还原清晰图（A=1.0）
        # J = (1/K) * I - 1/K + 1  →  J = (I - 1) / K + 1
        J = (x - 1) / (K + 1e-6) + 1
        return J.clamp(0, 1)
