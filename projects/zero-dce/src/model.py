import torch
import torch.nn as nn


class DCENet(nn.Module):
    """
    Zero-DCE: Zero-Reference Deep Curve Estimation (CVPR 2020)

    不预测清晰图，而是预测每个像素的"曲线参数" alpha，
    用一条二次曲线迭代地把暗图拉亮：

        LE_n = LE_{n-1} + alpha_n * LE_{n-1} * (1 - LE_{n-1})

    alpha > 0 → 拉亮；alpha < 0 → 压暗。
    重复 n_iter 次，不同区域用不同强度的曲线。
    """
    def __init__(self, n_iter=8):
        super().__init__()
        self.n_iter = n_iter

        self.conv1 = nn.Conv2d(3,  32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, 3, padding=1)   # 拼接 conv3+conv4
        self.conv6 = nn.Conv2d(64, 32, 3, padding=1)   # 拼接 conv2+conv5
        self.conv7 = nn.Conv2d(64, 3 * n_iter, 3, padding=1)  # 输出 24 通道 alpha
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        alphas = torch.tanh(self.conv7(torch.cat([x1, x6], dim=1)))
        # alphas: (B, 3*n_iter, H, W)，范围 (-1, 1)

        enhanced = x
        for i in range(self.n_iter):
            a = alphas[:, i*3:(i+1)*3, :, :]
            enhanced = enhanced + a * (enhanced - enhanced ** 2)

        return enhanced.clamp(0, 1), alphas
