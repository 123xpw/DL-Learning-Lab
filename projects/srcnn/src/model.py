import torch.nn as nn


class SRCNN(nn.Module):
    """
    SRCNN: Super-Resolution CNN（2014, Dong et al.）
    三层卷积，每层含义不同：
      conv1: 提取低分辨率图的局部特征（大卷积核，感受野宽）
      conv2: 在特征空间里做非线性映射（1×1 核，逐通道变换）
      conv3: 把特征重建回像素空间（输出 3 通道 RGB）
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,  64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32,  3, kernel_size=5, padding=2)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))  # 低清特征提取
        x = self.relu(self.conv2(x))  # 特征非线性映射
        x = self.conv3(x)             # 重建高清图（无激活函数，输出可以是任意值）
        return x
