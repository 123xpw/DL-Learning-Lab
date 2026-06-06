import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    ResNet 的基本单元：残差块（Residual Block）。

    核心思路：
      普通 CNN：  x → F(x)
      残差块：    x → F(x) + x   ← 多了一条"跳跃连接"

    好处：即使 F(x) 的梯度消失，x 本身的梯度还能直接流回去，
    所以可以堆很多层而不退化。
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 主路径：两层 3×3 卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # 跳跃连接（shortcut）：
        # 当 stride>1 或通道数变化时，x 的尺寸和主路径输出不一致，
        # 需要用 1×1 卷积对齐（projection shortcut）。
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))  # 第一层
        out = self.bn2(self.conv2(out))            # 第二层（注意：先不 ReLU）
        out = out + self.shortcut(x)               # 加上跳跃连接
        out = self.relu(out)                       # 最后再 ReLU
        return out


class ResNet18(nn.Module):
    """
    ResNet-18，适配 CIFAR-10（32×32 输入）。

    与 ImageNet 版本的区别：
    - 第一层用 3×3 conv（而非 7×7），stride=1（而非 2）
    - 去掉初始 MaxPool（图片太小，缩小两次就没了）
    - 其余结构完全一致

    网络结构（每行括号内为输出 shape，batch_size 省略）：
      conv1    →  [64, 32, 32]
      layer1   →  [64, 32, 32]   × 2 个 BasicBlock
      layer2   →  [128, 16, 16]  × 2 个 BasicBlock（stride=2 下采样）
      layer3   →  [256, 8, 8]    × 2 个 BasicBlock（stride=2 下采样）
      layer4   →  [512, 4, 4]    × 2 个 BasicBlock（stride=2 下采样）
      avgpool  →  [512, 1, 1]
      fc       →  [num_classes]
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # ── 第一层：提取基础特征 ─────────────────────────────────
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        # ── 四个阶段，每个阶段 2 个 BasicBlock ───────────────────
        self.layer1 = self._make_layer(64,  64,  stride=1)  # 尺寸不变
        self.layer2 = self._make_layer(64,  128, stride=2)  # 32→16
        self.layer3 = self._make_layer(128, 256, stride=2)  # 16→8
        self.layer4 = self._make_layer(256, 512, stride=2)  # 8→4

        # ── 分类头 ───────────────────────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool2d(1)   # [B, 512, 4, 4] → [B, 512, 1, 1]
        self.fc      = nn.Linear(512, num_classes)

        # 权重初始化（He 初始化，适合 ReLU 网络）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, stride):
        """创建一个阶段：第一个 Block 负责下采样，第二个 Block 保持尺寸。"""
        return nn.Sequential(
            BasicBlock(in_channels,  out_channels, stride=stride),  # 可能下采样
            BasicBlock(out_channels, out_channels, stride=1),        # 保持不变
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 64, 32, 32]

        x = self.layer1(x)   # [B, 64,  32, 32]
        x = self.layer2(x)   # [B, 128, 16, 16]
        x = self.layer3(x)   # [B, 256,  8,  8]
        x = self.layer4(x)   # [B, 512,  4,  4]

        x = self.avgpool(x)              # [B, 512, 1, 1]
        x = torch.flatten(x, 1)         # [B, 512]
        x = self.fc(x)                  # [B, 10]
        return x
