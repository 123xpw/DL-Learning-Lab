import torch.nn as nn


class DnCNN(nn.Module):
    """
    DnCNN: denoising via residual learning.
    Forward pass returns predicted noise; clean = noisy - predicted_noise.
    """
    def __init__(self, in_channels=3, n_feat=64, depth=9):
        super().__init__()
        layers = [nn.Conv2d(in_channels, n_feat, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=False),
                nn.BatchNorm2d(n_feat),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(n_feat, in_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # predicted noise residual
