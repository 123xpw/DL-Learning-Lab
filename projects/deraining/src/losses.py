import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """Smooth L1 variant: sqrt((pred - target)^2 + eps^2). More robust than MSE on outliers."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))
