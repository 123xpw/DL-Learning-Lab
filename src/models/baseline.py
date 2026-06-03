import torch
from torch import nn


class BaselineMLP(nn.Module):
    """A minimal MLP baseline for tabular or flattened vector inputs."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
        num_classes: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # x: [batch_size, input_dim] -> [batch_size, hidden_dim]
            nn.Linear(input_dim, hidden_dim),
            # x: [batch_size, hidden_dim] -> [batch_size, hidden_dim]
            nn.ReLU(),
            # x: [batch_size, hidden_dim] -> [batch_size, hidden_dim]
            nn.Dropout(dropout),
            # x: [batch_size, hidden_dim] -> [batch_size, hidden_dim]
            nn.Linear(hidden_dim, hidden_dim),
            # x: [batch_size, hidden_dim] -> [batch_size, hidden_dim]
            nn.ReLU(),
            # x: [batch_size, hidden_dim] -> [batch_size, num_classes]
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: [batch_size, input_dim]
        logits = self.net(x)
        # Output logits: [batch_size, num_classes]
        return logits
