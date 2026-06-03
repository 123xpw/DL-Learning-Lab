import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from models.baseline import BaselineMLP


def build_demo_loader(
    num_samples: int = 256,
    input_dim: int = 784,
    num_classes: int = 10,
    batch_size: int = 32,
) -> DataLoader:
    # features: [num_samples, input_dim]
    features = torch.randn(num_samples, input_dim)
    # labels: [num_samples]
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for features, labels in dataloader:
        # features: [batch_size, input_dim]
        features = features.to(device)
        # labels: [batch_size]
        labels = labels.to(device)

        # logits: [batch_size, num_classes]
        logits = model(features)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)

    return total_loss / len(dataloader.dataset)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = 784
    num_classes = 10
    model = BaselineMLP(input_dim=input_dim, num_classes=num_classes).to(device)
    dataloader = build_demo_loader(input_dim=input_dim, num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 4):
        avg_loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"epoch={epoch} loss={avg_loss:.4f}")


if __name__ == "__main__":
    main()
