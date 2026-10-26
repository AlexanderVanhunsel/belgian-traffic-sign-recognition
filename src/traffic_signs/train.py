import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from traffic_signs.model import TrafficSignCNN
from traffic_signs.splits import train_val_split


def run_epoch(
    model: nn.Module, #nn.module = base class for all NN.
    loader: DataLoader, #Dataloader iterates over the dataset and returns batches of data.
    criterion: nn.Module, #criterion has type nn.Module which has the loss function for training the model.
    device: torch.device, #device has type torch.device which specifies the device to run the model on.
    optimizer: torch.optim.Optimizer | None = None, #.optim contains optim. algorithms and Optimizer is the base class for all optimizers.
) -> tuple[float, float]:
    """Run one epoch. Pass an optimizer to train; omit it to only evaluate.

    Returns (average loss, accuracy) over the whole loader.
    """
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_training): #set.grad_enabled enables or disables gradient calculation.
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                optimizer.zero_grad() #zero_grad() sets the gradients of all optimized torch.Tensor s to zero. This is because by default, gradients are accumulated in buffers (i.e., not overwritten) whenever .backward() is called.
                loss.backward() 
                optimizer.step() #.step() updates parameters.

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def train_model(
    train_ds,
    val_ds,
    device: torch.device,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_workers: int = 0,
    output_dir: str | Path = "runs/baseline_cnn",
) -> dict:
    """Train a TrafficSignCNN and save the best checkpoint. Returns the training history."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = TrafficSignCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"epoch {epoch:2d}/{epochs} | "
            f"train loss {train_loss:.3f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.3f} acc {val_acc:.3f} | "
            f"{time.time() - start:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline traffic sign CNN.")
    parser.add_argument("--data-dir", default="data/Training")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="runs/baseline_cnn")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds = train_val_split(args.data_dir, val_fraction=args.val_fraction, seed=args.seed)
    print(f"Train: {len(train_ds)} images, Val: {len(val_ds)} images")

    train_model(
        train_ds,
        val_ds,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()