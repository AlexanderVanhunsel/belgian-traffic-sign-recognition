"""Run a trained model on a dataset and collect its predictions."""
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> nn.Module:
    """Load saved weights (from train_model's best_model.pt) into `model`."""
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    """Run `model` over every batch in `loader`. Returns (true labels, predicted labels)."""
    model = model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    return y_true, y_pred