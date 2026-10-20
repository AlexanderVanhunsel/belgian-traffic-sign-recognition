"""PyTorch Dataset for the BelgiumTS classification data."""
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


class BelgiumTSDataset(Dataset):
    def __init__(self, root: str | Path, transform: Callable | None = None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = int(class_dir.name)
            for image_path in sorted(class_dir.glob("*.ppm")):
                self.samples.append((image_path, label))

        if not self.samples:
            raise FileNotFoundError(f"No .ppm images found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label