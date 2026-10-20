"""Shared test fixtures: a tiny fake BelgiumTS folder, so tests run without the real data."""
from pathlib import Path

import pytest
from PIL import Image

N_CLASSES = 3
IMAGES_PER_CLASS = 10


@pytest.fixture
def fake_belgiumts(tmp_path: Path) -> Path:
    """Create data in the BelgiumTS layout, including the files that must be skipped."""
    root = tmp_path / "Training"
    for label in range(N_CLASSES):
        class_dir = root / f"{label:05d}"
        class_dir.mkdir(parents=True)
        for i in range(IMAGES_PER_CLASS):
            size = (30 + 5 * i, 40 + 3 * i)  # different sizes, like the real data
            Image.new("RGB", size, color=(40 * label, 100, 200)).save(class_dir / f"img_{i:05d}.ppm")
        (class_dir / f"GT-{label:05d}.csv").write_text("Filename;Width;Height\n")
    (root / "Readme.txt").write_text("not an image")
    return root