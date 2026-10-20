import pytest
import torch

from traffic_signs.dataset import BelgiumTSDataset
from traffic_signs.transforms import get_transforms

N_CLASSES = 3  # must match tests/conftest.py
IMAGES_PER_CLASS = 10


def test_counts_only_ppm_images(fake_belgiumts):
    ds = BelgiumTSDataset(fake_belgiumts)
    assert len(ds) == N_CLASSES * IMAGES_PER_CLASS


def test_label_comes_from_folder_name(fake_belgiumts):
    ds = BelgiumTSDataset(fake_belgiumts)
    for path, label in ds.samples:
        assert label == int(path.parent.name)


def test_getitem_returns_tensor_and_label(fake_belgiumts):
    ds = BelgiumTSDataset(fake_belgiumts, transform=get_transforms(train=False, image_size=64))
    image, label = ds[0]
    assert isinstance(image, torch.Tensor)
    assert tuple(image.shape) == (3, 64, 64)
    assert isinstance(label, int)


def test_empty_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        BelgiumTSDataset(tmp_path)