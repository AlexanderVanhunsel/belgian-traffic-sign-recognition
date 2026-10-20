from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from traffic_signs.dataset import BelgiumTSDataset
from traffic_signs.transforms import IMAGE_SIZE, get_transforms


def train_val_split(
    root: str | Path,
    val_fraction: float = 0.15,
    seed: int = 42,
    image_size: int = IMAGE_SIZE,
) -> tuple[Subset, Subset]:
    """
    The split is stratified => every class keeps the same proportion in both
    subsets (ex. if 10% of the samples are stopsigns, then 10% of the
    samples in both the training and validation subsets will be stop signs  ).
    The validation subset uses the evaluation transforms.
    """
    train_full = BelgiumTSDataset(root, transform=get_transforms(train=True, image_size=image_size))
    eval_full = BelgiumTSDataset(root, transform=get_transforms(train=False, image_size=image_size))

    labels = [label for _, label in train_full.samples]
    indices = list(range(len(labels)))
    train_idx, val_idx = train_test_split( # ex. train_idx = [0, 3, 4, 2], val_idx = [1, 5]
        indices, test_size=val_fraction, stratify=labels, random_state=seed
    )
    return Subset(train_full, train_idx), Subset(eval_full, val_idx) # Subset filters based on train_idx and val_idx.