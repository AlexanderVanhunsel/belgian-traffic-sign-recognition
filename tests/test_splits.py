from collections import Counter

from traffic_signs.splits import train_val_split

N_CLASSES = 3  # must match tests/conftest.py
IMAGES_PER_CLASS = 10


def test_split_is_disjoint_and_complete(fake_belgiumts):
    train_ds, val_ds = train_val_split(fake_belgiumts, val_fraction=0.2)
    train_idx, val_idx = set(train_ds.indices), set(val_ds.indices)
    assert train_idx.isdisjoint(val_idx)
    assert len(train_idx | val_idx) == N_CLASSES * IMAGES_PER_CLASS


def test_split_is_stratified(fake_belgiumts):
    _, val_ds = train_val_split(fake_belgiumts, val_fraction=0.2)
    labels = [val_ds.dataset.samples[i][1] for i in val_ds.indices]
    assert Counter(labels) == {label: 2 for label in range(N_CLASSES)}


def test_split_is_reproducible(fake_belgiumts):
    _, val_a = train_val_split(fake_belgiumts, seed=0)
    _, val_b = train_val_split(fake_belgiumts, seed=0)
    assert val_a.indices == val_b.indices