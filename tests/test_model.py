import torch

from traffic_signs.model import NUM_CLASSES, TrafficSignCNN


def test_output_shape():
    model = TrafficSignCNN()
    x = torch.randn(8, 3, 64, 64)  # a batch of 8 random "images"
    assert tuple(model(x).shape) == (8, NUM_CLASSES)


def test_works_with_other_image_sizes():
    model = TrafficSignCNN()
    for size in (32, 48, 96):
        x = torch.randn(2, 3, size, size)
        assert tuple(model(x).shape) == (2, NUM_CLASSES)