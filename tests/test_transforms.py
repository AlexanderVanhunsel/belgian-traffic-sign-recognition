import torch
from PIL import Image

from traffic_signs.transforms import get_transforms


def make_image() -> Image.Image:
    return Image.new("RGB", (120, 90), color=(200, 30, 30))


def test_output_shape():
    for train in (True, False):
        x = get_transforms(train=train, image_size=48)(make_image())
        assert tuple(x.shape) == (3, 48, 48)


def test_eval_transform_is_deterministic():
    t = get_transforms(train=False)
    image = make_image()
    assert torch.equal(t(image), t(image))