"""Image transforms for training and evaluation."""
from torchvision import transforms

IMAGE_SIZE = 64
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_transforms(train: bool, image_size: int = IMAGE_SIZE) -> transforms.Compose:
    if train: #moeilijker maken om robuuster model te trainen.
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD), #pixel waarden normaliseren pixel_n = pixel - mean / std
        ])
    #dit is als je data testdata is en wil je dus niet vervormen.
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])