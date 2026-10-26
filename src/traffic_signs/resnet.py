"""Transfer learning: a pretrained ResNet18 adapted to traffic sign classification."""
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from traffic_signs.model import NUM_CLASSES


def build_resnet18(
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = True,
    pretrained: bool = True,
) -> nn.Module:
    """Build a ResNet18 adapted for traffic sign classification.

    With freeze_backbone=True (the default), only the new final layer is
    trained; the pretrained feature extractor keeps its ImageNet weights.
    This trains fast and works well even with a small dataset.

    pretrained=False skips downloading the ImageNet weights (random init
    instead); used by tests so they run fast and without internet access.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replacing .fc creates a brand new layer, which is trainable by default
    # even though everything else above was just frozen.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model