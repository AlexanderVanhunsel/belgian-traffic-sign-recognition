from torch import Tensor, nn

NUM_CLASSES = 62


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,  # bijv. 32 filters: leert 32 verschillende patronen
            kernel_size=3,  # filtergrootte van 3x3 pixels
            padding=1,  # padding=1 zorgt bij kernel_size=3 voor behoud van dimensies
        ),
        nn.BatchNorm2d(
            out_channels
        ),  # normaliseert activaties voor stabielere training
        nn.ReLU(),  # voegt non-lineariteit toe (zet negatieve waarden op 0)
        nn.MaxPool2d(2),  # halveert hoogte en breedte (reduceert resolutie)
    )

#input is een tensor 3x64x64 (kanalen(RGB), hoogte, breedte) --> Maxpool halveert hoogte en breedte enkel. kanalen nemen toe.
class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> None: #dropout zet p% outputs van activaties op 0 om robuuster te maken.
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            nn.AdaptiveAvgPool2d((4, 4)), #maakt dat de output van de convolutielagen altijd 4x4 is ongeacht de inputgrootte.
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), #flatten van 4x4x128 naar 2048 vector
            nn.Dropout(dropout),
            nn.Linear(128 * 4 * 4, 256), #linear layer van 2048 naar 256
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(256, num_classes), #linear layer van 256 naar 62 berekent een score voor elke klasse
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.classifier(x)