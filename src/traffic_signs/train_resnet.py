"""Train a ResNet18 (transfer learning) on the BelgiumTS traffic signs."""
import argparse

import torch

from traffic_signs.resnet import build_resnet18
from traffic_signs.splits import train_val_split
from traffic_signs.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained ResNet18.")
    parser.add_argument("--data-dir", default="data/Training")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", default="runs/resnet18")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds = train_val_split(
        args.data_dir, val_fraction=args.val_fraction, seed=args.seed, image_size=args.image_size
    )
    print(f"Train: {len(train_ds)} images, Val: {len(val_ds)} images")

    model = build_resnet18(freeze_backbone=True)
    train_model(
        model,
        train_ds,
        val_ds,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()