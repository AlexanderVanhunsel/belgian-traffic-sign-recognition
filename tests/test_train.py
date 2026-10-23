import torch

from traffic_signs.splits import train_val_split
from traffic_signs.train import train_model


def test_train_model_runs_and_saves_checkpoint(fake_belgiumts, tmp_path):
    train_ds, val_ds = train_val_split(fake_belgiumts, val_fraction=0.2, image_size=32)

    history = train_model(
        train_ds,
        val_ds,
        device=torch.device("cpu"),
        epochs=2,
        batch_size=4,
        output_dir=tmp_path / "run",
    )

    assert len(history["train_loss"]) == 2
    assert all(0.0 <= acc <= 1.0 for acc in history["val_acc"])
    assert (tmp_path / "run" / "best_model.pt").exists()
    assert (tmp_path / "run" / "history.json").exists()