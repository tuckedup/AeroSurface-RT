import pytest
import torch

from aerosurface.config import load_config
from aerosurface.model import SurfaceUNet, load_model


def test_binary_checkpoint_rejected_for_five_class_deployment(tmp_path):
    config = load_config("configs/edge.json")
    path = tmp_path / "binary.pt"
    torch.save(
        {
            "state_dict": SurfaceUNet().state_dict(),
            "config": config,
            "label_space": "binary_collapsed",
        },
        path,
    )
    with pytest.raises(ValueError, match="Binary-adapted"):
        load_model(path)
