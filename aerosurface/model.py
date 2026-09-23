"""Small export-friendly U-Net; no downloaded or pretrained weights."""

import torch
from torch import nn
from torch.nn import functional as F


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.ReLU(),
    )


class SurfaceUNet(nn.Module):
    def __init__(self, base: int = 12, classes: int = 5):
        super().__init__()
        self.enc1 = block(3, base)
        self.enc2 = block(base, base * 2)
        self.bottleneck = block(base * 2, base * 4)
        self.dec2 = block(base * 6, base * 2)
        self.dec1 = block(base * 3, base)
        self.head = nn.Conv2d(base, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.bottleneck(F.max_pool2d(b, 2))
        d = self.dec2(torch.cat([F.interpolate(c, scale_factor=2, mode="nearest"), b], 1))
        return self.head(
            self.dec1(torch.cat([F.interpolate(d, scale_factor=2, mode="nearest"), a], 1))
        )


def load_model(path, device="cpu") -> tuple:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("label_space") == "binary_collapsed":
        raise ValueError("Binary-adapted weights cannot be used for five-class deployment")
    model = SurfaceUNet(checkpoint["config"]["base_channels"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint["config"]
