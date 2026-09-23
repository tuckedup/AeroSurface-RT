"""RGB/mask contracts shared across training and deployment."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def preprocess(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    """Floor-nearest resize, RGB uint8 -> contiguous NCHW float32 /255.

    Deliberately simple and identical to the portable C++ implementation.
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError("Expected nonempty HWC RGB uint8")
    if height <= 0 or width <= 0:
        raise ValueError("Invalid input dimensions")
    ys = np.arange(height) * rgb.shape[0] // height
    xs = np.arange(width) * rgb.shape[1] // width
    return (
        np.ascontiguousarray(rgb[ys[:, None], xs].transpose(2, 0, 1)[None], dtype=np.float32)
        / 255.0
    )


class SurfaceDataset(Dataset):
    def __init__(self, root: Path, split: str, config: dict, augment: bool = False):
        self.root, self.config, self.augment = root, config, augment
        self.rows = [
            r for r in json.loads((root / "manifest.json").read_text()) if r["split"] == split
        ]
        if not self.rows:
            raise ValueError(f"Empty split: {split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple:
        row = self.rows[index]
        rgb = np.array(Image.open(self.root / row["image"]).convert("RGB"))
        mask = np.array(Image.open(self.root / row["mask"]))
        if mask.shape != rgb.shape[:2] or mask.ndim != 2 or mask.max() > 4:
            raise ValueError("Invalid mask shape or class id")
        h, w = self.config["height"], self.config["width"]
        mask = mask[(np.arange(h) * mask.shape[0] // h)[:, None], np.arange(w) * mask.shape[1] // w]
        x = preprocess(rgb, h, w)[0]
        if self.augment and torch.rand(()) < 0.5:
            x, mask = x[:, :, ::-1].copy(), mask[:, ::-1].copy()
        return torch.from_numpy(x), torch.from_numpy(mask.astype(np.int64)), index
