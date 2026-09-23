"""Shared validated experiment configuration."""

import json
import math
from pathlib import Path


def load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text())
    required = {
        "seed",
        "width",
        "height",
        "base_channels",
        "batch_size",
        "epochs",
        "learning_rate",
        "train_samples",
        "val_samples",
        "test_samples",
        "threads",
        "amp",
        "confidence",
        "margin",
    }
    if set(config) != required:
        raise ValueError(f"Configuration keys differ: {set(config) ^ required}")
    for key in required - {"amp", "confidence", "learning_rate", "seed", "margin"}:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["width"] % 4 or config["height"] % 4:
        raise ValueError("Input dimensions must be divisible by four")
    if max(config["width"], config["height"]) > 8192:
        raise ValueError("Maximum supported image dimension is 8192")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (
        not 0 <= config["confidence"] <= 1
        or type(config["margin"]) is not int
        or not 0 <= config["margin"] <= 64
    ):
        raise ValueError("Invalid confidence or erosion margin")
    if (
        not math.isfinite(config["learning_rate"])
        or config["learning_rate"] <= 0
        or not isinstance(config["amp"], bool)
    ):
        raise ValueError("Invalid training configuration")
    return config
