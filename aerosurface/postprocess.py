"""Conservative image-space candidate ROI, not a robot safety guarantee."""

import numpy as np
from PIL import Image, ImageFilter

from aerosurface import PALETTE


def postprocess(logits: np.ndarray, threshold: float = 0.65, margin: int = 3) -> dict:
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 5:
        raise ValueError("Expected [1,5,H,W] logits")
    if not np.isfinite(logits).all() or not 0 <= threshold <= 1 or margin < 0:
        raise ValueError("Nonfinite logits or invalid parameters")
    values = logits[0] - logits[0].max(0)
    probs = np.exp(values)
    probs /= probs.sum(0)
    confidence = probs.max(0)
    mask = probs.argmax(0).astype(np.uint8)
    mask[confidence < threshold] = 0
    candidate = (mask == 1).astype(np.uint8) * 255
    # Explicit zero padding prevents an ROI reaching an unknown image boundary.
    padded = Image.fromarray(np.pad(candidate, margin))
    if margin:
        padded = padded.filter(ImageFilter.MinFilter(2 * margin + 1))
        candidate = np.array(padded)[margin:-margin, margin:-margin]
    return {
        "mask": mask,
        "confidence": confidence,
        "sandable": candidate,
        "avoid": np.where(candidate == 0, 255, 0).astype(np.uint8),
        "defect": (mask == 3).astype(np.uint8) * 255,
        "protected": (mask == 2).astype(np.uint8) * 255,
    }


def overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    colors = np.array(PALETTE, dtype=np.uint8)[mask]
    image = np.array(Image.fromarray(rgb).resize((mask.shape[1], mask.shape[0])))
    return (image.astype(float) * 0.55 + colors * 0.45).astype(np.uint8)
