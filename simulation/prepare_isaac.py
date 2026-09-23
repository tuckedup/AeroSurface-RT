"""Convert BasicWriter raw semantic IDs through its per-frame label mapping."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from aerosurface import CLASSES


def remap_ids(raw: np.ndarray, mapping: dict) -> np.ndarray:
    if raw.ndim == 3 and raw.dtype == np.uint8 and raw.shape[2] == 4:
        raw = np.ascontiguousarray(raw).view("<u4").reshape(raw.shape[:2])
    if raw.ndim != 2:
        raise ValueError("Expected raw integer IDs, not colorized semantic output")
    result = np.zeros(raw.shape, np.uint8)
    for key, value in mapping.items():
        name = value.get("class", "background_or_unknown")
        if name in CLASSES:
            result[raw == int(key)] = CLASSES.index(name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    files = sorted(args.source.glob("rgb_*.png"))
    if len(files) < 5:
        raise ValueError("Generate at least five frames for nonempty train/val/test splits")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, image in enumerate(files):
        suffix = image.stem.removeprefix("rgb_")
        label_path = args.source / f"semantic_segmentation_labels_{suffix}.json"
        raw_path = args.source / f"semantic_segmentation_{suffix}.png"
        mask = remap_ids(np.array(Image.open(raw_path)), json.loads(label_path.read_text()))
        if not np.any(mask == 1):
            raise ValueError(f"No sandable surface in frame {suffix}; inspect scene or ID mapping")
        Image.open(image).convert("RGB").save(args.output / f"{suffix}.png")
        Image.fromarray(mask).save(args.output / f"{suffix}_mask.png")
        split = (
            "train"
            if i < int(0.7 * len(files))
            else ("val" if i < int(0.85 * len(files)) else "test")
        )
        rows.append(
            {
                "id": suffix,
                "image": f"{suffix}.png",
                "mask": f"{suffix}_mask.png",
                "source": "Isaac_Sim_Replicator",
                "split": split,
                "condition": "isaac",
            }
        )
    (args.output / "manifest.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
