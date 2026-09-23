"""Deterministic 2-D raster fallback; NOT Isaac Sim or physical rendering."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SLICES = [
    "nominal",
    "low_light",
    "overexposure",
    "texture",
    "angle",
    "blur",
    "noise",
    "small_defect",
    "occlusion",
]


def scene(seed: int, width: int, height: int, condition: str = "nominal") -> tuple:
    if condition not in SLICES:
        raise ValueError(f"Unknown condition: {condition}")
    rng = np.random.default_rng(seed)
    w, h = width, height
    labels = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(labels)
    inset = int(rng.uniform(0.06, 0.17) * w)
    slant = int(rng.uniform(-0.09, 0.09) * w)
    if condition == "angle":
        slant = int(0.25 * w)
    polygon = [(inset + slant, 8), (w - inset, 8), (w - 5, h - 8), (5, h - 8)]
    draw.polygon(polygon, fill=1)
    px, py = int(rng.uniform(0.25, 0.7) * w), int(rng.uniform(0.25, 0.65) * h)
    protected = (px, py, px + w // 9, py + h // 4)
    draw.rectangle(protected, fill=2)
    # A seam is conservatively protected, never silently labelled sandable.
    seam_x = int(rng.uniform(0.2, 0.8) * w)
    draw.line([(seam_x, 10), (seam_x + slant // 2, h - 10)], fill=2, width=2)
    defects = []
    for _ in range(int(rng.integers(1, 4))):
        x, y = int(rng.uniform(0.18, 0.8) * w), int(rng.uniform(0.16, 0.8) * h)
        radius = 1 if condition == "small_defect" else int(rng.integers(3, 8))
        box = (x, y, x + radius * 2, y + radius)
        draw.ellipse(box, fill=3)
        defects.append(box)
    obstacle = None
    if rng.random() < 0.8 or condition == "occlusion":
        x, y = (
            (px, py)
            if condition == "occlusion"
            else (int(rng.uniform(0.15, 0.7) * w), int(rng.uniform(0.2, 0.7) * h))
        )
        obstacle = (x - 3, y - 4, x + w // 8, y + h // 8)
        draw.ellipse(obstacle, fill=4)
    mask = np.asarray(labels)
    colors = np.array(
        [[30, 38, 48], [160, 178, 188], [200, 150, 55], [105, 55, 50], [48, 70, 108]],
        dtype=np.float32,
    )
    material = rng.uniform(-22, 22, (5, 3))
    colors += material
    yy, xx = np.mgrid[:h, :w]
    direction = float(rng.uniform(-np.pi, np.pi))
    roughness, metallic = float(rng.uniform(0.2, 0.9)), float(rng.uniform(0.1, 0.9))
    curved = bool(rng.integers(2))
    shading = 0.82 + 0.22 * np.cos(xx / w * np.pi) if curved else np.ones((h, w))
    shading += 0.12 * (np.cos(direction) * xx / w + np.sin(direction) * yy / h)
    illumination = float(rng.uniform(0.8, 1.2))
    light_color = rng.uniform(0.9, 1.1, 3)
    rgb = colors[mask] * shading[..., None] * illumination * light_color
    rgb += rng.normal(0, 2 + roughness * 3, rgb.shape)
    texture_amplitude = 35 if condition == "texture" else 5
    rgb += texture_amplitude * np.sin(xx[..., None] * 0.45 + yy[..., None] * 0.1)
    if condition == "low_light":
        rgb *= 0.35
    elif condition == "overexposure":
        rgb = rgb * 1.8 + 35
    elif condition == "noise":
        rgb += rng.normal(0, 35, rgb.shape)
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    if condition == "blur":
        image = image.filter(ImageFilter.GaussianBlur(2.2))
    distance = float(rng.uniform(0.7, 1.4))
    depth = np.where(mask > 0, distance + 0.08 * (xx / w - 0.5) ** 2, 0).astype(np.float32)
    depth[mask == 4] -= 0.08
    metadata = {
        "source": "procedural_2d_NOT_Isaac_Sim",
        "seed": seed,
        "condition": condition,
        "curved_shading": curved,
        "panel": polygon,
        "illumination": illumination,
        "light_direction_rad": direction,
        "light_color": light_color.tolist(),
        "material_offset": material.tolist(),
        "roughness_proxy": roughness,
        "metallic_metadata_only": metallic,
        "camera_slant_proxy": slant,
        "distance_proxy_m": distance,
        "protected": protected,
        "defects": defects,
        "obstacle": obstacle,
        "depth_note": "analytic proxy, not calibrated geometry",
    }
    return np.asarray(image), mask.copy(), depth, metadata


def generate(root: Path, config: dict) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for split, offset in [("train", 0), ("val", 100000), ("test", 200000)]:
        for i in range(config[f"{split}_samples"]):
            # Test stress conditions are held out of training; val selects on nominal only.
            condition = SLICES[i % len(SLICES)] if split == "test" else "nominal"
            seed = config["seed"] + offset + i
            image, mask, depth, metadata = scene(seed, config["width"], config["height"], condition)
            stem = f"{split}_{i:05d}"
            Image.fromarray(image).save(root / f"{stem}.png")
            Image.fromarray(mask).save(root / f"{stem}_mask.png")
            np.save(root / f"{stem}_depth.npy", depth)
            manifest.append(
                {
                    "id": stem,
                    "split": split,
                    "image": f"{stem}.png",
                    "mask": f"{stem}_mask.png",
                    **metadata,
                }
            )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"samples": len(manifest), "source": "procedural_2d_NOT_Isaac_Sim"}
