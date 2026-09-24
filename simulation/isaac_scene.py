"""Isaac-independent scene sampling, sensor noise and label accounting.

Everything here is plain NumPy so geometry and noise can be unit-tested without Isaac Sim.
The panel is a parabolic shell z = h * (1 - (x / A)^2) spanning |x| <= A, |y| <= SPAN.
Objects are posed on the surface: translated along the local normal and rotated about Y so
their local +Z axis matches the normal. Placement uses rejection sampling so nothing overlaps.
"""

import math

import numpy as np

A = 0.65
SPAN = 0.55
MARGIN = 0.015

PAINTS = {
    "bare_aluminium": (0.72, 0.74, 0.76),
    "primer_green": (0.46, 0.55, 0.33),
    "zinc_chromate": (0.72, 0.66, 0.30),
    "white_paint": (0.86, 0.87, 0.88),
    "grey_paint": (0.45, 0.48, 0.52),
    "dark_grey": (0.24, 0.26, 0.28),
}
FLOORS = {
    "concrete": (0.55, 0.54, 0.52),
    "epoxy_grey": (0.38, 0.40, 0.42),
    "epoxy_green": (0.30, 0.42, 0.34),
    "epoxy_blue": (0.28, 0.34, 0.46),
    "dark": (0.14, 0.14, 0.15),
}
SEALANTS = [(0.15, 0.15, 0.16), (0.45, 0.45, 0.47), (0.62, 0.48, 0.25), (0.82, 0.60, 0.20)]
DEFECTS = {
    # name: (colour range low, colour range high, roughness, metallic)
    "stain": ((0.05, 0.04, 0.03), (0.25, 0.18, 0.12), 0.8, 0.0),
    "corrosion": ((0.35, 0.16, 0.05), (0.62, 0.36, 0.14), 0.95, 0.0),
    "scratch": ((0.70, 0.72, 0.75), (0.92, 0.93, 0.95), 0.25, 0.9),
    "paint_chip": ((0.55, 0.57, 0.60), (0.80, 0.82, 0.84), 0.4, 0.7),
}


def panel_z(x, h: float):
    return h * (1.0 - (np.asarray(x) / A) ** 2)


def panel_normal(x: float, h: float) -> np.ndarray:
    slope = -2.0 * h * x / A**2
    normal = np.array([-slope, 0.0, 1.0])
    return normal / np.linalg.norm(normal)


def surface_pose(x: float, y: float, h: float, offset: float = 0.0) -> tuple[list, float]:
    """Point on the shell pushed `offset` metres along the normal, plus Y rotation (deg)."""
    normal = panel_normal(x, h)
    position = np.array([x, y, float(panel_z(x, h))]) + offset * normal
    return position.tolist(), math.degrees(math.atan2(normal[0], normal[2]))


def tangent_sag(half_extent: float, h: float) -> float:
    """Max gap between the tangent plane and the shell over +-half_extent along x."""
    return h * half_extent**2 / A**2


class Placer:
    """Rejection sampler for non-overlapping circular footprints in panel (x, y)."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.circles: list[tuple[float, float, float]] = []
        self.x_bands: list[tuple[float, float]] = []  # spanwise seams at fixed x
        self.y_bands: list[tuple[float, float]] = []  # chordwise seams at fixed y

    def free(self, x: float, y: float, r: float) -> bool:
        if abs(x) + r > A - 0.05 or abs(y) + r > SPAN - 0.05:
            return False
        if any(math.hypot(x - cx, y - cy) < r + cr + MARGIN for cx, cy, cr in self.circles):
            return False
        if any(abs(x - bx) < r + bw + MARGIN for bx, bw in self.x_bands):
            return False
        return not any(abs(y - by) < r + bw + MARGIN for by, bw in self.y_bands)

    def place(self, r: float, tries: int = 400) -> tuple[float, float] | None:
        for _ in range(tries):
            x = float(self.rng.uniform(-(A - 0.05 - r), A - 0.05 - r))
            y = float(self.rng.uniform(-(SPAN - 0.05 - r), SPAN - 0.05 - r))
            if self.free(x, y, r):
                self.circles.append((x, y, r))
                return x, y
        return None


def _colour(rng, low, high) -> list:
    return rng.uniform(low, high).tolist()


def sample_layout(rng: np.random.Generator, layout_id: int) -> dict:
    """One physical arrangement. Several camera/light views are rendered per layout."""
    h = float(rng.uniform(0.05, 0.24))
    paint = str(rng.choice(list(PAINTS)))
    placer = Placer(rng)
    layout = {
        "layout": layout_id,
        "panel_height": h,
        "paint": paint,
        "panel_roughness": float(rng.uniform(0.15, 0.85)),
        "panel_metallic": float(rng.uniform(0.0, 0.9 if paint == "bare_aluminium" else 0.3)),
        "floor": str(rng.choice(list(FLOORS))),
        "dome_colour": _colour(rng, [0.6, 0.6, 0.6], [1.0, 1.0, 1.0]),
    }
    # Spanwise seam: straight along y at fixed x, so it lies exactly on the shell.
    width, thick = float(rng.uniform(0.010, 0.024)), float(rng.uniform(0.003, 0.008))
    sx = float(rng.uniform(-0.45, 0.45))
    pos, angle = surface_pose(sx, 0.0, h, thick / 2 - 0.001)
    layout["seam"] = {
        "x": sx,
        "position": pos,
        "angle": angle,
        "size": [width, 2 * SPAN, thick],
        "colour": list(SEALANTS[int(rng.integers(len(SEALANTS)))]),
    }
    placer.x_bands.append((sx, width / 2))
    # Optional chordwise seam: a strip mesh that follows the curvature along x.
    if rng.random() < 0.5:
        cy, cw = float(rng.uniform(-0.4, 0.4)), float(rng.uniform(0.010, 0.022))
        layout["chord_seam"] = {
            "y": cy,
            "width": cw,
            "thickness": float(rng.uniform(0.003, 0.007)),
            "colour": list(SEALANTS[int(rng.integers(len(SEALANTS)))]),
        }
        placer.y_bands.append((cy, cw / 2))
    else:
        layout["chord_seam"] = None

    # Window: flat framed pane on the tangent plane; the frame extends down by the sag so its
    # edges meet the curved shell instead of floating above it.
    scale = float(rng.uniform(0.7, 1.3))
    hx, hy = 0.125 * scale, 0.10 * scale
    spot = placer.place(math.hypot(hx, hy))
    if spot:
        pos, angle = surface_pose(*spot, h, 0.0)
        layout["window"] = {
            "position": pos,
            "angle": angle,
            "scale": scale,
            "yaw": float(rng.choice([0.0, 90.0])),
            "sag": tangent_sag(max(hx, hy), h),
        }
    else:
        layout["window"] = None

    kind = "mast" if rng.random() < 0.5 else "blade"
    if kind == "mast":
        size = [float(rng.uniform(0.015, 0.03)), float(rng.uniform(0.08, 0.25))]
        radius = size[0]
    else:
        size = [float(rng.uniform(0.08, 0.15)), 0.012, float(rng.uniform(0.06, 0.14))]
        radius = size[0] / 2
    spot = placer.place(radius)
    layout["antenna"] = None
    if spot:
        pos, angle = surface_pose(*spot, h, 0.0)
        layout["antenna"] = {
            "kind": kind,
            "size": size,
            "position": pos,
            "angle": angle,
            "yaw": float(rng.uniform(0, 180)),
        }
    layout["tooling"] = None
    if rng.random() < 0.5:
        size = rng.uniform([0.03, 0.03, 0.03], [0.09, 0.07, 0.08]).tolist()
        spot = placer.place(math.hypot(size[0], size[1]) / 2)
        if spot:
            pos, angle = surface_pose(*spot, h, size[2] / 2)
            layout["tooling"] = {
                "size": size,
                "position": pos,
                "angle": angle,
                "yaw": float(rng.uniform(0, 180)),
                "colour": _colour(rng, [0.05, 0.05, 0.05], [0.9, 0.3, 0.2]),
            }

    defects = []
    for _ in range(int(rng.integers(1, 4))):
        kind = str(rng.choice(list(DEFECTS)))
        if kind == "scratch":
            size = [float(rng.uniform(0.05, 0.14)), float(rng.uniform(0.004, 0.008))]
        else:
            size = rng.uniform([0.02, 0.015], [0.08, 0.05]).tolist()
        spot = placer.place(math.hypot(*size) / 2)
        if spot is None:
            continue
        thick = 0.0012
        pos, angle = surface_pose(*spot, h, thick / 2 + 0.0002)
        low, high, rough, metal = DEFECTS[kind]
        defects.append(
            {
                "kind": kind,
                "size": [*size, thick],
                "position": pos,
                "angle": angle,
                "yaw": float(rng.uniform(0, 180)),
                "colour": _colour(rng, low, high),
                "roughness": rough,
                "metallic": metal,
            }
        )
    layout["defects"] = defects

    # Second, flat workpiece lying beside the shell.
    layout["flat_panel"] = None
    if rng.random() < 0.6:
        side = 1 if rng.random() < 0.5 else -1
        layout["flat_panel"] = {
            "position": [
                side * float(rng.uniform(0.95, 1.25)),
                float(rng.uniform(-0.3, 0.3)),
                0.012,
            ],
            "size": [float(rng.uniform(0.35, 0.6)), float(rng.uniform(0.6, 1.1)), 0.024],
            "yaw": float(rng.uniform(-20, 20)),
            "paint": str(rng.choice(list(PAINTS))),
        }
    # Unlabelled clutter on the floor, sometimes painted like the workpiece to break colour cues.
    clutter = []
    keep_out = [(0.0, 0.0, A + 0.05, SPAN + 0.05)]  # centre x, y, half extents
    if layout["flat_panel"]:
        fx, fy, _ = layout["flat_panel"]["position"]
        fsx, fsy, _ = layout["flat_panel"]["size"]
        keep_out.append((fx, fy, fsx / 2 + 0.1, fsy / 2 + 0.1))
    for _ in range(int(rng.integers(0, 5))):
        angle, dist = float(rng.uniform(0, 2 * math.pi)), float(rng.uniform(1.0, 2.2))
        size = rng.uniform([0.08, 0.08, 0.05], [0.5, 0.5, 0.5]).tolist()
        cx, cy, r = dist * math.cos(angle), dist * math.sin(angle), math.hypot(*size[:2]) / 2
        if any(abs(cx - kx) < hx + r and abs(cy - ky) < hy + r for kx, ky, hx, hy in keep_out):
            continue
        colour = (
            list(PAINTS[str(rng.choice(list(PAINTS)))])
            if rng.random() < 0.4
            else _colour(rng, [0.05] * 3, [0.9] * 3)
        )
        keep_out.append((cx, cy, r, r))
        clutter.append(
            {
                "shape": "box" if rng.random() < 0.6 else "cylinder",
                "position": [cx, cy, size[2] / 2],
                "size": size,
                "yaw": float(rng.uniform(0, 180)),
                "colour": colour,
            }
        )
    layout["clutter"] = clutter
    return layout


def sample_view(rng: np.random.Generator, layout: dict) -> dict:
    h = layout["panel_height"]
    tx, ty = float(rng.uniform(-0.35, 0.45)), float(rng.uniform(-0.3, 0.3))
    target = [tx, ty, float(panel_z(tx, h))]
    distance = float(rng.uniform(0.9, 2.3))
    azimuth = math.radians(float(rng.uniform(0, 360)))
    elevation = math.radians(float(rng.uniform(30, 80)))
    camera = [
        target[0] + distance * math.cos(elevation) * math.cos(azimuth),
        target[1] + distance * math.cos(elevation) * math.sin(azimuth),
        target[2] + distance * math.sin(elevation),
    ]
    return {
        "camera_position": camera,
        "camera_target": target,
        "key_intensity": float(rng.uniform(300, 4000)),
        "key_colour": _colour(rng, [0.75] * 3, [1.0] * 3),
        "key_rotation": rng.uniform([-75, -60, -180], [-10, 60, 180]).tolist(),
        "dome_intensity": float(rng.uniform(60, 700)),
    }


def paint_texture(rng: np.random.Generator, base, size: int = 256, grime: float = 1.0):
    """Low-frequency blotches + fine grain + sanding streaks, as uint8 sRGB."""
    base = np.asarray(base, dtype=np.float32)
    coarse = rng.normal(0, 1, (8, 8, 1)).astype(np.float32)
    coarse = np.kron(coarse, np.ones((size // 8, size // 8, 1), np.float32))
    kernel = np.ones(size // 8, np.float32) / (size // 8)
    for axis in (0, 1):
        coarse = np.apply_along_axis(lambda v: np.convolve(v, kernel, "same"), axis, coarse)
    image = base * (1 + grime * float(rng.uniform(0.02, 0.10)) * coarse)
    image = image + grime * rng.normal(0, rng.uniform(0.005, 0.04), (size, size, 3))
    if rng.random() < 0.5:
        rows = np.arange(size)[:, None, None]
        streak = np.sin(rows * rng.uniform(0.5, 2.0) + rng.normal(0, 0.8, (1, size, 1)))
        image = image + grime * float(rng.uniform(0.005, 0.03)) * streak
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def depth_noise(raw: np.ndarray, rng: np.random.Generator, params: dict) -> np.ndarray:
    """Stereo/ToF-like depth: range-dependent noise, flying-pixel and random dropout.

    Invalid measurements are 0 (the common driver convention); the clean input is untouched.
    """
    depth = raw.astype(np.float32, copy=True)
    valid = np.isfinite(depth) & (depth > 0)
    z = np.where(valid, depth, 0.0)
    sigma = params["sigma0"] + params["sigma2"] * z**2
    noisy = z + rng.normal(0.0, 1.0, z.shape).astype(np.float32) * sigma
    # Depth discontinuities produce mixed/flying pixels on real sensors.
    padded = np.pad(np.where(valid, depth, 1e3), 1, mode="edge")
    jump = np.zeros_like(z)
    for dy, dx in [(0, 1), (2, 1), (1, 0), (1, 2)]:
        neighbour = padded[dy : dy + z.shape[0], dx : dx + z.shape[1]]
        jump = np.maximum(jump, np.abs(neighbour - np.where(valid, depth, 1e3)))
    edge = valid & (jump > params["edge_threshold"] * np.maximum(z, 1e-3))
    drop = edge & (rng.random(z.shape) < params["edge_dropout"])
    drop |= valid & (rng.random(z.shape) < params["dropout"])
    noisy[~valid | drop] = 0.0
    return np.maximum(noisy, 0.0).astype(np.float32)


DEFAULT_DEPTH_NOISE = {
    "sigma0": 0.0008,
    "sigma2": 0.0012,
    "edge_threshold": 0.03,
    "edge_dropout": 0.7,
    "dropout": 0.003,
}


def class_pixels(semantic: np.ndarray, labels: dict) -> dict:
    """Visible pixels per semantic class name from BasicWriter raw IDs + label JSON."""
    if semantic.ndim == 3:
        semantic = np.ascontiguousarray(semantic).view("<u4").reshape(semantic.shape[:2])
    ids, counts = np.unique(semantic, return_counts=True)
    out: dict[str, int] = {}
    for i, n in zip(ids.tolist(), counts.tolist()):
        name = labels.get(str(i), {}).get("class", "UNKNOWN")
        out[name] = out.get(name, 0) + int(n)
    return out


def instance_pixels(instance: np.ndarray, mapping: dict) -> dict:
    """Visible pixels per prim path from colourised instance PNG + mapping JSON."""
    if instance.ndim != 3:
        ids, counts = np.unique(instance, return_counts=True)
        keys = [str(int(i)) for i in ids]
    else:
        flat = instance.reshape(-1, instance.shape[-1])
        colours, counts = np.unique(flat, axis=0, return_counts=True)
        keys = [str(tuple(int(c) for c in colour)) for colour in colours]
    return {mapping[k]: int(n) for k, n in zip(keys, counts.tolist()) if k in mapping}
