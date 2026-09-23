"""Export, numerical parity, held-out evaluation and representative artifacts."""

import json
from pathlib import Path

import numpy as np
import onnx
import torch
from PIL import Image, ImageDraw

from aerosurface import PALETTE
from aerosurface.data import SurfaceDataset, preprocess
from aerosurface.metrics import confusion, summarize
from aerosurface.model import load_model
from aerosurface.postprocess import overlay, postprocess
from aerosurface.runtime import ort_session


def export(checkpoint: Path, output: Path, half: bool = False) -> dict:
    model, config = load_model(checkpoint)
    x = torch.zeros(1, 3, config["height"], config["width"])
    if half:
        model, x = model.half(), x.half()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        x,
        str(output),
        opset_version=17,
        input_names=["rgb"],
        output_names=["logits"],
        dynamo=False,
    )
    onnx.checker.check_model(str(output))
    return {"onnx_bytes": output.stat().st_size, "dtype": "float16" if half else "float32"}


@torch.inference_mode()
def parity(checkpoint: Path, onnx_path: Path, data: Path, output: Path) -> dict:
    model, config = load_model(checkpoint)
    torch.set_num_threads(config["threads"])
    session = ort_session(onnx_path, config["threads"])
    ds = SurfaceDataset(data, "test", config)
    errors, agreement = [], []
    for i in range(min(12, len(ds))):
        x = ds[i][0][None]
        a, b = model(x).numpy(), session.run(None, {"rgb": x.numpy()})[0]
        if a.shape != (1, 5, config["height"], config["width"]) or a.shape != b.shape:
            raise AssertionError("Incorrect exported logits shape")
        np.testing.assert_allclose(a, b, rtol=2e-4, atol=1e-4)
        errors.append(float(np.max(np.abs(a - b))))
        agreement.append(float(np.mean(a.argmax(1) == b.argmax(1))))
    result = {
        "passed": True,
        "samples": len(errors),
        "max_absolute_error": max(errors),
        "mean_argmax_agreement": float(np.mean(agreement)),
        "rtol": 2e-4,
        "atol": 1e-4,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


@torch.inference_mode()
def evaluate(checkpoint: Path, data: Path, output: Path) -> dict:
    model, config = load_model(checkpoint)
    torch.set_num_threads(config["threads"])
    ds = SurfaceDataset(data, "test", config)
    cms, per_image = {}, []
    all_cm = np.zeros((5, 5), dtype=np.int64)
    false_safe, true_safe, candidate_pixels = 0, 0, 0
    for x, y, i in ds:
        logits = model(x[None]).numpy()
        pred = logits.argmax(1)[0]
        cm = confusion(y.numpy(), pred)
        all_cm += cm
        condition = ds.rows[i]["condition"]
        cms.setdefault(condition, np.zeros((5, 5), dtype=np.int64))
        cms[condition] += cm
        regions = postprocess(logits, config["confidence"], config["margin"])
        safe = regions["sandable"] > 0
        false_safe += int(np.sum(safe & (y.numpy() != 1)))
        true_safe += int(np.sum(y.numpy() == 1))
        candidate_pixels += int(safe.sum())
        per_image.append(
            {"id": ds.rows[i]["id"], "condition": condition, "miou": summarize(cm)["miou"]}
        )
    result = {
        "source": sorted({row.get("source", "unspecified") for row in ds.rows}),
        "samples": len(ds),
        "raw_argmax": summarize(all_cm),
        "slices": {key: summarize(cm) for key, cm in cms.items()},
        "roi": {
            "false_sandable_pixels": false_safe,
            "candidate_pixels": candidate_pixels,
            "false_sandable_fraction": false_safe / max(candidate_pixels, 1),
            "sandable_coverage": (candidate_pixels - false_safe) / max(true_safe, 1),
            "confidence": config["confidence"],
            "erosion_margin_px": config["margin"],
        },
        "per_image": per_image,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


def demo(checkpoint: Path, onnx_path: Path, data: Path, output: Path) -> dict:
    _, config = load_model(checkpoint)
    ds = SurfaceDataset(data, "test", config)
    session = ort_session(onnx_path, config["threads"])
    output.mkdir(parents=True, exist_ok=True)
    w, h = config["width"], config["height"]
    canvas = Image.new("RGB", (w * 4, (h + 24) * 9), (12, 18, 30))
    draw = ImageDraw.Draw(canvas)
    for i in range(min(9, len(ds))):
        row = ds.rows[i]
        rgb = np.array(Image.open(data / row["image"]).convert("RGB"))
        logits = session.run(None, {"rgb": preprocess(rgb, h, w)})[0]
        regions = postprocess(logits, config["confidence"], config["margin"])
        pred = overlay(rgb, regions["mask"])
        truth = np.array(PALETTE, dtype=np.uint8)[ds[i][1].numpy()]
        safe = np.repeat(regions["sandable"][..., None], 3, 2)
        for j, (frame, title) in enumerate(
            zip(
                [rgb, truth, pred, safe],
                [row["condition"], "Ground truth", "ORT overlay", "Candidate ROI"],
            )
        ):
            y = i * (h + 24)
            draw.text((j * w + 4, y + 4), title, fill="white")
            canvas.paste(Image.fromarray(frame), (j * w, y + 24))
        if i == 0:
            Image.fromarray(rgb).save(output / "input.ppm")
            np.save(output / "input.npy", preprocess(rgb, h, w))
            np.save(output / "logits.npy", logits)
            for name in ["mask", "sandable", "avoid", "defect", "protected"]:
                Image.fromarray(regions[name]).save(output / f"{name}.png")
            Image.fromarray(pred).save(output / "overlay.png")
    canvas.save(output / "contact_sheet.png")
    result = {
        "backend": "ONNX Runtime CPU",
        "frames": min(9, len(ds)),
        "source": sorted({row.get("source", "unspecified") for row in ds.rows}),
    }
    (output / "demo.json").write_text(json.dumps(result, indent=2))
    return result
