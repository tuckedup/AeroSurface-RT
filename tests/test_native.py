"""Native integration tests run when the built executable and trained model exist."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aerosurface.data import preprocess
from aerosurface.postprocess import postprocess
from aerosurface.runtime import ort_session


def native_paths():
    candidates = [
        Path("build/cpp/Release/aerosurface_infer.exe"),
        Path("build/cpp/aerosurface_infer"),
    ]
    executable = next((p for p in candidates if p.exists()), None)
    model = Path("models/edge/model.onnx")
    if executable is None or not model.exists():
        pytest.skip("Build native ORT runtime and export edge model to enable integration")
    return executable.resolve(), model.resolve()


def test_native_rejects_resolution_and_truncation(tmp_path):
    executable, model = native_paths()
    bad = tmp_path / "bad.ppm"
    bad.write_bytes(b"P6\n160 128\n255\nabc")
    for w, h, reason in [(160, 128, "Truncated"), (128, 128, "static float32")]:
        run = subprocess.run(
            [str(executable), str(model), str(bad), str(tmp_path / "out"), str(w), str(h)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode != 0 and reason in run.stderr


def test_native_inference_resize_and_regions(tmp_path):
    executable, model = native_paths()
    # Non-model dimensions exercise C++ resize as well as tensor and ROI contracts.
    rgb = np.asarray(Image.open("data/procedural/test_00001.png").convert("RGB").resize((113, 97)))
    image = tmp_path / "input.ppm"
    Image.fromarray(rgb).save(image)
    prefix = tmp_path / "result"
    subprocess.run(
        [str(executable), str(model), str(image), str(prefix), "160", "128"],
        capture_output=True,
        text=True,
        check=True,
    )
    logits = ort_session(model).run(None, {"rgb": preprocess(rgb, 128, 160)})[0]
    result = postprocess(logits)
    for name in ["mask", "sandable", "avoid", "protected", "defect"]:
        np.testing.assert_array_equal(
            result[name], np.array(Image.open(tmp_path / f"result_{name}.pgm"))
        )
