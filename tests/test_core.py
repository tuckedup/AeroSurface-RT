import json

import numpy as np
import pytest
import torch

from aerosurface.config import load_config
from aerosurface.data import SurfaceDataset, preprocess
from aerosurface.experiments import demo, export, parity
from aerosurface.metrics import confusion, summarize
from aerosurface.model import SurfaceUNet
from aerosurface.postprocess import postprocess
from simulation.procedural import SLICES, generate, scene


@pytest.mark.parametrize("condition", SLICES)
def test_scene_contract_and_seed(condition):
    a = scene(42, 64, 48, condition)
    b = scene(42, 64, 48, condition)
    for i in range(3):
        np.testing.assert_array_equal(a[i], b[i])
    assert a[0].shape == (48, 64, 3)
    assert set(np.unique(a[1])).issubset(set(range(5)))
    assert np.isfinite(a[2]).all()


def test_preprocess_rgb_and_stride():
    rgb = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], np.uint8)
    x = preprocess(rgb[:, ::-1], 4, 4)
    assert x.shape == (1, 3, 4, 4) and x.flags.c_contiguous
    np.testing.assert_array_equal(x[0, :, 0, 0], [0, 1, 0])
    with pytest.raises(ValueError):
        preprocess(rgb.astype(float), 2, 2)


def test_metrics_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 1, 1])
    result = summarize(confusion(y, p, 2), ["a", "b"])
    assert result["miou"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert result["per_class"]["b"]["recall"] == 1
    assert summarize(np.diag([2, 0]), ["a", "b"])["per_class"]["b"]["iou"] is None
    with pytest.raises(ValueError):
        confusion(y, np.array([0, 1, 2, 3]), 2)


def test_fail_closed_regions():
    logits = np.zeros((1, 5, 11, 11), np.float32)
    assert not postprocess(logits)["sandable"].any()
    logits[:, 1] = 10
    logits[:, 2, 5, 5] = 20
    out = postprocess(logits, 0.65, 2)
    assert not out["sandable"][3:8, 3:8].any()
    assert not out["sandable"][:2].any()
    assert out["sandable"][2, 2] == 255
    logits[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        postprocess(logits)


def test_config_validation(tmp_path):
    cfg = load_config("configs/edge.json")
    cfg["width"] = 161
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        load_config(path)


def test_split_disjoint_and_dataset(tmp_path):
    cfg = load_config("configs/edge.json")
    cfg.update(train_samples=3, val_samples=2, test_samples=2)
    generate(tmp_path, cfg)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len({r["seed"] for r in manifest}) == 7
    x, y, _ = SurfaceDataset(tmp_path, "train", cfg)[0]
    assert x.shape == (3, 128, 160) and y.dtype == torch.int64


def test_onnx_parity_and_end_to_end(tmp_path):
    cfg = load_config("configs/edge.json")
    cfg.update(train_samples=1, val_samples=1, test_samples=2, width=32, height=32)
    torch.manual_seed(7)
    model = SurfaceUNet(cfg["base_channels"])
    checkpoint = tmp_path / "test.pt"
    torch.save({"config": cfg, "state_dict": model.state_dict()}, checkpoint)
    generate(tmp_path / "data", cfg)
    export(checkpoint, tmp_path / "test.onnx")
    result = parity(checkpoint, tmp_path / "test.onnx", tmp_path / "data", tmp_path / "parity.json")
    assert result["passed"]
    demo(checkpoint, tmp_path / "test.onnx", tmp_path / "data", tmp_path / "demo")
    assert (tmp_path / "demo/mask.png").exists()
