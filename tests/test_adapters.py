import numpy as np
import pytest
import torch
from PIL import Image

from simulation.prepare_isaac import remap_ids
from training.real_domain import RealDataset, binary_logits, pairs


def test_replicator_id_mapping():
    raw = np.array([[0, 42], [7, 99]], np.uint32)
    mapping = {"42": {"class": "surface_defect"}, "7": {"class": "sandable_surface"}}
    np.testing.assert_array_equal(remap_ids(raw, mapping), [[0, 3], [1, 0]])
    rgba = raw.view(np.uint8).reshape(2, 2, 4)
    np.testing.assert_array_equal(remap_ids(rgba, mapping), [[0, 3], [1, 0]])


def test_binary_collapse_preserves_probability():
    torch.manual_seed(1)
    logits = torch.randn(2, 5, 8, 8)
    torch.testing.assert_close(binary_logits(logits).softmax(1)[:, 1], logits.softmax(1)[:, 3])


def test_ksdd2_adapter_rejects_missing_masks(tmp_path):
    directory = tmp_path / "train"
    directory.mkdir()
    Image.new("RGB", (32, 80)).save(directory / "10000.png")
    with pytest.raises(ValueError, match="Missing mask"):
        pairs(tmp_path, "train")
    mask = np.zeros((80, 32), np.uint8)
    mask[20:30, 10:15] = 255
    Image.fromarray(mask).save(directory / "10000_GT.png")
    rows = pairs(tmp_path, "train")
    assert len(rows) == 1 and rows[0]["positive"]
    x, y = RealDataset(rows)[0]
    assert x.shape == (3, 320, 128) and set(y.unique().tolist()) == {0, 1}
    (directory / "10000 (copy).png").write_bytes((directory / "10000.png").read_bytes())
    assert len(pairs(tmp_path, "train")) == 1
