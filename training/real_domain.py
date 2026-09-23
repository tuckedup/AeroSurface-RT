"""KSDD2 binary transfer experiment; nondefect is NOT synonymous with sandable.

Collapse five logits into defect (class 3) versus logsumexp of the other four.
Uses official test split untouched and a deterministic subset of official train.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from aerosurface.data import preprocess
from aerosurface.metrics import confusion, summarize
from aerosurface.model import SurfaceUNet, load_model
from training.train import seed_all


def binary_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.logsumexp(logits[:, [0, 1, 2, 4]], dim=1), logits[:, 3]], dim=1)


def pairs(root: Path, split: str) -> list[dict]:
    rows = []
    for image in sorted((root / split).glob("*.png")):
        if image.stem.endswith("_GT"):
            continue
        mask = image.with_name(image.stem + "_GT.png")
        if not mask.exists():
            original = image.with_name(image.name.replace(" (copy)", ""))
            if (
                original != image
                and original.exists()
                and hashlib.sha256(image.read_bytes()).digest()
                == hashlib.sha256(original.read_bytes()).digest()
            ):
                print(
                    f"Skipping verified byte-identical unlabelled duplicate: {image.name}",
                    flush=True,
                )
                continue
            raise ValueError(f"Missing mask for {image.name}")
        with Image.open(mask) as source:
            positive = bool(np.any(np.asarray(source) > 0))
        rows.append({"image": image, "mask": mask, "positive": positive})
    if not rows:
        raise ValueError(f"No official KSDD2 {split} images found")
    return rows


class RealDataset(Dataset):
    def __init__(self, rows: list[dict], augment: bool = False):
        self.rows, self.augment = rows, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image"]) as image:
            rgb = np.array(image.convert("RGB"))
        with Image.open(row["mask"]) as image:
            mask = (np.array(image.convert("L")) > 0).astype(np.uint8)
        if rgb.shape[:2] != mask.shape:
            raise ValueError("Real-domain image and mask are not aligned")
        # Approximate original tall aspect ratio; nearest keeps categorical labels.
        h, w = 320, 128
        y = mask[(np.arange(h) * mask.shape[0] // h)[:, None], np.arange(w) * mask.shape[1] // w]
        x = preprocess(rgb, h, w)[0]
        if self.augment and torch.rand(()) < 0.5:
            x, y = x[:, :, ::-1].copy(), y[:, ::-1].copy()
        return torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))


@torch.inference_mode()
def evaluate_binary(model, loader, device) -> dict:
    model.eval()
    cm = np.zeros((2, 2), np.int64)
    images = 0
    for x, y in loader:
        prediction = binary_logits(model(x.to(device))).argmax(1).cpu().numpy()
        cm += confusion(y.numpy(), prediction, 2)
        images += len(x)
    return {"images": images, **summarize(cm, ["nondefect", "defect"])}


def run(root: Path, checkpoint: Path, output: Path, epochs: int, device: str) -> dict:
    seed_all(123)
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    all_train, test_rows = pairs(root, "train"), pairs(root, "test")
    rng = np.random.default_rng(123)
    positive = [r for r in all_train if r["positive"]]
    negative = [r for r in all_train if not r["positive"]]
    rng.shuffle(positive)
    rng.shuffle(negative)
    # No overlapping identities; selection uses official train masks only.
    train_rows = positive[:64] + negative[:192]
    val_rows = positive[64:80] + negative[192:240]
    train_loader = DataLoader(RealDataset(train_rows, True), batch_size=8, shuffle=True)
    val_loader = DataLoader(RealDataset(val_rows), batch_size=8)
    test_loader = DataLoader(RealDataset(test_rows), batch_size=8)
    counts = np.zeros(2)
    for _, y in RealDataset(train_rows):
        counts += np.bincount(y.numpy().ravel(), minlength=2)
    weight = np.sqrt(counts.sum() / np.maximum(counts, 1))
    weight = torch.tensor(weight / weight.mean(), dtype=torch.float32, device=device)
    output.mkdir(parents=True, exist_ok=True)
    config = load_model(checkpoint)[1]
    results = {}
    for name in ["synthetic_only", "real_only", "synthetic_then_real"]:
        seed_all(123)
        model = (
            SurfaceUNet(config["base_channels"])
            if name == "real_only"
            else load_model(checkpoint)[0]
        )
        model.to(device)
        history, start = [], time.perf_counter()
        best, best_state = -1.0, None
        if name != "synthetic_only":
            optim = torch.optim.AdamW(model.parameters(), lr=0.001)
            for epoch in range(epochs):
                model.train()
                total = 0.0
                for x, y in train_loader:
                    optim.zero_grad(set_to_none=True)
                    logits = binary_logits(model(x.to(device)))
                    loss = torch.nn.functional.cross_entropy(
                        logits.permute(0, 2, 3, 1).reshape(-1, 2), y.to(device).reshape(-1), weight
                    )
                    loss.backward()
                    optim.step()
                    total += loss.item() * len(x)
                validation = evaluate_binary(model, val_loader, device)
                score = validation["per_class"]["defect"]["iou"] or 0.0
                row = {"epoch": epoch + 1, "loss": total / len(train_rows), "val_defect_iou": score}
                history.append(row)
                print(name, json.dumps(row), flush=True)
                if score > best:
                    best = score
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
            model.load_state_dict(best_state)
            torch.save(
                {"state_dict": best_state, "config": config, "label_space": "binary_collapsed"},
                output / f"{name}.pt",
            )
        results[name] = {
            "metrics": evaluate_binary(model, test_loader, device),
            "history": history,
            "seconds": time.perf_counter() - start,
        }
        print(name, json.dumps(results[name]["metrics"]), flush=True)
        del model
    result = {
        "dataset": "KolektorSDD2",
        "license": "CC BY-NC-SA 4.0",
        "source": "https://www.vicos.si/resources/kolektorsdd2/",
        "seed": 123,
        "input_shape": [1, 3, 320, 128],
        "epochs": epochs,
        "device": device,
        "train_images": [r["image"].name for r in train_rows],
        "val_images": [r["image"].name for r in val_rows],
        "test_count": len(test_rows),
        "selection": "best validation defect IoU",
        "comparison": "binary defect only; no sandability/protected/obstacle claims",
        "class_counts": counts.tolist(),
        "results": results,
    }
    (output / "real_domain.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/ksdd2"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/edge/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/real_domain"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    run(args.data, args.checkpoint, args.output, args.epochs, args.device)


if __name__ == "__main__":
    main()
