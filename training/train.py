"""Short reproducible supervised training with best-validation selection."""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aerosurface.data import SurfaceDataset
from aerosurface.metrics import confusion, summarize
from aerosurface.model import SurfaceUNet


def seed_all(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


@torch.inference_mode()
def score(model, loader, device) -> dict:
    model.eval()
    cm = np.zeros((5, 5), dtype=np.int64)
    for x, y, _ in loader:
        prediction = model(x.to(device)).argmax(1).cpu().numpy()
        cm += confusion(y.numpy(), prediction)
    return summarize(cm)


def train(root: Path, out: Path, config: dict, device: str = "auto") -> dict:
    seed_all(config["seed"])
    torch.set_num_threads(config["threads"])
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    ds = SurfaceDataset(root, "train", config, augment=True)
    val = SurfaceDataset(root, "val", config)
    loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    vloader = DataLoader(val, batch_size=config["batch_size"], num_workers=0)
    counts = np.zeros(5)
    for _, y, _ in SurfaceDataset(root, "train", config):
        counts += np.bincount(y.numpy().ravel(), minlength=5)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1))
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
    model = SurfaceUNet(config["base_channels"]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    amp = config["amp"] and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    out.mkdir(parents=True, exist_ok=True)
    best, history, start = -1, [], time.perf_counter()
    for epoch in range(config["epochs"]):
        model.train()
        total = 0.0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if amp else "cpu", enabled=amp):
                logits = model(x)
                # Flatten to N,C to avoid CUDA's nondeterministic 2-D NLL reduction.
                loss = loss_fn(logits.permute(0, 2, 3, 1).reshape(-1, 5), y.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            total += loss.item() * len(x)
        metrics = score(model, vloader, device)
        row = {"epoch": epoch + 1, "loss": total / len(ds), "val_miou": metrics["miou"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["val_miou"] > best:
            best = row["val_miou"]
            torch.save(
                {
                    "state_dict": model.cpu().state_dict(),
                    "config": config,
                    "epoch": epoch + 1,
                    "val_miou": best,
                },
                out / "best.pt",
            )
            model.to(device)
    result = {
        "device": device,
        "seconds": time.perf_counter() - start,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_val_miou": best,
        "history": history,
        "config": config,
        "class_counts": counts.tolist(),
        "pretrained": False,
    }
    (out / "training.json").write_text(json.dumps(result, indent=2))
    return result
