"""Batch-one serialized host-to-host latency: copies included, startup excluded."""

import csv
import datetime
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime
import psutil
import torch

from aerosurface.data import SurfaceDataset
from aerosurface.metrics import confusion, summarize
from aerosurface.model import load_model
from aerosurface.runtime import TensorRTRuntime, ort_session


def timed(fn, inputs: list, warmup: int, iterations: int) -> dict:
    for i in range(warmup):
        fn(inputs[i % len(inputs)])
    process = psutil.Process()
    start_cpu = sum(process.cpu_times()[:2])
    values = []
    for i in range(iterations):
        start = time.perf_counter_ns()
        fn(inputs[i % len(inputs)])
        values.append((time.perf_counter_ns() - start) / 1e6)
    cpu_seconds = sum(process.cpu_times()[:2]) - start_cpu
    elapsed = sum(values) / 1000
    return {
        "warmup": warmup,
        "iterations": iterations,
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "fps": 1000 / float(np.mean(values)),
        "process_cpu_percent_one_core": 100 * cpu_seconds / elapsed,
        "process_rss_bytes": process.memory_info().rss,
        "latencies_ms": values,
    }


@torch.inference_mode()
def benchmark(
    checkpoint: Path,
    onnx_path: Path,
    data: Path,
    output: Path,
    iterations: int = 200,
    warmup: int = 30,
) -> dict:
    if iterations < 20 or warmup < 1:
        raise ValueError("Use >=20 timed iterations and >=1 warmup")
    model, config = load_model(checkpoint)
    torch.set_num_threads(config["threads"])
    # FP32 comparison must not silently use CUDA TF32 convolution math.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    ds = SurfaceDataset(data, "test", config)
    inputs = [ds[i][0][None].numpy() for i in range(min(9, len(ds)))]
    reference = [model(torch.from_numpy(x)).numpy() for x in inputs]
    session = ort_session(onnx_path, config["threads"])
    factories = {
        "pytorch_cpu_fp32": lambda: lambda x: model(torch.from_numpy(x)).numpy(),
        "onnxruntime_cpu_fp32": lambda: lambda x: session.run(None, {"rgb": x})[0],
    }
    if torch.cuda.is_available():

        def gpu_factory():
            gpu, _ = load_model(checkpoint, "cuda")
            return lambda x: gpu(torch.from_numpy(x).cuda()).cpu().numpy()

        factories["pytorch_cuda_fp32"] = gpu_factory
    for precision in ["fp32", "fp16"]:
        engine = onnx_path.with_name(f"model_{precision}.engine")
        if engine.exists():
            factories[f"tensorrt_{precision}"] = lambda p=engine: TensorRTRuntime(p)
    rows = []
    for name, factory in factories.items():
        print(f"Benchmarking {name}", flush=True)
        fn = factory()
        outputs = [fn(x) for x in inputs]
        max_error = max(float(np.max(np.abs(a - b))) for a, b in zip(reference, outputs))
        agreement = float(
            np.mean([np.mean(a.argmax(1) == b.argmax(1)) for a, b in zip(reference, outputs)])
        )
        relative_l2 = max(
            float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))
            for a, b in zip(reference, outputs)
        )
        tolerance = 0.005 if name.endswith("fp16") else 0.0001
        if relative_l2 > tolerance or agreement < 0.99:
            raise AssertionError(f"{name}: parity failed: error={max_error}, agreement={agreement}")
        cm = np.zeros((5, 5), dtype=np.int64)
        for x, y, _ in ds:
            cm += confusion(y.numpy(), fn(x[None].numpy()).argmax(1)[0])
        if "cuda" in name or "tensorrt" in name:
            torch.cuda.reset_peak_memory_stats()
        row = {
            "backend": name,
            "status": "MEASURED",
            "batch_size": 1,
            "shape": list(inputs[0].shape),
            "max_absolute_error": max_error,
            "argmax_agreement": agreement,
            "test_miou": summarize(cm)["miou"],
            "relative_l2_error": relative_l2,
            "relative_l2_tolerance": tolerance,
            **timed(fn, inputs, warmup, iterations),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated()
            if "cuda" in name or "tensorrt" in name
            else None,
            "gpu_utilization_percent": None,
        }
        # PyTorch allocator telemetry excludes TensorRT-owned allocations.
        rows.append(row)
        del fn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {
        "measured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "method": "serialized host NCHW input to host logits; preprocessing excluded",
        "hardware": {
            "os": platform.platform(),
            "cpu": platform.processor(),
            "logical_cpus": psutil.cpu_count(),
            "ram_bytes": psutil.virtual_memory().total,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_pytorch": torch.version.cuda,
            "onnxruntime": onnxruntime.__version__,
        },
        "threads": config["threads"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "onnx_bytes": onnx_path.stat().st_size,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "results": rows,
        "unavailable": {
            "tensorrt_int8": "NOT RUN: no calibration experiment",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2))
    fields = [
        "backend",
        "status",
        "test_miou",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "fps",
        "max_absolute_error",
        "argmax_agreement",
        "process_cpu_percent_one_core",
    ]
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return result
