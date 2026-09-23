"""One cross-platform command interface; run from repository root."""

import argparse
import json
from pathlib import Path

from aerosurface.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "data",
            "train",
            "evaluate",
            "export",
            "parity",
            "tensorrt",
            "benchmark",
            "demo",
            "smoke",
        ],
    )
    parser.add_argument("--config", type=Path, default=Path("configs/edge.json"))
    parser.add_argument("--data", type=Path, default=Path("data/procedural"))
    parser.add_argument("--run", type=Path, default=Path("models/edge"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint, onnx_path = args.run / "best.pt", args.run / "model.onnx"
    if args.command == "data":
        from simulation.procedural import generate

        result = generate(args.data, config)
    elif args.command == "train":
        from training.train import train

        result = train(args.data, args.run, config, args.device)
    elif args.command == "evaluate":
        from aerosurface.experiments import evaluate

        result = evaluate(checkpoint, args.data, args.output or Path("benchmarks/evaluation.json"))
        result = {"samples": result["samples"], "miou": result["raw_argmax"]["miou"]}
    elif args.command == "export":
        from aerosurface.experiments import export

        result = export(checkpoint, onnx_path)
    elif args.command == "parity":
        from aerosurface.experiments import parity

        result = parity(
            checkpoint, onnx_path, args.data, args.output or Path("benchmarks/onnx_parity.json")
        )
    elif args.command == "tensorrt":
        from aerosurface.experiments import export
        from aerosurface.runtime import build_engine

        export(checkpoint, args.run / "model_fp16.onnx", half=True)
        result = {
            p: build_engine(
                onnx_path if p == "fp32" else args.run / "model_fp16.onnx",
                args.run / f"model_{p}.engine",
            )
            for p in ["fp32", "fp16"]
        }
        (args.run / "tensorrt_build.json").write_text(json.dumps(result, indent=2))
    elif args.command == "benchmark":
        from aerosurface.benchmark import benchmark

        result = benchmark(
            checkpoint,
            onnx_path,
            args.data,
            args.output or Path("benchmarks"),
            args.iterations,
            args.warmup,
        )
        result = {
            r["backend"]: {k: r[k] for k in ["p50_ms", "p95_ms", "fps", "test_miou"]}
            for r in result["results"]
        }
    elif args.command == "demo":
        from aerosurface.experiments import demo

        result = demo(checkpoint, onnx_path, args.data, args.output or Path("artifacts/demo"))
    else:
        import tempfile

        from aerosurface.experiments import demo, export, parity
        from simulation.procedural import generate
        from training.train import train

        with tempfile.TemporaryDirectory(prefix="aerosurface_") as tmp:
            root = Path(tmp)
            config.update(train_samples=8, val_samples=2, test_samples=9, epochs=1)
            generate(root / "data", config)
            train(root / "data", root / "model", config, args.device)
            export(root / "model/best.pt", root / "model/model.onnx")
            result = parity(
                root / "model/best.pt",
                root / "model/model.onnx",
                root / "data",
                root / "parity.json",
            )
            demo(root / "model/best.pt", root / "model/model.onnx", root / "data", root / "demo")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
