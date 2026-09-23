"""Regenerate human-readable reports strictly from measured machine-readable results."""

import json
import shutil
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def main():
    docs, assets = Path("docs"), Path("assets")
    docs.mkdir(exist_ok=True)
    assets.mkdir(exist_ok=True)
    evaluation = read("benchmarks/evaluation.json")
    benchmark = read("benchmarks/results.json")
    baseline = read("benchmarks/baseline/results.json")
    parity = read("benchmarks/onnx_parity.json")
    real = read("artifacts/real_domain/real_domain.json")
    shutil.copy2("artifacts/real_domain/real_domain.json", "benchmarks/real_domain.json")
    shutil.copy2("models/edge/training.json", "benchmarks/training_edge.json")
    shutil.copy2("models/baseline/training.json", "benchmarks/training_baseline.json")
    shutil.copy2("artifacts/demo/contact_sheet.png", assets / "demo.png")
    shutil.copy2("artifacts/demo/cpp_benchmark.json", "benchmarks/cpp_results.json")
    cpp = read("benchmarks/cpp_results.json")
    rows = benchmark["results"]
    report = [
        "# Measured benchmark report",
        "",
        "Actual local execution on 2026-09-07; no projected results.",
        "",
        "## Edge model",
        "",
        (
            f"{benchmark['parameters']:,} parameters; {benchmark['onnx_bytes']:,} byte FP32 ONNX; "
            "160×128 RGB, batch 1, five logits per pixel. Two CPU inference threads."
        ),
        "",
        "| Runtime | Test mIoU | Mean ms | p50 ms | p95 ms | Serialized FPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['backend']} | {row['test_miou']:.4f} | {row['mean_ms']:.3f} | "
            f"{row['p50_ms']:.3f} | {row['p95_ms']:.3f} | {row['fps']:.1f} |"
        )
    report += [
        "",
        (
            "30 warmup + 200 measured serialized calls per backend. Input cycles over nine held-out "
            "conditions. Timing begins with an already preprocessed host NCHW array and ends with host logits; "
            "GPU input/output copies and synchronization are included. Disk IO, preprocessing, softmax/ROI, "
            "ROS transport, visualization, model loading and engine build are excluded. FPS is 1000/mean ms, "
            "not concurrent throughput or full robot-loop frequency."
        ),
        "",
        (
            "FP32 explicitly disables TF32. FP16 uses a typed half-precision ONNX graph. "
            "Accuracy is separately evaluated on all 64 test images using raw argmax; confidence-filtered ROI "
            "metrics are a different output. Test labels are synthetic procedural data, not Isaac or aircraft imagery."
        ),
        "",
        "## Numerical validation",
        "",
        (
            f"PyTorch/ORT: {parity['samples']} inputs, max absolute logit error {parity['max_absolute_error']:.8f}, "
            f"argmax agreement {parity['mean_argmax_agreement']:.8f}. Asserted rtol={parity['rtol']}, atol={parity['atol']}."
        ),
        "",
        "| Runtime | Max absolute logit error | Relative L2 error | Argmax agreement |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['backend']} | {row['max_absolute_error']:.8f} | "
            f"{row['relative_l2_error']:.8f} | {row['argmax_agreement']:.8f} |"
        )
    report += [
        "",
        (
            "TensorRT FP16 has some large absolute differences on large-magnitude logits; it is not "
            "FP32-identical. The benchmark gate uses relative L2 ≤0.005 for FP16 (≤0.0001 FP32) and "
            "argmax agreement ≥0.99. Confidence calibration and all possible threshold flips are unvalidated."
        ),
        "",
        "## Larger baseline",
        "",
        (
            f"{baseline['parameters']:,} parameters and {baseline['onnx_bytes']:,} ONNX bytes. "
            "Same dataset and 12-epoch budget; selection on nominal validation only."
        ),
        "",
        "| Runtime | Test mIoU | p50 ms | p95 ms |",
        "|---|---:|---:|---:|",
    ]
    for row in baseline["results"]:
        report.append(
            f"| {row['backend']} | {row['test_miou']:.4f} | {row['p50_ms']:.3f} | {row['p95_ms']:.3f} |"
        )
    report += [
        "",
        (
            "The larger model did not improve held-out stress accuracy. Both runs are short and single-seed; "
            "this does not establish that greater capacity is intrinsically worse. TensorRT baseline engines were not built."
        ),
        "",
        "## Native C++ and measurement limits",
        "",
        (
            f"Windows C++ ORT preprocessing + inference + postprocessing: mean {cpp['mean_ms']:.3f} ms, "
            f"p50 {cpp['p50_ms']:.3f} ms, p95 {cpp['p95_ms']:.3f} ms, {cpp['iterations']} iterations after "
            f"{cpp['warmup']} warmups on one frame. Overlay/file output excluded. This is a separate scope and run, "
            "so do not interpret the difference against Python as a controlled language speedup."
        ),
        "",
        (
            "The shared RTX 4060 Laptop GPU ran under WDDM with other applications active. CPU utilization and "
            "process RSS are recorded per backend in JSON. CPU percent is normalized to one logical core and can "
            "exceed 100%. PyTorch peak allocation excludes driver/context/TensorRT-owned memory, so it is not total "
            "GPU memory. Device-wide GPU utilization is N/A; no utilization percentage was inferred from timing. "
            "There is no hard real-time scheduling or isolated thermal/power control. ROS integration timing is "
            "reported separately, not substituted for a steady-state ROS benchmark."
        ),
        "",
        (
            "Run `python -m aerosurface.cli benchmark --iterations 200 --warmup 30` to reproduce the Python "
            "comparison. Engines must first be built locally to include TensorRT. Raw latency samples, versions "
            "and checkpoint hashes are in `benchmarks/results.json`."
        ),
    ]
    if Path("benchmarks/ros_results.json").exists():
        ros = read("benchmarks/ros_results.json")
        report += [
            "",
            "## ROS2 message round trips",
            "",
            (
                f"Separately measured under Ubuntu 24.04 WSL2 / ROS2 Jazzy: "
                f"{ros['iterations']} serialized calls after {ros['warmup']} warmups; "
                f"mean {ros['mean_ms']:.3f} ms, p50 {ros['p50_ms']:.3f} ms, "
                f"p95 {ros['p95_ms']:.3f} ms. Timing starts at the Python RGB publisher and "
                "ends when all six output images from the C++ node arrive. Includes DDS, "
                "preprocessing, CPU inference, region processing, overlay and output copies. "
                "Disk loading is excluded. This is a different platform and boundary from "
                "the native Windows model benchmarks above. Raw samples: benchmarks/ros_results.json."
            ),
        ]
    (docs / "benchmark_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    failure = [
        "# Failure analysis",
        "",
        (
            "These failures come from executed held-out evaluation. "
            "Representative input/truth/overlay/candidate frames are shown below."
        ),
        "",
        "![Measured demo](../assets/demo.png)",
        "",
        "| Test condition | mIoU | Defect recall |",
        "|---|---:|---:|",
    ]
    for condition, value in evaluation["slices"].items():
        recall = value["per_class"]["surface_defect"]["recall"]
        failure.append(f"| {condition} | {value['miou']:.4f} | {recall:.4f} |")
    roi = evaluation["roi"]
    failure += [
        "",
        "## Observed cases",
        "",
        "| Input / observed error | Probable cause | Severity | Mitigation / tested? |",
        "|---|---|---|---|",
        (
            "| Low light: protected fixture becomes fragmented, obstacles/defects confused; lowest slice mIoU | "
            "Color shortcuts and limited training exposure range | High: may admit prohibited pixels | "
            "Exposure augmentation, photometric checks and real data; not tested as a five-class mitigation |"
        ),
        (
            "| Strong stripe texture: defect/protected confusion | Synthetic texture competes with color cues | "
            "High | Texture diversity and real material priors; not tested |"
        ),
        (
            "| Blur: thin seam erodes/disappears and small defects lose boundaries | 160×128 sampling and local evidence loss | "
            "High | Higher resolution, boundary supervision and multi-frame confirmation; not tested |"
        ),
        (
            "| Small defects: missing/expanded few-pixel regions | Very few supervised pixels and imbalance | "
            "Medium/high | Native-resolution crops and recall-focused sampling; not tested |"
        ),
        (
            "| Real-only six-epoch baseline predicts no defects | Sparse positives, short training and random initialization | "
            "High | Longer training, defect-balanced crops and pretrained features; not tested |"
        ),
        (
            "| Synthetic-only real images: almost all defects missed | Severe material/geometry/color domain gap | "
            "High | Real fine-tuning was tested; see transfer report |"
        ),
        "",
        (
            f"Default candidate ROI still contains **{roi['false_sandable_pixels']:,} non-sandable-labelled pixels** "
            f"out of {roi['candidate_pixels']:,} candidates ({100 * roi['false_sandable_fraction']:.3f}%). "
            f"Coverage of true sandable pixels is {100 * roi['sandable_coverage']:.2f}%. "
            "Confidence and erosion do not certify safety. They exclude uncertain pixels and margins, but confident "
            "misclassification remains. Do not use this project to authorize sanding without independent geometric "
            "and process constraints."
        ),
        "",
        "## Execution failures that were fixed",
        "",
        (
            "- Strict deterministic training rejected CUDA 2-D NLL reduction. Flattening logits/targets used a "
            "supported deterministic reduction and the smoke run passed."
        ),
        (
            "- Trained ONNX parity exposed small near-zero FP32 cancellation differences. A documented mixed "
            "absolute/relative tolerance was used; argmax equality was independently checked."
        ),
        (
            "- Initial GPU/TensorRT FP32 validation caught default TF32 math. TF32 was disabled and engines rebuilt; "
            "true FP32 parity passed without relaxing its relative error gate."
        ),
        (
            "- Official KSDD2 contained unlabelled duplicate files. Only byte-identical duplicate copies were "
            "excluded; unexpected missing masks still fail loudly."
        ),
        (
            "- Native MSVC initially could not access its SDK under the sandbox. A permitted build with SDK "
            "access compiled the same code and passed CTest."
        ),
    ]
    (docs / "failure_analysis.md").write_text("\n".join(failure) + "\n", encoding="utf-8")

    transfer = [
        "# Executed binary transfer experiment",
        "",
        (
            "KolektorSDD2; CC BY-NC-SA 4.0. "
            "256 train / 64 validation / 1,004 official test images, one seed, six training epochs. "
            "This compares defect versus nondefect, never five-class sandability accuracy."
        ),
        "",
        "| Initialization/training | Defect IoU | Defect Dice | Precision | Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in real["results"].items():
        d = row["metrics"]["per_class"]["defect"]
        values = [
            "N/A" if d[k] is None else f"{d[k]:.4f}" for k in ["iou", "dice", "precision", "recall"]
        ]
        transfer.append(f"| {name} | " + " | ".join(values) + " |")
    transfer += [
        "",
        (
            "Synthetic pretraining helped this short fine-tuning run, but the scratch baseline collapsed "
            "to nondefect. Its zero recall must not be advertised as a competitive real-only baseline. "
            "The comparisons share an architecture, training subset, epoch budget and validation selection rule, "
            "but synthetic pretraining adds compute. There are no confidence intervals, repeated seeds, full "
            "training-set sweeps or pretrained industrial baseline. Accuracy at downsampled resolution is not "
            "native-resolution defect accuracy. See `docs/datasets.md` for label mapping and exact split policy."
        ),
        "",
        (
            "Raw confusion matrices, histories and train/validation identities are saved in "
            "`benchmarks/real_domain.json`. Adapted weights and real imagery remain ignored under `artifacts/`."
        ),
    ]
    (docs / "sim_to_real.md").write_text("\n".join(transfer) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
