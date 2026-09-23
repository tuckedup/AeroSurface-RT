# Measured benchmark report

Actual local execution on 2026-09-07; no projected results.

## Edge model

66,725 parameters; 271,909 byte FP32 ONNX; 160×128 RGB, batch 1, five logits per pixel. Two CPU inference threads.

| Runtime | Test mIoU | Mean ms | p50 ms | p95 ms | Serialized FPS |
|---|---:|---:|---:|---:|---:|
| pytorch_cpu_fp32 | 0.7561 | 15.019 | 14.696 | 19.530 | 66.6 |
| onnxruntime_cpu_fp32 | 0.7561 | 9.852 | 9.510 | 12.895 | 101.5 |
| pytorch_cuda_fp32 | 0.7561 | 1.537 | 1.413 | 2.270 | 650.6 |
| tensorrt_fp32 | 0.7561 | 0.633 | 0.560 | 1.067 | 1579.4 |
| tensorrt_fp16 | 0.7562 | 0.640 | 0.581 | 1.157 | 1562.2 |

30 warmup + 200 measured serialized calls per backend. Input cycles over nine held-out conditions. Timing begins with an already preprocessed host NCHW array and ends with host logits; GPU input/output copies and synchronization are included. Disk IO, preprocessing, softmax/ROI, ROS transport, visualization, model loading and engine build are excluded. FPS is 1000/mean ms, not concurrent throughput or full robot-loop frequency.

FP32 explicitly disables TF32. FP16 uses a typed half-precision ONNX graph. Accuracy is separately evaluated on all 64 test images using raw argmax; confidence-filtered ROI metrics are a different output. Test labels are synthetic procedural data, not Isaac or aircraft imagery.

## Numerical validation

PyTorch/ORT: 12 inputs, max absolute logit error 0.00042725, argmax agreement 1.00000000. Asserted rtol=0.0002, atol=0.0001.

| Runtime | Max absolute logit error | Relative L2 error | Argmax agreement |
|---|---:|---:|---:|
| pytorch_cpu_fp32 | 0.00000000 | 0.00000000 | 1.00000000 |
| onnxruntime_cpu_fp32 | 0.00042725 | 0.00000018 | 1.00000000 |
| pytorch_cuda_fp32 | 0.00054932 | 0.00000025 | 1.00000000 |
| tensorrt_fp32 | 0.00042725 | 0.00000028 | 1.00000000 |
| tensorrt_fp16 | 2.27227783 | 0.00117234 | 0.99992405 |

TensorRT FP16 has some large absolute differences on large-magnitude logits; it is not FP32-identical. The benchmark gate uses relative L2 ≤0.005 for FP16 (≤0.0001 FP32) and argmax agreement ≥0.99. Confidence calibration and all possible threshold flips are unvalidated.

## Larger baseline

265,637 parameters and 1,067,574 ONNX bytes. Same dataset and 12-epoch budget; selection on nominal validation only.

| Runtime | Test mIoU | p50 ms | p95 ms |
|---|---:|---:|---:|
| pytorch_cpu_fp32 | 0.5867 | 35.760 | 43.618 |
| onnxruntime_cpu_fp32 | 0.5867 | 25.164 | 33.846 |
| pytorch_cuda_fp32 | 0.5867 | 1.748 | 2.563 |

The larger model did not improve held-out stress accuracy. Both runs are short and single-seed; this does not establish that greater capacity is intrinsically worse. TensorRT baseline engines were not built.

## Native C++ and measurement limits

Windows C++ ORT preprocessing + inference + postprocessing: mean 5.839 ms, p50 5.689 ms, p95 6.619 ms, 100 iterations after 20 warmups on one frame. Overlay/file output excluded. This is a separate scope and run, so do not interpret the difference against Python as a controlled language speedup.

The shared RTX 4060 Laptop GPU ran under WDDM with other applications active. CPU utilization and process RSS are recorded per backend in JSON. CPU percent is normalized to one logical core and can exceed 100%. PyTorch peak allocation excludes driver/context/TensorRT-owned memory, so it is not total GPU memory. Device-wide GPU utilization is N/A; no utilization percentage was inferred from timing. There is no hard real-time scheduling or isolated thermal/power control. ROS integration timing is reported separately, not substituted for a steady-state ROS benchmark.

Run `python -m aerosurface.cli benchmark --iterations 200 --warmup 30` to reproduce the Python comparison. Engines must first be built locally to include TensorRT. Raw latency samples, versions and checkpoint hashes are in `benchmarks/results.json`.

## ROS2 message round trips

Separately measured under Ubuntu 24.04 WSL2 / ROS2 Jazzy: 100 serialized calls after 10 warmups; mean 8.056 ms, p50 7.976 ms, p95 8.934 ms. Timing starts at the Python RGB publisher and ends when all six output images from the C++ node arrive. Includes DDS, preprocessing, CPU inference, region processing, overlay and output copies. Disk loading is excluded. This is a different platform and boundary from the native Windows model benchmarks above. Raw samples: benchmarks/ros_results.json.
