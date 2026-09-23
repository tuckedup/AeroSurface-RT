# Executed validation and remaining boundaries

Validation date: 2026-09-07. This distinguishes actual execution from implementation only.

| Component | Result | Evidence |
|---|---|---|
| Procedural data | 296 synchronized RGB/mask/depth-proxy samples generated | data manifest and generator tests |
| Edge CUDA training | 12 epochs; best validation mIoU 0.9304 | `benchmarks/training_edge.json` |
| Larger CUDA baseline | 12 epochs; best validation mIoU 0.9299 | `benchmarks/training_baseline.json` |
| Held-out five-class evaluation | 64 images, nine stress conditions | `benchmarks/evaluation.json` |
| ONNX export/parity | Passed on 12 trained-model inputs; exact argmax agreement | `benchmarks/onnx_parity.json` |
| TensorRT FP32/FP16 | Both engines built, numerically validated and benchmarked | `benchmarks/results.json` |
| C++ Windows build | MSVC Release build succeeded | CMake build and CTest commands below |
| C++ unit/error tests | 2/2 CTest cases passed | core contracts and invalid-model exit |
| Python/native tests | 21/21 tests passed | `artifacts/pytest.xml` |
| Python lint/format | Ruff configured, checked and formatted | commands below |
| C++ formatting | clang-format applied and build repeated | `.clang-format` |
| Native parity | Five masks exactly match Python, including resized/padded cases | `benchmarks/cpp_parity.json` and native integration tests |
| Headless demo | Actual RGB, segmentation, overlay and ROI images saved | `assets/demo.png`, `artifacts/demo/` |
| ROS2 Linux build | Jazzy C++ package compiled in Ubuntu 24.04 WSL2 | `artifacts/ros_build_log/` |
| ROS live integration | Nine contract/failure/replay checks passed | `benchmarks/ros_integration.json` |
| ROS bag regression | Actual sqlite3 storage, six re-stamped replay frames; identical masks | `benchmarks/ros_integration.json` |
| ROS launch | Actual launch graph produced 20 overlays and OK diagnostics | `benchmarks/ros_launch.json` |
| ROS round trips | 100 calls after 10 warmups; all six outputs included | `benchmarks/ros_results.json` |
| Real-domain experiment | Three binary configurations evaluated on 1,004 official test images | `benchmarks/real_domain.json` |
| Isaac Sim | NOT RUN: simulator unavailable; recipe syntax and ID adapter tested | `simulation/isaac_generate.py`, adapter tests |
| Docker | NOT RUN: Docker unavailable | separate ML/ROS recipes provided |
| INT8 / NITROS | NOT IMPLEMENTED / NOT RUN | no claims or fabricated measurements |
| C++ TensorRT ROS backend | NOT IMPLEMENTED | measured TensorRT uses Python; C++ ROS uses ORT CPU |

## Final commands

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/pytest.xml
.\.venv\Scripts\python.exe -m aerosurface.cli smoke
.\.venv\Scripts\python.exe -m aerosurface.cli benchmark
.\.venv\Scripts\python.exe -m aerosurface.cli demo
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
.\build\cpp\Release\aerosurface_infer.exe models/edge/model.onnx artifacts/demo/input.ppm artifacts/demo/cpp 160 128
.\.venv\Scripts\python.exe scripts/check_cpp_parity.py
```

From the repository root in WSL, after installing the dependencies and building ROS:

```bash
bash scripts/run_ros_tests.sh
bash scripts/final_ros_check.sh
```

The WSL test wrapper preserves ROS library paths while adding ORT. A first manually composed
Windows-to-WSL command lost those paths; the checked-in wrapper corrected it. The ROS diagnostic
test also had to compare byte-valued levels against generated message constants rather than integer
literals. The node had already published the expected ERROR status; the original assertion was wrong.

The shared Python environment changed mid-session, so a local CUDA PyTorch and NumPy installation
was needed before final benchmark reproduction. The original benchmark is preserved in
`benchmarks/initial_results.json`; `benchmarks/results.json` is the final rerun. No code or packages
were committed to Git; Git was initialized and ignored/generated content audited. Real data, weights,
engines, build trees and SDK downloads are excluded from deliverables.
