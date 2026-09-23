# Reproduction and runtime compatibility

## Windows / Python

Use Python 3.10 and a CUDA-capable PyTorch installation for the GPU results. The authoring run
initially inherited an existing environment; final validation uses local CUDA PyTorch and NumPy
wheels after shared packages changed. For an independent setup use a clean virtualenv:

```powershell
# Replace python with your Python 3.10 executable if needed.
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m aerosurface.cli smoke
powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1
```

CPU-only users may install PyTorch from its CPU index; data/training/evaluation/ONNX/demo/tests
work without a GPU. The smoke test trains a tiny disposable dataset for one epoch and validates
the full export/inference path; it does not overwrite the portfolio model.

`configs/edge.json` and `configs/baseline.json` record experiments. `models/*/training.json` contains
loss/validation trajectories and counts. `benchmarks/` holds small measured reports, hashes and
latency samples. `.gitignore` excludes datasets, model weights, engines, build trees and SDKs.
Deterministic algorithms were required; 2-D CUDA NLL loss was replaced with its flattened form
after actual execution rejected the nondeterministic kernel. Bitwise reproduction across different
hardware/library versions is not promised.

## C++ standalone

```powershell
.\.venv\Scripts\python.exe scripts/fetch_ort.py
cmake -S . -B build -A x64 "-DONNXRUNTIME_ROOT=$PWD/third_party/onnxruntime-win-x64-1.19.2"
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
.\build\cpp\Release\aerosurface_infer.exe models/edge/model.onnx artifacts/demo/input.ppm artifacts/demo/cpp 160 128
.\.venv\Scripts\python.exe scripts/check_cpp_parity.py
```

Linux: use `python3 scripts/fetch_ort.py`, configure a fresh build directory and set
`ONNXRUNTIME_ROOT=$PWD/third_party/onnxruntime-linux-x64-1.19.2`. The standalone parser accepts
binary P6 RGB PPM without comments; ROS supports ordinary image messages. C++ core tests can build
without any ORT SDK by omitting its CMake root.

## TensorRT

The measured Windows setup uses TensorRT 11.2.1.2's Python/CUDA 13 wheel libraries and an NVIDIA
driver that supports CUDA 13; the training runtime separately uses CUDA 11.8. Reproduce against
NVIDIA's documented requirements for your chosen release rather than mixing system DLL versions.

```bash
python -m aerosurface.cli tensorrt
python -m aerosurface.cli benchmark --iterations 200 --warmup 30
```

This builds explicit FP32 and FP16 typed graphs with a 256 MiB TensorRT workspace limit, disables
TF32 for FP32 and validates outputs before timing. INT8 calibration/quantization is not implemented.
Equivalent TensorRT 11 CLI builds, if `trtexec` is installed:

```bash
trtexec --onnx=models/edge/model.onnx --saveEngine=models/edge/model_fp32.engine --noTF32
trtexec --onnx=models/edge/model_fp16.onnx --saveEngine=models/edge/model_fp16.engine --noTF32
```

Those CLI commands are documentation alternatives, not the commands used for the measured engines.
TensorRT 11 removed legacy precision-enabling flags: see NVIDIA's
[migration guide](https://docs.nvidia.com/deeplearning/tensorrt/11.2.1/api/migration/tensorrt-10x-to-11x-python-api-patterns.html).
Never load an engine from an untrusted source; build it locally from the project graph.

## ROS2 Jazzy

Ubuntu 24.04 is the selected ROS platform. `scripts/setup_ros_ubuntu.sh` explicitly installs ROS
Jazzy and build dependencies with root access, following the
[official installation source](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Installation/Ubuntu-Install-Debs.rst).

```bash
source /opt/ros/jazzy/setup.bash
python3 scripts/fetch_ort.py
colcon --log-base artifacts/ros_build_log build --base-paths ros2_ws/src \
  --build-base artifacts/ros_build --install-base artifacts/ros_install \
  --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT="$PWD/third_party/onnxruntime-linux-x64-1.19.2"
source artifacts/ros_install/setup.bash
export LD_LIBRARY_PATH="$PWD/third_party/onnxruntime-linux-x64-1.19.2/lib:${LD_LIBRARY_PATH:-}"
ros2 launch aerosurface_perception demo.launch.py \
  model:="$PWD/models/edge/model.onnx" images:="$PWD/data/procedural"
# In a separate sourced shell:
python3 scripts/ros_integration.py
```

The integration harness uses isolated ROS domain 73 and localhost discovery. Replay uses actual
sqlite3 rosbag storage but re-stamps frames to current time; it tests deterministic content, not
simulated-clock timing. Real camera/bag use should align ROS clock types and `use_sim_time`.

## Docker and Isaac separation

```bash
docker build -f docker/Dockerfile.ml -t aerosurface-ml .
docker run --rm aerosurface-ml
docker build -f docker/Dockerfile.ros -t aerosurface-ros .
docker run --rm --network host -v "$PWD/models:/workspace/models:ro" \
  -v "$PWD/data:/workspace/data:ro" aerosurface-ros
```

Docker is absent on the authoring host, so both images are provided but not built or run. The ML
image is CPU-only and intentionally excludes NVIDIA/ROS/Isaac dependencies. GPU containers need
NVIDIA Container Toolkit, `--gpus all`, and a base image matching the chosen driver/runtime support
matrix. Isaac Sim requires its own supported NVIDIA distribution and EULA; it is not bundled into
either image. No claim is made that this 8 GB laptop can comfortably run Isaac and training together.
