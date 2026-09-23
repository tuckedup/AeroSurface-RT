# Environment audit

Audit performed 2026-09-07. Personal paths, process lists and credentials are intentionally omitted.

| Component | Observed state |
|---|---|
| Native OS | Windows 11 Home, build 10.0.26200 |
| CPU | AMD Ryzen 9 8945HS, 8 cores / 16 logical processors |
| RAM | 16,380,862,464 bytes; only about 0.46–0.75 GB available during initial audit |
| Storage | About 54.6 GB free on workspace volume initially |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB VRAM, WDDM |
| Driver | 581.80; driver advertises CUDA 13.0 compatibility |
| CUDA toolkit | nvcc 11.8.89 |
| Training runtime | Python 3.10.16, PyTorch 2.5.1, CUDA runtime 11.8, cuDNN 9.1 |
| Default Python | 3.14.7, no ML packages; not used for experiments |
| ONNX / ORT | ONNX 1.16.2 / ONNX Runtime 1.19.2 |
| TensorRT | Python TensorRT 11.2.1.2 with its CUDA 13 libraries; builder executed successfully |
| Native compiler | MSVC 19.43.34808; CMake 3.31.6; Windows SDK 10.0.26100.0 |
| ROS2 at initial audit | Absent on Windows and WSL |
| WSL | Existing Ubuntu 24.04.1, WSL2 Linux 5.15.167.4, GCC available; initially stopped |
| Docker | No docker executable found in Windows or WSL; images not built |
| Isaac Sim / Isaac ROS | Not found in inspected installations/Python; no generation or NITROS run |
| Git initially | Empty directory, not a Git repository; no user code to preserve |

The project-local `.venv` initially inherited the existing CUDA environment without modifying it.
During execution, that shared installation changed to CPU-only PyTorch 2.14.0. Final validation
therefore uses CUDA PyTorch 2.5.1+cu118 and NumPy 2.2.2 wheels installed locally. The local NumPy
wheel also removes an OpenMP collision with the inherited numerical stack; no duplicate-runtime
override was used. Other dependencies remain inherited. Use a clean virtualenv for an independent setup.
pytest and Ruff were installed locally. Official ONNX Runtime C++ SDKs are kept under ignored
`third_party/`. ROS2 Jazzy and build tools were subsequently installed in the existing Ubuntu WSL
instance using `scripts/setup_ros_ubuntu.sh`; execution status is recorded in `docs/validation.md`.

Initial Windows hardware queries and MSVC SDK access required expanded sandbox access. Subsequent
builds used the actual installed compiler. WSL installation adds packages to that existing Ubuntu
instance; it does not alter the Windows Python environment.

Small images, two CPU inference threads, zero data-loader subprocesses and sequential GPU experiments
limit pressure on this shared laptop. Other applications were active; this is not an isolated
real-time performance certification. Driver CUDA, toolkit CUDA, PyTorch CUDA and TensorRT CUDA are
different version domains; success here does not establish arbitrary cross-version compatibility.
