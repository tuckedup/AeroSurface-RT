param([string]$Python = '.\.venv\Scripts\python.exe', [switch]$TensorRT, [switch]$Native)
$ErrorActionPreference = 'Stop'
foreach ($step in @('data','train','export','parity','evaluate','demo')) {
    & $Python -m aerosurface.cli $step
    if ($LASTEXITCODE -ne 0) { throw "Failed: $step" }
}
if ($TensorRT) {
    & $Python -m aerosurface.cli tensorrt
    if ($LASTEXITCODE -ne 0) { throw 'TensorRT build failed' }
}
& $Python -m aerosurface.cli benchmark
if ($LASTEXITCODE -ne 0) { throw 'Benchmark failed' }
if ($Native) {
    & $Python scripts/fetch_ort.py
    if ($LASTEXITCODE -ne 0) { throw 'SDK download failed' }
    cmake -S . -B build -A x64 "-DONNXRUNTIME_ROOT=$PWD/third_party/onnxruntime-win-x64-1.19.2"
    if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed' }
    cmake --build build --config Release
    if ($LASTEXITCODE -ne 0) { throw 'C++ build failed' }
    ctest --test-dir build -C Release --output-on-failure
    if ($LASTEXITCODE -ne 0) { throw 'C++ tests failed' }
    .\build\cpp\Release\aerosurface_infer.exe models/edge/model.onnx artifacts/demo/input.ppm artifacts/demo/cpp 160 128
    if ($LASTEXITCODE -ne 0) { throw 'Native demo failed' }
    & $Python scripts/check_cpp_parity.py
    if ($LASTEXITCODE -ne 0) { throw 'Native parity failed' }
}
& $Python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
