#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
find cpp ros2_ws/src -type f \( -name '*.cpp' -o -name '*.hpp' \) -exec clang-format -i {} +
source /opt/ros/jazzy/setup.bash
colcon --log-base artifacts/ros_build_log build --base-paths ros2_ws/src \
  --build-base artifacts/ros_build --install-base artifacts/ros_install \
  --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT="$PWD/third_party/onnxruntime-linux-x64-1.19.2"
source artifacts/ros_install/setup.bash
export LD_LIBRARY_PATH="$PWD/third_party/onnxruntime-linux-x64-1.19.2/lib:${LD_LIBRARY_PATH:-}"
python3 scripts/ros_launch_smoke.py
