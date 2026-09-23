#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
source /opt/ros/jazzy/setup.bash
source artifacts/ros_install/setup.bash
export LD_LIBRARY_PATH="$PWD/third_party/onnxruntime-linux-x64-1.19.2/lib:${LD_LIBRARY_PATH:-}"
python3 scripts/ros_integration.py
