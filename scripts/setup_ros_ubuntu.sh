#!/usr/bin/env bash
# Ubuntu 24.04 only. Run explicitly as root to install ROS2 Jazzy dependencies.
set -euo pipefail
source /etc/os-release
[[ ${VERSION_CODENAME} == noble ]] || { echo 'Ubuntu 24.04 required'; exit 1; }
[[ $(id -u) == 0 ]] || { echo 'Run with sudo'; exit 1; }
export DEBIAN_FRONTEND=noninteractive
version=$(curl --fail --silent --show-error --location https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')
curl --fail --location --output /tmp/aerosurface-ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${version}/ros2-apt-source_${version}.noble_all.deb"
dpkg -i /tmp/aerosurface-ros2-apt-source.deb
apt-get update
apt-get install -y --no-install-recommends ros-jazzy-ros-base ros-jazzy-diagnostic-msgs \
  ros-jazzy-sensor-msgs python3-colcon-common-extensions python3-pil python3-numpy \
  cmake build-essential clang-format
