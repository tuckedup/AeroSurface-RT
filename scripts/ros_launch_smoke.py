#!/usr/bin/env python3
"""Run the actual launch graph and require overlay frames and OK diagnostics."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def main():
    root = Path.cwd()
    os.environ["ROS_DOMAIN_ID"] = "74"
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    rclpy.init()
    node = rclpy.create_node("aerosurface_launch_observer")
    frames, diagnostics = [], []
    sub = node.create_subscription(
        Image, "/aerosurface/overlay", frames.append, qos_profile_sensor_data
    )
    diag = node.create_subscription(DiagnosticArray, "/diagnostics", diagnostics.append, 10)
    log = (root / "artifacts/ros/launch.log").open("w")
    process = subprocess.Popen(
        [
            "ros2",
            "launch",
            "aerosurface_perception",
            "demo.launch.py",
            f"model:={root / 'models/edge/model.onnx'}",
            f"images:={root / 'data/procedural'}",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and len(frames) < 20:
            if process.poll() is not None:
                raise RuntimeError("Launch exited before producing frames")
            rclpy.spin_once(node, timeout_sec=0.1)
        assert len(frames) >= 20, "Launch did not produce 20 overlay frames"
        assert any(s.level == s.OK for d in diagnostics for s in d.status)
        assert all(m.width == 160 and m.height == 128 and m.encoding == "rgb8" for m in frames)
        result = {
            "status": "PASS",
            "overlay_frames": len(frames),
            "OK_diagnostics": True,
            "launch": "aerosurface_perception demo.launch.py",
        }
        (root / "benchmarks/ros_launch.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        node.destroy_subscription(sub)
        node.destroy_subscription(diag)
        node.destroy_node()
        rclpy.shutdown()
        log.close()


if __name__ == "__main__":
    main()
