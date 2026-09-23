#!/usr/bin/env python3
"""Live ROS2 C++ inference, failure behavior and rosbag storage/replay regression.

Run after sourcing Jazzy and this workspace; no PyTorch needed in ROS Python.
"""

import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import rclpy
import rosbag2_py
from diagnostic_msgs.msg import DiagnosticArray
from PIL import Image
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import Image as ImageMsg


def main():
    root = Path.cwd()
    output = root / "artifacts/ros"
    output.mkdir(parents=True, exist_ok=True)
    os.environ["ROS_DOMAIN_ID"] = "73"
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    log = (output / "node.log").open("w")
    process = subprocess.Popen(
        [
            "ros2",
            "run",
            "aerosurface_perception",
            "perception_node",
            "--ros-args",
            "-p",
            f"model_path:={root / 'models/edge/model.onnx'}",
            "-p",
            "benchmark_mode:=true",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    rclpy.init()
    node = rclpy.create_node("aerosurface_integration")
    publisher = node.create_publisher(ImageMsg, "/camera/image_raw", qos_profile_sensor_data)
    received, diagnostics = {}, []
    subscriptions = []

    def on_image(name, message):
        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        received.setdefault(stamp, {})[name] = message

    for name in ["mask", "sandable", "avoid", "defect", "protected", "overlay"]:
        subscriptions.append(
            node.create_subscription(
                ImageMsg,
                f"/aerosurface/{name}",
                lambda msg, key=name: on_image(key, msg),
                qos_profile_sensor_data,
            )
        )
    subscriptions.append(
        node.create_subscription(DiagnosticArray, "/diagnostics", diagnostics.append, 10)
    )

    def spin_until(predicate, seconds=15):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("C++ ROS node exited; inspect artifacts/ros/node.log")
            rclpy.spin_once(node, timeout_sec=0.05)
            if predicate():
                return
        raise TimeoutError("ROS graph did not meet test condition")

    def send(message, expected=6):
        message.header.stamp = node.get_clock().now().to_msg()
        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        publisher.publish(message)
        spin_until(lambda: len(received.get(stamp, {})) >= expected)
        return received[stamp]

    try:
        spin_until(lambda: publisher.get_subscription_count() > 0)
        # Give subscriber discovery time to complete in both directions.
        spin_until(
            lambda: all(
                node.count_publishers(f"/aerosurface/{key}") > 0
                for key in ["mask", "sandable", "overlay"]
            )
        )
        rgb = Image.open(root / "artifacts/demo/input.ppm").convert("RGB")
        message = ImageMsg()
        message.header.frame_id = "camera_optical_frame"
        message.width, message.height = rgb.size
        message.encoding, message.step = "rgb8", rgb.width * 3
        message.data = rgb.tobytes()
        first = send(message)
        for name in ["mask", "sandable", "avoid", "defect", "protected"]:
            expected = np.array(Image.open(root / f"artifacts/demo/{name}.png"))
            actual = np.array(first[name].data, np.uint8).reshape(128, 160)
            np.testing.assert_array_equal(actual, expected)
            assert first[name].header.frame_id == message.header.frame_id
            Image.fromarray(actual).save(output / f"{name}.png")
        Image.fromarray(np.array(first["overlay"].data, np.uint8).reshape(128, 160, 3)).save(
            output / "overlay.png"
        )
        # Storage-backed replay twice. Re-stamp because the node rejects stale sensor data.
        with tempfile.TemporaryDirectory(prefix="aerosurface_bag_") as tmp:
            uri = str(Path(tmp) / "regression")
            writer = rosbag2_py.SequentialWriter()
            writer.open(
                rosbag2_py.StorageOptions(uri=uri, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""),
            )
            writer.create_topic(
                rosbag2_py.TopicMetadata(
                    id=0,
                    name="/camera/image_raw",
                    type="sensor_msgs/msg/Image",
                    serialization_format="cdr",
                )
            )
            for i in range(3):
                writer.write(
                    "/camera/image_raw", serialize_message(message), 1000000000 + i * 100000000
                )
            del writer
            replay_frames = 0
            for _ in range(2):
                reader = rosbag2_py.SequentialReader()
                reader.open(
                    rosbag2_py.StorageOptions(uri=uri, storage_id="sqlite3"),
                    rosbag2_py.ConverterOptions("", ""),
                )
                while reader.has_next():
                    _, data, _ = reader.read_next()
                    result = send(deserialize_message(data, ImageMsg))
                    for name in ["mask", "sandable", "avoid", "defect", "protected"]:
                        assert bytes(result[name].data) == bytes(first[name].data)
                    replay_frames += 1
        # End-to-end serialized ROS round trips, including all six image outputs.
        for _ in range(10):
            send(message)
        roundtrips = []
        for _ in range(100):
            begin = time.perf_counter_ns()
            send(message)
            roundtrips.append((time.perf_counter_ns() - begin) / 1e6)
        ros_benchmark = {
            "scope": "Python publish to receipt of all six C++ ROS image outputs; serialized",
            "platform": "Ubuntu 24.04 WSL2, ROS2 Jazzy, ORT CPU, 2 inference threads",
            "warmup": 10,
            "iterations": 100,
            "batch_size": 1,
            "shape": [1, 3, 128, 160],
            "mean_ms": float(np.mean(roundtrips)),
            "p50_ms": float(np.percentile(roundtrips, 50)),
            "p95_ms": float(np.percentile(roundtrips, 95)),
            "latencies_ms": roundtrips,
        }
        (root / "benchmarks/ros_results.json").write_text(json.dumps(ros_benchmark, indent=2))
        # Padded BGR input validates ROS stride/channel conversion.
        rgb_array = np.array(rgb)
        padded = np.zeros((128, 160 * 3 + 7), np.uint8)
        padded[:, :480] = rgb_array[:, :, ::-1].reshape(128, 480)
        message.encoding, message.step, message.data = "bgr8", 487, padded.tobytes()
        result = send(message)
        assert bytes(result["mask"].data) == bytes(first["mask"].data)
        # Unsupported encoding must emit unknown/all-avoid and an ERROR diagnostic.
        message.encoding = "32FC1"
        result = send(message, expected=5)
        assert not any(result["mask"].data) and not any(result["sandable"].data)
        assert all(value == 255 for value in result["avoid"].data)
        spin_until(lambda: any(s.level == s.ERROR for d in diagnostics for s in d.status))
        # Stale frame rejection separately from encoding validation.
        message.encoding = "bgr8"
        message.header.stamp.sec = 1
        message.header.stamp.nanosec = 0
        publisher.publish(message)
        spin_until(lambda: len(received.get((1, 0), {})) >= 5)
        assert not any(received[(1, 0)]["sandable"].data)
        spin_until(
            lambda: any("Camera timeout" in s.message for d in diagnostics for s in d.status)
        )
        latencies = [
            float(k.value)
            for d in diagnostics
            for s in d.status
            if s.level == s.OK
            for k in s.values
            if k.key == "callback_ms"
        ]
        result = {
            "status": "PASS",
            "runtime": "ROS2 Jazzy C++ ONNX Runtime CPU under WSL2",
            "checks": [
                "five exact Python/native masks",
                "overlay publication",
                "header preservation",
                "padded BGR conversion",
                "invalid encoding fail closed",
                "stale frame fail closed",
                "ERROR diagnostics",
                "camera dropout watchdog",
                "sqlite3 rosbag replay equality",
            ],
            "replay_frames": replay_frames,
            "callback_ms": latencies,
            "latency_note": "Integration observations, not steady-state benchmark",
        }
        (root / "benchmarks/ros_integration.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        log.close()


if __name__ == "__main__":
    main()
