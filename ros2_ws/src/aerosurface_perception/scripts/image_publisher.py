#!/usr/bin/env python3
"""Hardware-free ROS demo: loop through recorded RGB frames at 10 Hz."""

from pathlib import Path

import rclpy
from PIL import Image
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as ImageMsg


class ImagePublisher(Node):
    def __init__(self):
        super().__init__("aerosurface_image_publisher")
        directory = self.declare_parameter("image_directory", "data/procedural").value
        self.files = sorted(
            p for p in Path(directory).glob("test_*.png") if not p.stem.endswith("_mask")
        )
        if not self.files:
            raise ValueError("No test RGB PNG frames found")
        self.publisher = self.create_publisher(
            ImageMsg, "/camera/image_raw", qos_profile_sensor_data
        )
        self.index = 0
        self.timer = self.create_timer(0.1, self.publish_image)

    def publish_image(self):
        with Image.open(self.files[self.index % len(self.files)]) as frame:
            rgb = frame.convert("RGB")
        message = ImageMsg()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "camera_optical_frame"
        message.width, message.height = rgb.size
        message.encoding = "rgb8"
        message.step = message.width * 3
        message.data = rgb.tobytes()
        self.publisher.publish(message)
        self.index += 1


def main():
    rclpy.init()
    node = ImagePublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
