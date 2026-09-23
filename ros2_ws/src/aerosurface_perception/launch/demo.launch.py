from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("model", description="Absolute path to exported FP32 model.onnx"),
            DeclareLaunchArgument(
                "images", description="Absolute path to procedural image directory"
            ),
            Node(
                package="aerosurface_perception",
                executable="image_publisher.py",
                parameters=[{"image_directory": LaunchConfiguration("images")}],
            ),
            Node(
                package="aerosurface_perception",
                executable="perception_node",
                output="screen",
                parameters=[{"model_path": LaunchConfiguration("model"), "benchmark_mode": True}],
            ),
        ]
    )
