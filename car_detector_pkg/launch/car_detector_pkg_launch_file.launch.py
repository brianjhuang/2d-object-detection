from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='car_detector_pkg',
            executable='car_detector',
            output='screen'),
    ])