#!/usr/bin/env python3

"""Launch file for Theta camera streaming to ROS2 topic."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    theta_src_node = Node(
        package='isaac_ros_gestures',
        executable='theta_uvc_src',
        name='theta_uvc_src',
        parameters=[{
            'width': 1920,
            'height': 960,
            'frame_id': 'theta_camera',
        }],
        output='screen',
    )

    return LaunchDescription([
        theta_src_node
    ])
