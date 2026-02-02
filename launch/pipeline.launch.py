#!/usr/bin/env python3
# Copyright 2024 Isaac ROS YOLOv8 Pose
# SPDX-License-Identifier: MIT

"""
Launch file for YOLOv8 Pose Estimation Pipeline with Theta Camera.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description for YOLOv8 Pose pipeline."""
    
    # Declare launch arguments
    engine_file_path_arg = DeclareLaunchArgument(
        'engine_file_path',
        default_value='/workspaces/isaac_ros-dev/models/yolov8s-pose.plan',
        description='Path to the TensorRT engine file (.plan)'
    )
    
    score_threshold_arg = DeclareLaunchArgument(
        'score_threshold',
        default_value='0.25',
        description='Minimum confidence score for detections'
    )
    
    nms_threshold_arg = DeclareLaunchArgument(
        'nms_threshold',
        default_value='0.45',
        description='IoU threshold for Non-Maximum Suppression'
    )
    
    # Theta camera source node (GStreamer-based)
    theta_src_node = Node(
        package='isaac_ros_yolov8_pose',
        executable='theta_uvc_src',
        name='theta_uvc_src',
        remappings=[
            ('image_raw', '/image_raw'),
        ],
        parameters=[{
            'width': 1920,
            'height': 1080,
            'framerate': 30,
            'frame_id': 'theta_camera',
            'output_topic': 'image_raw',
        }],
        output='screen',
    )
    
    # TensorRT inference node (ComposableNode for performance)
    tensorrt_node = ComposableNode(
        name='tensorrt_node',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        remappings=[
            ('image', '/image_raw'),
            ('tensor_pub', '/raw_tensor_output'),
        ],
        parameters=[{
            'engine_file_path': LaunchConfiguration('engine_file_path'),
            'output_binding_names': ['output0'],
            'output_tensor_names': ['output0'],
            'input_tensor_names': ['images'],
            'input_binding_names': ['images'],
            'force_engine_update': False,
            'verbose': False,
        }]
    )
    
    # Container for composable nodes
    tensorrt_container = ComposableNodeContainer(
        name='tensorrt_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[tensorrt_node],
        output='screen',
    )
    
    # Custom Python decoder node
    decoder_node = Node(
        package='isaac_ros_yolov8_pose',
        executable='yolov8_pose_decoder',
        name='yolov8_pose_decoder',
        remappings=[
            ('tensor_pub', '/raw_tensor_output'),
            ('pose_markers', '/pose_markers'),
        ],
        parameters=[{
            'score_threshold': LaunchConfiguration('score_threshold'),
            'nms_threshold': LaunchConfiguration('nms_threshold'),
            'num_keypoints': 17,
            'input_width': 640,
            'input_height': 640,
            'frame_id': 'theta_camera',
        }],
        output='screen',
    )
    
    return LaunchDescription([
        # Launch arguments
        engine_file_path_arg,
        score_threshold_arg,
        nms_threshold_arg,
        # Nodes - Theta camera source first, then inference pipeline
        theta_src_node,
        tensorrt_container,
        decoder_node,
    ])
