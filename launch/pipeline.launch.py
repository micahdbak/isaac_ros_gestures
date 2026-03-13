#!/usr/bin/env python3

"""Launch file for Handpose Estimation Pipeline with Theta Camera."""

from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description."""
    
    model_file_path = '/workspaces/isaac_ros-dev/src/isaac_ros_gestures/yolo26l-pose-hand.onnx'
    engine_file_path1 = '/workspaces/isaac_ros-dev/src/isaac_ros_gestures/yolo26l-pose-hand.plan'
    engine_file_path2 = '/workspaces/isaac_ros-dev/src/isaac_ros_gestures/yolo26l-pose-hand2.plan'
    
    # Theta camera source
    theta_src_node = Node(
        package='isaac_ros_gestures',
        executable='theta_uvc_src',
        name='theta_uvc_src',
        remappings=[
            ('image_raw', '/image_raw'),
        ],
        output='screen',
    )
    
    # Video folder source/collector (hot-swappable with theta_src_node)
    video_collector_node = Node(
        package='isaac_ros_gestures',
        executable='video_collector_node',
        name='video_collector_node',
        parameters=[{
            'width': 1920,
            'height': 960,
            'frame_id': 'theta_camera',
            'video_dir': '/workspaces/isaac_ros-dev/recordings/right',
            'output_dir': '/workspaces/isaac_ros-dev/training/right',
        }],
        output='screen',
    )
    
    # subscribes to /image_raw and publishes to /tensor_view_handbox
    handbox_encoder_node = ComposableNode(
        name='dnn_image_encoder_handbox',
        package='isaac_ros_dnn_image_encoder',
        plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
        remappings=[
            ('image', '/image_raw'),
            ('encoded_tensor', '/tensor_view_handbox'),
        ],
        parameters=[{
            'input_image_width': 1920,
            'input_image_height': 960,
            'network_image_width': 640,
            'network_image_height': 640,
            'image_mean': [0.0, 0.0, 0.0],
            'image_stddev': [1.0, 1.0, 1.0],
            'enable_padding': False,
            'keep_aspect_ratio': True,
            'crop_mode': 'CENTER',
        }]
    )

    # subscribes to /tensor_view_handbox and publishes to /tensor_output_handbox
    handbox_tensorrt_node = ComposableNode(
        name='tensorrt_node_handbox',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        remappings=[
            ('tensor_pub', '/tensor_view_handbox'),
            ('tensor_sub', '/tensor_output_handbox'),
        ],
        parameters=[{
            'model_file_path': model_file_path,
            'engine_file_path': engine_file_path1,
            'input_binding_names': ['images'],
            'input_tensor_names': ['input_tensor'],
            'output_binding_names': ['output0'],
            'output_tensor_names': ['tensor_output'],
            'force_engine_update': False,
            'verbose': True,
        }]
    )

    # subscribes to /tensor_output_handbox and publishes to /image_cropped and /image_roi
    handbox_decoder_node = Node(
        package='isaac_ros_gestures',
        executable='handbox_decoder',
        name='handbox_decoder',
        remappings=[
            ('image_raw', '/image_raw'),
            ('tensor_output_handbox', '/tensor_output_handbox'),
            ('image_cropped', '/image_cropped'),
            ('image_roi', '/image_roi'),
        ],
        output='screen',
    )
    
    encoder_node = ComposableNode(
        name='dnn_image_encoder',
        package='isaac_ros_dnn_image_encoder',
        plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
        remappings=[
            ('image', '/image_cropped'),
            ('encoded_tensor', '/tensor_view'),
        ],
        parameters=[{
            'input_image_width': 640,
            'input_image_height': 640,
            'network_image_width': 640,
            'network_image_height': 640,
            'image_mean': [0.0, 0.0, 0.0],
            'image_stddev': [1.0, 1.0, 1.0],
            'enable_padding': False,
        }]
    )
    
    tensorrt_node = ComposableNode(
        name='tensorrt_node',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        remappings=[
            ('tensor_pub', '/tensor_view'),
            ('tensor_sub', '/tensor_output'),
        ],
        parameters=[{
            'model_file_path': model_file_path,
            'engine_file_path': engine_file_path2,
            'input_binding_names': ['images'],
            'input_tensor_names': ['input_tensor'],
            'output_binding_names': ['output0'],
            'output_tensor_names': ['tensor_output'],
            'force_engine_update': False,
            'verbose': True,
        }]
    )
    
    inference_container = ComposableNodeContainer(
        name='inference_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[handbox_encoder_node, handbox_tensorrt_node, encoder_node, tensorrt_node],
        output='screen',
    )
    
    handpose_decoder_node = Node(
        package='isaac_ros_gestures',
        executable='handpose_decoder',
        name='handpose_decoder',
        remappings=[
            ('tensor_output', '/tensor_output'),
        ],
        output='screen',
    )

    return LaunchDescription([
        inference_container,

        #theta_src_node,
        video_collector_node,
        handbox_decoder_node,
        handpose_decoder_node,
    ])
