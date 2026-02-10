#!/usr/bin/env python3

"""Launch file for Handpose Estimation Pipeline with Theta Camera."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description."""
    
    # Launch arguments
    model_file_path_arg = DeclareLaunchArgument(
        'model_file_path',
        default_value='/workspaces/isaac_ros-dev/models/handpose_nchw.onnx',
        description='Path to the ONNX model file'
    )

    engine_file_path_arg = DeclareLaunchArgument(
        'engine_file_path',
        default_value='/workspaces/isaac_ros-dev/models/handpose_nchw.plan',
        description='Path to the TensorRT engine file'
    )
    
    score_threshold_arg = DeclareLaunchArgument(
        'score_threshold',
        default_value='0.05',
        description='Minimum confidence score for detections'
    )
    
    # Theta camera source
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
    
    # DNN Image Encoder
    encoder_node = ComposableNode(
        name='dnn_image_encoder',
        package='isaac_ros_dnn_image_encoder',
        plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
        remappings=[
            ('image', '/image_raw'),
            ('encoded_tensor', '/tensor_view'),
        ],
        parameters=[{
            'input_image_width': 1920,
            'input_image_height': 960,
            'network_image_width': 224,
            'network_image_height': 224,
            'image_mean': [0.5, 0.5, 0.5],
            'image_stddev': [0.5, 0.5, 0.5],
            'num_blocks': 40,
            'enable_padding': False,
        }]
    )
    
    # TensorRT inference
    tensorrt_node = ComposableNode(
        name='tensorrt_node',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        remappings=[
            ('tensor_pub', '/tensor_view'),
            ('tensor_sub', '/tensor_output'),
        ],
        parameters=[{
            'model_file_path': LaunchConfiguration('model_file_path'),
            'engine_file_path': LaunchConfiguration('engine_file_path'),
            'input_binding_names': ['input_tensor'],
            'input_tensor_names': ['input_tensor'],
            'output_binding_names': ['Identity', 'Identity_1', 'Identity_2', 'Identity_3'],
            'output_tensor_names': ['landmarks', 'handedness', 'score', 'hand_presence'],
            'force_engine_update': False,
            'verbose': True,
        }]
    )
    
    # Composable Node Container
    inference_container = ComposableNodeContainer(
        name='inference_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[encoder_node, tensorrt_node],
        output='screen',
    )
    
    # Handpose decoder
    decoder_node = Node(
        package='isaac_ros_gestures',
        executable='handpose_decoder',
        name='handpose_decoder',
        remappings=[
            ('tensor_output', '/tensor_output'),
        ],
        parameters=[{
            'score_threshold': LaunchConfiguration('score_threshold'),
            'frame_id': 'theta_camera',
        }],
        output='screen',
    )
    
    # Debug visualizer
    visualizer_node = Node(
        package='isaac_ros_gestures',
        executable='tensor_visualizer',
        name='tensor_visualizer',
        remappings=[
            ('tensor_pub', '/tensor_pub'),
        ],
        output='screen'
    )

    return LaunchDescription([
        model_file_path_arg,
        engine_file_path_arg,
        score_threshold_arg,
        theta_src_node,
        inference_container,
        decoder_node,
        visualizer_node
    ])
