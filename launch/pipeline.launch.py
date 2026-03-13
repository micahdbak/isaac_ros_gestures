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
        default_value='/workspaces/isaac_ros-dev/src/isaac_ros_gestures/yolo26s-pose-hands.onnx',
        description='Path to the ONNX model file'
    )

    engine_file_path_arg = DeclareLaunchArgument(
        'engine_file_path',
        default_value='/workspaces/isaac_ros-dev/src/isaac_ros_gestures/yolo26s-pose-hands.plan',
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
    
    # Video folder source/collector (hot-swappable with theta_src_node)
    #video_collector_node = Node(
    #    package='isaac_ros_gestures',
    #    executable='video_collector_node',
    #    name='video_collector_node',
    #    parameters=[{
    #        'width': 1920,
    #        'height': 960,
    #        'frame_id': 'theta_camera',
    #        'video_dir': '/workspaces/isaac_ros-dev/recordings/right',
    #        'output_dir': '/workspaces/isaac_ros-dev/training/right',
    #    }],
    #    output='screen',
    #)
    
    # --- Full-frame hand detector (YOLO26 TensorRT) -> handbox_decoder -> /image_cropped ---
    handdet_encoder_node = ComposableNode(
        name='dnn_image_encoder_handdet',
        package='isaac_ros_dnn_image_encoder',
        plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
        remappings=[
            ('image', '/image_raw'),
            ('encoded_tensor', '/tensor_view_handdet'),
        ],
        parameters=[{
            'input_image_width': 640,
            'input_image_height': 640,
            'network_image_width': 640,
            'network_image_height': 640,
            'image_mean': [0.0, 0.0, 0.0],
            'image_stddev': [1.0, 1.0, 1.0],
            'enable_padding': True,
        }]
    )

    handdet_tensorrt_node = ComposableNode(
        name='tensorrt_node_handdet',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        remappings=[
            ('tensor_pub', '/tensor_view_handdet'),
            ('tensor_sub', '/tensor_output_handdet'),
        ],
        parameters=[{
            'model_file_path': LaunchConfiguration('model_file_path'),
            'engine_file_path': LaunchConfiguration('engine_file_path'),
            'input_binding_names': ['images'],
            'input_tensor_names': ['input_tensor'],
            'output_binding_names': ['output0'],
            'output_tensor_names': ['tensor_output'],
            'force_engine_update': False,
            'verbose': True,
        }]
    )

    handdet_container = ComposableNodeContainer(
        name='handdet_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[handdet_encoder_node, handdet_tensorrt_node],
        output='screen',
    )

    handbox_decoder_node = Node(
        package='isaac_ros_gestures',
        executable='handbox_decoder',
        name='handbox_decoder',
        remappings=[
            ('image_raw', '/image_raw'),
            ('tensor_output_handdet', '/tensor_output_handdet'),
            ('image_cropped', '/image_cropped'),
        ],
        parameters=[{
            'score_threshold': LaunchConfiguration('score_threshold'),
            'network_size': 640.0,
            'padding_ratio': 2.0,
        }],
        output='screen',
    )
    
    # --- Original /image_cropped -> TensorRT -> handpose_decoder pipeline ---
    # DNN Image Encoder
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
            'enable_padding': True,
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
            'input_binding_names': ['images'],
            'input_tensor_names': ['input_tensor'],
            'output_binding_names': ['output0'],
            'output_tensor_names': ['tensor_output'],
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
        }],
        output='screen',
    )
    
    # Static TF Publisher (map -> theta_camera) for RViz
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'theta_camera'],
        output='screen'
    )

    return LaunchDescription([
        model_file_path_arg,
        engine_file_path_arg,
        score_threshold_arg,
        
        # Uncomment video_collector_node and comment theta_src_node to swap
        theta_src_node,
        #video_collector_node,
        
        handdet_container,
        handbox_decoder_node,
        inference_container,
        decoder_node,
        static_tf_node
    ])
