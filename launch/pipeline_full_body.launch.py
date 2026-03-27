#!/usr/bin/env python3

"""Launch file for Handpose Estimation Pipeline with Theta Camera."""

from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description."""
    
    bbox_model_file_path = '/workspaces/isaac_ros-dev/src/isaac_ros_gestures/yolo26n-pose-fullbody.onnx'
    hand_model_file_path = '/workspaces/isaac_ros-dev/src/isaac_ros_gestures/yolo26s-pose-hands.onnx'
    bbox_engine_file_path = '/workspaces/isaac_ros-dev/src/isaac_ros_gestures/yolo26n-pose-fullbody.plan'
    hand_engine_file_path = '/workspaces/isaac_ros-dev/src/isaac_ros_gestures/yolo26s-pose-hands.plan'
    classifier_model_file_path= '/workspaces/isaac_ros-dev/src/isaac_ros_gestures/hand_classifier.onnx'
    classifier_engine_file_path= '/workspaces/isaac_ros-dev/src/isaac_ros_gestures/hand_classifier.plan'

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
    image_gate_node = Node(
        package='isaac_ros_gestures',
        executable='image_gate',
        name='image_gate',
        parameters=[{
            'input_topic': '/image_raw',
            'output_topic': '/image_raw_gated',
            'button_topic': '/arduino_buttons',
            'default_enabled': False,
        }],
        output='screen',
    )

    # subscribes to /image_raw and publishes to /tensor_view_handbox
    handbox_encoder_node = ComposableNode(
        name='dnn_image_encoder_handbox',
        package='isaac_ros_dnn_image_encoder',
        plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
        remappings=[
            ('image', '/image_raw_gated'),
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
            'keep_aspect_ratio': False,
            #'crop_mode': 'CENTER',
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
            'model_file_path': bbox_model_file_path,
            'engine_file_path': bbox_engine_file_path,
            'input_binding_names': ['images'],
            'input_tensor_names': ['input_tensor'],
            'output_binding_names': ['output0'],
            'output_tensor_names': ['tensor_output'],
            'force_engine_update': False,
            'verbose': True,
        }]
    )

    # subscribes to /tensor_output_handbox and publishes to /image_cropped and /image_roi
    full_body_pose_decoder_node = Node(
        package='isaac_ros_gestures',
        executable='full_body_pose_decoder',
        name='full_body_pose_decoder',
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
            'model_file_path': hand_model_file_path,
            'engine_file_path': hand_engine_file_path,
            'input_binding_names': ['images'],
            'input_tensor_names': ['input_tensor'],
            'output_binding_names': ['output0'],
            'output_tensor_names': ['tensor_output'],
            'force_engine_update': False,
            'verbose': True,
        }]
    )
    classifier_tensorrt_node = ComposableNode(
        name='classifier_tensorrt_node',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        remappings=[
            ('tensor_pub', '/gesture_input_tensor'),
            ('tensor_sub', '/classifier_tensor_output'),
        ],
        parameters=[{
            'model_file_path': classifier_model_file_path,
            'engine_file_path': classifier_engine_file_path,
            'input_binding_names': ['body', 'hand'],
            'input_tensor_names': ['body', 'hand'],
            'output_binding_names': ['logits'],
            'output_tensor_names': ['logits'],
            'force_engine_update': False,
            'verbose': True,
        }]
    )
    inference_container = ComposableNodeContainer(
        name='inference_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[handbox_encoder_node, handbox_tensorrt_node, encoder_node, tensorrt_node,classifier_tensorrt_node],
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

    handbox_tensor_view_viz_node = Node(
        package='isaac_ros_gestures',
        executable='handbox_tensor_view_viz',
        name='handbox_tensor_view_viz',
        remappings=[
            ('tensor_view_handbox', '/tensor_view_handbox'),
            ('handbox_detector_input_image', '/handbox_detector_input_image'),
        ],
        output='screen',
    )
    session_collector_node = Node(
        package='isaac_ros_gestures',
        executable='session_collector_node',
        name='session_collector_node',
        parameters=[{
            'button_topic': '/arduino_buttons',
            'marker_topic': '/pose_markers',
            'image_topic': '/image_cropped',
            'save_dir': '/workspaces/isaac_ros-dev/src/isaac_ros_gestures/data_27/null',
            'video_fps': 20.0,
            'video_codec': 'mp4v',
        }],
        output='screen',
    )
    classifier_collector_node = Node(
        package='isaac_ros_gestures',
        executable='classifier_collector_node',
        name='classifier_collector_node',
        parameters=[{
            'button_topic': '/arduino_buttons',
            'hand_marker_topic': '/pose_markers',
            'body_marker_topic': '/fullbody_pose_markers',
            'tensor_topic': '/gesture_input_tensor',
            'sequence_length': 64,
            'hand_landmarks': 21,
            'body_landmarks': 8,

            # IMPORTANT:
            # replace with the exact 8 body keypoint indices used during training
            'body_indices': [5, 6, 8, 10,11,12],
        }],
        output='screen',
    )

    # NEW: decoder node for classifier output
    classifier_output_decoder_node = Node(
        package='isaac_ros_gestures',
        executable='classifier_output_decoder',
        name='classifier_output_decoder',
        parameters=[{
            'input_topic': '/classifier_tensor_output',
            'output_topic': '/classifier_output',
        }],
        output='screen',
    )

    return LaunchDescription([
        inference_container,
        image_gate_node,
        # theta_src_node,
        # video_collector_node,
        full_body_pose_decoder_node,
        handpose_decoder_node,
        session_collector_node,
        #classifier_collector_node,
        #classifier_output_decoder_node,
    ])