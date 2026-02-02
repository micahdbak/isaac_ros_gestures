#!/usr/bin/env python3
# Copyright 2024 Isaac ROS YOLOv8 Pose
# SPDX-License-Identifier: Apache-2.0

"""
YOLOv8 Pose Decoder Node

Subscribes to raw tensors from TensorRTNode, parses YOLOv8-Pose output,
applies NMS, and publishes skeleton visualization markers.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from isaac_ros_tensor_list_interfaces.msg import TensorList
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point

# Try torch for fast NMS, fallback to OpenCV
try:
    import torch
    import torchvision
    USE_TORCH = True
except ImportError:
    import cv2
    USE_TORCH = False


# COCO skeleton connectivity (17 keypoints)
# Format: (from_keypoint, to_keypoint)
COCO_SKELETON = [
    (0, 1), (0, 2),      # Nose -> Left Eye, Nose -> Right Eye
    (1, 3), (2, 4),      # Left Eye -> Left Ear, Right Eye -> Right Ear
    (5, 6),              # Left Shoulder -> Right Shoulder
    (5, 7), (7, 9),      # Left Shoulder -> Left Elbow -> Left Wrist
    (6, 8), (8, 10),     # Right Shoulder -> Right Elbow -> Right Wrist
    (5, 11), (6, 12),    # Left/Right Shoulder -> Left/Right Hip
    (11, 12),            # Left Hip -> Right Hip
    (11, 13), (13, 15),  # Left Hip -> Left Knee -> Left Ankle
    (12, 14), (14, 16),  # Right Hip -> Right Knee -> Right Ankle
]

# Keypoint names for reference
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


class Yolov8PoseDecoder(Node):
    """
    ROS 2 Node for decoding YOLOv8-Pose tensor output.
    
    Subscribes to TensorList from TensorRTNode, decodes bounding boxes
    and keypoints, applies NMS, and publishes MarkerArray for visualization.
    """

    def __init__(self):
        super().__init__('yolov8_pose_decoder')
        
        # Declare parameters
        self.declare_parameter('score_threshold', 0.25)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('num_keypoints', 17)
        self.declare_parameter('input_width', 640)
        self.declare_parameter('input_height', 640)
        self.declare_parameter('frame_id', 'camera_link')
        
        # Get parameters
        self.score_threshold = self.get_parameter('score_threshold').value
        self.nms_threshold = self.get_parameter('nms_threshold').value
        self.num_keypoints = self.get_parameter('num_keypoints').value
        self.input_width = self.get_parameter('input_width').value
        self.input_height = self.get_parameter('input_height').value
        self.frame_id = self.get_parameter('frame_id').value
        
        # Calculate expected channels: 4 (box) + 1 (conf) + num_keypoints * 3
        self.expected_channels = 4 + 1 + self.num_keypoints * 3
        
        self.get_logger().info(
            f'Yolov8PoseDecoder initialized with score_threshold={self.score_threshold}, '
            f'nms_threshold={self.nms_threshold}, num_keypoints={self.num_keypoints}'
        )
        self.get_logger().info(f'Using {"PyTorch" if USE_TORCH else "OpenCV"} for NMS')
        
        # Subscriber for tensor input
        self.tensor_sub = self.create_subscription(
            TensorList,
            'tensor_pub',
            self.tensor_callback,
            10
        )
        
        # Publisher for visualization markers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            'pose_markers',
            10
        )
        
        self.marker_id = 0

    def tensor_callback(self, msg: TensorList):
        """Process incoming tensor data from TensorRTNode."""
        if len(msg.tensors) == 0:
            self.get_logger().warn('Received empty TensorList')
            return
        
        tensor = msg.tensors[0]
        
        # Convert bytes to numpy array
        try:
            data = np.frombuffer(bytes(tensor.data), dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f'Failed to parse tensor data: {e}')
            return
        
        # Expected shape: (1, 56, 8400) for YOLOv8-Pose
        # Reshape based on tensor dimensions
        expected_shape = tuple(tensor.shape)
        try:
            data = data.reshape(expected_shape)
        except ValueError as e:
            self.get_logger().error(
                f'Failed to reshape tensor. Expected shape {expected_shape}, '
                f'got {data.size} elements: {e}'
            )
            return
        
        # Process detections
        detections = self.decode_predictions(data)
        
        # Publish markers
        self.publish_markers(detections, msg.header)

    def decode_predictions(self, tensor: np.ndarray) -> list:
        """
        Decode YOLOv8-Pose predictions from raw tensor.
        
        Args:
            tensor: Raw output tensor of shape (1, 56, 8400)
            
        Returns:
            List of detections, each containing:
            - 'box': [x1, y1, x2, y2]
            - 'confidence': float
            - 'keypoints': array of shape (17, 3) with (x, y, visibility)
        """
        # Remove batch dimension and transpose: (1, 56, 8400) -> (8400, 56)
        predictions = tensor[0].T  # Shape: (8400, 56)
        
        # Extract confidence scores (index 4)
        confidences = predictions[:, 4]
        
        # Filter by confidence threshold
        mask = confidences > self.score_threshold
        filtered = predictions[mask]
        filtered_conf = confidences[mask]
        
        if len(filtered) == 0:
            return []
        
        # Extract boxes: (xc, yc, w, h) -> (x1, y1, x2, y2)
        boxes_xywh = filtered[:, :4]
        boxes_xyxy = self.xywh_to_xyxy(boxes_xywh)
        
        # Extract keypoints: columns 5-55 -> (N, 17, 3)
        keypoints = filtered[:, 5:5 + self.num_keypoints * 3]
        keypoints = keypoints.reshape(-1, self.num_keypoints, 3)
        
        # Apply NMS
        keep_indices = self.apply_nms(boxes_xyxy, filtered_conf)
        
        # Build detection results
        detections = []
        for idx in keep_indices:
            detections.append({
                'box': boxes_xyxy[idx],
                'confidence': filtered_conf[idx],
                'keypoints': keypoints[idx]
            })
        
        self.get_logger().debug(f'Detected {len(detections)} poses after NMS')
        return detections

    def xywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert boxes from (xc, yc, w, h) to (x1, y1, x2, y2) format."""
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        return xyxy

    def apply_nms(self, boxes: np.ndarray, scores: np.ndarray) -> list:
        """
        Apply Non-Maximum Suppression to filter overlapping boxes.
        
        Uses PyTorch if available, otherwise falls back to OpenCV.
        """
        if USE_TORCH:
            boxes_tensor = torch.from_numpy(boxes).float()
            scores_tensor = torch.from_numpy(scores).float()
            keep = torchvision.ops.nms(boxes_tensor, scores_tensor, self.nms_threshold)
            return keep.numpy().tolist()
        else:
            # OpenCV NMS expects boxes as list of [x, y, w, h]
            boxes_xywh = []
            for box in boxes:
                x1, y1, x2, y2 = box
                boxes_xywh.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
            
            indices = cv2.dnn.NMSBoxes(
                boxes_xywh,
                scores.tolist(),
                self.score_threshold,
                self.nms_threshold
            )
            return indices.flatten().tolist() if len(indices) > 0 else []

    def publish_markers(self, detections: list, header):
        """Publish visualization markers for detected poses."""
        marker_array = MarkerArray()
        
        # Clear previous markers
        delete_marker = Marker()
        delete_marker.header = header
        delete_marker.header.frame_id = self.frame_id
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        for det_idx, detection in enumerate(detections):
            keypoints = detection['keypoints']
            confidence = detection['confidence']
            
            # Create skeleton lines marker
            skeleton_marker = self.create_skeleton_marker(
                keypoints, det_idx, header, confidence
            )
            marker_array.markers.append(skeleton_marker)
            
            # Create keypoint spheres marker
            joints_marker = self.create_joints_marker(
                keypoints, det_idx, header, confidence
            )
            marker_array.markers.append(joints_marker)
        
        self.marker_pub.publish(marker_array)

    def create_skeleton_marker(
        self, keypoints: np.ndarray, det_idx: int, header, confidence: float
    ) -> Marker:
        """Create LINE_LIST marker for skeleton connections."""
        marker = Marker()
        marker.header = header
        marker.header.frame_id = self.frame_id
        marker.ns = 'skeleton'
        marker.id = det_idx * 2
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        
        # Scale for line width
        marker.scale.x = 0.01  # Line width in meters
        
        # Color based on confidence (green = high, red = low)
        marker.color = ColorRGBA(
            r=1.0 - confidence,
            g=confidence,
            b=0.2,
            a=0.8
        )
        
        # Add skeleton connections
        for start_idx, end_idx in COCO_SKELETON:
            start_kp = keypoints[start_idx]
            end_kp = keypoints[end_idx]
            
            # Check visibility (third value in keypoint)
            if start_kp[2] < 0.5 or end_kp[2] < 0.5:
                continue
            
            # Normalize coordinates to meters (assuming image coords)
            # Convert from pixel coordinates to normalized 3D space
            start_point = Point(
                x=float(start_kp[0]) / self.input_width,
                y=float(start_kp[1]) / self.input_height,
                z=0.0
            )
            end_point = Point(
                x=float(end_kp[0]) / self.input_width,
                y=float(end_kp[1]) / self.input_height,
                z=0.0
            )
            
            marker.points.append(start_point)
            marker.points.append(end_point)
        
        return marker

    def create_joints_marker(
        self, keypoints: np.ndarray, det_idx: int, header, confidence: float
    ) -> Marker:
        """Create POINTS marker for joint positions."""
        marker = Marker()
        marker.header = header
        marker.header.frame_id = self.frame_id
        marker.ns = 'joints'
        marker.id = det_idx * 2 + 1
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        
        # Scale for point size
        marker.scale.x = 0.02  # Point width
        marker.scale.y = 0.02  # Point height
        
        # Add visible keypoints
        for kp_idx, kp in enumerate(keypoints):
            if kp[2] < 0.5:  # Skip low visibility keypoints
                continue
            
            point = Point(
                x=float(kp[0]) / self.input_width,
                y=float(kp[1]) / self.input_height,
                z=0.0
            )
            marker.points.append(point)
            
            # Color keypoints based on body part
            if kp_idx < 5:  # Head keypoints
                color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=1.0)  # Yellow
            elif kp_idx < 11:  # Upper body
                color = ColorRGBA(r=0.0, g=0.8, b=1.0, a=1.0)  # Cyan
            else:  # Lower body
                color = ColorRGBA(r=1.0, g=0.4, b=0.7, a=1.0)  # Pink
            
            marker.colors.append(color)
        
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = Yolov8PoseDecoder()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
