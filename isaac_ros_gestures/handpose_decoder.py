#!/usr/bin/env python3

"""Handpose Decoder Node for parsing MediaPipe output and publishing visualization."""

import numpy as np
import rclpy
from rclpy.node import Node
from isaac_ros_tensor_list_interfaces.msg import TensorList
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, PoseArray, Pose


# MediaPipe Hand Keypoint Names
HAND_KEYPOINT_NAMES = [
    'WRIST',
    'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
    'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
    'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
    'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
]

# Finger Color Mapping
FINGER_COLORS = {
    'wrist': (0, 1),
    'thumb': (1, 5),
    'index': (5, 9),
    'middle': (9, 13),
    'ring': (13, 17),
    'pinky': (17, 21),
}


class HandposeDecoder(Node):
    """Decodes MediaPipe handpose tensors and publishes visualization markers."""

    def __init__(self):
        super().__init__('handpose_decoder')
        
        # Parameters
        self.declare_parameter('score_threshold', 0.25)
        self.declare_parameter('num_keypoints', 21)
        self.declare_parameter('image_width', 1920.0)
        self.declare_parameter('image_height', 960.0)
        self.declare_parameter('frame_id', 'camera_link')
        
        self.score_threshold = self.get_parameter('score_threshold').value
        self.num_keypoints = self.get_parameter('num_keypoints').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.frame_id = self.get_parameter('frame_id').value
        
        self.roi_cache = {}
        
        # Subscribe to CameraInfo to track the bounding box (which contains our ROI and a Header)
        self.roi_sub = self.create_subscription(
            CameraInfo,
            'palm_roi',
            self.roi_callback,
            10
        )
        
        # Wrapper for TensorRT output
        self.tensor_sub = self.create_subscription(
            TensorList,
            'tensor_output',
            self.tensor_callback,
            10
        )
        
        # Visual markers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            'pose_markers',
            10
        )

    def roi_callback(self, msg: CameraInfo):
        """Cache incoming Regions of Interest by timestamp."""
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self.roi_cache[key] = msg.roi

    def tensor_callback(self, msg: TensorList):
        """Process incoming tensor data."""
        if len(msg.tensors) == 0:
            return

        # Locate tensors
        landmarks_tensor = None
        score_tensor = None

        #self.get_logger().info(f'{msg.tensors}')
        
        for tensor in msg.tensors:
            if tensor.name == 'landmarks':
                landmarks_tensor = tensor
            elif tensor.name == 'score':
                score_tensor = tensor
        
        # Fallback to indices
        if landmarks_tensor is None and len(msg.tensors) > 0:
            landmarks_tensor = msg.tensors[0]
        if score_tensor is None and len(msg.tensors) > 2:
            score_tensor = msg.tensors[2]
            
        if landmarks_tensor is None:
            return
            
        # Parse landmarks
        try:
            landmarks_data = np.frombuffer(bytes(landmarks_tensor.data), dtype=np.float32)
            landmarks_shape = tuple(d for d in landmarks_tensor.shape.dims)
            landmarks_data = landmarks_data.reshape(landmarks_shape)
        except Exception as e:
            self.get_logger().error(f'Failed to parse landmarks: {e}')
            return

        # Parse score
        score_data = None
        if score_tensor:
            try:
                score_data = np.frombuffer(bytes(score_tensor.data), dtype=np.float32)
                score_shape = tuple(d for d in score_tensor.shape.dims)
                score_data = score_data.reshape(score_shape)
            except Exception as e:
                self.get_logger().error(f'Failed to parse score: {e}')

        # Decode and publish
        hands = self.decode_predictions(landmarks_data, score_data)
        
        # Match ROI
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        matched_roi = self.roi_cache.get(key)
        
        # Memory constraint: Clean up the cache, explicitly dropping all older or matched ROIs
        current_time = key[0] * 1e9 + key[1]
        self.roi_cache = {
            k: v for k, v in self.roi_cache.items()
            if (k[0] * 1e9 + k[1]) > current_time
        }
        
        self.publish_markers(hands, msg.header, matched_roi)

    def decode_predictions(self, landmarks_tensor: np.ndarray, score_tensor: np.ndarray = None) -> list:
        """Decode predictions from tensors."""
        landmarks_tensor = np.squeeze(landmarks_tensor)
        
        keypoints = None
        if landmarks_tensor.size == self.num_keypoints * 3:
            keypoints = landmarks_tensor.reshape(self.num_keypoints, 3)
        else:
            return []
        
        # Determine confidence
        confidence = 0.0
        if score_tensor is not None:
             confidence = float(np.max(score_tensor))
        
        if confidence < self.score_threshold:
            return []
        
        return [{
            'keypoints': keypoints,
            'confidence': confidence
        }]

    def publish_markers(self, hands: list, header, roi):
        """Publish visualization markers."""
        marker_array = MarkerArray()
        
        # Delete previous markers
        delete_marker = Marker()
        delete_marker.header = header
        delete_marker.header.frame_id = self.frame_id
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        for hand_idx, hand in enumerate(hands):
            keypoints = hand['keypoints']
            confidence = hand['confidence']
            
            joints_marker = self.create_joints_marker(
                keypoints, hand_idx, header, confidence, roi
            )
            marker_array.markers.append(joints_marker)
        
        self.marker_pub.publish(marker_array)

    def create_joints_marker(
        self, keypoints: np.ndarray, hand_idx: int, header, confidence: float, roi
    ) -> Marker:
        """Create Marker for hand keypoints."""
        marker = Marker()
        marker.header = header
        marker.header.frame_id = self.frame_id
        marker.ns = 'hand_keypoints'
        marker.id = hand_idx
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        
        # Point size
        marker.scale.x = 0.02
        marker.scale.y = 0.02
        
        # Base DNN crop dimension
        dnn_dim = 224.0
        
        for kp_idx, kp in enumerate(keypoints):
            x_val = float(kp[0])
            y_val = float(kp[1])
            z_val = float(kp[2]) if len(kp) > 2 else 0.0
            
            pct_x = x_val
            pct_y = y_val
            pct_z = z_val
            
            # The model often outputs literal [0, 224] pixel coordinates!
            if abs(x_val) > 1.0 or abs(y_val) > 1.0:
                pct_x /= dnn_dim
                pct_y /= dnn_dim
                pct_z /= dnn_dim
            
            if roi is not None:
                # Calculate absolute pixel coordinates relative to the full image
                abs_x = roi.x_offset + pct_x * roi.width
                abs_y = roi.y_offset + pct_y * roi.height
                
                # Transform directly into a [-1, 1] space where 0,0 is the center of the total image
                x_val = (abs_x / self.image_width) * 2.0 - 1.0
                y_val = (abs_y / self.image_height) * 2.0 - 1.0
                
                # Scale Z equivalently to maintain spatial proportions
                z_val = pct_z * (roi.width / self.image_width)
            else:
                x_val = pct_x * 2.0 - 1.0
                y_val = pct_y * 2.0 - 1.0
                z_val = pct_z
            
            # Z offset
            z_offset = 1.0
            point = Point(x=x_val, y=y_val, z=z_val + z_offset)
            marker.points.append(point)
            
            # Color by finger
            color = self._get_finger_color(kp_idx)
            marker.colors.append(color)
        
        return marker

    def _get_finger_color(self, kp_idx: int) -> ColorRGBA:
        """Return color for finger index."""
        if kp_idx == 0:
            return ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        elif 1 <= kp_idx <= 4:
            return ColorRGBA(r=1.0, g=0.2, b=0.2, a=1.0)
        elif 5 <= kp_idx <= 8:
            return ColorRGBA(r=1.0, g=0.6, b=0.0, a=1.0)
        elif 9 <= kp_idx <= 12:
            return ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        elif 13 <= kp_idx <= 16:
            return ColorRGBA(r=0.0, g=1.0, b=0.4, a=1.0)
        else:
            return ColorRGBA(r=0.4, g=0.6, b=1.0, a=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = HandposeDecoder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
